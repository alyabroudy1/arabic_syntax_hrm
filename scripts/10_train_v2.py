#!/usr/bin/env python3
"""
Script 10: Train HRM-Grid Parser v2 on Arabic Syntax Grids
==========================================================

Architecture Overhaul:
    - Multi-Scale CharCNN / Morphological Highway
    - Variational Tree Manager (Mixture Prior + VAE + Goals)
    - 3-Layer Stacked PreNorm Transformer
    - Fast Worker HRM Grid Processor
    - TreeGNN Refinement
    - Differentiable Tree-CRF, Second order scoring, etc.

Usage:
    # Quick test (executes a dummy batch to ensure End-To-End compilation)
    python scripts/10_train_v2.py --quick-test
"""

import argparse
import time
import numpy as np
from pathlib import Path
import math

import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
from models.v2.parser import ArabicHRMGridParserV2, ParserConfig

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "arabic_syntax_grid"
OUTPUT_DIR = PROJECT_ROOT / "models" / "v2_arabic_syntax"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class SentenceLengthCurriculum:
    def __init__(self, max_epochs: int = 30, initial_max_len: int = 15, final_max_len: int = 128):
        self.max_epochs = max_epochs
        self.initial = initial_max_len
        self.final = final_max_len
    
    def get_max_length(self, epoch: int) -> int:
        progress = min(1.0, epoch / (self.max_epochs * 0.5))
        return int(self.initial + progress * (self.final - self.initial))

class ExponentialMovingAverage:
    def __init__(self, parameters, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.parameters = list(parameters)
        for i, p in enumerate(self.parameters):
            if p.requires_grad:
                self.shadow[i] = p.data.clone()
    
    def update(self):
        for i, p in enumerate(self.parameters):
            if p.requires_grad and i in self.shadow:
                self.shadow[i] = self.decay * self.shadow[i] + (1 - self.decay) * p.data

    def apply_shadow(self):
        for i, p in enumerate(self.parameters):
            if p.requires_grad and i in self.shadow:
                self.backup[i] = p.data.clone()
                p.data.copy_(self.shadow[i])

    def restore(self):
        for i, p in enumerate(self.parameters):
            if p.requires_grad and i in self.backup:
                p.data.copy_(self.backup[i])


class DummyArabicDataset(Dataset):
    """
    Since the current `train_grids` only have 8 categorical features, we mock the 
    morphological and subword pieces (char_ids, bpe_ids, etc.) deterministically 
    based on word_ids to allow network compilation, training, and scaling tests.
    """
    def __init__(self, size=100, max_len=32, device='cpu'):
        self.size = size
        self.max_len = max_len
        
        # We must limit randint so it does not exceed vocabulary bounds
        self.grids = torch.zeros((size, max_len, 8), dtype=torch.long, device=device)
        self.grids[:, :, 0] = torch.randint(1, 256, (size, max_len), device=device) # word_ids
        self.grids[:, :, 1] = torch.randint(1, 64, (size, max_len), device=device)  # pos_tags
        self.grids[:, :, 2] = torch.randint(1, 64, (size, max_len), device=device)  # morph
        
        self.masks = torch.ones((size, max_len), device=device)
        # Randomly pad the end
        for i in range(size):
            actual_len = torch.randint(5, max_len, (1,)).item()
            self.masks[i, actual_len:] = 0
            self.grids[i, actual_len:, :] = 0
            
        self.solutions = torch.zeros((size, max_len, 8), device=device)
        self.solutions[:, :, 3] = torch.randint(0, 4, (size, max_len)) # cases: 0-3 (None,Nom,Acc,Gen)
        self.solutions[:, :, 4] = torch.randint(0, max_len//2, (size, max_len)) # heads
        self.solutions[:, :, 5] = torch.randint(0, 29, (size, max_len)) # deprels: 29 unique in data
        
        # Root is at index 0
        self.masks[:, 0] = 1
        self.grids[:, 0, :] = 0
        self.solutions[:, 0, :] = 0
        # dep_mask: all valid tokens except root words (index 0 for dummy)
        self.dep_masks = self.masks.clone()
        self.dep_masks[:, 0] = 0
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        grid = self.grids[idx]
        sol = self.solutions[idx]
        mask = self.masks[idx]
        
        word_ids = grid[:, 0]
        pos_tags = grid[:, 1]
        
        W = self.max_len
        char_ids = (word_ids.unsqueeze(-1) + torch.arange(8, device=grid.device)).clamp(0, 299)
        bpe_ids = (word_ids.unsqueeze(-1) + torch.arange(4, device=grid.device)).clamp(0, 7999)
        root_ids = word_ids % 5000
        pattern_ids = word_ids % 200
        proclitic_ids = word_ids % 200
        enclitic_ids = word_ids % 100
        diac_ids = char_ids % 20
        
        # Mask out padded features
        char_ids = (char_ids * mask.unsqueeze(-1)).long()
        bpe_ids = (bpe_ids * mask.unsqueeze(-1)).long()
        
        return {
            'word_ids': word_ids.long(),
            'pos_tags': pos_tags.long(),
            'char_ids': char_ids,
            'bpe_ids': bpe_ids,
            'root_ids': root_ids.long(),
            'pattern_ids': pattern_ids.long(),
            'proclitic_ids': proclitic_ids.long(),
            'enclitic_ids': enclitic_ids.long(),
            'diac_ids': diac_ids.long(),
            'mask': mask.long(),
            'dep_mask': self.dep_masks[idx].long(),
            'heads': sol[:, 4].long(),
            'relations': sol[:, 5].long(),
            'cases': sol[:, 3].long()
        }


class ArabicRootExtractor:
    """
    Lightweight algorithmic Arabic root extractor.
    Strips known prefixes/suffixes and extracts the trilateral consonant skeleton.
    No external dependencies required.
    """
    # Common Arabic prefixes (proclitics + verb prefixes)
    PREFIXES = [
        'وال', 'فال', 'بال', 'كال', 'لل',
        'وب', 'ول', 'وي', 'فب', 'فل', 'في',
        'ال', 'لي', 'لن', 'لا', 'ست', 'سي', 'سن', 'سأ', 'سـ',
        'و', 'ف', 'ب', 'ل', 'ك', 'ي', 'ت', 'ن', 'أ', 'س',
    ]
    # Common Arabic suffixes (enclitics + verb/noun suffixes)
    SUFFIXES = [
        'كما', 'هما', 'كم', 'هم', 'هن', 'نا', 'ها', 'ون', 'ين', 'ان', 'ات', 'وا',
        'تم', 'تن', 'ية', 'ته', 'تك',
        'ك', 'ه', 'ي', 'ا', 'ة', 'ت', 'ن',
    ]
    # Arabic vowel diacritics and weak letters to strip for consonant skeleton
    VOWELS_AND_DIACRITICS = set('\u064e\u064f\u0650\u0651\u0652\u064b\u064c\u064d\u0670اوىيءئؤإأآ')
    
    def extract_root(self, word: str) -> str:
        """Extract approximate trilateral root from Arabic word."""
        if not word or len(word) < 2:
            return word
        
        w = word
        # Strip prefixes (longest match first)
        for prefix in self.PREFIXES:
            if w.startswith(prefix) and len(w) - len(prefix) >= 2:
                w = w[len(prefix):]
                break
        
        # Strip suffixes (longest match first)
        for suffix in self.SUFFIXES:
            if w.endswith(suffix) and len(w) - len(suffix) >= 2:
                w = w[:-len(suffix)]
                break
        
        # Extract consonant skeleton
        consonants = [c for c in w if c not in self.VOWELS_AND_DIACRITICS]
        
        # Return trilateral root if possible
        if len(consonants) >= 3:
            return consonants[0] + consonants[1] + consonants[2]
        elif len(consonants) == 2:
            return consonants[0] + consonants[1]
        elif consonants:
            return consonants[0]
        return w
    
    def extract_pattern(self, word: str, root: str) -> str:
        """Derive approximate morphological pattern by replacing root chars with فعل."""
        if not root or len(root) < 2:
            return 'other'
        pattern = word
        fa, ain, lam = 'ف', 'ع', 'ل'
        replacements = list(zip(root[:3], [fa, ain, lam]))
        for old, new in replacements:
            pattern = pattern.replace(old, new, 1)
        return pattern


class RealArabicV2Dataset(Dataset):
    """
    Real Arabic dataset that reads offline generated grids and uses
    algorithmic root extraction for authentic morphological features.
    """
    def __init__(self, data_dir: Path, split: str = "train", max_len: int = 32, device='cpu'):
        self.max_len = max_len
        self.device = device
        
        # Load offline generated syntax grids
        self.grids = torch.tensor(np.load(data_dir / f"{split}_grids.npy"), dtype=torch.long)
        self.masks = torch.tensor(np.load(data_dir / f"{split}_masks.npy"), dtype=torch.long)
        self.solutions = torch.tensor(np.load(data_dir / f"{split}_solutions.npy"), dtype=torch.long)
        
        import json
        with open(data_dir / f"{split}_texts.json", "r", encoding="utf-8") as f:
            self.texts = json.load(f)
        
        import hashlib
        root_extractor = ArabicRootExtractor()
        
        # Caches to avoid extracting morphology per step during training
        self.char_ids_cache = []
        self.bpe_ids_cache = []
        self.root_ids_cache = []
        self.pattern_ids_cache = []
        
        def stable_hash(s, bins):
            if not s: return 0
            return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % bins
            
        print(f"Pre-computing morphology for {len(self.texts)} sentences...")
        from tqdm import tqdm
        for i, text in enumerate(tqdm(self.texts, desc="Root Extraction")):
            words = text.split()[:max_len]
            words += [""] * (max_len - len(words))  # padding
            
            c_row, b_row, r_row, p_row = [], [], [], []
            for w in words:
                if not w:
                    c_row.append([0]*8)
                    b_row.append([0]*4)
                    r_row.append(0)
                    p_row.append(0)
                    continue
                    
                chars = [ord(c)%300 for c in w[:8]] + [0]*max(0, 8-len(w))
                bpes = [(stable_hash(w, 8000))]*4
                c_row.append(chars)
                b_row.append(bpes)
                
                # True trilateral root extraction
                root = root_extractor.extract_root(w)
                pattern = root_extractor.extract_pattern(w, root)
                
                root_id = stable_hash(root, 5000)
                pattern_id = stable_hash(pattern, 200)
                
                r_row.append(root_id)
                p_row.append(pattern_id)
                
            self.char_ids_cache.append(c_row)
            self.bpe_ids_cache.append(b_row)
            self.root_ids_cache.append(r_row)
            self.pattern_ids_cache.append(p_row)
            
        self.char_ids_cache = torch.tensor(self.char_ids_cache, device='cpu').clamp(0, 299)
        self.bpe_ids_cache = torch.tensor(self.bpe_ids_cache, device='cpu').clamp(0, 7999)
        self.root_ids_cache = torch.tensor(self.root_ids_cache, device='cpu')
        self.pattern_ids_cache = torch.tensor(self.pattern_ids_cache, device='cpu')

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        grid = self.grids[idx].to(self.device)
        sol = self.solutions[idx].to(self.device)
        mask = self.masks[idx].to(self.device)

        word_ids = grid[:, 0]
        pos_tags = grid[:, 1]

        char_ids = self.char_ids_cache[idx].to(self.device)
        bpe_ids = self.bpe_ids_cache[idx].to(self.device)
        root_ids = self.root_ids_cache[idx].to(self.device).long()
        pattern_ids = self.pattern_ids_cache[idx].to(self.device).long()
        
        proclitic_ids = word_ids % 200
        enclitic_ids = word_ids % 100
        diac_ids = (char_ids % 20).long()

        char_ids = (char_ids * mask.unsqueeze(-1)).long()
        bpe_ids = (bpe_ids * mask.unsqueeze(-1)).long()

        # UD heads are 1-indexed: head=k means parent is UD token k (grid position k-1)
        # head=0 means this word is root (no valid head in grid — must be masked from arc loss)
        raw_heads = sol[:, 4].long()
        is_root = (raw_heads == 0)  # root words in UD have head=0
        heads_0idx = (raw_heads - 1).clamp(min=0)  # UD 1-indexed → grid 0-indexed
        
        # dep_mask: 1 for tokens with valid heads, 0 for root words and padding
        dep_mask = mask.long() * (~is_root).long()
        
        return {
            'word_ids': word_ids.long(),
            'pos_tags': pos_tags.long(),
            'char_ids': char_ids,
            'bpe_ids': bpe_ids,
            'root_ids': root_ids,
            'pattern_ids': pattern_ids,
            'proclitic_ids': proclitic_ids.long(),
            'enclitic_ids': enclitic_ids.long(),
            'diac_ids': diac_ids,
            'mask': mask.long(),
            'dep_mask': dep_mask,
            'heads': heads_0idx,
            'relations': sol[:, 5].long(),
            'cases': sol[:, 3].long()
        }


def get_layer_wise_lr_groups(model, base_lr=2e-3, decay=0.85):
    groups = [
        {'params': model.morph_encoder.parameters(), 'lr': base_lr * decay**4},
        {'params': list(model.word_embed.parameters()) + 
                   list(model.pos_embed.parameters()) +
                   list(model.struct_pos_encoder.parameters()), 
         'lr': base_lr * decay**3},
        {'params': model.transformer.parameters(), 'lr': base_lr * decay**2},
        {'params': list(model.manager.parameters()) + 
                   list(model.grid_processor.parameters()), 
         'lr': base_lr * decay},
        {'params': list(model.biaffine_arc.parameters()) +
                   list(model.biaffine_rel.parameters()) + 
                   list(model.gnn_refine.parameters()) +
                   list(model.second_order.parameters()) +
                   list(model.case_classifier.parameters()) +
                   list(model.arc_head_mlp.parameters()) +
                   list(model.arc_dep_mlp.parameters()) +
                   list(model.rel_head_mlp.parameters()) +
                   list(model.rel_dep_mlp.parameters()) +
                   list(model.uncertainty_loss.parameters()),
         'lr': base_lr},
    ]
    return groups

def compute_accuracy(output_dict, batch):
    # Handle both training output (logits) and inference output (pred_*)
    if 'pred_heads' in output_dict:
        pred_heads = output_dict['pred_heads']
    else:
        pred_heads = output_dict['arc_scores'].argmax(dim=2)
    
    if 'pred_rels' in output_dict:
        pred_rels = output_dict['pred_rels']
    else:
        pred_rels = output_dict['rel_logits'].argmax(dim=-1)
    
    if 'pred_cases' in output_dict:
        pred_cases = output_dict['pred_cases']
    else:
        pred_cases = output_dict['case_logits'].argmax(dim=-1)
    
    gold_heads = batch['heads']
    gold_rels = batch['relations']
    gold_cases = batch['cases']
    m = batch['mask'].bool()
    # dep_mask excludes root words (no valid head target)
    dm = batch.get('dep_mask', batch['mask']).bool()
    
    case_correct = ((pred_cases == gold_cases) & m).sum().item()
    head_correct = ((pred_heads == gold_heads) & dm).sum().item()
    # LAS: both head AND relation must be correct (standard definition)
    las_correct = ((pred_heads == gold_heads) & (pred_rels == gold_rels) & dm).sum().item()
    total = m.sum().item()
    dep_total = dm.sum().item()
    
    return {
        'case': case_correct / max(total, 1),
        'head': head_correct / max(dep_total, 1),
        'deprel': las_correct / max(dep_total, 1),
    }

def train_hrm_v2(args):
    print(f"\n{'='*60}")
    print(f"Training HRM-Grid Parser v2 on {DEVICE.upper()}")
    print(f"{'='*60}")
    
    if args.quick_test:
        dataset = DummyArabicDataset(size=32, max_len=32, device=DEVICE)
    else:
        dataset = RealArabicV2Dataset(DATA_DIR, split="train", max_len=32, device=DEVICE)
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    config = ParserConfig(
        word_dim=256 if args.quick_test else 384,
        n_heads=4 if args.quick_test else 6,
        n_transformer_layers=2 if args.quick_test else 3,
        n_gnn_rounds=2 if args.quick_test else 3,
        n_relations=50,
        n_cases=4,  # None/Mabni, Nom, Acc, Gen (Jussive absent in data)
    )
    
    model = ArabicHRMGridParserV2(config).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params/1e6:.2f}M")
    
    param_groups = get_layer_wise_lr_groups(model, base_lr=3e-3, decay=0.85)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.02)
    steps_per_epoch = len(dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g['lr'] for g in param_groups],
        epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.05, anneal_strategy='cos', final_div_factor=100
    )
    
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load DEV set for proper evaluation (not training data!)
    if not args.quick_test:
        dev_dataset = RealArabicV2Dataset(DATA_DIR, split="dev", max_len=32, device=DEVICE)
        dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
        print(f"Dev set: {len(dev_dataset)} sentences")
    
    best_uas = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            output = model(batch, epoch=epoch, training=True)
            loss = output['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            ema.update()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            if args.quick_test:
                break
        

        
        # Evaluate on DEV set (or train sample for quick_test)
        ema.apply_shadow()
        model.eval()
        avg_acc = {'case': 0, 'head': 0, 'deprel': 0}
        eval_batches = 0
        eval_loader = dataloader if args.quick_test else dev_loader
        with torch.no_grad():
            for batch in eval_loader:
                out = model(batch, epoch=epoch, training=False)
                acc = compute_accuracy(out, batch)
                avg_acc['case'] += acc['case']
                avg_acc['head'] += acc['head']
                avg_acc['deprel'] += acc['deprel']
                eval_batches += 1
                if args.quick_test and eval_batches >= 1:
                    break
        
        n = max(eval_batches, 1)
        uas = avg_acc['head'] / n * 100
        las = avg_acc['deprel'] / n * 100
        case_acc = avg_acc['case'] / n * 100
        
        print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss/max(n_batches,1):.4f} | "
              f"UAS: {uas:.2f}% | LAS: {las:.2f}% | "
              f"Case: {case_acc:.2f}% | Time: {time.time()-t_start:.1f}s")
              
        ema.restore()
        
        # Checkpoint best model
        if uas > best_uas and not args.quick_test:
            best_uas = uas
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'uas': uas,
                'las': las,
            }, OUTPUT_DIR / 'best_model.pt')
            print(f"  >> Saved best model (UAS={uas:.2f}%)")
        
        # Periodic checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0 and not args.quick_test:
            torch.save(model.state_dict(), OUTPUT_DIR / f'checkpoint_ep{epoch+1}.pt')
        
        if args.quick_test and epoch == 1:
            print("Quick test passed! Model successfully compiled and ran forward/backward passes.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true", help="Run 2 epochs on 1 batch")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    train_hrm_v2(args)
