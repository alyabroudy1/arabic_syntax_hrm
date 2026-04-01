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
        
        self.grids = torch.randint(1, 512, (size, max_len, 8), device=device)
        self.masks = torch.ones((size, max_len), device=device)
        # Randomly pad the end
        for i in range(size):
            actual_len = torch.randint(5, max_len, (1,)).item()
            self.masks[i, actual_len:] = 0
            self.grids[i, actual_len:, :] = 0
            
        self.solutions = torch.zeros((size, max_len, 8), device=device)
        self.solutions[:, :, 3] = torch.randint(1, 5, (size, max_len)) # cases
        self.solutions[:, :, 4] = torch.randint(0, max_len//2, (size, max_len)) # heads
        self.solutions[:, :, 5] = torch.randint(1, 33, (size, max_len)) # deprels
        
        # Root is at index 0
        self.masks[:, 0] = 1
        self.grids[:, 0, :] = 0
        self.solutions[:, 0, :] = 0
        
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
            'heads': sol[:, 4].long(),
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
    pred_heads = output_dict['arc_scores'].argmax(dim=-1)
    pred_rels = output_dict['rel_logits'].argmax(dim=-1)
    pred_cases = output_dict['case_logits'].argmax(dim=-1)
    
    gold_heads = batch['heads']
    gold_rels = batch['relations']
    gold_cases = batch['cases']
    m = batch['mask'].bool()
    
    case_correct = ((pred_cases == gold_cases) & m).sum().item()
    head_correct = ((pred_heads == gold_heads) & m).sum().item()
    rel_correct = ((pred_rels == gold_rels) & m).sum().item()
    total = m.sum().item()
    
    return {
        'case': case_correct / max(total, 1),
        'head': head_correct / max(total, 1),
        'deprel': rel_correct / max(total, 1),
    }

def train_hrm_v2(args):
    print(f"\n{'='*60}")
    print(f"Training HRM-Grid Parser v2 on {DEVICE.upper()}")
    print(f"{'='*60}")
    
    dataset = DummyArabicDataset(size=512 if not args.quick_test else 32, max_len=32, device=DEVICE)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    config = ParserConfig(
        word_dim=256 if args.quick_test else 384,
        n_heads=4 if args.quick_test else 6,
        n_transformer_layers=2 if args.quick_test else 3,
        n_gnn_rounds=2 if args.quick_test else 3,
        n_relations=50,
        n_cases=5,
    )
    
    model = ArabicHRMGridParserV2(config).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params/1e6:.2f}M")
    
    param_groups = get_layer_wise_lr_groups(model, base_lr=2e-3, decay=0.85)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()
        
        # Simple loops
        for batch in dataloader:
            optimizer.zero_grad()
            
            output = model(batch, epoch=epoch, training=True)
            loss = output['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            ema.update()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            if args.quick_test:
                break
        
        scheduler.step()
        
        # Eval
        ema.apply_shadow()
        model.eval()
        avg_acc = {'case':0, 'head':0, 'deprel':0}
        with torch.no_grad():
            for batch in dataloader:
                out = model(batch, epoch=epoch, training=True) # use training=True just to get all metrics
                acc = compute_accuracy(out, batch)
                avg_acc['case'] += acc['case']
                avg_acc['head'] += acc['head']
                avg_acc['deprel'] += acc['deprel']
                if args.quick_test: break
        
        n = 1 if args.quick_test else len(dataloader)
        print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss/n_batches:.4f} | "
              f"UAS (Head): {avg_acc['head']/n*100:.2f}% | "
              f"LAS (Rel): {avg_acc['deprel']/n*100:.2f}% | "
              f"Case: {avg_acc['case']/n*100:.2f}% | "
              f"Time: {time.time()-t_start:.1f}s")
              
        ema.restore()
        
        if args.quick_test and epoch == 1:
            print("Quick test passed! Model successfully compiled and ran forward/backward passes.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true", help="Run 2 epochs on 1 batch")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train_hrm_v2(args)
