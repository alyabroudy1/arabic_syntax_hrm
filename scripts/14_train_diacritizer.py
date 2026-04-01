#!/usr/bin/env python3
"""
Script 14: Train Arabic Diacritizer
=====================================
Trains the character-level diacritizer on PADT + Quran data.

Usage:
    python scripts/14_train_diacritizer.py --epochs 50
    python scripts/14_train_diacritizer.py --quick-test
"""

import argparse
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.v2.diacritizer import ArabicDiacritizer

DATA_DIR = PROJECT_ROOT / "data"
PADT_DIR = DATA_DIR / "ud_arabic_padt"
QURAN_DIR = DATA_DIR / "quran"
OUTPUT_DIR = PROJECT_ROOT / "models" / "v2_diacritizer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_WORDS = 32
MAX_CHARS = 16

# ─── Diacritic Constants ───
ALL_DIACRITICS = set('\u064E\u064F\u0650\u0652\u0651\u064B\u064C\u064D\u0670\u0653\u0654\u0655')

FATHA    = '\u064E'
DAMMA    = '\u064F'
KASRA    = '\u0650'
SUKUN    = '\u0652'
SHADDA   = '\u0651'
TANWEEN_FATH = '\u064B'
TANWEEN_DAMM = '\u064C'
TANWEEN_KASR = '\u064D'

DIAC_LABELS = {
    'NONE': 0, 'FATHA': 1, 'DAMMA': 2, 'KASRA': 3, 'SUKUN': 4,
    'SHADDA': 5, 'SHADDA_FATHA': 6, 'SHADDA_DAMMA': 7, 'SHADDA_KASRA': 8,
    'TANWEEN_FATH': 9, 'TANWEEN_DAMM': 10, 'TANWEEN_KASR': 11,
    'SHADDA_TANWEEN_FATH': 12, 'SHADDA_TANWEEN_DAMM': 13, 'SHADDA_TANWEEN_KASR': 14,
}
DIAC_LABELS_INV = {v: k for k, v in DIAC_LABELS.items()}

# ─── Character Vocabulary ───
# Build a simple Arabic char vocab
ARABIC_CHARS = 'ءآأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهوي' \
               'ىةﻻﻷﻹﻵ' \
               'ٱ' \
               '٠١٢٣٤٥٦٧٨٩' \
               '0123456789' \
               ' .,:;!?()-/"\''

CHAR_TO_ID = {c: i + 1 for i, c in enumerate(ARABIC_CHARS)}  # 0 = padding


def char_to_id(c):
    return CHAR_TO_ID.get(c, len(CHAR_TO_ID) + 1)  # UNK


def extract_char_diacritics(diacritized_word):
    """Extract (base_char, diac_label) pairs from a diacritized word."""
    result = []
    i = 0
    chars = list(diacritized_word)
    
    while i < len(chars):
        c = chars[i]
        if c in ALL_DIACRITICS:
            i += 1
            continue
        
        diacs = set()
        j = i + 1
        while j < len(chars) and chars[j] in ALL_DIACRITICS:
            diacs.add(chars[j])
            j += 1
        
        has_shadda = SHADDA in diacs
        if has_shadda and FATHA in diacs: label = 6
        elif has_shadda and DAMMA in diacs: label = 7
        elif has_shadda and KASRA in diacs: label = 8
        elif has_shadda and TANWEEN_FATH in diacs: label = 12
        elif has_shadda and TANWEEN_DAMM in diacs: label = 13
        elif has_shadda and TANWEEN_KASR in diacs: label = 14
        elif has_shadda: label = 5
        elif FATHA in diacs: label = 1
        elif DAMMA in diacs: label = 2
        elif KASRA in diacs: label = 3
        elif SUKUN in diacs: label = 4
        elif TANWEEN_FATH in diacs: label = 9
        elif TANWEEN_DAMM in diacs: label = 10
        elif TANWEEN_KASR in diacs: label = 11
        else: label = 0
        
        result.append((c, label))
        i = j
    
    return result


def strip_diacritics(text):
    return ''.join(c for c in text if c not in ALL_DIACRITICS)


# ─── Dataset ───

class DiacritizationDataset(Dataset):
    """Dataset for diacritizer training."""
    
    def __init__(self, sentences, max_words=MAX_WORDS, max_chars=MAX_CHARS, device='cpu'):
        """
        sentences: list of lists of dicts with 'form', 'vform', 'chars', 'char_labels'
        """
        self.device = device
        self.items = []
        
        for sent in sentences:
            char_ids = torch.zeros(max_words, max_chars, dtype=torch.long)
            diac_labels = torch.zeros(max_words, max_chars, dtype=torch.long)
            diac_mask = torch.zeros(max_words, max_chars, dtype=torch.long)
            word_mask = torch.zeros(max_words, dtype=torch.long)
            
            for wi, word in enumerate(sent[:max_words]):
                word_mask[wi] = 1
                chars = word.get('chars', list(strip_diacritics(word.get('vform', word.get('form', '')))))
                labels = word.get('char_labels', [0] * len(chars))
                
                for ci, (c, l) in enumerate(zip(chars[:max_chars], labels[:max_chars])):
                    char_ids[wi, ci] = char_to_id(c)
                    diac_labels[wi, ci] = l
                    diac_mask[wi, ci] = 1
            
            self.items.append({
                'char_ids': char_ids.to(device),
                'word_mask': word_mask.to(device),
                'diac_labels': diac_labels.to(device),
                'diac_mask': diac_mask.to(device),
            })
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


# ─── Data Loading ───

def load_padt_sentences(split):
    """Load diacritized sentences from PADT CoNLL-U."""
    path = PADT_DIR / f"ar_padt-ud-{split}.conllu"
    if not path.exists():
        return []
    
    sentences = []
    current_words = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences.append(current_words)
                    current_words = []
                continue
            if line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 10 or '-' in parts[0] or '.' in parts[0]:
                continue
            
            form = parts[1]
            misc = parts[9]
            vform = form
            if 'Vform=' in misc:
                for field in misc.split('|'):
                    if field.startswith('Vform='):
                        vform = field.split('=', 1)[1]
                        break
            
            char_diacs = extract_char_diacritics(vform)
            current_words.append({
                'form': form,
                'vform': vform,
                'char_labels': [l for _, l in char_diacs],
                'chars': [c for c, _ in char_diacs],
            })
    
    if current_words:
        sentences.append(current_words)
    return sentences


def load_quran_sentences():
    """Load diacritized sentences from Quran."""
    path = QURAN_DIR / "quran-uthmani.txt"
    if not path.exists():
        return []
    
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            words = line.split()
            word_data = []
            for w in words:
                char_diacs = extract_char_diacritics(w)
                if char_diacs:
                    word_data.append({
                        'form': strip_diacritics(w),
                        'vform': w,
                        'char_labels': [l for _, l in char_diacs],
                        'chars': [c for c, _ in char_diacs],
                    })
            if word_data:
                sentences.append(word_data)
    return sentences


# ─── Metrics ───

def compute_diac_metrics(pred_diacs, gold_diacs, mask):
    """Compute DER and WER."""
    m = mask.bool()
    
    # DER: character-level
    total_chars = m.sum().item()
    correct_chars = ((pred_diacs == gold_diacs) & m).sum().item()
    der = 1.0 - correct_chars / max(total_chars, 1)
    
    # WER: word-level (any char wrong = word wrong)
    B, W, C = pred_diacs.shape
    word_mask = m.any(dim=-1)  # (B, W)
    char_correct = ((pred_diacs == gold_diacs) | ~m)  # (B, W, C)
    word_correct = char_correct.all(dim=-1) & word_mask  # (B, W)
    total_words = word_mask.sum().item()
    correct_words = word_correct.sum().item()
    wer = 1.0 - correct_words / max(total_words, 1)
    
    return {
        'der': der * 100,  # Diacritic Error Rate %
        'wer': wer * 100,  # Word Error Rate %
        'char_acc': (1 - der) * 100,
        'word_acc': (1 - wer) * 100,
    }


# ─── Training ───

def train(args):
    print(f"\n{'='*60}")
    print(f"Training Arabic Diacritizer on {DEVICE.upper()}")
    print(f"{'='*60}")
    
    # Load data
    print("\nLoading data...")
    train_padt = load_padt_sentences('train')
    quran = load_quran_sentences()
    train_data = train_padt + quran
    dev_data = load_padt_sentences('dev')
    
    print(f"  Train: {len(train_data)} sentences (PADT:{len(train_padt)} + Quran:{len(quran)})")
    print(f"  Dev:   {len(dev_data)} sentences")
    
    if args.quick_test:
        train_data = train_data[:50]
        dev_data = dev_data[:20]
    
    train_dataset = DiacritizationDataset(train_data, device=DEVICE)
    dev_dataset = DiacritizationDataset(dev_data, device=DEVICE)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
    
    # Model
    model = ArabicDiacritizer(
        char_vocab=256,
        char_embed_dim=64,
        char_hidden=128,
        word_dim=256,
        n_heads=4,
        n_transformer_layers=3,
        n_diac_classes=15,
        max_chars=MAX_CHARS,
        max_words=MAX_WORDS,
        dropout=0.3,
    ).to(DEVICE)
    
    print(f"  Model: {model.count_parameters()/1e6:.2f}M parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_char_acc = 0
    
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                char_ids=batch['char_ids'],
                word_mask=batch['word_mask'],
                diac_labels=batch['diac_labels'],
                diac_mask=batch['diac_mask'],
            )
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        
        # Dev evaluation
        model.eval()
        all_preds, all_golds, all_masks = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                out = model(
                    char_ids=batch['char_ids'],
                    word_mask=batch['word_mask'],
                )
                all_preds.append(out['pred_diacs'])
                all_golds.append(batch['diac_labels'])
                all_masks.append(batch['diac_mask'])
        
        preds = torch.cat(all_preds)
        golds = torch.cat(all_golds)
        masks = torch.cat(all_masks)
        metrics = compute_diac_metrics(preds, golds, masks)
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | "
              f"CharAcc: {metrics['char_acc']:.2f}% | WordAcc: {metrics['word_acc']:.2f}% | "
              f"DER: {metrics['der']:.2f}% | Time: {elapsed:.1f}s")
        
        if metrics['char_acc'] > best_char_acc and not args.quick_test:
            best_char_acc = metrics['char_acc']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'char_acc': best_char_acc,
                'der': metrics['der'],
                'wer': metrics['wer'],
            }, OUTPUT_DIR / 'best_diacritizer.pt')
            print(f"  >> Saved best model (CharAcc={best_char_acc:.2f}%)")
        
        if args.quick_test and epoch >= 1:
            print("Quick test passed!")
            break
    
    print(f"\nBest CharAcc: {best_char_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    
    train(args)
