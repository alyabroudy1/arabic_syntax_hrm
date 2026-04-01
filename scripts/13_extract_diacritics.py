#!/usr/bin/env python3
"""
Script 13: Extract Diacritics from PADT CoNLL-U + Quran
========================================================
Builds character-level diacritic labels for training the auto-diacritizer.

Sources:
1. PADT Vform (CoNLL-U MISC column) — ~6K MSA sentences  
2. Quran Uthmani text (Tanzil) — 6236 verses, fully diacritized

Output:
  data/arabic_syntax_grid/{split}_diac.npy  — (N, max_words, max_chars) diac labels
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PADT_DIR = DATA_DIR / "ud_arabic_padt"
QURAN_DIR = DATA_DIR / "quran"
OUTPUT_DIR = DATA_DIR / "arabic_syntax_grid"

MAX_WORDS = 32
MAX_CHARS = 16

# ─── Diacritic Constants ───

# Unicode diacritics
FATHA    = '\u064E'  # فتحة
DAMMA    = '\u064F'  # ضمة
KASRA    = '\u0650'  # كسرة
SUKUN    = '\u0652'  # سكون
SHADDA   = '\u0651'  # شدة
TANWEEN_FATH = '\u064B'  # تنوين فتح
TANWEEN_DAMM = '\u064C'  # تنوين ضم
TANWEEN_KASR = '\u064D'  # تنوين كسر
SUPERSCRIPT_ALEF = '\u0670'  # ألف خنجرية
MADDAH   = '\u0653'  # مدة

ALL_DIACRITICS = set([FATHA, DAMMA, KASRA, SUKUN, SHADDA,
                      TANWEEN_FATH, TANWEEN_DAMM, TANWEEN_KASR,
                      SUPERSCRIPT_ALEF, MADDAH, '\u0654', '\u0655'])

# Diacritic label mapping (15 classes)
DIAC_LABELS = {
    'NONE': 0,
    'FATHA': 1,
    'DAMMA': 2,
    'KASRA': 3,
    'SUKUN': 4,
    'SHADDA': 5,
    'SHADDA_FATHA': 6,
    'SHADDA_DAMMA': 7,
    'SHADDA_KASRA': 8,
    'TANWEEN_FATH': 9,
    'TANWEEN_DAMM': 10,
    'TANWEEN_KASR': 11,
    'SHADDA_TANWEEN_FATH': 12,
    'SHADDA_TANWEEN_DAMM': 13,
    'SHADDA_TANWEEN_KASR': 14,
}

# Reverse for display
DIAC_LABELS_INV = {v: k for k, v in DIAC_LABELS.items()}


def extract_char_diacritics(diacritized_word: str) -> List[Tuple[str, int]]:
    """Extract (base_char, diac_label) pairs from a diacritized word.
    
    Example: 'كِتَابٌ' → [('ك', KASRA), ('ت', FATHA), ('ا', NONE), ('ب', TANWEEN_DAMM)]
    """
    result = []
    i = 0
    chars = list(diacritized_word)
    
    while i < len(chars):
        c = chars[i]
        
        # Skip diacritics as standalone — they're attached to previous char
        if c in ALL_DIACRITICS:
            i += 1
            continue
        
        # Base letter found — collect its diacritics
        diacs = set()
        j = i + 1
        while j < len(chars) and chars[j] in ALL_DIACRITICS:
            diacs.add(chars[j])
            j += 1
        
        # Map diacritic combination to label
        has_shadda = SHADDA in diacs
        
        if has_shadda and FATHA in diacs:
            label = DIAC_LABELS['SHADDA_FATHA']
        elif has_shadda and DAMMA in diacs:
            label = DIAC_LABELS['SHADDA_DAMMA']
        elif has_shadda and KASRA in diacs:
            label = DIAC_LABELS['SHADDA_KASRA']
        elif has_shadda and TANWEEN_FATH in diacs:
            label = DIAC_LABELS['SHADDA_TANWEEN_FATH']
        elif has_shadda and TANWEEN_DAMM in diacs:
            label = DIAC_LABELS['SHADDA_TANWEEN_DAMM']
        elif has_shadda and TANWEEN_KASR in diacs:
            label = DIAC_LABELS['SHADDA_TANWEEN_KASR']
        elif has_shadda:
            label = DIAC_LABELS['SHADDA']
        elif FATHA in diacs:
            label = DIAC_LABELS['FATHA']
        elif DAMMA in diacs:
            label = DIAC_LABELS['DAMMA']
        elif KASRA in diacs:
            label = DIAC_LABELS['KASRA']
        elif SUKUN in diacs:
            label = DIAC_LABELS['SUKUN']
        elif TANWEEN_FATH in diacs:
            label = DIAC_LABELS['TANWEEN_FATH']
        elif TANWEEN_DAMM in diacs:
            label = DIAC_LABELS['TANWEEN_DAMM']
        elif TANWEEN_KASR in diacs:
            label = DIAC_LABELS['TANWEEN_KASR']
        else:
            label = DIAC_LABELS['NONE']
        
        result.append((c, label))
        i = j
    
    return result


def strip_diacritics(text: str) -> str:
    """Remove all diacritics from Arabic text."""
    return ''.join(c for c in text if c not in ALL_DIACRITICS)


# ─── PADT Extraction ───

def extract_padt_diacritics(conllu_path: Path) -> List[Dict]:
    """Extract (undiacritized_form, diacritized_form, diac_labels) from CoNLL-U."""
    sentences = []
    current_words = []
    
    with open(conllu_path, 'r', encoding='utf-8') as f:
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
            if len(parts) < 10:
                continue
            
            # Skip multi-word tokens
            if '-' in parts[0] or '.' in parts[0]:
                continue
            
            form = parts[1]  # undiacritized
            misc = parts[9]
            
            # Extract Vform from MISC
            vform = form  # fallback
            if 'Vform=' in misc:
                for field in misc.split('|'):
                    if field.startswith('Vform='):
                        vform = field.split('=', 1)[1]
                        break
            
            char_diacs = extract_char_diacritics(vform)
            current_words.append({
                'form': form,
                'vform': vform,
                'char_labels': [label for _, label in char_diacs],
                'chars': [c for c, _ in char_diacs],
            })
    
    if current_words:
        sentences.append(current_words)
    
    return sentences


# ─── Quran Extraction ───

def extract_quran_diacritics(quran_path: Path) -> List[Dict]:
    """Extract diacritized words from Quran Uthmani text."""
    sentences = []
    
    with open(quran_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Each line is a verse — split into words
            words = line.split()
            word_data = []
            
            for w in words:
                char_diacs = extract_char_diacritics(w)
                if char_diacs:
                    word_data.append({
                        'form': strip_diacritics(w),
                        'vform': w,
                        'char_labels': [label for _, label in char_diacs],
                        'chars': [c for c, _ in char_diacs],
                    })
            
            if word_data:
                sentences.append(word_data)
    
    return sentences


def build_diac_arrays(sentences: List, max_words=MAX_WORDS, max_chars=MAX_CHARS):
    """Convert sentence list to numpy arrays."""
    N = len(sentences)
    diac_labels = np.zeros((N, max_words, max_chars), dtype=np.int64)
    diac_mask = np.zeros((N, max_words, max_chars), dtype=np.int64)
    
    for si, words in enumerate(sentences):
        for wi, word in enumerate(words[:max_words]):
            labels = word['char_labels'][:max_chars]
            for ci, label in enumerate(labels):
                diac_labels[si, wi, ci] = label
                diac_mask[si, wi, ci] = 1
    
    return diac_labels, diac_mask


def print_stats(sentences, name):
    """Print dataset statistics."""
    n_words = sum(len(s) for s in sentences)
    n_chars = sum(len(w['char_labels']) for s in sentences for w in s)
    
    # Count label distribution
    label_counts = {}
    for s in sentences:
        for w in s:
            for l in w['char_labels']:
                label_counts[l] = label_counts.get(l, 0) + 1
    
    print(f"\n  {name}:")
    print(f"    Sentences: {len(sentences)}")
    print(f"    Words: {n_words}")
    print(f"    Characters: {n_chars}")
    print(f"    Label distribution:")
    for label_id in sorted(label_counts.keys()):
        name_str = DIAC_LABELS_INV.get(label_id, '?')
        count = label_counts[label_id]
        pct = count / n_chars * 100
        print(f"      {name_str:25s}: {count:>8d} ({pct:5.1f}%)")


def main():
    print("="*60)
    print("Extracting Arabic Diacritics")
    print("="*60)
    
    # ─── Extract PADT ───
    all_padt = {}
    for split in ['train', 'dev', 'test']:
        path = PADT_DIR / f"ar_padt-ud-{split}.conllu"
        if path.exists():
            sentences = extract_padt_diacritics(path)
            all_padt[split] = sentences
            print_stats(sentences, f"PADT {split}")
        else:
            print(f"  ⚠️  {path} not found")
    
    # ─── Extract Quran ───
    quran_path = QURAN_DIR / "quran-uthmani.txt"
    quran_sentences = []
    if quran_path.exists():
        quran_sentences = extract_quran_diacritics(quran_path)
        print_stats(quran_sentences, "Quran Uthmani")
    else:
        print(f"  ⚠️  Quran file not found: {quran_path}")
    
    # ─── Combine: PADT train + Quran → combined train ───
    combined_train = all_padt.get('train', []) + quran_sentences
    print(f"\n  Combined training set: {len(combined_train)} sentences")
    
    # ─── Save arrays ───
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for split, sentences in [('train', combined_train), 
                              ('dev', all_padt.get('dev', [])),
                              ('test', all_padt.get('test', []))]:
        if not sentences:
            continue
        
        diac_labels, diac_mask = build_diac_arrays(sentences)
        np.save(OUTPUT_DIR / f"{split}_diac_labels.npy", diac_labels)
        np.save(OUTPUT_DIR / f"{split}_diac_mask.npy", diac_mask)
        print(f"  ✅ Saved {split}_diac_labels.npy: {diac_labels.shape}")
    
    # Save diacritic label map
    with open(OUTPUT_DIR / "diac_labels.json", 'w', encoding='utf-8') as f:
        json.dump(DIAC_LABELS, f, ensure_ascii=False, indent=2)
    
    # ─── Show examples ───
    print(f"\n{'='*60}")
    print("Sample Extractions:")
    print("="*60)
    
    for s in (all_padt.get('test', []) or all_padt.get('train', []))[:3]:
        for w in s[:5]:
            labels_str = ' '.join(DIAC_LABELS_INV[l][:4] for l in w['char_labels'])
            print(f"  {w['form']:>15s} → {w['vform']:>20s} | {labels_str}")
        print()


if __name__ == "__main__":
    main()
