#!/usr/bin/env python3
"""
Script 02: Build Arabic Syntax Grid Dataset for HRM
====================================================

Priority 2 — This is the NOVEL CONTRIBUTION of the project.
Converts Arabic dependency-parsed sentences into HRM-compatible grid format.

The key insight: Arabic iʻrāb IS a constraint satisfaction problem,
just like Sudoku. The grid encodes structural relationships between words,
and HRM learns to propagate constraints to fill in missing values
(case markings, dependency heads, dependency relations).

Grid Layout:
    32 rows  (words, padded to max sentence length)
    8 columns (features per word)
    
Column Definitions:
    0: word_bucket    (hash of word form → 0-255)
    1: pos_tag        (17 universal POS tags)
    2: morph_pattern  (lemma hash → 0-63)
    3: case_marking   (Nom=1, Acc=2, Gen=3, Jus=4) ← TARGET TO PREDICT
    4: dep_head       (pointer to row index)          ← TARGET TO PREDICT
    5: dep_relation   (25 relation types)              ← TARGET TO PREDICT
    6: agreement      (gender×number×person = 18 values)
    7: definiteness   (0=indef, 1=def, 2=construct)

Usage:
    python scripts/02_build_syntax_grids.py
    
    # Verify on a small sample first:
    python scripts/02_build_syntax_grids.py --verify-only --sample-size 10
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MAX_WORDS = 32   # max sentence length (pad/truncate)
NUM_FEATURES = 8  # feature columns per word

# ─────────────────────────────────────────────
# Feature Vocabulary Mappings
# ─────────────────────────────────────────────

# Universal POS tags (from UD specification)
POS_TAGS = {
    '_': 0, 'NOUN': 1, 'VERB': 2, 'ADJ': 3, 'ADV': 4, 'ADP': 5,
    'CONJ': 6, 'CCONJ': 6, 'SCONJ': 7, 'PART': 8, 'DET': 9,
    'PRON': 10, 'PROPN': 11, 'NUM': 12, 'PUNCT': 13, 'AUX': 14,
    'INTJ': 15, 'X': 16, 'SYM': 17,
}

# Arabic case markings (الإعراب)
# Nom (مرفوع/رفع), Acc (منصوب/نصب), Gen (مجرور/جر)
CASE_TAGS = {
    '_': 0,     # no case / مبني
    'Nom': 1,   # مرفوع (ضمة)
    'Acc': 2,   # منصوب (فتحة)
    'Gen': 3,   # مجرور (كسرة)
}

# Universal Dependency Relations (common ones in Arabic PADT)
DEP_RELS = {
    '_': 0, 'root': 1, 'nsubj': 2, 'obj': 3, 'iobj': 4, 'obl': 5,
    'advmod': 6, 'amod': 7, 'nmod': 8, 'det': 9, 'case': 10,
    'conj': 11, 'cc': 12, 'punct': 13, 'flat': 14, 'compound': 15,
    'appos': 16, 'acl': 17, 'advcl': 18, 'cop': 19, 'mark': 20,
    'dep': 21, 'parataxis': 22, 'fixed': 23, 'vocative': 24,
    'nummod': 25, 'flat:foreign': 26, 'nsubj:pass': 27, 'csubj': 28,
    'xcomp': 29, 'ccomp': 30, 'orphan': 31,
}

# Morphological agreement features
GENDER_MAP = {'_': 0, 'Masc': 1, 'Fem': 2}
NUMBER_MAP = {'_': 0, 'Sing': 1, 'Dual': 2, 'Plur': 3}
PERSON_MAP = {'_': 0, '1': 1, '2': 2, '3': 3}

# Definiteness
DEFINITE_MAP = {'_': 0, 'Ind': 0, 'Def': 1, 'Com': 2, 'Cons': 2}


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class ArabicSyntaxGrid:
    """One Arabic sentence encoded as an HRM-compatible grid."""
    text: str
    sent_id: str
    grid: np.ndarray       # shape: (MAX_WORDS, NUM_FEATURES) — input (partially masked)
    mask: np.ndarray        # shape: (MAX_WORDS,) — 1 for real words, 0 for padding
    solution: np.ndarray    # shape: (MAX_WORDS, NUM_FEATURES) — full solution
    num_words: int
    difficulty: int         # 0=hard, 1=medium, 2=easy


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────

def parse_ud_features(feat_string: str) -> Dict[str, str]:
    """Parse UD morphological features string (e.g., 'Case=Nom|Gender=Masc|Number=Sing')."""
    if feat_string == '_' or not feat_string:
        return {}
    features = {}
    for feat in feat_string.split('|'):
        if '=' in feat:
            key, val = feat.split('=', 1)
            features[key] = val
    return features


def encode_agreement(feats: Dict[str, str]) -> int:
    """
    Encode gender × number × person into single integer (0-35).
    
    Mapping: gender * 12 + number * 3 + person
    This gives unique indices for all combinations.
    """
    g = GENDER_MAP.get(feats.get('Gender', '_'), 0)
    n = NUMBER_MAP.get(feats.get('Number', '_'), 0)
    p = PERSON_MAP.get(feats.get('Person', '_'), 0)
    return g * 12 + n * 3 + p


def word_to_bucket(word: str, num_buckets: int = 256) -> int:
    """Hash word form to a fixed bucket (0 to num_buckets-1)."""
    # Use a simple but deterministic hash
    h = 0
    for ch in word:
        h = (h * 31 + ord(ch)) % num_buckets
    return max(1, h)  # reserve 0 for padding


def lemma_to_pattern(lemma: str, num_buckets: int = 64) -> int:
    """Hash lemma to a morphological pattern bucket."""
    h = 0
    for ch in lemma:
        h = (h * 37 + ord(ch)) % num_buckets
    return max(1, h)


# ─────────────────────────────────────────────
# Grid Encoding
# ─────────────────────────────────────────────

def sentence_to_grid(sentence: Dict) -> ArabicSyntaxGrid:
    """
    Convert a UD-parsed sentence to an HRM grid.
    
    The SOLUTION grid has all features filled in.
    The INPUT grid has case marking (col 3), dep head (col 4), 
    and dep relation (col 5) MASKED — these are what HRM must predict.
    """
    tokens = sentence['tokens']
    num_words = min(len(tokens), MAX_WORDS)
    
    # Initialize grids
    grid = np.zeros((MAX_WORDS, NUM_FEATURES), dtype=np.int32)
    mask = np.zeros(MAX_WORDS, dtype=np.int32)
    solution = np.zeros((MAX_WORDS, NUM_FEATURES), dtype=np.int32)
    
    for i, tok in enumerate(tokens[:MAX_WORDS]):
        feats = parse_ud_features(tok['feats'])
        
        # Column 0: word bucket (given in input)
        word_bucket = word_to_bucket(tok['form'])
        grid[i, 0] = word_bucket
        solution[i, 0] = word_bucket
        
        # Column 1: POS tag (given in input)
        pos = POS_TAGS.get(tok['upos'], 0)
        grid[i, 1] = pos
        solution[i, 1] = pos
        
        # Column 2: morphological pattern (given in input)
        morph = lemma_to_pattern(tok['lemma'])
        grid[i, 2] = morph
        solution[i, 2] = morph
        
        # Column 3: CASE MARKING → TARGET (masked in input)
        case = CASE_TAGS.get(feats.get('Case', '_'), 0)
        grid[i, 3] = 0       # MASKED in input
        solution[i, 3] = case  # ground truth
        
        # Column 4: DEPENDENCY HEAD → TARGET (masked in input)
        head = min(tok['head'], MAX_WORDS - 1)
        # Convert from 1-indexed (UD) to 0-indexed (grid)
        # HEAD=0 in UD means root, we keep it as 0
        grid[i, 4] = 0        # MASKED in input
        solution[i, 4] = head  # ground truth
        
        # Column 5: DEPENDENCY RELATION → TARGET (masked in input)
        deprel = DEP_RELS.get(tok['deprel'], DEP_RELS.get('dep', 0))
        grid[i, 5] = 0          # MASKED in input
        solution[i, 5] = deprel  # ground truth
        
        # Column 6: agreement features (given in input)
        agreement = encode_agreement(feats)
        grid[i, 6] = agreement
        solution[i, 6] = agreement
        
        # Column 7: definiteness (given in input)
        definite = DEFINITE_MAP.get(feats.get('Definite', '_'), 0)
        grid[i, 7] = definite
        solution[i, 7] = definite
        
        # Mark as real word
        mask[i] = 1
    
    return ArabicSyntaxGrid(
        text=sentence.get('text', ''),
        sent_id=sentence.get('sent_id', ''),
        grid=grid,
        mask=mask,
        solution=solution,
        num_words=num_words,
        difficulty=0,  # default: hardest
    )


def create_difficulty_variants(base_grid: ArabicSyntaxGrid, 
                                 num_variants: int = 3,
                                 rng: Optional[np.random.Generator] = None
                                 ) -> List[ArabicSyntaxGrid]:
    """
    Create training variants with different masking patterns.
    Like Sudoku puzzles with different numbers of given cells.
    
    Difficulty 0 (Hard):   Only word + POS + agreement given → predict case + deps
    Difficulty 1 (Medium): 50% of cases revealed → predict rest + deps
    Difficulty 2 (Easy):   All cases given → predict deps only
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    variants = []
    
    for difficulty in range(min(num_variants, 3)):
        new_grid = base_grid.grid.copy()
        
        if difficulty == 0:
            # Hard: cols 3, 4, 5 all masked (default)
            new_grid[:, 3] = 0
            new_grid[:, 4] = 0
            new_grid[:, 5] = 0
            
        elif difficulty == 1:
            # Medium: reveal ~50% of case markings randomly
            reveal_mask = rng.random(MAX_WORDS) > 0.5
            new_grid[:, 3] = np.where(
                reveal_mask & (base_grid.mask == 1),
                base_grid.solution[:, 3],
                0
            )
            new_grid[:, 4] = 0  # deps still masked
            new_grid[:, 5] = 0
            
        elif difficulty == 2:
            # Easy: all cases revealed, predict deps only
            new_grid[:, 3] = np.where(
                base_grid.mask == 1,
                base_grid.solution[:, 3],
                0
            )
            new_grid[:, 4] = 0  # dep heads masked
            new_grid[:, 5] = 0  # dep rels masked
        
        variants.append(ArabicSyntaxGrid(
            text=base_grid.text,
            sent_id=base_grid.sent_id,
            grid=new_grid,
            mask=base_grid.mask.copy(),
            solution=base_grid.solution.copy(),
            num_words=base_grid.num_words,
            difficulty=difficulty,
        ))
    
    return variants


# ─────────────────────────────────────────────
# Pretty-Print Grid (for verification)
# ─────────────────────────────────────────────

CASE_NAMES = {0: '—', 1: 'Nom/رفع', 2: 'Acc/نصب', 3: 'Gen/جر'}
POS_NAMES = {v: k for k, v in POS_TAGS.items()}
DEP_NAMES = {v: k for k, v in DEP_RELS.items()}

def print_grid(grid_obj: ArabicSyntaxGrid, show_solution: bool = True):
    """Pretty-print a grid for manual verification."""
    print(f"\n  Text: {grid_obj.text}")
    print(f"  Words: {grid_obj.num_words}, Difficulty: {grid_obj.difficulty}")
    
    header = (f"  {'#':>2} {'Word':>8} {'POS':>6} {'Morph':>5} "
              f"{'Case':>10} {'Head':>4} {'DepRel':>10} {'Agree':>5} {'Def':>3}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    
    for i in range(grid_obj.num_words):
        g = grid_obj.grid[i]
        s = grid_obj.solution[i]
        
        # Show input vs solution for masked columns
        case_in = CASE_NAMES.get(g[3], '?')
        case_sol = CASE_NAMES.get(s[3], '?')
        case_str = f"{case_in}" if g[3] == s[3] else f"?→{case_sol}" if show_solution else "?"
        
        head_in = g[4]
        head_sol = s[4]
        head_str = f"{head_in}" if g[4] == s[4] else f"?→{head_sol}" if show_solution else "?"
        
        deprel_in = DEP_NAMES.get(g[5], '?')
        deprel_sol = DEP_NAMES.get(s[5], '?')
        deprel_str = f"{deprel_in}" if g[5] == s[5] else f"?→{deprel_sol}" if show_solution else "?"
        
        pos_str = POS_NAMES.get(g[1], '?')
        
        print(f"  {i:>2} {g[0]:>8} {pos_str:>6} {g[2]:>5} "
              f"{case_str:>10} {head_str:>4} {deprel_str:>10} {g[6]:>5} {g[7]:>3}")


# ─────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────

def build_hrm_dataset(ud_data_path: str, output_dir: str,
                       min_tokens: int = 3, max_tokens: int = MAX_WORDS,
                       num_variants: int = 3, verify_only: bool = False,
                       sample_size: int = 0):
    """
    Main dataset builder: UD parsed sentences → HRM grids.
    
    Args:
        ud_data_path: path to ud_padt_all.json
        output_dir: where to save numpy arrays
        min_tokens: minimum sentence length
        max_tokens: maximum sentence length
        num_variants: difficulty variants per sentence
        verify_only: if True, only print samples, don't save
        sample_size: if > 0, only process this many sentences
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Building HRM Grid Dataset")
    print(f"{'='*60}")
    print(f"  Source: {ud_data_path}")
    print(f"  Output: {output_dir}")
    print(f"  Grid size: {MAX_WORDS} × {NUM_FEATURES} = {MAX_WORDS * NUM_FEATURES} cells")
    print(f"  Sentence length: {min_tokens}–{max_tokens} tokens")
    print(f"  Difficulty variants: {num_variants}")
    
    # Load data
    with open(ud_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    rng = np.random.default_rng(42)
    stats = {'total': 0, 'skipped_short': 0, 'skipped_long': 0}
    
    for split_name in ['train', 'dev', 'test']:
        if split_name not in data:
            print(f"\n  ⚠️  Split '{split_name}' not found, skipping.")
            continue
        
        sentences = data[split_name]
        if sample_size > 0:
            sentences = sentences[:sample_size]
        
        split_grids = []
        
        for sent in sentences:
            num_tok = len(sent['tokens'])
            
            if num_tok < min_tokens:
                stats['skipped_short'] += 1
                continue
            if num_tok > max_tokens:
                stats['skipped_long'] += 1
                continue
            
            base_grid = sentence_to_grid(sent)
            
            if split_name == 'train':
                variants = create_difficulty_variants(
                    base_grid, num_variants=num_variants, rng=rng
                )
                split_grids.extend(variants)
            else:
                # Eval/test: only hardest variant
                variants = create_difficulty_variants(
                    base_grid, num_variants=1, rng=rng
                )
                split_grids.extend(variants)
        
        stats['total'] += len(split_grids)
        
        # Print samples for verification
        print(f"\n  --- {split_name} split: {len(split_grids)} examples ---")
        
        num_to_show = min(3, len(split_grids))
        for i in range(num_to_show):
            print_grid(split_grids[i], show_solution=True)
        
        if verify_only:
            continue
        
        # Save as numpy arrays
        grids = np.stack([g.grid for g in split_grids])
        masks = np.stack([g.mask for g in split_grids])
        solutions = np.stack([g.solution for g in split_grids])
        texts = [g.text for g in split_grids]
        
        np.save(output_path / f"{split_name}_grids.npy", grids)
        np.save(output_path / f"{split_name}_masks.npy", masks)
        np.save(output_path / f"{split_name}_solutions.npy", solutions)
        with open(output_path / f"{split_name}_texts.json", "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        
        print(f"\n  💾 Saved {split_name}: {grids.shape}")
    
    if not verify_only:
        # Dataset statistics summary
        print(f"\n{'='*60}")
        print(f"DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"  Total examples: {stats['total']}")
        print(f"  Skipped (too short <{min_tokens}): {stats['skipped_short']}")
        print(f"  Skipped (too long >{max_tokens}): {stats['skipped_long']}")
        print(f"  Grid shape per example: ({MAX_WORDS}, {NUM_FEATURES})")
        print(f"  Cell value range: 0–{max(max(v for v in POS_TAGS.values()), max(v for v in DEP_RELS.values()))}")
        
        total_size = sum(
            f.stat().st_size
            for f in output_path.rglob('*.npy')
        ) / 1e6
        print(f"  Total disk size: {total_size:.1f} MB")
        print(f"  Saved to: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"NEXT STEP")
    print(f"{'='*60}")
    print(f"  1. Review the grids above — do they make linguistic sense?")
    print(f"  2. Run: python -m pytest tests/test_grid_encoding.py -v")
    print(f"  3. If grids look good → python scripts/06_train_hrm.py")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build Arabic Syntax Grid Dataset for HRM"
    )
    parser.add_argument(
        "--input", default=str(DATA_DIR / "ud_padt_all.json"),
        help="Path to parsed UD data JSON"
    )
    parser.add_argument(
        "--output", default=str(DATA_DIR / "arabic_syntax_grid"),
        help="Output directory for grid arrays"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only print sample grids, don't save"
    )
    parser.add_argument(
        "--sample-size", type=int, default=0,
        help="Process only this many sentences per split (0=all)"
    )
    parser.add_argument(
        "--min-tokens", type=int, default=3,
        help="Minimum sentence length"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_WORDS,
        help="Maximum sentence length"
    )
    parser.add_argument(
        "--variants", type=int, default=3,
        help="Number of difficulty variants per sentence"
    )
    
    args = parser.parse_args()
    
    build_hrm_dataset(
        ud_data_path=args.input,
        output_dir=args.output,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        num_variants=args.variants,
        verify_only=args.verify_only,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
