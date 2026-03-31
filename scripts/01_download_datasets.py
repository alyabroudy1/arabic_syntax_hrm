#!/usr/bin/env python3
"""
Script 01: Download and Prepare Arabic Syntax Datasets
======================================================

Priority 1 — This is the FIRST thing to run.
Validates the entire data pipeline by downloading and parsing real data.

Datasets:
  1. Universal Dependencies Arabic PADT (free, CoNLL-U format)
  2. CIDAR (10k Arabic instruction pairs)
  3. Arabic-LLM-Parsing (1,100+ parsed sentences)

Usage:
    python scripts/01_download_datasets.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# CoNLL-U Parser
# ─────────────────────────────────────────────
def parse_conllu(filepath: str) -> List[Dict]:
    """
    Parse CoNLL-U format into structured sentences.
    
    CoNLL-U columns:
        ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
    
    Returns:
        List of sentence dicts with 'text' and 'tokens' keys.
    """
    sentences = []
    current_sentence = []
    current_text = ""
    sent_id = ""
    
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"  ⚠️  File not found: {filepath}")
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Metadata comments
            if line.startswith("# text ="):
                current_text = line.split("= ", 1)[1] if "= " in line else ""
            elif line.startswith("# sent_id ="):
                sent_id = line.split("= ", 1)[1] if "= " in line else ""
            elif line.startswith("#"):
                continue  # skip other comments
            
            # Empty line = sentence boundary
            elif line == "":
                if current_sentence:
                    sentences.append({
                        "sent_id": sent_id,
                        "text": current_text,
                        "tokens": current_sentence,
                        "num_tokens": len(current_sentence),
                    })
                    current_sentence = []
                    current_text = ""
                    sent_id = ""
            
            # Token line
            else:
                parts = line.split("\t")
                if len(parts) < 10:
                    continue
                
                # Skip multi-word tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
                token_id = parts[0]
                if "-" in token_id or "." in token_id:
                    continue
                
                try:
                    current_sentence.append({
                        "id": int(token_id),
                        "form": parts[1],        # word form
                        "lemma": parts[2],        # lemma
                        "upos": parts[3],         # universal POS tag
                        "xpos": parts[4],         # language-specific POS
                        "feats": parts[5],        # morphological features
                        "head": int(parts[6]),    # dependency head
                        "deprel": parts[7],       # dependency relation
                        "deps": parts[8],         # enhanced dependencies
                        "misc": parts[9],         # miscellaneous
                    })
                except (ValueError, IndexError) as e:
                    print(f"  ⚠️  Parse error at line {line_num}: {e}")
                    continue
    
    # Don't forget the last sentence if file doesn't end with blank line
    if current_sentence:
        sentences.append({
            "sent_id": sent_id,
            "text": current_text,
            "tokens": current_sentence,
            "num_tokens": len(current_sentence),
        })
    
    return sentences


def print_sentence_sample(sentence: Dict, max_tokens: int = 10):
    """Pretty-print a parsed sentence for verification."""
    print(f"\n  Text: {sentence['text']}")
    print(f"  Tokens: {sentence['num_tokens']}")
    print(f"  {'ID':>3} {'FORM':>15} {'UPOS':>6} {'FEATS':>30} {'HEAD':>4} {'DEPREL':>10}")
    print(f"  {'─'*3} {'─'*15} {'─'*6} {'─'*30} {'─'*4} {'─'*10}")
    for tok in sentence['tokens'][:max_tokens]:
        feats_short = tok['feats'][:30] if tok['feats'] != '_' else '-'
        print(f"  {tok['id']:>3} {tok['form']:>15} {tok['upos']:>6} "
              f"{feats_short:>30} {tok['head']:>4} {tok['deprel']:>10}")
    if len(sentence['tokens']) > max_tokens:
        print(f"  ... ({len(sentence['tokens']) - max_tokens} more tokens)")


# ─────────────────────────────────────────────
# 1. Download UD Arabic PADT
# ─────────────────────────────────────────────
def download_ud_arabic_padt():
    """Download Universal Dependencies Arabic PADT dataset."""
    
    print("\n" + "=" * 60)
    print("1. UNIVERSAL DEPENDENCIES — ARABIC PADT")
    print("=" * 60)
    
    ud_dir = DATA_DIR / "ud_arabic_padt"
    
    if ud_dir.exists() and any(ud_dir.glob("*.conllu")):
        print("  ✅ Already downloaded.")
    else:
        print("  📥 Cloning UD_Arabic-PADT...")
        try:
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/UniversalDependencies/UD_Arabic-PADT.git",
                str(ud_dir)
            ], check=True, capture_output=True, text=True)
            print("  ✅ Downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Download failed: {e.stderr}")
            return None
    
    # Parse all splits
    splits = {}
    for split_name in ['train', 'dev', 'test']:
        conllu_file = ud_dir / f"ar_padt-ud-{split_name}.conllu"
        if conllu_file.exists():
            sentences = parse_conllu(str(conllu_file))
            splits[split_name] = sentences
            print(f"  📊 {split_name}: {len(sentences)} sentences")
            
            # Show sample
            if sentences:
                print(f"\n  Sample from {split_name}:")
                print_sentence_sample(sentences[0])
        else:
            print(f"  ⚠️  {split_name} file not found: {conllu_file}")
    
    # Save parsed data
    output_path = DATA_DIR / "ud_padt_all.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    
    total = sum(len(s) for s in splits.values())
    size_mb = output_path.stat().st_size / 1e6
    print(f"\n  💾 Saved to: {output_path}")
    print(f"  📊 Total: {total} sentences ({size_mb:.1f} MB)")
    
    # Compute statistics
    all_sents = []
    for split_sents in splits.values():
        all_sents.extend(split_sents)
    
    lengths = [s['num_tokens'] for s in all_sents]
    if lengths:
        print(f"\n  Sentence length stats:")
        print(f"    Min: {min(lengths)}, Max: {max(lengths)}, "
              f"Mean: {sum(lengths)/len(lengths):.1f}")
        
        # POS tag distribution
        pos_counts = {}
        for s in all_sents:
            for t in s['tokens']:
                pos_counts[t['upos']] = pos_counts.get(t['upos'], 0) + 1
        print(f"\n  POS tag distribution (top 10):")
        for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {pos:>8}: {count:>6}")
        
        # Case feature distribution
        case_counts = {}
        for s in all_sents:
            for t in s['tokens']:
                feats = t['feats']
                if 'Case=' in feats:
                    for feat in feats.split('|'):
                        if feat.startswith('Case='):
                            case_val = feat.split('=')[1]
                            case_counts[case_val] = case_counts.get(case_val, 0) + 1
        if case_counts:
            print(f"\n  Case marking distribution:")
            for case, count in sorted(case_counts.items(), key=lambda x: -x[1]):
                print(f"    {case:>8}: {count:>6}")
    
    return splits


# ─────────────────────────────────────────────
# 2. Download CIDAR Dataset
# ─────────────────────────────────────────────
def download_cidar():
    """Download CIDAR Arabic instruction dataset."""
    
    print("\n" + "=" * 60)
    print("2. CIDAR DATASET (Arabic Instructions)")
    print("=" * 60)
    
    output_path = DATA_DIR / "cidar_raw.json"
    
    if output_path.exists():
        print("  ✅ Already downloaded.")
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  📊 {len(data)} examples")
        return data
    
    try:
        from datasets import load_dataset
        
        print("  📥 Downloading CIDAR from HuggingFace...")
        cidar = load_dataset("arbml/CIDAR", split="train")
        
        cidar_processed = []
        for item in cidar:
            cidar_processed.append({
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "source": "cidar"
            })
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cidar_processed, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ Downloaded: {len(cidar_processed)} examples")
        print(f"  💾 Saved to: {output_path}")
        
        # Show sample
        if cidar_processed:
            sample = cidar_processed[0]
            print(f"\n  Sample:")
            print(f"    Instruction: {sample['instruction'][:80]}...")
            print(f"    Output: {sample['output'][:80]}...")
        
        return cidar_processed
        
    except ImportError:
        print("  ⚠️  'datasets' library not installed. Install with: pip install datasets")
        print("  Skipping CIDAR download.")
        return None
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return None


# ─────────────────────────────────────────────
# 3. Download Arabic-LLM-Parsing
# ─────────────────────────────────────────────
def download_arabic_parsing():
    """Download Arabic LLM Parsing dataset."""
    
    print("\n" + "=" * 60)
    print("3. ARABIC-LLM-PARSING DATASET")
    print("=" * 60)
    
    parsing_dir = DATA_DIR / "arabic_llm_parsing"
    
    if parsing_dir.exists():
        print("  ✅ Already downloaded.")
    else:
        print("  📥 Cloning Arabic-LLM-Parsing...")
        try:
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/alsabahi2030/Arabic-LLM-Parsing.git",
                str(parsing_dir)
            ], check=True, capture_output=True, text=True)
            print("  ✅ Downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Download failed: {e.stderr}")
            return None
    
    # List available files
    print("  📁 Contents:")
    for f in sorted(parsing_dir.rglob("*")):
        if f.is_file() and not str(f).startswith('.git'):
            rel = f.relative_to(parsing_dir)
            size = f.stat().st_size
            if size > 0:
                print(f"    {rel} ({size/1024:.1f} KB)")
    
    return parsing_dir


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Arabic Syntax HRM — Dataset Download & Preparation    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # 1. UD Arabic PADT (most important — validates parser)
    ud_data = download_ud_arabic_padt()
    
    # 2. CIDAR
    cidar_data = download_cidar()
    
    # 3. Arabic-LLM-Parsing
    parsing_dir = download_arabic_parsing()
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    if ud_data:
        total_ud = sum(len(s) for s in ud_data.values())
        print(f"  ✅ UD Arabic PADT: {total_ud} sentences")
    else:
        print(f"  ❌ UD Arabic PADT: FAILED")
    
    if cidar_data:
        print(f"  ✅ CIDAR: {len(cidar_data)} examples")
    else:
        print(f"  ⚠️  CIDAR: skipped (install 'datasets' library)")
    
    if parsing_dir:
        print(f"  ✅ Arabic-LLM-Parsing: downloaded")
    else:
        print(f"  ⚠️  Arabic-LLM-Parsing: failed")
    
    data_size = sum(
        f.stat().st_size 
        for f in DATA_DIR.rglob('*') 
        if f.is_file()
    ) / 1e6
    print(f"\n  Total data directory size: {data_size:.1f} MB")
    print(f"  Location: {DATA_DIR}")
    
    # Next step
    print("\n" + "=" * 60)
    print("NEXT STEP")
    print("=" * 60)
    print("  Run: python scripts/02_build_syntax_grids.py")
    print("  This will convert parsed sentences to HRM-compatible grids.")


if __name__ == "__main__":
    main()
