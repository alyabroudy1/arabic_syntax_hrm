#!/usr/bin/env python3
"""
Script 17: Export Android Assets
=================================

Exports all data files needed by the Android Arabiya TTS Engine:
  1. lexicon.json     — 18K word stem lexicon
  2. diptotes.json    — diptote lemma set  
  3. manqus_lemmas.json — manqus word set
  4. foreign_lemmas.json — foreign indeclinable words

Output: android/src/main/assets/
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ASSETS_DIR = PROJECT_ROOT / "android" / "src" / "main" / "assets"

def export_lexicon():
    """Copy lexicon.json to assets."""
    src = PROJECT_ROOT / "arabiya" / "data" / "lexicon.json"
    if not src.exists():
        print(f"  ⚠  Lexicon not found: {src}")
        return 0
    
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dst = ASSETS_DIR / "lexicon.json"
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    
    size_mb = dst.stat().st_size / 1e6
    print(f"  ✓ lexicon.json: {len(data)} words ({size_mb:.1f} MB)")
    return len(data)

def export_diptotes():
    """Export diptote lemma set."""
    src = PROJECT_ROOT / "arabiya" / "data" / "diptotes.json"
    if not src.exists():
        print(f"  ⚠  Diptotes not found: {src}")
        return 0
    
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract unique lemmas
    if isinstance(data, dict):
        lemmas = sorted(set(data.get('lemmas', [])))
    elif isinstance(data, list):
        lemmas = sorted(set(data))
    else:
        lemmas = []
    
    dst = ASSETS_DIR / "diptotes.json"
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(lemmas, f, ensure_ascii=False, separators=(',', ':'))
    
    print(f"  ✓ diptotes.json: {len(lemmas)} lemmas")
    return len(lemmas)

def export_manqus():
    """Export manqus lemma set from case_engine.py."""
    from models.v2.case_engine import _MANQUS_LEMMAS
    
    lemmas = sorted(_MANQUS_LEMMAS)
    dst = ASSETS_DIR / "manqus_lemmas.json"
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(lemmas, f, ensure_ascii=False, separators=(',', ':'))
    
    print(f"  ✓ manqus_lemmas.json: {len(lemmas)} lemmas")
    return len(lemmas)

def export_foreign():
    """Export foreign indeclinable word set."""
    from models.v2.case_engine import _FOREIGN_INDECL_LEMMAS
    
    lemmas = sorted(_FOREIGN_INDECL_LEMMAS)
    dst = ASSETS_DIR / "foreign_lemmas.json"
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(lemmas, f, ensure_ascii=False, separators=(',', ':'))
    
    print(f"  ✓ foreign_lemmas.json: {len(lemmas)} lemmas")
    return len(lemmas)

def main():
    print("=" * 60)
    print("  Arabiya Engine — Android Asset Export")
    print("=" * 60)
    
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    n_lex = export_lexicon()
    n_dip = export_diptotes()
    n_man = export_manqus()
    n_for = export_foreign()
    
    total_size = sum(f.stat().st_size for f in ASSETS_DIR.glob("*.json")) / 1e6
    
    print(f"\n{'─' * 40}")
    print(f"  Total assets: {total_size:.1f} MB")
    print(f"  - Lexicon:  {n_lex} words")
    print(f"  - Diptotes: {n_dip} lemmas")
    print(f"  - Manqus:   {n_man} lemmas")
    print(f"  - Foreign:  {n_for} lemmas")
    print(f"  Output: {ASSETS_DIR}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
