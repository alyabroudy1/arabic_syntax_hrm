"""
Extract diptote word list from PADT data.

Strategy: Scan for words that are:
  - Genitive (Case=Gen) + Indefinite (Definite=Ind)
  - But have FATHA ending instead of KASRA in the Vform
  
These are diptotes (ممنوع من الصرف).
Also check for any explicit Diptote feature.
"""
import sys, os, json, re
from pathlib import Path
from collections import Counter

sys.path.insert(0, '.')

DIACRITIC_PATTERN = re.compile('[\u0617-\u061A\u064B-\u0655\u0670]')
ALL_DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0670')
FATHA = '\u064E'
KASRA = '\u0650'


def strip_diacritics(t):
    return DIACRITIC_PATTERN.sub('', t)


def get_final_diac(vform):
    """Get diacritic on last base letter."""
    chars = list(vform)
    i = len(chars) - 1
    trailing = ''
    while i >= 0 and chars[i] in ALL_DIACRITICS:
        trailing = chars[i] + trailing
        i -= 1
    return trailing


def main():
    padt_dir = Path('data/ud_arabic_padt')
    diptote_lemmas = Counter()
    diptote_words = set()
    
    total_checked = 0
    
    for split in ['train', 'dev', 'test']:
        path = padt_dir / f'ar_padt-ud-{split}.conllu'
        if not path.exists():
            continue
        
        print(f"Scanning {path.name}...")
        
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 10 or '-' in parts[0] or '.' in parts[0]:
                    continue
                
                form, lemma, upos, feats, misc = parts[1], parts[2], parts[3], parts[5], parts[9]
                
                # Method 1: Explicit Diptote feature
                if 'Diptote=Yes' in feats:
                    diptote_lemmas[lemma] += 1
                    diptote_words.add(strip_diacritics(form))
                    continue
                
                # Method 2: Genitive + Indefinite + Fatha ending
                if 'Case=Gen' not in feats:
                    continue
                if 'Definite=Ind' not in feats:
                    continue
                
                # Get Vform
                vform = ''
                for m in misc.split('|'):
                    if m.startswith('Vform='):
                        vform = m.split('=', 1)[1]
                        break
                if not vform or vform == '-':
                    continue
                
                total_checked += 1
                ending = get_final_diac(vform)
                
                # Diptote signature: Gen+Indef should have KASRA or KASRATAN
                # But if it has FATHA instead, it's a diptote
                if FATHA in ending and KASRA not in ending:
                    bare = strip_diacritics(form)
                    diptote_lemmas[lemma] += 1
                    diptote_words.add(bare)
    
    # Save
    os.makedirs('arabiya/data', exist_ok=True)
    
    result = {
        'lemmas': sorted(diptote_lemmas.keys()),
        'words': sorted(diptote_words),
        'stats': {
            'unique_lemmas': len(diptote_lemmas),
            'unique_surface_forms': len(diptote_words),
            'total_occurrences': sum(diptote_lemmas.values()),
        }
    }
    
    with open('arabiya/data/diptotes.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"  DIPTOTE EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"  Unique lemmas:        {len(diptote_lemmas):,}")
    print(f"  Unique surface forms: {len(diptote_words):,}")
    print(f"  Total occurrences:    {sum(diptote_lemmas.values()):,}")
    print(f"  Saved to: arabiya/data/diptotes.json")
    
    # Show top diptotes
    print(f"\n  Top 20 diptote lemmas:")
    for lemma, count in diptote_lemmas.most_common(20):
        print(f"    {count:4d}x  {lemma}")
    
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
