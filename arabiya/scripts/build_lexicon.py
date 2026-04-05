"""
Build stem diacritization lexicon from PADT CoNLL-U data.

Usage:
    python -m arabiya.scripts.build_lexicon --padt-dir data/ud_arabic_padt/
    python -m arabiya.scripts.build_lexicon --conllu data/ud_arabic_padt/ar_padt-ud-train.conllu
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from arabiya.stem_diacritizer import StemDiacritizer


def build_lexicon(conllu_paths=None, text_paths=None,
                  output_path='arabiya/data/lexicon.json'):
    diacritizer = StemDiacritizer()
    total = 0

    if conllu_paths:
        for path in conllu_paths:
            if os.path.exists(path):
                print(f"Processing CoNLL-U: {path}")
                n = diacritizer.build_from_conllu(path)
                total += n
                print(f"  -> {n} entries extracted")
            else:
                print(f"[WARNING] Not found: {path}")

    if text_paths:
        for path in text_paths:
            if os.path.exists(path):
                print(f"Processing text: {path}")
                n = diacritizer.build_from_diacritized_file(path)
                total += n
            else:
                print(f"[WARNING] Not found: {path}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    diacritizer.save_lexicon(output_path)

    stats = diacritizer.get_stats()
    print(f"\n{'=' * 50}")
    print(f"  LEXICON BUILD COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Unique words:    {stats['unique_words']:,}")
    print(f"  Total forms:     {stats['total_forms']:,}")
    print(f"  Ambiguous words: {stats['ambiguous_words']:,}")
    print(f"  Saved to:        {output_path}")
    sz = os.path.getsize(output_path)
    print(f"  File size:       {sz / 1024:.1f} KB")
    print(f"{'=' * 50}")

    return diacritizer


def build_from_padt_directory(padt_dir, output_path):
    conllu_files = []
    for split in ['train', 'dev', 'test']:
        path = os.path.join(padt_dir, f'ar_padt-ud-{split}.conllu')
        if os.path.exists(path):
            conllu_files.append(path)
            print(f"  Found: {path}")
    if not conllu_files:
        print(f"[ERROR] No CoNLL-U files in {padt_dir}")
        return None
    return build_lexicon(conllu_paths=conllu_files, output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Arabiya lexicon')
    parser.add_argument('--conllu', nargs='*', default=[])
    parser.add_argument('--text', nargs='*', default=[])
    parser.add_argument('--padt-dir', type=str, default=None)
    parser.add_argument('--output', type=str, default='arabiya/data/lexicon.json')

    args = parser.parse_args()

    if args.padt_dir:
        build_from_padt_directory(args.padt_dir, args.output)
    elif args.conllu or args.text:
        build_lexicon(args.conllu, args.text, args.output)
    else:
        print("Usage: python -m arabiya.scripts.build_lexicon "
              "--padt-dir data/ud_arabic_padt/")
