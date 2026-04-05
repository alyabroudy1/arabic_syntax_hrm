"""
Arabiya Engine Demo — quick test + interactive + comparison modes.

Usage:
    python -m arabiya.scripts.demo                # quick test
    python -m arabiya.scripts.demo --compare      # show case ending impact
    python -m arabiya.scripts.demo --interactive   # type sentences
"""

import argparse
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set UTF-8 stdout once at module level
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from arabiya.engine import ArabiyaEngine, quick_test


def run_comparison():
    engine = ArabiyaEngine.create_with_mock()
    tests = [
        ('ذهب الطالب إلى المدرسة',
         'Basic VSO: subject=Nom, prep object=Gen'),
        ('كتب الطالب الدرس',
         'Transitive: subject=Nom, direct object=Acc'),
        ('إن العلم نور',
         'Inn: subject=Acc (ism inn), predicate=Nom (khabar)'),
        ('يكتب المعلمون الدروس في المدارس الكبيرة',
         'Present + SMP + multiple case assignments'),
    ]

    print("=" * 70)
    print("    ARABIYA ENGINE -- Case Ending Impact")
    print("=" * 70)

    for text, note in tests:
        result = engine.process(text)
        stems = []
        for s in result.sentences:
            for w in s.words:
                stems.append(w.stem_diacritized or w.bare)

        print(f"\n{'_' * 65}")
        print(f"  Input:       {text}")
        print(f"  Stem only:   {' '.join(stems)}")
        print(f"  + Case ends: {result.diacritized}")
        print(f"  Note:        {note}")

        modified = []
        for s in result.sentences:
            for w in s.words:
                if w.case_diacritic:
                    modified.append(f"[{w.bare}->{w.case_display}]")
        if modified:
            print(f"  Case marks:  {', '.join(modified)}")

    print(f"\n{'=' * 70}")


def run_interactive():
    print("=" * 60)
    print("    ARABIYA ENGINE -- Interactive")
    print("    Type Arabic text, 'quit' to exit")
    print("=" * 60)

    engine = ArabiyaEngine.create_with_mock()
    while True:
        try:
            text = input("\n  Arabic > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text in ('quit', 'exit', 'q'):
            break

        result = engine.process(text)
        print(f"  Output:   {result.diacritized}")
        print(f"  Coverage: {result.lexicon_coverage:.0%} | "
              f"{result.case_applied} case endings")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arabiya Demo')
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--compare', '-c', action='store_true')
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.compare:
        run_comparison()
    else:
        quick_test()
