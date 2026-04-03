#!/usr/bin/env python3
"""
Case Ending Evaluation on PADT Gold Data
==========================================

PROOF OF CONCEPT for Arabiya Engine's key innovation:
syntax-driven diacritization of case endings.

This script:
1. Reads PADT CoNLL-U test data (gold case tags + gold Vform diacritics)
2. For each word with a Case feature, applies the CaseEndingRuleEngine
3. Compares the predicted ending diacritic against the gold Vform's final diacritic
4. Reports accuracy (target: >85%)

If this works, the entire Arabiya Engine concept is proven.
"""

import sys
import os
import re
from pathlib import Path
from collections import defaultdict, Counter

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Direct import to avoid loading heavy parser
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "case_engine", PROJECT_ROOT / "models" / "v2" / "case_engine.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CaseEndingRuleEngine = _mod.CaseEndingRuleEngine
strip_diacritics = _mod.strip_diacritics
ALL_DIACRITICS = _mod.ALL_DIACRITICS
FATHA = _mod.FATHA
DAMMA = _mod.DAMMA
KASRA = _mod.KASRA
SUKUN = _mod.SUKUN
SHADDA = _mod.SHADDA
TANWEEN_F = _mod.TANWEEN_F
TANWEEN_D = _mod.TANWEEN_D
TANWEEN_K = _mod.TANWEEN_K


# ═══════════════════════════════════════════════════
# CoNLL-U Parser
# ═══════════════════════════════════════════════════

def parse_conllu(filepath):
    """Parse CoNLL-U file, yield word-level dicts."""
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 10:
                continue
            # Skip multi-word tokens (e.g., "1-2")
            if '-' in parts[0] or '.' in parts[0]:
                continue
            
            idx, form, lemma, upos, xpos, feats, head, deprel, deps, misc = parts
            
            # Extract Vform from misc
            vform = ""
            for m in misc.split('|'):
                if m.startswith('Vform='):
                    vform = m.split('=', 1)[1]
                    break
            
            # Extract Case from feats
            case = ""
            for f2 in feats.split('|'):
                if f2.startswith('Case='):
                    case = f2.split('=')[1]
                    break
            
            yield {
                'idx': idx,
                'form': form,
                'lemma': lemma,
                'upos': upos,
                'xpos': xpos,
                'feats': feats,
                'head': head,
                'deprel': deprel,
                'misc': misc,
                'vform': vform,
                'case': case,
            }


def get_final_diacritics(vform):
    """Extract the case-relevant diacritic from the Vform.
    
    Returns tuple: (last_base_letter, diacritics_on_it)
    
    Special handling for tanween fath + alef (تنوين النصب):
    In Arabic, accusative indefinite is written as ًا where the tanween
    is on the letter BEFORE the final alef. e.g., كتابًا
    We need to detect this and return the tanween_f as the case diacritic.
    """
    if not vform or vform == '-':
        return '', ''
    
    chars = list(vform)
    i = len(chars) - 1
    
    # Collect trailing diacritics on last letter
    trailing_diacs = ''
    while i >= 0 and chars[i] in ALL_DIACRITICS:
        trailing_diacs = chars[i] + trailing_diacs
        i -= 1
    
    if i < 0:
        return '', trailing_diacs
    
    last_letter = chars[i]
    
    # SPECIAL CASE: tanween fath + alef (تنوين النصب)
    # If last base letter is alef (ا) with no diacritic,
    # check the previous letter for tanween fath
    if last_letter == '\u0627' and not trailing_diacs:
        j = i - 1
        prev_diacs = ''
        while j >= 0 and chars[j] in ALL_DIACRITICS:
            prev_diacs = chars[j] + prev_diacs
            j -= 1
        if TANWEEN_F in prev_diacs:
            return last_letter, TANWEEN_F
    
    return last_letter, trailing_diacs


def simplify_diac(diac_str):
    """Simplify a diacritic string to its 'case-relevant' component.
    
    We only care about the case vowel:
    - FATHA, DAMMA, KASRA, SUKUN
    - TANWEEN_F, TANWEEN_D, TANWEEN_K
    - SHADDA + any of the above
    
    Returns the primary case diacritic.
    """
    if not diac_str:
        return ''
    
    # If shadda is present, we care about what's WITH it
    has_shadda = SHADDA in diac_str
    
    # Check for tanween first (highest specificity)
    if TANWEEN_F in diac_str:
        return TANWEEN_F
    if TANWEEN_D in diac_str:
        return TANWEEN_D
    if TANWEEN_K in diac_str:
        return TANWEEN_K
    
    # Then simple vowels
    if DAMMA in diac_str:
        return DAMMA
    if FATHA in diac_str:
        return FATHA
    if KASRA in diac_str:
        return KASRA
    if SUKUN in diac_str:
        return SUKUN
    
    return diac_str if diac_str else ''


# ═══════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════

def evaluate(conllu_path, max_words=None, verbose=False):
    """Run case ending evaluation on PADT data.
    
    Uses GOLD case tags (from the treebank) to test the rule engine.
    This isolates the rule engine's accuracy from the parser's case prediction.
    """
    engine = CaseEndingRuleEngine()
    
    total = 0
    correct = 0
    correct_class = 0  # correct case class (ignoring tanween)
    
    errors_by_type = Counter()
    errors_by_case = Counter()
    case_counts = Counter()
    wtype_counts = Counter()
    wtype_correct = Counter()
    
    examples = []
    
    for word_data in parse_conllu(conllu_path):
        case = word_data['case']
        vform = word_data['vform']
        upos = word_data['upos']
        
        # Only evaluate words that HAVE a case feature and a valid Vform
        if not case or not vform or vform == '-':
            continue
        
        # Apply rule engine with GOLD case
        result = engine.apply(
            word=word_data['form'],
            case=case,
            upos=upos,
            feats=word_data['feats'],
            lemma=word_data['lemma'],
            deprel=word_data['deprel'],
        )
        
        # Get gold final diacritic from Vform
        _, gold_diac_str = get_final_diacritics(vform)
        gold_diac = simplify_diac(gold_diac_str)
        
        predicted_diac = result.ending_diacritic
        
        # Skip indeclinable words (no case ending to predict)
        if result.word_type == "indeclinable" or result.word_type == "verb_past":
            continue
        
        total += 1
        case_counts[case] += 1
        wtype_counts[result.word_type] += 1
        
        # Exact match (including tanween)
        is_correct = (predicted_diac == gold_diac)
        
        # Class match (same case vowel, ignoring tanween)
        def to_class(d):
            if d in (DAMMA, TANWEEN_D): return 'damm'
            if d in (FATHA, TANWEEN_F): return 'fath'
            if d in (KASRA, TANWEEN_K): return 'kasr'
            if d == SUKUN: return 'sukun'
            return 'other'
        
        is_class_correct = (to_class(predicted_diac) == to_class(gold_diac))
        
        if is_correct:
            correct += 1
            wtype_correct[result.word_type] += 1
        else:
            errors_by_type[result.word_type] += 1
            errors_by_case[case] += 1
            
            if verbose or len(examples) < 30:
                examples.append({
                    'word': word_data['form'],
                    'vform': vform,
                    'case': case,
                    'upos': upos,
                    'wtype': result.word_type,
                    'predicted': repr(predicted_diac),
                    'gold': repr(gold_diac),
                    'desc': result.ending_description_ar,
                    'feats': word_data['feats'][:60],
                })
        
        if is_class_correct:
            correct_class += 1
        
        if max_words and total >= max_words:
            break
    
    return {
        'total': total,
        'correct': correct,
        'correct_class': correct_class,
        'accuracy': correct / total if total > 0 else 0,
        'class_accuracy': correct_class / total if total > 0 else 0,
        'errors_by_type': errors_by_type,
        'errors_by_case': errors_by_case,
        'case_counts': case_counts,
        'wtype_counts': wtype_counts,
        'wtype_correct': wtype_correct,
        'examples': examples,
    }


def main():
    import sys as _sys
    import io
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    conllu_path = PROJECT_ROOT / "data" / "ud_arabic_padt" / "ar_padt-ud-test.conllu"
    
    if not conllu_path.exists():
        print(f"ERROR: {conllu_path} not found")
        return
    
    print("="*70)
    print("Arabiya Engine -- Case Ending Proof of Concept")
    print("Testing CaseEndingRuleEngine against PADT Gold Data")
    print("="*70)
    print(f"\nData: {conllu_path.name}")
    print("Using: GOLD case tags (isolates rule engine accuracy)")
    
    results = evaluate(conllu_path, verbose=False)
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Total words with case:    {results['total']:,}")
    print(f"  Exact match (w/ tanween): {results['correct']:,} / {results['total']:,} = {results['accuracy']:.1%}")
    print(f"  Case class match:         {results['correct_class']:,} / {results['total']:,} = {results['class_accuracy']:.1%}")
    
    # ── By case ──
    print(f"\n  By Case:")
    for case in ['Nom', 'Acc', 'Gen']:
        ct = results['case_counts'].get(case, 0)
        err = results['errors_by_case'].get(case, 0)
        acc = (ct - err) / ct if ct > 0 else 0
        print(f"    {case}: {ct - err:,}/{ct:,} = {acc:.1%}")
    
    # ── By word type ──
    print(f"\n  By Word Type:")
    for wtype, count in sorted(results['wtype_counts'].items(), key=lambda x: -x[1]):
        corr = results['wtype_correct'].get(wtype, 0)
        acc = corr / count if count > 0 else 0
        err = results['errors_by_type'].get(wtype, 0)
        marker = " <<" if acc < 0.80 else ""
        print(f"    {wtype:30s}: {corr:5d}/{count:5d} = {acc:.1%}{marker}")
    
    # ── Error examples ──
    if results['examples']:
        print(f"\n  Sample Errors (first 20):")
        print(f"  {'word':15s} {'vform':20s} {'case':4s} {'wtype':20s} {'predicted':12s} {'gold':12s}")
        print(f"  {'-'*90}")
        for ex in results['examples'][:20]:
            print(f"  {ex['word']:15s} {ex['vform']:20s} {ex['case']:4s} "
                  f"{ex['wtype']:20s} {ex['predicted']:12s} {ex['gold']:12s}")
    
    # ── Verdict ──
    print(f"\n{'='*70}")
    acc = results['accuracy']
    if acc >= 0.85:
        print(f"  >>> PROOF OF CONCEPT: PASSED ({acc:.1%} >= 85%)")
        print(f"  >>> Arabiya Engine is VIABLE. Case endings from syntax WORK.")
    elif acc >= 0.70:
        print(f"  >>> PROMISING ({acc:.1%}) but needs rule refinement")
    else:
        print(f"  >>> NEEDS WORK ({acc:.1%}) — review error patterns")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
