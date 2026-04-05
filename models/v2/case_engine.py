#!/usr/bin/env python3
"""
Arabic Case Ending Rule Engine (قواعد الإعراب)
================================================

Module 3B of the Arabiya Engine.

Given a word + syntactic case + morphological features from the HRM parser,
deterministically applies the correct case ending diacritic.

This is the KEY INNOVATION: case endings are NOT guessed —
they are DERIVED from the parser's syntactic analysis.

Case mapping:
  مرفوع (Nom) -> ضمة (ـُ) / واو (ـون) / ألف (ـان)
  منصوب (Acc) -> فتحة (ـَ) / ياء (ـين) / ألف (ـان → ـَيْن)
  مجرور (Gen) -> كسرة (ـِ) / ياء (ـين) / ألف (ـَيْن)
  مجزوم (Jus) -> سكون (ـْ) / حذف حرف العلة
  مبني  (Ind) -> keep as-is (indeclinable)
"""

import re
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set
from enum import Enum

# ═══════════════════════════════════════════════════
# Constants — Arabic Diacritics
# ═══════════════════════════════════════════════════

FATHA     = '\u064E'   # فتحة  ـَ
DAMMA     = '\u064F'   # ضمة   ـُ
KASRA     = '\u0650'   # كسرة  ـِ
SUKUN     = '\u0652'   # سكون  ـْ
SHADDA    = '\u0651'   # شدة   ـّ
TANWEEN_F = '\u064B'   # تنوين فتح  ـً
TANWEEN_D = '\u064C'   # تنوين ضم   ـٌ
TANWEEN_K = '\u064D'   # تنوين كسر  ـٍ

ALL_DIACRITICS = set([FATHA, DAMMA, KASRA, SUKUN, SHADDA,
                      TANWEEN_F, TANWEEN_D, TANWEEN_K,
                      '\u0670', '\u0653', '\u0654', '\u0655'])

# ═══════════════════════════════════════════════════
# Morphological Categories
# ═══════════════════════════════════════════════════

class Case(Enum):
    NOM = "Nom"      # مرفوع
    ACC = "Acc"      # منصوب
    GEN = "Gen"      # مجرور
    # JUS = "Jus"    # مجزوم (verbs only — handled separately)

class WordType(Enum):
    """Morphological word type affecting case ending realization."""
    REGULAR_SINGULAR = "regular_singular"          # مفرد عادي
    REGULAR_SINGULAR_ALEF_MAQSURA = "alef_maq"    # المقصور - ends in ى
    REGULAR_SINGULAR_TAA_MARBUTA = "taa_marbuta"   # ends in ة
    MANQUS = "manqus"                              # المنقوص - ends in ي (estimated Nom/Gen)
    SOUND_MASC_PLURAL = "sound_masc_plural"        # جمع مذكر سالم (-ون/-ين)
    SOUND_FEM_PLURAL = "sound_fem_plural"          # جمع مؤنث سالم (-ات)
    DUAL = "dual"                                  # مثنى (-ان/-ين)
    BROKEN_PLURAL = "broken_plural"                # جمع تكسير (like singular)
    FIVE_NOUNS = "five_nouns"                      # الأسماء الخمسة (أبو/أخو/حمو/فو/ذو)
    DIPTOTE = "diptote"                            # ممنوع من الصرف
    INDECLINABLE = "indeclinable"                  # مبني (particles, demonstratives)
    VERB_IMPERFECT = "verb_imperfect"              # فعل مضارع
    VERB_PAST = "verb_past"                        # فعل ماض (indeclinable base)
    VERB_IMPERATIVE = "verb_imperative"            # فعل أمر


@dataclass
class CaseEndingResult:
    """Result of applying case ending rules."""
    word: str                           # original word
    case: str                           # Nom/Acc/Gen
    word_type: str                      # detected word type
    ending_diacritic: str               # the diacritic to apply
    ending_description_ar: str          # Arabic description
    is_definite: bool                   # ال or construct state
    has_tanween: bool                   # requires tanween?
    confidence: float = 1.0             # rule confidence


# ═══════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════

def strip_diacritics(text: str) -> str:
    """Remove all diacritical marks."""
    return ''.join(c for c in text if c not in ALL_DIACRITICS)


def get_last_letter(word: str) -> str:
    """Get the last base letter (ignoring diacritics)."""
    bare = strip_diacritics(word)
    return bare[-1] if bare else ''


def get_last_two(word: str) -> str:
    """Get last two base letters."""
    bare = strip_diacritics(word)
    return bare[-2:] if len(bare) >= 2 else bare


def ends_with_taa_marbuta(word: str) -> bool:
    bare = strip_diacritics(word)
    return bare.endswith('ة')


def ends_with_alef_maqsura(word: str) -> bool:
    bare = strip_diacritics(word)
    return bare.endswith('ى')


def ends_with_yaa(word: str) -> bool:
    """Check if word ends in ي (U+064A) — potential manqus."""
    bare = strip_diacritics(word)
    return bare.endswith('\u064A')


# ─── Ibn Malik: المنقوص والمقصور والممدود ───
# المنقوص = noun ending in ي with kasra before it
# Case: Nom/Gen → estimated (hidden), Acc → visible fatha
#
# Detection signals (from PADT features):
#   1. Construct dual (Number=Dual + Definite=Cons) → ي replaces يْن
#   2. Active/passive participle of weak-laam roots (القاضي, الماضي)
#   3. Lemma differs from nisba pattern
#
# Nisba adjectives (ـيّ) are NOT manqus — they have shadda on ي
# and take normal case endings on the shadda.

# Known manqus lemma endings (roots with weak laam)
_MANQUS_LEMMAS = {
    # Active participles from weak-laam roots
    'ماضي', 'ثاني', 'قاضي', 'داعي', 'جاري', 'ساعي', 'باقي',
    'تالي', 'عالي', 'غالي', 'حاوي', 'ناحي', 'رامي', 'ساري',
    'هادي', 'وادي', 'آتي', 'باني', 'ناجي', 'راضي', 'حامي',
    'ماشي', 'موالي', 'أهالي', 'معاني', 'أراضي', 'ليالي',
    # Derived forms
    'مبني', 'معني', 'محامي', 'مقتضي', 'منتهي', 'مستوي',
    'مصطفي', 'متعاطي', 'متلقي', 'مستشفي',
    # Masdar/noun forms from weak roots
    'تعاطي', 'تلقي', 'تفادي', 'توالي', 'تراضي',
}

def is_manqus(word: str, feats: str, lemma: str = '') -> bool:
    """Detect المنقوص per Ibn Malik.
    
    Returns True if the word is manqus (estimated case in Nom/Gen).
    Manqus = noun/adj ending in ي from a weak-laam root.
    NOT nisba adjectives (which have shadda on the ي).
    
    Signals:
      1. Construct dual → ي from يْن deletion
      2. Known manqus lemma pattern
      3. Construct state + singular + not nisba
    """
    bare = strip_diacritics(word)
    if not bare.endswith('\u064A'):  # Must end in ي
        return False
    
    bare_lemma = strip_diacritics(lemma) if lemma else ''
    
    # 1. Construct dual: ي replaces يْن → no case on ي
    if 'Number=Dual' in feats and 'Definite=Cons' in feats:
        return True
    
    # 2. Known manqus lemma
    if bare_lemma in _MANQUS_LEMMAS:
        return True
    
    # 3. Construct state singular ending in ي where lemma ≠ form base
    #    (possessive: وزيري lemma=وزير → ي is suffix, treat as construct)
    if 'Definite=Cons' in feats and 'Number=Sing' not in feats:
        # Plural/dual constructs with ي → manqus-like behavior
        return True
    if 'Definite=Cons' in feats:
        # If lemma doesn't end in ي but form does → possessive suffix
        if bare_lemma and not bare_lemma.endswith('\u064A'):
            return True
    
    # 4. Pattern detection for broken plural manqus
    #    Words like: الأهالي, الأراضي, الموالي, الليالي, التعاطي
    #    These are broken plurals where the pattern ends in ي
    #    Pattern: أَفَاعِي / مَوَاعِي / فَوَاعِي (مفاعل but with ي ending)
    stem = bare[2:] if bare.startswith('ال') else bare
    if len(stem) >= 4:
        # If it's a broken plural ending in ي — typically manqus
        if 'Number=Plur' in feats:
            return True
        # Active/passive participle patterns: فَاعِي, مُفَاعِي, مُتَفَاعِي
        # If the letter before ي could be kasra (typical manqus indicator)
        # and the lemma ends in ي → manqus from weak root
        if bare_lemma and bare_lemma.endswith('\\u064A') and bare_lemma.endswith('\\u0649'):
            return True
    
    return False


def has_definite_article(word: str) -> bool:
    """Check for ال prefix."""
    bare = strip_diacritics(word)
    return bare.startswith('ال') or bare.startswith('ٱل')


def is_sound_masc_plural(word: str, feats: str) -> bool:
    """Check for جمع مذكر سالم (-ون/-ين ending).
    
    Detects SMP even when Gender=Masc is missing from features.
    Key distinction: SMP adds -ون/-ين to a SINGULAR STEM of 3+ letters.
    فعول plurals (شؤون, عيون, ديون) have ون as part of the pattern, not a suffix.
    """
    bare = strip_diacritics(word)
    if bare.startswith('ال'):
        stem = bare[2:]
    else:
        stem = bare
    
    if 'Number=Plur' not in feats:
        return False
    
    # Must end in ون or ين
    if not (stem.endswith('ون') or stem.endswith('ين')):
        # Handle construct state: مواطنو (ون minus final noon)
        if stem.endswith('و') and 'Definite=Cons' in feats:
            base = stem[:-1]
            return len(base) >= 3
        return False
    
    # The singular stem (before ون/ين) must be at least 3 characters
    # This prevents فعول patterns like شؤون (stem شؤ = 2 chars) from matching
    base = stem[:-2]
    if len(base) < 3:
        return False
    
    return True


# ─── Ibn Malik's صيغة منتهى الجموع ───
# Per Alfiyyat Ibn Malik:
#   وَكُلُّ جَمْعٍ مُشَبَّهٌ مَفَاعِلاَ أَوِ الْمَفَاعِيلَ بِمَنْعٍ كَافِلاَ
#
# Diptote broken plurals are EXACTLY those matching:
#   1. مَفَاعِل pattern: alef after letter 2, then 2 consonants (e.g., مساجد, قوالب, فواعل)
#   2. مَفَاعِيل pattern: alef after letter 2, then 3 letters with ي/و (e.g., مصابيح, أساليب)
# ALL other broken plurals are TRIPTOTE (take kasra/tanween normally).

def is_diptote_plural_pattern(word: str, lemma: str = '') -> bool:
    """Detect صيغة منتهى الجموع (Ibn Malik's diptote plural patterns).
    
    A broken plural is diptote IFF:
      - Pattern has ألف ساكنة after the 2nd radical
      - Followed by 2 consonants (مفاعل) or 3 with middle ي/و (مفاعيل)
    
    Examples of DIPTOTE plurals:
      مساجد, مكاتب, مدارس, فواعل, قوالب     (مفاعل)
      مصابيح, أساليب, مفاتيح, تماثيل          (مفاعيل)
    
    Examples of TRIPTOTE plurals (NOT this pattern):
      أسباب, أعمال, أقوال     (أفعال)
      دروس, شروط, جهود       (فعول)
      رجال, جبال              (فعال)
      غرف, دول, صور           (فعل)
      كتب, رسل               (فعل)
    """
    bare = strip_diacritics(word)
    if bare.startswith('ال'):
        bare = bare[2:]
    
    n = len(bare)
    if n < 4:
        return False  # Too short for منتهى الجموع
    
    # Look for ا (alef) at position 2 (3rd character, 0-indexed)
    # This captures: مَ-فَ-ا-عِ-ل, فَ-وَ-ا-عِ-ل, etc.
    # Position 2 means after the first two radicals
    
    # مفاعل pattern: 5 letters, alef at position 2
    # Examples: مساجد(مسأجد→5), قوالب(5), مدارس(5), فواعل(5)
    if n == 5 and bare[2] in ('ا', 'آ'):
        return True
    
    # مفاعيل pattern: 6+ letters, alef at position 2, then ي/و before last letter
    # Examples: مصابيح(6), أساليب(6), تماثيل(6), مفاتيح(6)
    if n == 6 and bare[2] in ('ا', 'آ') and bare[4] in ('ي', 'و'):
        return True
    
    # Also: 5 letters with alef at position 3 (فعالل, فعاليل)
    # Examples: دنانير(6), سراويل(6), عصافير(6)
    if n == 6 and bare[3] in ('ا', 'آ') and bare[4] in ('ي', 'و'):
        return True
    if n == 6 and bare[2] in ('ا', 'آ'):
        return True  # 6-letter with alef at 2
    
    # 7+ letters: extended patterns like تفاعيل, مستفاعل
    if n >= 7 and bare[2] in ('ا', 'آ'):
        return True
    
    return False


def is_sound_fem_plural(word: str, feats: str) -> bool:
    """Check for جمع مؤنث سالم (-ات ending)."""
    bare = strip_diacritics(word)
    if 'Number=Plur' in feats and 'Gender=Fem' in feats:
        return bare.endswith('ات')
    return bare.endswith('ات') and 'Number=Plur' in feats


def is_dual(word: str, feats: str) -> bool:
    """Check for مثنى (-ان/-ين ending)."""
    bare = strip_diacritics(word)
    if 'Number=Dual' in feats:
        return bare.endswith('ان') or bare.endswith('ين')
    return False


def is_five_nouns(word: str, lemma: str) -> bool:
    """Check for الأسماء الخمسة (أب أخ حم فم ذو)."""
    bare_lemma = strip_diacritics(lemma) if lemma else ''
    five = {'اب', 'أب', 'اخ', 'أخ', 'حم', 'فم', 'ذو', 'ذي', 'ذا'}
    return bare_lemma in five


def is_diptote(feats: str) -> bool:
    """Check for ممنوع من الصرف (diptote).
    
    Common diptotes: proper nouns, certain broken plural patterns (مفاعل، فواعل),
    adjectives on أفعل pattern, foreign words.
    """
    if 'Diptote=Yes' in feats:
        return True
    # Foreign words are diptotes
    if 'Foreign=Yes' in feats:
        return True
    return False


def _load_diptote_set() -> Set[str]:
    """Load PADT-extracted diptote word list (surface forms + lemmas)."""
    diptote_set = set()
    # Try multiple possible locations
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'arabiya', 'data', 'diptotes.json'),
        os.path.join(os.path.dirname(__file__), '..', 'arabiya', 'data', 'diptotes.json'),
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                diptote_set.update(data.get('words', []))
                diptote_set.update(data.get('lemmas', []))
                return diptote_set
            except Exception:
                pass
    return diptote_set


# Load once at module level
_DIPTOTE_SET = _load_diptote_set()


def is_known_diptote(word: str, lemma: str = '') -> bool:
    """Check if word or lemma is in the PADT-extracted diptote list."""
    bare = strip_diacritics(word)
    if bare in _DIPTOTE_SET:
        return True
    if lemma and strip_diacritics(lemma) in _DIPTOTE_SET:
        return True
    return False


def is_construct_state(deprel: str, feats: str) -> bool:
    """Check if word is in إضافة (construct/idafa) state.
    
    Construct state = no tanween, even if indefinite.
    In UD PADT, this is Definite=Cons or Definite=Com.
    """
    return 'Definite=Cons' in feats or 'Definite=Com' in feats


def is_definite(word: str, feats: str) -> bool:
    """Check if word is definite (no tanween)."""
    if has_definite_article(word):
        return True
    if 'Definite=Def' in feats:
        return True
    if 'Definite=Com' in feats or 'Definite=Cons' in feats:  # construct state
        return True
    return False


# ═══════════════════════════════════════════════════
# Case Ending Rule Engine
# ═══════════════════════════════════════════════════

class CaseEndingRuleEngine:
    """Deterministic case ending rules for Arabic nouns and adjectives.
    
    Given a word, its grammatical case (from the HRM parser),
    and morphological features, produces the correct final diacritic.
    
    Usage:
        engine = CaseEndingRuleEngine()
        result = engine.apply(word="الطالب", case="Nom", upos="NOUN",
                              feats="Case=Nom|Definite=Def|...", deprel="nsubj")
    """
    
    def detect_word_type(self, word: str, upos: str, feats: str,
                        lemma: str = "", deprel: str = "") -> WordType:
        """Detect morphological word type from features."""
        
        # Verbs
        if upos == 'VERB':
            if 'VerbForm=Fin' in feats and 'Mood=' in feats:
                return WordType.VERB_IMPERFECT
            if 'Aspect=Perf' in feats:
                return WordType.VERB_PAST
            if 'Mood=Imp' in feats:
                return WordType.VERB_IMPERATIVE
            return WordType.VERB_PAST  # default for verbs
        
        # Particles, prepositions, conjunctions — indeclinable
        if upos in ('ADP', 'CCONJ', 'SCONJ', 'PART', 'INTJ', 'PUNCT',
                     'DET', 'AUX', 'X'):
            return WordType.INDECLINABLE
        
        # Pronouns — mostly indeclinable
        if upos == 'PRON':
            return WordType.INDECLINABLE  # most pronouns are مبني
        
        # Relative pronouns (الذي/التي) — indeclinable even if tagged as other POS
        if 'PronType=Rel' in feats:
            return WordType.INDECLINABLE
        
        # Five nouns check
        if is_five_nouns(word, lemma):
            return WordType.FIVE_NOUNS
        
        # ══ المقصور (Alef maqsura) — Ibn Malik ══
        # Case endings are ALWAYS estimated (تعذّر) — alif cannot bear vowel.
        # Covers: ى ending AND ا ending when it's a maqsur plural
        # Examples: الفتى, أخرى, قتلى, ضحايا, قضايا
        #
        # EXCEPTION: Nisba adjectives written with ى instead of ي (Egyptian orthography)
        # e.g., الامريكى (lemma=أمريكي) — these are NOT maqsur, they take normal case.
        # Detection: if lemma ends in ي (U+064A), the ى is actually a dotless yaa.
        if ends_with_alef_maqsura(word):
            bare_lemma = strip_diacritics(lemma) if lemma else ''
            # If lemma ends in ي → this is a nisba/regular word, NOT maqsur
            if bare_lemma and bare_lemma.endswith('\u064A'):
                pass  # Fall through to regular/nisba handling below
            else:
                return WordType.REGULAR_SINGULAR_ALEF_MAQSURA
        
        # Also: words ending in ا (alef) that are broken plurals with estimated case
        # Pattern: فعايا/فعالا (like ضحايا, قضايا)
        # BUT NOT tanween-fath+alef (مجالًا) which is regular Acc indefinite!
        bare = strip_diacritics(word)
        if bare.endswith('\u0627') and len(bare) > 2:  # ends in alef
            stem = bare[2:] if bare.startswith('ال') else bare
            # Broken plurals ending in ا with pattern CاCيا or similar
            if stem.endswith('ايا'):
                return WordType.REGULAR_SINGULAR_ALEF_MAQSURA
            # BUT: single alef after consonant is NOT maqsur if it could be 
            # tanween fath + alef (كتابًا). Only count if it's a genuine maqsur plural.
        
        # ══ المنقوص (Manqus) — Ibn Malik ══
        # Nom/Gen: estimated (hidden), Acc: visible fatha (unless construct)
        # Also catches broken plural manqus: الأهالي, الأراضي, الموالي
        if ends_with_yaa(word) and upos in ('NOUN', 'ADJ', 'PROPN', 'NUM'):
            if is_manqus(word, feats, lemma):
                return WordType.MANQUS
        
        # Dual
        if is_dual(word, feats):
            return WordType.DUAL
        
        # Sound masculine plural
        if is_sound_masc_plural(word, feats):
            return WordType.SOUND_MASC_PLURAL
        
        # Sound feminine plural
        if is_sound_fem_plural(word, feats):
            return WordType.SOUND_FEM_PLURAL
        
        # Broken plural — Ibn Malik's صيغة منتهى الجموع rule
        # Diptote = مفاعل/مفاعيل patterns; everything else = triptote
        if 'Number=Plur' in feats and upos in ('NOUN', 'ADJ'):
            # Check if it's a known diptote from PADT list
            if is_known_diptote(word, lemma):
                return WordType.DIPTOTE
            # Check Ibn Malik's مفاعل/مفاعيل pattern
            if is_diptote_plural_pattern(word, lemma):
                return WordType.DIPTOTE
            # All other broken plurals are TRIPTOTE (normal endings)
            return WordType.BROKEN_PLURAL
        
        # Diptote — explicit mark, proper nouns, or in PADT diptote list
        if is_diptote(feats):
            return WordType.DIPTOTE
        if upos == 'PROPN':
            return WordType.DIPTOTE
        # Check PADT-extracted diptote set for regular nouns/adjs
        if is_known_diptote(word, lemma):
            return WordType.DIPTOTE
        
        # Taa marbuta ending
        if ends_with_taa_marbuta(word):
            return WordType.REGULAR_SINGULAR_TAA_MARBUTA
        
        # Default: regular singular
        if upos in ('NOUN', 'ADJ', 'PROPN', 'NUM', 'ADV'):
            return WordType.REGULAR_SINGULAR
        
        return WordType.INDECLINABLE
    
    def apply(self, word: str, case: str, upos: str, feats: str,
              lemma: str = "", deprel: str = "") -> CaseEndingResult:
        """Apply case ending rules to a word.
        
        Args:
            word: The Arabic word (may have partial diacritics)
            case: "Nom", "Acc", or "Gen" from parser
            upos: Universal POS tag
            feats: Morphological features string
            lemma: Lemma form
            deprel: Dependency relation
            
        Returns:
            CaseEndingResult with the diacritic to apply
        """
        word_type = self.detect_word_type(word, upos, feats, lemma, deprel)
        definite = is_definite(word, feats)
        construct = is_construct_state(deprel, feats)
        
        # ══ INDECLINABLE (مبني) ══
        if word_type == WordType.INDECLINABLE:
            return CaseEndingResult(
                word=word, case=case, word_type=word_type.value,
                ending_diacritic="", ending_description_ar="مبني",
                is_definite=definite, has_tanween=False, confidence=1.0
            )
        
        if word_type == WordType.VERB_PAST:
            return CaseEndingResult(
                word=word, case=case, word_type=word_type.value,
                ending_diacritic=FATHA, ending_description_ar="فعل ماض مبني على الفتح",
                is_definite=False, has_tanween=False, confidence=0.9
            )
        
        if word_type == WordType.VERB_IMPERATIVE:
            return CaseEndingResult(
                word=word, case=case, word_type=word_type.value,
                ending_diacritic=SUKUN, ending_description_ar="فعل أمر مبني على السكون",
                is_definite=False, has_tanween=False, confidence=0.9
            )
        
        # ══ VERB IMPERFECT (فعل مضارع) ══
        if word_type == WordType.VERB_IMPERFECT:
            return self._apply_verb_imperfect(word, case, feats, definite)
        
        # ══ ALEF MAQSURA (ـى) ══
        # Case is estimated (hidden) — no visible change
        # ══ المقصور (ـى) ══
        if word_type == WordType.REGULAR_SINGULAR_ALEF_MAQSURA:
            return CaseEndingResult(
                word=word, case=case, word_type=word_type.value,
                ending_diacritic="", ending_description_ar="اسم مقصور - إعراب تقديري",
                is_definite=definite, has_tanween=False, confidence=0.95
            )
        
        # ══ المنقوص (ـي) — Ibn Malik ══
        # Nom/Gen: estimated (hidden) — no visible diacritic
        # Acc: visible fatha (or fathatan if indefinite)
        if word_type == WordType.MANQUS:
            # In construct state with possessive suffix, ي IS the suffix
            # → no case diacritic on it (case is on the base word)
            if construct:
                return CaseEndingResult(
                    word=word, case=case, word_type=word_type.value,
                    ending_diacritic="",
                    ending_description_ar="مضاف - الإعراب على المضاف إليه",
                    is_definite=definite, has_tanween=False, confidence=0.9
                )
            if case == "Acc":
                needs_tanween = not definite
                diac = TANWEEN_F if needs_tanween else FATHA
                desc = "منصوب بفتحة ظاهرة على الياء" + (" المنونة" if needs_tanween else "")
            else:
                diac = ""
                if case == "Nom":
                    desc = "مرفوع بضمة مقدرة على الياء للثقل"
                else:
                    desc = "مجرور بكسرة مقدرة على الياء للثقل"
            return CaseEndingResult(
                word=word, case=case, word_type=word_type.value,
                ending_diacritic=diac, ending_description_ar=desc,
                is_definite=definite, has_tanween=False, confidence=0.9
            )
        
        # ══ FIVE NOUNS (الأسماء الخمسة) ══
        if word_type == WordType.FIVE_NOUNS:
            return self._apply_five_nouns(word, case, definite, construct)
        
        # ══ DUAL (المثنى) ══
        if word_type == WordType.DUAL:
            return self._apply_dual(word, case, definite)
        
        # ══ SOUND MASCULINE PLURAL (جمع مذكر سالم) ══
        if word_type == WordType.SOUND_MASC_PLURAL:
            return self._apply_sound_masc_plural(word, case, definite)
        
        # ══ SOUND FEMININE PLURAL (جمع مؤنث سالم) ══
        if word_type == WordType.SOUND_FEM_PLURAL:
            return self._apply_sound_fem_plural(word, case, definite, construct)
        
        # ══ DIPTOTE (ممنوع من الصرف) ══
        if word_type == WordType.DIPTOTE:
            return self._apply_diptote(word, case, definite)
        
        # ══ TAA MARBUTA ENDING (ـة) ══
        if word_type == WordType.REGULAR_SINGULAR_TAA_MARBUTA:
            return self._apply_taa_marbuta(word, case, definite, construct)
        
        # ══ REGULAR SINGULAR / BROKEN PLURAL ══
        return self._apply_regular(word, case, word_type.value, definite, construct)
    
    # ── Specific rule implementations ──
    
    def _apply_regular(self, word, case, wtype, definite, construct):
        """Regular singular nouns and broken plurals.
        
        Nom: ـُ (definite) / ـٌ (indefinite)
        Acc: ـَ (definite) / ـً (indefinite)
        Gen: ـِ (definite) / ـٍ (indefinite)
        """
        needs_tanween = not definite and not construct
        
        if case == "Nom":
            diac = TANWEEN_D if needs_tanween else DAMMA
            desc = "مرفوع بالضمة" + (" المنونة" if needs_tanween else "")
        elif case == "Acc":
            diac = TANWEEN_F if needs_tanween else FATHA
            desc = "منصوب بالفتحة" + (" المنونة" if needs_tanween else "")
        elif case == "Gen":
            diac = TANWEEN_K if needs_tanween else KASRA
            desc = "مجرور بالكسرة" + (" المنونة" if needs_tanween else "")
        else:
            diac = ""
            desc = "غير معرب"
        
        return CaseEndingResult(
            word=word, case=case, word_type=wtype,
            ending_diacritic=diac, ending_description_ar=desc,
            is_definite=definite, has_tanween=needs_tanween
        )
    
    def _apply_taa_marbuta(self, word, case, definite, construct):
        """Words ending in ة — same diacritics as regular but on the ة."""
        needs_tanween = not definite and not construct
        
        if case == "Nom":
            diac = TANWEEN_D if needs_tanween else DAMMA
            desc = "مرفوع بالضمة على التاء المربوطة"
        elif case == "Acc":
            diac = TANWEEN_F if needs_tanween else FATHA
            desc = "منصوب بالفتحة على التاء المربوطة"
        elif case == "Gen":
            diac = TANWEEN_K if needs_tanween else KASRA
            desc = "مجرور بالكسرة على التاء المربوطة"
        else:
            diac = ""
            desc = "غير معرب"
        
        return CaseEndingResult(
            word=word, case=case, word_type="taa_marbuta",
            ending_diacritic=diac, ending_description_ar=desc,
            is_definite=definite, has_tanween=needs_tanween
        )
    
    def _apply_sound_masc_plural(self, word, case, definite):
        """Sound masculine plural: -ون (Nom) / -ين (Acc/Gen).
        
        The FINAL letter is always نون which gets FATHA (ـونَ / ـينَ).
        Case is reflected in the suffix change (ون vs ين), not the final diacritic.
        """
        if case == "Nom":
            desc = "مرفوع بالواو (جمع مذكر سالم)"
        else:
            desc = "منصوب/مجرور بالياء (جمع مذكر سالم)"
        
        # Final نون always gets FATHA
        diac = FATHA
        
        return CaseEndingResult(
            word=word, case=case, word_type="sound_masc_plural",
            ending_diacritic=diac, ending_description_ar=desc,
            is_definite=definite, has_tanween=False
        )
    
    def _apply_sound_fem_plural(self, word, case, definite, construct):
        """Sound feminine plural (-ات).
        
        IMPORTANT EXCEPTION: Acc uses كسرة, NOT فتحة!
        Nom: -اتُ
        Acc: -اتِ (not -اتَ!)
        Gen: -اتِ
        """
        needs_tanween = not definite and not construct
        
        if case == "Nom":
            diac = TANWEEN_D if needs_tanween else DAMMA
            desc = "مرفوع بالضمة (جمع مؤنث سالم)"
        else:  # Acc or Gen — BOTH use kasra!
            diac = TANWEEN_K if needs_tanween else KASRA
            desc = "منصوب/مجرور بالكسرة (جمع مؤنث سالم)"
        
        return CaseEndingResult(
            word=word, case=case, word_type="sound_fem_plural",
            ending_diacritic=diac, ending_description_ar=desc,
            is_definite=definite, has_tanween=needs_tanween
        )
    
    def _apply_dual(self, word, case, definite):
        """Dual: -ان (Nom) / -ين (Acc/Gen)."""
        if case == "Nom":
            diac = KASRA  # on the نون: ـانِ
            desc = "مرفوع بالألف (مثنى)"
        else:  # Acc or Gen
            diac = KASRA  # ـَيْنِ
            desc = "منصوب/مجرور بالياء (مثنى)"
        
        return CaseEndingResult(
            word=word, case=case, word_type="dual",
            ending_diacritic=diac, ending_description_ar=desc,
            is_definite=definite, has_tanween=False
        )
    
    def _apply_five_nouns(self, word, case, definite, construct):
        """Five nouns: أب أخ حم فم ذو.
        
        Nom: -و (e.g., أبو)
        Acc: -ا (e.g., أبا)
        Gen: -ي (e.g., أبي)
        
        Only work in construct state (مضاف).
        """
        if not construct:
            # Fall back to regular case marking
            return self._apply_regular(word, case, "five_nouns_non_construct",
                                       definite, construct)
        
        if case == "Nom":
            diac = DAMMA
            desc = "مرفوع بالواو (أسماء خمسة)"
        elif case == "Acc":
            diac = FATHA
            desc = "منصوب بالألف (أسماء خمسة)"
        else:  # Gen
            diac = KASRA
            desc = "مجرور بالياء (أسماء خمسة)"
        
        return CaseEndingResult(
            word=word, case=case, word_type="five_nouns",
            ending_diacritic=diac, ending_description_ar=desc,
            is_definite=definite, has_tanween=False
        )
    
    def _apply_diptote(self, word, case, definite):
        """Diptote (ممنوع من الصرف).
        
        KEY EXCEPTION: Gen uses فتحة instead of كسرة!
        No tanween ever (unless preceded by ال or in construct).
        
        Nom: ـُ
        Acc: ـَ
        Gen: ـَ (NOT كسرة!)
        """
        if definite:
            # With ال, diptotes decline normally
            if case == "Nom":
                diac = DAMMA
                desc = "مرفوع بالضمة (ممنوع من الصرف معرّف)"
            elif case == "Acc":
                diac = FATHA
                desc = "منصوب بالفتحة (ممنوع من الصرف)"
            else:
                diac = KASRA  # with ال, genitive is normal kasra
                desc = "مجرور بالكسرة (ممنوع من الصرف معرّف)"
        else:
            if case == "Nom":
                diac = DAMMA
                desc = "مرفوع بالضمة (ممنوع من الصرف)"
            elif case == "Acc":
                diac = FATHA
                desc = "منصوب بالفتحة (ممنوع من الصرف)"
            else:
                diac = FATHA  # THIS IS THE KEY EXCEPTION
                desc = "مجرور بالفتحة نيابة عن الكسرة (ممنوع من الصرف)"
        
        return CaseEndingResult(
            word=word, case=case, word_type="diptote",
            ending_diacritic=diac, ending_description_ar=desc,
            is_definite=definite, has_tanween=False  # diptotes never get tanween
        )
    
    def _apply_verb_imperfect(self, word, case, feats, definite):
        """Imperfect/present tense verb (فعل مضارع).
        
        Nom (مرفوع): ـُ (default) or ثبوت النون (five verbs)
        Acc (منصوب): ـَ after أن/لن/كي
        Jus (مجزوم): ـْ after لم/لا الناهية
        """
        # Check if it's one of the "five verbs" (الأفعال الخمسة)
        bare = strip_diacritics(word)
        is_five_verb = (bare.endswith('ون') or bare.endswith('ان') or
                        bare.endswith('ين'))
        
        if case == "Nom":
            if is_five_verb:
                desc = "مرفوع بثبوت النون (أفعال خمسة)"
                diac = FATHA  # the نون at end: ـونَ
            else:
                diac = DAMMA
                desc = "مرفوع بالضمة الظاهرة"
        elif case == "Acc":
            if is_five_verb:
                desc = "منصوب بحذف النون (أفعال خمسة)"
                diac = FATHA
            else:
                diac = FATHA
                desc = "منصوب بالفتحة"
        elif case == "Gen":
            # Gen on verbs is rare, treat like Nom
            diac = DAMMA
            desc = "مرفوع بالضمة (فعل مضارع)"
        else:
            diac = SUKUN
            desc = "مجزوم بالسكون"
        
        return CaseEndingResult(
            word=word, case=case, word_type="verb_imperfect",
            ending_diacritic=diac, ending_description_ar=desc,
            is_definite=False, has_tanween=False
        )
    
    def get_expected_final_diac(self, result: CaseEndingResult) -> str:
        """Get the expected final diacritic character for comparison."""
        return result.ending_diacritic


# ═══════════════════════════════════════════════════
# Quick Test
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    engine = CaseEndingRuleEngine()
    
    test_cases = [
        # (word, case, upos, feats, lemma, deprel, expected_diac)
        ("الطالب", "Nom", "NOUN", "Case=Nom|Definite=Def|Gender=Masc|Number=Sing", "طالب", "nsubj", DAMMA),
        ("الكتاب", "Acc", "NOUN", "Case=Acc|Definite=Def|Gender=Masc|Number=Sing", "كتاب", "obj", FATHA),
        ("المدرسة", "Gen", "NOUN", "Case=Gen|Definite=Def|Gender=Fem|Number=Sing", "مدرسة", "nmod", KASRA),
        ("طالب", "Nom", "NOUN", "Case=Nom|Definite=Ind|Gender=Masc|Number=Sing", "طالب", "nsubj", TANWEEN_D),
        ("كتابا", "Acc", "NOUN", "Case=Acc|Definite=Ind|Gender=Masc|Number=Sing", "كتاب", "obj", TANWEEN_F),
        ("المعلمون", "Nom", "NOUN", "Case=Nom|Definite=Def|Gender=Masc|Number=Plur", "معلم", "nsubj", DAMMA),
        ("المعلمين", "Gen", "NOUN", "Case=Gen|Definite=Def|Gender=Masc|Number=Plur", "معلم", "nmod", KASRA),
    ]
    
    print("="*70)
    print("Case Ending Rule Engine -- Quick Test")
    print("="*70)
    
    correct = 0
    for word, case, upos, feats, lemma, deprel, expected in test_cases:
        result = engine.apply(word, case, upos, feats, lemma, deprel)
        match = result.ending_diacritic == expected
        correct += match
        status = "OK" if match else "FAIL"
        print(f"  [{status}] {word:15s} case={case:3s} -> {result.ending_description_ar}")
    
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.0f}%)")
