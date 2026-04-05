"""
Core Arabic utilities, Unicode constants, and shared data types.

FOUNDATION of the Arabiya Engine. Every Arabic character
manipulation flows through here.

CRITICAL: Arabic diacritics are Unicode COMBINING CHARACTERS.
They appear AFTER the base letter: "بُ" = U+0628 (baa) + U+064F (damma).
Every function here is aware of this.

CASE TAG STANDARD (Decision #1):
    Internal format is UD-style strings: "Nom", "Acc", "Gen", "Jus", None
    Arabic display labels are OUTPUT-ONLY via CASE_DISPLAY_MAP.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import re

# ═══════════════════════════════════════════════════════════
#                 ARABIC UNICODE CONSTANTS
# ═══════════════════════════════════════════════════════════

FATHATAN  = '\u064B'  # ً  — tanween fath
DAMMATAN  = '\u064C'  # ٌ  — tanween damm
KASRATAN  = '\u064D'  # ٍ  — tanween kasr
FATHA     = '\u064E'  # َ  — fatha
DAMMA     = '\u064F'  # ُ  — damma
KASRA     = '\u0650'  # ِ  — kasra
SHADDA    = '\u0651'  # ّ  — shadda (gemination)
SUKUN     = '\u0652'  # ْ  — sukun (no vowel)

SUPERSCRIPT_ALEF = '\u0670'
MADDAH    = '\u0653'
HAMZA_ABOVE = '\u0654'
HAMZA_BELOW = '\u0655'

ALL_DIACRITICS = frozenset({
    FATHATAN, DAMMATAN, KASRATAN,
    FATHA, DAMMA, KASRA,
    SHADDA, SUKUN,
    SUPERSCRIPT_ALEF, MADDAH,
    HAMZA_ABOVE, HAMZA_BELOW,
    '\u0617', '\u0618', '\u0619', '\u061A',
})

CASE_DIACRITICS = frozenset({
    FATHATAN, DAMMATAN, KASRATAN,
    FATHA, DAMMA, KASRA, SUKUN,
})

DIACRITIC_PATTERN = re.compile('[\u0617-\u061A\u064B-\u0655\u0670]')

# Special characters
TATWEEL   = '\u0640'
ALEF      = '\u0627'
ALEF_MADDA = '\u0622'
ALEF_HAMZA_ABOVE = '\u0623'
ALEF_HAMZA_BELOW = '\u0625'
ALEF_WASLA = '\u0671'
ALEF_MAQSURA = '\u0649'
TAA_MARBUTA = '\u0629'
HAMZA     = '\u0621'
WAW       = '\u0648'
YAA       = '\u064A'
LAM       = '\u0644'
NOON      = '\u0646'

# ═══════════════════════════════════════════════════════════
#                 CASE TAG STANDARD (UD internally)
# ═══════════════════════════════════════════════════════════

CASE_DISPLAY_MAP = {
    "Nom": "مرفوع",
    "Acc": "منصوب",
    "Gen": "مجرور",
    "Jus": "مجزوم",
    None:  "مبني",
}

CASE_FROM_ARABIC = {v: k for k, v in CASE_DISPLAY_MAP.items()}

def case_to_arabic(case_tag: Optional[str]) -> str:
    """Convert UD case tag to Arabic display label."""
    return CASE_DISPLAY_MAP.get(case_tag, "مبني")

def case_from_arabic(arabic_label: str) -> Optional[str]:
    """Convert Arabic label to UD case tag."""
    return CASE_FROM_ARABIC.get(arabic_label)

# ═══════════════════════════════════════════════════════════
#                    UNICODE UTILITIES
# ═══════════════════════════════════════════════════════════

def strip_diacritics(text: str) -> str:
    return DIACRITIC_PATTERN.sub('', text)

def strip_tatweel(text: str) -> str:
    return text.replace(TATWEEL, '')

def normalize_arabic(text: str) -> str:
    text = strip_tatweel(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub('[\u200B-\u200F\u202A-\u202E\uFEFF]', '', text)
    return text

def is_arabic_letter(char: str) -> bool:
    if len(char) != 1:
        return False
    cp = ord(char)
    return (0x0621 <= cp <= 0x064A) or cp == 0x0671

def is_diacritic(char: str) -> bool:
    return len(char) == 1 and char in ALL_DIACRITICS

def decompose_word(word: str) -> List[Tuple[str, List[str]]]:
    """Decompose Arabic word into (base_letter, [diacritics]) pairs."""
    result = []
    current_letter = None
    current_diacritics = []

    for char in word:
        if is_arabic_letter(char):
            if current_letter is not None:
                result.append((current_letter, current_diacritics))
            current_letter = char
            current_diacritics = []
        elif is_diacritic(char):
            current_diacritics.append(char)
        else:
            if current_letter is not None:
                result.append((current_letter, current_diacritics))
                current_letter = None
                current_diacritics = []
            result.append((char, []))

    if current_letter is not None:
        result.append((current_letter, current_diacritics))
    return result

def recompose_word(decomposed: List[Tuple[str, List[str]]]) -> str:
    parts = []
    for letter, diacritics in decomposed:
        parts.append(letter)
        parts.extend(diacritics)
    return ''.join(parts)

def get_last_letter_index(decomposed: List[Tuple[str, List[str]]]) -> int:
    for i in range(len(decomposed) - 1, -1, -1):
        if is_arabic_letter(decomposed[i][0]):
            return i
    return -1

def replace_case_ending(word: str, new_diacritic: str) -> str:
    """Replace case diacritic on last Arabic letter. Preserves SHADDA."""
    if not word or not new_diacritic:
        return word

    decomposed = decompose_word(word)
    last_idx = get_last_letter_index(decomposed)
    if last_idx < 0:
        return word

    letter, existing = decomposed[last_idx]
    preserved = [d for d in existing if d not in CASE_DIACRITICS]

    new_marks = []
    if SHADDA in preserved:
        new_marks.append(SHADDA)
        preserved.remove(SHADDA)
    new_marks.extend(preserved)
    new_marks.append(new_diacritic)

    decomposed[last_idx] = (letter, new_marks)
    return recompose_word(decomposed)

def has_definite_article(word: str) -> bool:
    bare = strip_diacritics(word)
    return bare.startswith('ال') and len(bare) > 2

def ends_with_taa_marbuta(word: str) -> bool:
    bare = strip_diacritics(word)
    return bare.endswith(TAA_MARBUTA) if bare else False

def ends_with_alef_maqsura(word: str) -> bool:
    bare = strip_diacritics(word)
    return bare.endswith(ALEF_MAQSURA) if bare else False

# ═══════════════════════════════════════════════════════════
#                      DATA TYPES
# ═══════════════════════════════════════════════════════════

@dataclass
class WordInfo:
    """Complete linguistic information for a single word."""
    original: str
    cleaned: str
    bare: str
    position: int

    # Parser output
    pos: str = ''
    head: int = -1
    relation: str = ''
    case_tag: Optional[str] = None  # UD format: "Nom"/"Acc"/"Gen"/"Jus"/None

    # Morphological features
    is_definite: bool = False
    number: str = 'sing'
    gender: str = 'masc'
    person: str = ''
    verb_form: str = ''
    is_construct: bool = False

    # BUG FIX #1: features dict was missing from dataclass
    features: Dict[str, str] = field(default_factory=dict)

    # Diacritization
    stem_diacritized: str = ''
    case_diacritic: str = ''
    final_diacritized: str = ''
    diac_source: str = 'none'

    # Confidence
    parser_confidence: float = 0.0
    diac_confidence: float = 0.0

    @property
    def case_display(self) -> str:
        """Arabic display label for case tag."""
        return case_to_arabic(self.case_tag)


@dataclass
class SentenceInfo:
    original: str
    words: List[WordInfo] = field(default_factory=list)

    @property
    def diacritized(self) -> str:
        return ' '.join(w.final_diacritized or w.bare for w in self.words)

    @property
    def bare(self) -> str:
        return ' '.join(w.bare for w in self.words)


@dataclass
class ArabiyaResult:
    input_text: str
    diacritized: str = ''
    sentences: List[SentenceInfo] = field(default_factory=list)
    total_words: int = 0
    lexicon_hits: int = 0
    lexicon_misses: int = 0
    case_applied: int = 0

    @property
    def lexicon_coverage(self) -> float:
        total = self.lexicon_hits + self.lexicon_misses
        return self.lexicon_hits / total if total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"=== Arabiya Engine Result ===\n"
            f"Input:       {self.input_text}\n"
            f"Diacritized: {self.diacritized}\n"
            f"Words:       {self.total_words}\n"
            f"Lexicon:     {self.lexicon_hits}/{self.total_words} "
            f"({self.lexicon_coverage:.0%})\n"
            f"Case applied: {self.case_applied}"
        )

    def detail(self) -> str:
        lines = [self.summary(), "", "-- Per-Word --"]
        for sent in self.sentences:
            for w in sent.words:
                lines.append(
                    f"  {w.bare:>15} -> {w.final_diacritized:<20} "
                    f"[{w.pos:<5} {w.case_display:<8} {w.relation:<10} "
                    f"src={w.diac_source}]"
                )
        return '\n'.join(lines)
