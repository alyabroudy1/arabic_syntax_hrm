#!/usr/bin/env python3
"""
Arabic Tajweed Engine (أحكام التجويد)
======================================

Deterministic, rule-based engine that analyzes diacritized Arabic text
and produces tajweed annotations for TTS prosody control.

Covers all 7 categories:
1. أحكام النون الساكنة والتنوين  (Noon Sakinah & Tanween)
2. أحكام الميم الساكنة           (Meem Sakinah)
3. أحكام المد                    (Prolongation)
4. القلقلة                       (Qalqalah / Echo)
5. لام التعريف                   (Definite Article Laam)
6. التفخيم والترقيق              (Heavy/Light Pronunciation)
7. أحكام الراء                   (Raa Rules)
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum, auto


# ═══════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════

# Diacritics
FATHA    = '\u064E'
DAMMA    = '\u064F'
KASRA    = '\u0650'
SUKUN    = '\u0652'
SHADDA   = '\u0651'
TANWEEN_FATH = '\u064B'
TANWEEN_DAMM = '\u064C'
TANWEEN_KASR = '\u064D'

ALL_DIACRITICS = set([FATHA, DAMMA, KASRA, SUKUN, SHADDA,
                      TANWEEN_FATH, TANWEEN_DAMM, TANWEEN_KASR,
                      '\u0670', '\u0653', '\u0654', '\u0655'])

TANWEEN_SET = {TANWEEN_FATH, TANWEEN_DAMM, TANWEEN_KASR}

# Letter groups
NOON = 'ن'
MEEM = 'م'
BAA  = 'ب'
HAMZA = 'ء'
ALEF = 'ا'
WAW  = 'و'
YAA  = 'ي'

# ── أحكام النون الساكنة والتنوين ──
THROAT_LETTERS = set('ءهعحغخ')          # حروف الحلق → إظهار
IDGHAAM_GHUNNA_LETTERS = set('ينمو')     # → إدغام بغنة
IDGHAAM_NO_GHUNNA_LETTERS = set('لر')    # → إدغام بغير غنة
IQLAAB_LETTER = 'ب'                     # → إقلاب
IKHFAA_LETTERS = set('صذثكجشقسدطزفتضظ') # → إخفاء

# ── القلقلة ──
QALQALA_LETTERS = set('قطبجد')  # قُطْبُ جَدّ

# ── لام التعريف ──
SUN_LETTERS = set('تثدذرزسشصضطظنل')   # الحروف الشمسية
MOON_LETTERS = set('ابجحخعغفقكمهوي')   # الحروف القمرية

# ── التفخيم ──
ALWAYS_HEAVY = set('خصضغطقظ')   # حروف الاستعلاء — always tafkheem
LAAM_LAFZ_JALAALAH = True  # لام لفظ الجلالة


# ═══════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════

class TajweedRule(Enum):
    """All tajweed rule types."""
    NONE = auto()
    
    # النون الساكنة والتنوين
    ITHAAR_HALQI = auto()       # إظهار حلقي
    IDGHAAM_GHUNNA = auto()     # إدغام بغنة
    IDGHAAM_NO_GHUNNA = auto()  # إدغام بغير غنة
    IQLAAB = auto()             # إقلاب
    IKHFAA_HAQIQI = auto()     # إخفاء حقيقي
    
    # الميم الساكنة
    IKHFAA_SHAFAWI = auto()    # إخفاء شفوي
    IDGHAAM_MUTAMATHILAIN = auto()  # إدغام متماثلين
    ITHAAR_SHAFAWI = auto()    # إظهار شفوي
    
    # المد
    MADD_TABII = auto()        # مد طبيعي (2 beats)
    MADD_MUTTASIL = auto()     # مد واجب متصل (4-5 beats)
    MADD_MUNFASIL = auto()     # مد جائز منفصل (4-5 beats)
    MADD_LAAZIM = auto()       # مد لازم (6 beats)
    MADD_AARID = auto()        # مد عارض للسكون (2-6 beats)
    MADD_LEEN = auto()         # مد لين
    MADD_BADAL = auto()        # مد بدل (2 beats)
    MADD_SILAH = auto()        # مد صلة
    
    # القلقلة
    QALQALA_SUGHRA = auto()    # قلقلة صغرى
    QALQALA_KUBRA = auto()     # قلقلة كبرى
    
    # لام التعريف
    LAAM_SHAMSIYA = auto()     # لام شمسية (assimilated)
    LAAM_QAMARIYA = auto()     # لام قمرية (pronounced)
    
    # التفخيم والترقيق
    TAFKHEEM = auto()          # تفخيم (heavy)
    TARQEEQ = auto()           # ترقيق (light)
    
    # أحكام الراء
    RAA_TAFKHEEM = auto()      # راء مفخمة
    RAA_TARQEEQ = auto()       # راء مرققة
    
    # غنة
    GHUNNA = auto()            # غنة (nasalization)


@dataclass
class TajweedAnnotation:
    """A tajweed annotation on a character or sequence."""
    rule: TajweedRule
    char_index: int           # position in the word
    word_index: int           # position in the sentence
    description_ar: str = ""  # Arabic description
    description_en: str = ""  # English description
    beats: int = 0            # duration in beats (for madd)
    color: str = ""           # display color (for visualization)


@dataclass
class AnalyzedChar:
    """A single character with its tajweed analysis."""
    char: str
    diacritic: str = ""
    rules: List[TajweedRule] = field(default_factory=list)
    

@dataclass 
class AnalyzedWord:
    """A word with full tajweed analysis."""
    text: str                 # diacritized text
    plain: str               # undiacritized
    chars: List[AnalyzedChar] = field(default_factory=list)
    annotations: List[TajweedAnnotation] = field(default_factory=list)


# ═══════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════

def strip_diacritics(text: str) -> str:
    return ''.join(c for c in text if c not in ALL_DIACRITICS)


def split_chars_and_diacs(word: str) -> List[AnalyzedChar]:
    """Split diacritized word into (base_char, combined_diacritics) pairs."""
    result = []
    i = 0
    chars = list(word)
    
    while i < len(chars):
        c = chars[i]
        if c in ALL_DIACRITICS:
            i += 1
            continue
        
        diacs = ""
        j = i + 1
        while j < len(chars) and chars[j] in ALL_DIACRITICS:
            diacs += chars[j]
            j += 1
        
        result.append(AnalyzedChar(char=c, diacritic=diacs))
        i = j
    
    return result


def has_sukun(diac: str) -> bool:
    return SUKUN in diac


def has_shadda(diac: str) -> bool:
    return SHADDA in diac


def has_tanween(diac: str) -> bool:
    return bool(set(diac) & TANWEEN_SET)


def is_noon_sakinah(char: AnalyzedChar) -> bool:
    return char.char == NOON and has_sukun(char.diacritic)


def is_meem_sakinah(char: AnalyzedChar) -> bool:
    return char.char == MEEM and has_sukun(char.diacritic)


def is_madd_letter(char: AnalyzedChar, prev: Optional[AnalyzedChar]) -> bool:
    """Check if this char is a madd letter (elongation)."""
    if prev is None:
        return False
    if char.char == ALEF and FATHA in prev.diacritic:
        return True
    if char.char == WAW and DAMMA in prev.diacritic and has_sukun(char.diacritic):
        return True
    if char.char == YAA and KASRA in prev.diacritic and has_sukun(char.diacritic):
        return True
    return False


# ═══════════════════════════════════════════════════
# Tajweed Analysis Engine
# ═══════════════════════════════════════════════════

class TajweedEngine:
    """Full Arabic Tajweed rule engine.
    
    Analyzes diacritized text and produces tajweed annotations
    for each character.
    """
    
    # Color scheme for tajweed visualization
    RULE_COLORS = {
        TajweedRule.ITHAAR_HALQI: '#4CAF50',       # green
        TajweedRule.IDGHAAM_GHUNNA: '#FF9800',      # orange
        TajweedRule.IDGHAAM_NO_GHUNNA: '#9C27B0',   # purple
        TajweedRule.IQLAAB: '#2196F3',              # blue
        TajweedRule.IKHFAA_HAQIQI: '#00BCD4',       # cyan
        TajweedRule.IKHFAA_SHAFAWI: '#00BCD4',
        TajweedRule.QALQALA_SUGHRA: '#F44336',      # red
        TajweedRule.QALQALA_KUBRA: '#D32F2F',       # dark red
        TajweedRule.MADD_TABII: '#795548',           # brown
        TajweedRule.MADD_MUTTASIL: '#E91E63',       # pink
        TajweedRule.MADD_MUNFASIL: '#E91E63',
        TajweedRule.MADD_LAAZIM: '#3F51B5',         # indigo
        TajweedRule.LAAM_SHAMSIYA: '#607D8B',        # grey
        TajweedRule.GHUNNA: '#FF5722',               # deep orange
        TajweedRule.TAFKHEEM: '#8BC34A',             # light green
        TajweedRule.RAA_TAFKHEEM: '#8BC34A',
        TajweedRule.RAA_TARQEEQ: '#CDDC39',          # lime
    }
    
    # Arabic descriptions  
    RULE_DESC_AR = {
        TajweedRule.ITHAAR_HALQI: 'إظهار حلقي',
        TajweedRule.IDGHAAM_GHUNNA: 'إدغام بغنة',
        TajweedRule.IDGHAAM_NO_GHUNNA: 'إدغام بغير غنة',
        TajweedRule.IQLAAB: 'إقلاب',
        TajweedRule.IKHFAA_HAQIQI: 'إخفاء حقيقي',
        TajweedRule.IKHFAA_SHAFAWI: 'إخفاء شفوي',
        TajweedRule.IDGHAAM_MUTAMATHILAIN: 'إدغام متماثلين',
        TajweedRule.ITHAAR_SHAFAWI: 'إظهار شفوي',
        TajweedRule.MADD_TABII: 'مد طبيعي',
        TajweedRule.MADD_MUTTASIL: 'مد واجب متصل',
        TajweedRule.MADD_MUNFASIL: 'مد جائز منفصل',
        TajweedRule.MADD_LAAZIM: 'مد لازم',
        TajweedRule.MADD_AARID: 'مد عارض للسكون',
        TajweedRule.MADD_LEEN: 'مد لين',
        TajweedRule.MADD_BADAL: 'مد بدل',
        TajweedRule.QALQALA_SUGHRA: 'قلقلة صغرى',
        TajweedRule.QALQALA_KUBRA: 'قلقلة كبرى',
        TajweedRule.LAAM_SHAMSIYA: 'لام شمسية',
        TajweedRule.LAAM_QAMARIYA: 'لام قمرية',
        TajweedRule.TAFKHEEM: 'تفخيم',
        TajweedRule.TARQEEQ: 'ترقيق',
        TajweedRule.RAA_TAFKHEEM: 'راء مفخمة',
        TajweedRule.RAA_TARQEEQ: 'راء مرققة',
        TajweedRule.GHUNNA: 'غنة',
    }
    
    def analyze(self, text: str) -> List[AnalyzedWord]:
        """Analyze full diacritized text and return tajweed annotations."""
        words = text.split()
        analyzed_words = []
        
        for wi, word in enumerate(words):
            chars = split_chars_and_diacs(word)
            aw = AnalyzedWord(
                text=word,
                plain=strip_diacritics(word),
                chars=chars,
            )
            analyzed_words.append(aw)
        
        # Apply all rules
        self._apply_noon_sakinah_rules(analyzed_words)
        self._apply_meem_sakinah_rules(analyzed_words)
        self._apply_madd_rules(analyzed_words)
        self._apply_qalqala_rules(analyzed_words)
        self._apply_laam_rules(analyzed_words)
        self._apply_tafkheem_tarqeeq(analyzed_words)
        self._apply_raa_rules(analyzed_words)
        
        return analyzed_words
    
    def _get_next_letter(self, words: List[AnalyzedWord], wi: int, ci: int) -> Optional[AnalyzedChar]:
        """Get the next base letter after position (wi, ci), crossing word boundaries."""
        # Try next char in same word
        if ci + 1 < len(words[wi].chars):
            return words[wi].chars[ci + 1]
        # Try first char of next word
        if wi + 1 < len(words) and words[wi + 1].chars:
            return words[wi + 1].chars[0]
        return None
    
    def _get_prev_letter(self, words: List[AnalyzedWord], wi: int, ci: int) -> Optional[AnalyzedChar]:
        """Get the previous base letter."""
        if ci > 0:
            return words[wi].chars[ci - 1]
        if wi > 0 and words[wi - 1].chars:
            return words[wi - 1].chars[-1]
        return None
    
    # ── 1. أحكام النون الساكنة والتنوين ──
    
    def _apply_noon_sakinah_rules(self, words: List[AnalyzedWord]):
        for wi, word in enumerate(words):
            for ci, char in enumerate(word.chars):
                is_noon_sak = is_noon_sakinah(char)
                is_tanw = has_tanween(char.diacritic)
                
                if not (is_noon_sak or is_tanw):
                    continue
                
                next_char = self._get_next_letter(words, wi, ci)
                if next_char is None:
                    continue
                
                next_letter = next_char.char
                
                if next_letter in THROAT_LETTERS:
                    rule = TajweedRule.ITHAAR_HALQI
                elif next_letter in IDGHAAM_GHUNNA_LETTERS:
                    # Idghaam only across word boundaries for noon sakinah
                    if is_noon_sak and ci + 1 < len(word.chars):
                        continue  # Same word — no idghaam (e.g., دُنْيَا)
                    rule = TajweedRule.IDGHAAM_GHUNNA
                elif next_letter in IDGHAAM_NO_GHUNNA_LETTERS:
                    if is_noon_sak and ci + 1 < len(word.chars):
                        continue
                    rule = TajweedRule.IDGHAAM_NO_GHUNNA
                elif next_letter == IQLAAB_LETTER:
                    rule = TajweedRule.IQLAAB
                elif next_letter in IKHFAA_LETTERS:
                    rule = TajweedRule.IKHFAA_HAQIQI
                else:
                    continue
                
                word.annotations.append(TajweedAnnotation(
                    rule=rule,
                    char_index=ci,
                    word_index=wi,
                    description_ar=self.RULE_DESC_AR.get(rule, ''),
                    color=self.RULE_COLORS.get(rule, ''),
                ))
    
    # ── 2. أحكام الميم الساكنة ──
    
    def _apply_meem_sakinah_rules(self, words: List[AnalyzedWord]):
        for wi, word in enumerate(words):
            for ci, char in enumerate(word.chars):
                if not is_meem_sakinah(char):
                    continue
                
                next_char = self._get_next_letter(words, wi, ci)
                if next_char is None:
                    continue
                
                if next_char.char == BAA:
                    rule = TajweedRule.IKHFAA_SHAFAWI
                elif next_char.char == MEEM:
                    rule = TajweedRule.IDGHAAM_MUTAMATHILAIN
                else:
                    rule = TajweedRule.ITHAAR_SHAFAWI
                
                word.annotations.append(TajweedAnnotation(
                    rule=rule,
                    char_index=ci,
                    word_index=wi,
                    description_ar=self.RULE_DESC_AR.get(rule, ''),
                    color=self.RULE_COLORS.get(rule, ''),
                ))
    
    # ── 3. أحكام المد ──
    
    def _apply_madd_rules(self, words: List[AnalyzedWord]):
        for wi, word in enumerate(words):
            for ci, char in enumerate(word.chars):
                prev = self._get_prev_letter(words, wi, ci)
                
                if not is_madd_letter(char, prev):
                    continue
                
                next_char = self._get_next_letter(words, wi, ci)
                
                if next_char is None:
                    # End of text — madd tabii
                    rule = TajweedRule.MADD_TABII
                    beats = 2
                elif next_char.char == HAMZA or next_char.char == 'أ' or next_char.char == 'إ' or next_char.char == 'ئ' or next_char.char == 'ؤ':
                    # Hamza after madd
                    if ci + 1 < len(word.chars):
                        rule = TajweedRule.MADD_MUTTASIL  # same word
                        beats = 4
                    else:
                        rule = TajweedRule.MADD_MUNFASIL  # next word
                        beats = 4
                elif has_sukun(next_char.diacritic) or has_shadda(next_char.diacritic):
                    if has_shadda(next_char.diacritic):
                        rule = TajweedRule.MADD_LAAZIM  # permanent sukun/shadda
                        beats = 6
                    else:
                        rule = TajweedRule.MADD_AARID  # temporary sukun (at stop)
                        beats = 2
                else:
                    rule = TajweedRule.MADD_TABII
                    beats = 2
                
                word.annotations.append(TajweedAnnotation(
                    rule=rule,
                    char_index=ci,
                    word_index=wi,
                    description_ar=self.RULE_DESC_AR.get(rule, ''),
                    color=self.RULE_COLORS.get(rule, ''),
                    beats=beats,
                ))
    
    # ── 4. القلقلة ──
    
    def _apply_qalqala_rules(self, words: List[AnalyzedWord]):
        for wi, word in enumerate(words):
            for ci, char in enumerate(word.chars):
                if char.char not in QALQALA_LETTERS:
                    continue
                
                if has_sukun(char.diacritic):
                    is_at_end = (ci == len(word.chars) - 1) and (wi == len(words) - 1 or True)
                    
                    if ci == len(word.chars) - 1:
                        rule = TajweedRule.QALQALA_KUBRA
                    else:
                        rule = TajweedRule.QALQALA_SUGHRA
                    
                    word.annotations.append(TajweedAnnotation(
                        rule=rule,
                        char_index=ci,
                        word_index=wi,
                        description_ar=self.RULE_DESC_AR.get(rule, ''),
                        color=self.RULE_COLORS.get(rule, ''),
                    ))
    
    # ── 5. لام التعريف ──
    
    def _apply_laam_rules(self, words: List[AnalyzedWord]):
        for wi, word in enumerate(words):
            plain = word.plain
            if not (plain.startswith('ال') or plain.startswith('ٱل')):
                continue
            
            # Find the laam character index
            laam_ci = None
            for ci, char in enumerate(word.chars):
                if char.char == 'ل' and ci <= 2:
                    laam_ci = ci
                    break
            
            if laam_ci is None:
                continue
            
            next_char = self._get_next_letter(words, wi, laam_ci)
            if next_char is None:
                continue
            
            if next_char.char in SUN_LETTERS:
                rule = TajweedRule.LAAM_SHAMSIYA
            else:
                rule = TajweedRule.LAAM_QAMARIYA
            
            word.annotations.append(TajweedAnnotation(
                rule=rule,
                char_index=laam_ci,
                word_index=wi,
                description_ar=self.RULE_DESC_AR.get(rule, ''),
                color=self.RULE_COLORS.get(rule, ''),
            ))
    
    # ── 6. التفخيم والترقيق ──
    
    def _apply_tafkheem_tarqeeq(self, words: List[AnalyzedWord]):
        for wi, word in enumerate(words):
            for ci, char in enumerate(word.chars):
                if char.char in ALWAYS_HEAVY:
                    word.annotations.append(TajweedAnnotation(
                        rule=TajweedRule.TAFKHEEM,
                        char_index=ci,
                        word_index=wi,
                        description_ar=self.RULE_DESC_AR[TajweedRule.TAFKHEEM],
                        color=self.RULE_COLORS.get(TajweedRule.TAFKHEEM, ''),
                    ))
    
    # ── 7. أحكام الراء ──
    
    def _apply_raa_rules(self, words: List[AnalyzedWord]):
        for wi, word in enumerate(words):
            for ci, char in enumerate(word.chars):
                if char.char != 'ر':
                    continue
                
                prev = self._get_prev_letter(words, wi, ci)
                
                if FATHA in char.diacritic or DAMMA in char.diacritic:
                    rule = TajweedRule.RAA_TAFKHEEM
                elif KASRA in char.diacritic:
                    rule = TajweedRule.RAA_TARQEEQ
                elif has_sukun(char.diacritic):
                    # Sakin raa — depends on previous diacritic
                    if prev and (FATHA in prev.diacritic or DAMMA in prev.diacritic):
                        rule = TajweedRule.RAA_TAFKHEEM
                    elif prev and KASRA in prev.diacritic:
                        rule = TajweedRule.RAA_TARQEEQ
                    else:
                        rule = TajweedRule.RAA_TAFKHEEM  # default
                else:
                    continue
                
                word.annotations.append(TajweedAnnotation(
                    rule=rule,
                    char_index=ci,
                    word_index=wi,
                    description_ar=self.RULE_DESC_AR.get(rule, ''),
                    color=self.RULE_COLORS.get(rule, ''),
                ))
    
    # ═══════════════════════════════════════════════════
    # Output Formatting
    # ═══════════════════════════════════════════════════
    
    def to_kokoro_text(self, words: List[AnalyzedWord]) -> str:
        """Convert tajweed analysis to Kokoro-friendly text.
        
        Uses punctuation and spacing to control prosody since
        Kokoro doesn't support SSML.
        """
        output_parts = []
        
        for wi, word in enumerate(words):
            # Check for pauses/breaks
            has_ikhfaa = any(a.rule == TajweedRule.IKHFAA_HAQIQI for a in word.annotations)
            has_madd_long = any(a.rule in (TajweedRule.MADD_MUTTASIL, TajweedRule.MADD_LAAZIM) 
                               and a.beats >= 4 for a in word.annotations)
            
            text = word.text
            
            # Add slight pause markers for ghunna/ikhfaa
            if has_ikhfaa:
                text = text + ","  # comma triggers natural pause in Kokoro
            
            output_parts.append(text)
        
        return ' '.join(output_parts)
    
    def to_html(self, words: List[AnalyzedWord]) -> str:
        """Generate color-coded HTML for tajweed visualization."""
        html_parts = []
        
        for word in words:
            # Build character-level spans
            char_spans = []
            annotation_map = {}
            for ann in word.annotations:
                annotation_map.setdefault(ann.char_index, []).append(ann)
            
            for ci, char in enumerate(word.chars):
                anns = annotation_map.get(ci, [])
                if anns:
                    primary = anns[0]
                    color = primary.color or '#333'
                    title = primary.description_ar
                    char_spans.append(
                        f'<span style="color:{color}" title="{title}">'
                        f'{char.char}{char.diacritic}</span>'
                    )
                else:
                    char_spans.append(f'{char.char}{char.diacritic}')
            
            html_parts.append(''.join(char_spans))
        
        return ' '.join(html_parts)
    
    def get_summary(self, words: List[AnalyzedWord]) -> dict:
        """Get summary statistics of tajweed rules found."""
        counts = {}
        for word in words:
            for ann in word.annotations:
                name = self.RULE_DESC_AR.get(ann.rule, ann.rule.name)
                counts[name] = counts.get(name, 0) + 1
        return counts


# ═══════════════════════════════════════════════════
# CLI Demo
# ═══════════════════════════════════════════════════

def main():
    engine = TajweedEngine()
    
    # Test with Quran verses
    test_verses = [
        "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
        "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ",
        "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
        "قُلْ هُوَ ٱللَّهُ أَحَدٌ",
        "ٱللَّهُ ٱلصَّمَدُ",
        "لَمْ يَلِدْ وَلَمْ يُولَدْ",
        "وَلَمْ يَكُن لَّهُۥ كُفُوًا أَحَدٌۢ",
        "مِن شَرِّ ٱلْوَسْوَاسِ ٱلْخَنَّاسِ",
    ]
    
    print("="*60)
    print("أحكام التجويد — Tajweed Engine Demo")
    print("="*60)
    
    for verse in test_verses:
        words = engine.analyze(verse)
        summary = engine.get_summary(words)
        
        print(f"\n📖 {verse}")
        if summary:
            for rule_name, count in sorted(summary.items(), key=lambda x: -x[1]):
                print(f"   • {rule_name}: {count}")
        else:
            print("   (no special tajweed rules)")


if __name__ == "__main__":
    main()
