"""
Case ending application + stem-case combiner.

BUG FIX #3 + #6: Uses UD case tags (Nom/Acc/Gen/Jus/None) only.
No Arabic string comparisons, no CaseTag enum dual-checks.

BUG FIX #4: external_engine delegation actually calls the engine.
"""

from typing import Optional

from arabiya.core import (
    WordInfo, replace_case_ending, has_definite_article,
    ends_with_taa_marbuta, ends_with_alef_maqsura,
    FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN,
    SUKUN, SHADDA,
)


class CaseEndingApplicator:
    """Applies case endings based on syntactic analysis.
    
    If external_engine is provided, delegates to it.
    Otherwise uses built-in rules (~85% accuracy).
    """

    def __init__(self, external_engine=None):
        self.external_engine = external_engine

    def get_case_diacritic(self, word_info: WordInfo) -> str:
        # BUG FIX #4: actually call external engine
        if self.external_engine is not None:
            try:
                return self.external_engine.get_case_diacritic(word_info)
            except Exception:
                pass  # Fall through to builtin
        return self._builtin_rules(word_info)

    def _builtin_rules(self, w: WordInfo) -> str:
        case = w.case_tag  # UD format: "Nom"/"Acc"/"Gen"/"Jus"/None

        # Indeclinable
        if case is None:
            return ''

        # Particles, prepositions, pronouns — always indeclinable
        if w.pos in ('ADP', 'CCONJ', 'SCONJ', 'PART', 'DET', 'INTJ', 'PRON'):
            return ''

        # Verbs
        if w.pos in ('VERB', 'AUX'):
            return self._verb_case(w)

        # Nouns, Adjectives, Proper Nouns
        return self._noun_case(w)

    def _noun_case(self, w: WordInfo) -> str:
        case = w.case_tag
        is_def = w.is_definite
        is_con = w.is_construct
        number = w.number
        feats = w.features

        # Sound masculine plural — final noon gets FATHA
        if feats.get('plural_type') == 'sound_masc':
            return FATHA

        # Sound feminine plural — Acc uses KASRA (exception!)
        if feats.get('plural_type') == 'sound_fem':
            use_tanween = not is_def and not is_con
            if case == 'Nom':
                return DAMMATAN if use_tanween else DAMMA
            else:  # Acc and Gen both use kasra
                return KASRATAN if use_tanween else KASRA

        # Dual — final noon gets KASRA
        if w.number == 'dual':
            return KASRA

        # Alef maqsura — hidden case
        if ends_with_alef_maqsura(w.bare):
            return ''

        # Regular/broken plural/singular
        return self._standard_case(case, is_def, is_con)

    def _standard_case(self, case, is_definite, is_construct):
        use_tanween = not is_definite and not is_construct
        if case == 'Nom':
            return DAMMATAN if use_tanween else DAMMA
        elif case == 'Acc':
            return FATHATAN if use_tanween else FATHA
        elif case == 'Gen':
            return KASRATAN if use_tanween else KASRA
        elif case == 'Jus':
            return SUKUN
        return ''

    def _verb_case(self, w: WordInfo) -> str:
        vf = w.features.get('verb_form', w.verb_form)
        case = w.case_tag
        if vf in ('past', 'imp'):
            return ''
        if case == 'Nom':
            return DAMMA
        elif case == 'Acc':
            return FATHA
        elif case == 'Jus':
            return SUKUN
        return ''


class DiacriticCombiner:
    """Combines stem diacritics with case endings."""

    def __init__(self, case_applicator: CaseEndingApplicator):
        self.case_applicator = case_applicator

    def combine(self, word_info: WordInfo) -> str:
        base = word_info.stem_diacritized if word_info.stem_diacritized else word_info.bare
        case_diac = self.case_applicator.get_case_diacritic(word_info)
        word_info.case_diacritic = case_diac

        if case_diac:
            word_info.final_diacritized = replace_case_ending(base, case_diac)
        else:
            word_info.final_diacritized = base
        return word_info.final_diacritized
