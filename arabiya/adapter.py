"""
Adapter — bridges YOUR CaseEndingRuleEngine (93% accuracy) to the pipeline.

BUG FIX #4: Actually calls your engine instead of raising NotImplementedError.
BUG FIX #3: Translates UD case tags to the format your engine expects.

YOUR ENGINE API (from models/v2/case_engine.py):
    engine.apply(word, case, upos, feats, lemma, deprel) -> CaseEndingResult
    result.ending_diacritic -> str  (the diacritic character)
"""

from typing import Any
from arabiya.core import WordInfo


# UD -> your engine's expected case format
_CASE_UD_TO_ENGINE = {
    'Nom': 'Nom',
    'Acc': 'Acc',
    'Gen': 'Gen',
    'Jus': 'Jus',
}


class CaseEngineAdapter:
    """Wraps your CaseEndingRuleEngine for the Arabiya pipeline."""

    def __init__(self, engine: Any):
        self.engine = engine

    def get_case_diacritic(self, word_info: WordInfo) -> str:
        """Called by CaseEndingApplicator for each word."""
        case_tag = word_info.case_tag
        if case_tag is None:
            return ''

        engine_case = _CASE_UD_TO_ENGINE.get(case_tag, '')
        if not engine_case:
            return ''

        # Build UD-style features string from WordInfo
        feats_parts = []
        if word_info.case_tag:
            feats_parts.append(f'Case={word_info.case_tag}')
        if word_info.is_definite:
            feats_parts.append('Definite=Def')
        elif word_info.is_construct:
            feats_parts.append('Definite=Cons')
        else:
            feats_parts.append('Definite=Ind')
        if word_info.number == 'plur':
            feats_parts.append('Number=Plur')
        elif word_info.number == 'dual':
            feats_parts.append('Number=Dual')
        else:
            feats_parts.append('Number=Sing')
        if word_info.gender == 'fem':
            feats_parts.append('Gender=Fem')
        else:
            feats_parts.append('Gender=Masc')

        feats_str = '|'.join(feats_parts)

        try:
            result = self.engine.apply(
                word=word_info.bare,
                case=engine_case,
                upos=word_info.pos,
                feats=feats_str,
                lemma=word_info.bare,  # Best guess
                deprel=word_info.relation,
            )
            return result.ending_diacritic
        except Exception as e:
            return ''


def connect_case_engine(case_engine, lexicon_path=None, model_dir=None):
    """One-step function to create an ArabiyaEngine with your CaseEndingRuleEngine."""
    from arabiya.engine import ArabiyaEngine

    if lexicon_path:
        from arabiya.stem_diacritizer import StemDiacritizer
        stem = StemDiacritizer()
        stem.load_lexicon(lexicon_path)
        engine = ArabiyaEngine.create_with_mock()
        engine.stem_diacritizer = stem
    else:
        engine = ArabiyaEngine.create_with_mock()

    adapter = CaseEngineAdapter(case_engine)
    from arabiya.diacritizer import CaseEndingApplicator, DiacriticCombiner
    engine.case_applicator = CaseEndingApplicator(external_engine=adapter)
    engine.combiner = DiacriticCombiner(engine.case_applicator)

    return engine
