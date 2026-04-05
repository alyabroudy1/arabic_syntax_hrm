"""
Arabiya Engine — Main pipeline orchestrator.

BUG FIX #5: Complete file (not truncated).
BUG FIX #1: Uses WordInfo.features dict properly.
BUG FIX #3: UD case tags throughout.
"""

import os
from typing import List, Optional

from arabiya.core import (
    WordInfo, SentenceInfo, ArabiyaResult,
    strip_diacritics, has_definite_article,
    ends_with_taa_marbuta, case_to_arabic,
)
from arabiya.preprocessor import ArabicPreprocessor
from arabiya.parser import ParserInterface, MockParser, ParseResult
from arabiya.stem_diacritizer import StemDiacritizer
from arabiya.diacritizer import CaseEndingApplicator, DiacriticCombiner


class ArabiyaEngine:

    def __init__(self, preprocessor, parser, stem_diacritizer,
                 case_applicator, combiner):
        self.preprocessor = preprocessor
        self.parser = parser
        self.stem_diacritizer = stem_diacritizer
        self.case_applicator = case_applicator
        self.combiner = combiner

    # ══════════════════════════════════════════════
    #               FACTORY METHODS
    # ══════════════════════════════════════════════

    @classmethod
    def create_with_mock(cls, lexicon_data=None):
        prep = ArabicPreprocessor(strip_diacritics_from_input=True)
        parser = MockParser()
        stem = StemDiacritizer()
        if lexicon_data:
            stem.build_from_inline_data(lexicon_data)
        else:
            stem.build_from_inline_data(cls._minimal_test_lexicon())
        case_app = CaseEndingApplicator(external_engine=None)
        combiner = DiacriticCombiner(case_app)
        return cls(prep, parser, stem, case_app, combiner)

    @classmethod
    def load(cls, model_dir, lexicon_path=None,
             external_case_engine=None, device='cpu'):
        from arabiya.adapter import CaseEngineAdapter
        prep = ArabicPreprocessor(strip_diacritics_from_input=True)

        # Parser — try real, fall back to mock
        try:
            from arabiya.parser_hrm import HRMParserAdapter
            parser = HRMParserAdapter(model_dir, device=device)
        except Exception as e:
            print(f"[WARNING] HRM parser not available: {e}")
            parser = MockParser()

        stem = StemDiacritizer()
        if lexicon_path and os.path.exists(lexicon_path):
            stem.load_lexicon(lexicon_path)

        if external_case_engine is not None:
            adapter = CaseEngineAdapter(external_case_engine)
            case_app = CaseEndingApplicator(external_engine=adapter)
        else:
            case_app = CaseEndingApplicator(external_engine=None)

        combiner = DiacriticCombiner(case_app)
        return cls(prep, parser, stem, case_app, combiner)

    # ══════════════════════════════════════════════
    #               MAIN PROCESSING
    # ══════════════════════════════════════════════

    def process(self, text: str) -> ArabiyaResult:
        result = ArabiyaResult(input_text=text)
        sentences = self.preprocessor.process(text)
        if not sentences:
            result.diacritized = text
            return result

        all_diacritized = []
        for sent_tokens in sentences:
            sent_info = self._process_sentence(sent_tokens, result)
            result.sentences.append(sent_info)
            all_diacritized.append(sent_info.diacritized)

        result.diacritized = ' '.join(all_diacritized)
        return result

    def _process_sentence(self, tokens, result):
        sent_info = SentenceInfo(original=' '.join(tokens))
        tokens = [t for t in tokens if t.strip()]
        if not tokens:
            return sent_info

        parse_result = self._safe_parse(tokens)

        for i, token in enumerate(tokens):
            word_info = self._build_word_info(i, token, parse_result)
            self._apply_stem(word_info, result)
            self.combiner.combine(word_info)
            if word_info.case_diacritic:
                result.case_applied += 1
            result.total_words += 1
            sent_info.words.append(word_info)

        return sent_info

    def _safe_parse(self, tokens):
        try:
            if self.parser.is_loaded():
                return self.parser.parse(tokens)
        except Exception as e:
            print(f"[WARNING] Parser error: {e}")
        return None

    def _build_word_info(self, idx, token, parse):
        bare = strip_diacritics(token)
        info = WordInfo(
            original=token, cleaned=token, bare=bare, position=idx,
        )

        if parse and idx < len(parse.words):
            info.pos = parse.pos_tags[idx]
            info.head = parse.heads[idx]
            info.relation = parse.relations[idx]
            info.case_tag = parse.case_tags[idx]  # UD: Nom/Acc/Gen/None
            info.parser_confidence = (
                parse.case_confidences[idx]
                if idx < len(parse.case_confidences) else 0.0
            )

            feats = (parse.features[idx]
                     if idx < len(parse.features) else {})
            info.features = feats  # BUG FIX #1: field exists now
            info.is_definite = (
                feats.get('definite', 'no') == 'yes'
                or has_definite_article(bare)
            )
            info.number = feats.get('number', 'sing')
            info.gender = feats.get('gender', 'masc')
            info.person = feats.get('person', '')
            info.verb_form = feats.get('verb_form', '')
            info.is_construct = feats.get('construct', 'no') == 'yes'
        else:
            info.is_definite = has_definite_article(bare)
            info.pos = 'X'
            info.case_tag = None
            info.features = {}

        return info

    def _apply_stem(self, word_info, result):
        if not any('\u0621' <= c <= '\u064A' for c in word_info.bare):
            word_info.stem_diacritized = word_info.bare
            word_info.diac_source = 'passthrough'
            return

        diac = self.stem_diacritizer.lookup(word_info.bare, word_info.pos)
        if diac:
            word_info.stem_diacritized = diac
            word_info.diac_source = 'lexicon'
            word_info.diac_confidence = 0.85
            result.lexicon_hits += 1
        else:
            word_info.stem_diacritized = ''
            word_info.diac_source = 'none'
            result.lexicon_misses += 1

    # ══════════════════════════════════════════════
    #            CONVENIENCE METHODS
    # ══════════════════════════════════════════════

    def diacritize(self, text: str) -> str:
        return self.process(text).diacritized

    # ══════════════════════════════════════════════
    #           BUILT-IN TEST LEXICON
    # ══════════════════════════════════════════════

    @staticmethod
    def _minimal_test_lexicon():
        return {
            # Common Verbs
            'ذهب': {'VERB': 'ذَهَبَ'},
            'كتب': {'VERB': 'كَتَبَ', 'NOUN': 'كُتُب'},
            'قرأ': {'VERB': 'قَرَأَ'},
            'علم': {'VERB': 'عَلِمَ', 'NOUN': 'عِلْم'},
            'قال': {'VERB': 'قَالَ'},
            'كان': {'VERB': 'كَانَ', 'AUX': 'كَانَ'},
            'رجع': {'VERB': 'رَجَعَ'},
            'وجد': {'VERB': 'وَجَدَ'},
            'أكل': {'VERB': 'أَكَلَ'},
            'دخل': {'VERB': 'دَخَلَ'},
            'خرج': {'VERB': 'خَرَجَ'},
            'فتح': {'VERB': 'فَتَحَ'},
            'درس': {'VERB': 'دَرَسَ', 'NOUN': 'دَرْس'},
            'فهم': {'VERB': 'فَهِمَ'},
            # Present verbs
            'يذهب': {'VERB': 'يَذْهَبُ'},
            'يكتب': {'VERB': 'يَكْتُبُ'},
            'يقرأ': {'VERB': 'يَقْرَأُ'},
            # Common Nouns
            'كتاب': {'NOUN': 'كِتَاب'},
            'طالب': {'NOUN': 'طَالِب'},
            'معلم': {'NOUN': 'مُعَلِّم'},
            'مدرسة': {'NOUN': 'مَدْرَسَة'},
            'بيت': {'NOUN': 'بَيْت'},
            'ولد': {'NOUN': 'وَلَد'},
            'رجل': {'NOUN': 'رَجُل'},
            'يوم': {'NOUN': 'يَوْم'},
            'ماء': {'NOUN': 'مَاء'},
            'سماء': {'NOUN': 'سَمَاء'},
            'أرض': {'NOUN': 'أَرْض'},
            'نور': {'NOUN': 'نُور'},
            'عمل': {'NOUN': 'عَمَل'},
            'قلم': {'NOUN': 'قَلَم'},
            'باب': {'NOUN': 'بَاب'},
            'حديقة': {'NOUN': 'حَدِيقَة'},
            'سيارة': {'NOUN': 'سَيَّارَة'},
            'جامعة': {'NOUN': 'جَامِعَة'},
            'دروس': {'NOUN': 'دُرُوس'},
            # With definite article
            'الكتاب': {'NOUN': 'الْكِتَاب'},
            'الطالب': {'NOUN': 'الطَّالِب'},
            'المعلم': {'NOUN': 'الْمُعَلِّم'},
            'المدرسة': {'NOUN': 'الْمَدْرَسَة'},
            'البيت': {'NOUN': 'الْبَيْت'},
            'اليوم': {'NOUN': 'الْيَوْم'},
            'السماء': {'NOUN': 'السَّمَاء'},
            'العلم': {'NOUN': 'الْعِلْم'},
            'الدرس': {'NOUN': 'الدَّرْس'},
            'المعلمون': {'NOUN': 'الْمُعَلِّمُون'},
            'الدروس': {'NOUN': 'الدُّرُوس'},
            'المدارس': {'NOUN': 'الْمَدَارِس'},
            # Adjectives
            'كبير': {'ADJ': 'كَبِير'},
            'جديد': {'ADJ': 'جَدِيد'},
            'جميل': {'ADJ': 'جَمِيل'},
            'الكبيرة': {'ADJ': 'الْكَبِيرَة'},
            'الجديدة': {'ADJ': 'الْجَدِيدَة'},
            # Particles
            'إلى': {'ADP': 'إِلَى'},
            'في': {'ADP': 'فِي'},
            'من': {'ADP': 'مِنْ'},
            'على': {'ADP': 'عَلَى'},
            'عن': {'ADP': 'عَنْ'},
            'إن': {'PART': 'إِنَّ'},
            'ما': {'PART': 'مَا'},
            'لا': {'PART': 'لَا'},
            'قد': {'PART': 'قَدْ'},
            'هل': {'PART': 'هَلْ'},
            # Pronouns / demonstratives
            'هذا': {'DET': 'هٰذَا'},
            'هو': {'PRON': 'هُوَ'},
            'هي': {'PRON': 'هِيَ'},
            'أنا': {'PRON': 'أَنَا'},
            # Special
            'أجمل': {'VERB': 'أَجْمَلَ', 'ADJ': 'أَجْمَل'},
        }


def quick_test():
    """Run pipeline test with mock parser."""
    import sys, io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      errors='replace')

    engine = ArabiyaEngine.create_with_mock()
    tests = [
        "ذهب الطالب إلى المدرسة",
        "كتب الطالب الدرس",
        "إن العلم نور",
        "قرأ المعلمون كتبا جديدة",
        "ما أجمل السماء",
        "يكتب المعلمون الدروس في المدارس الكبيرة",
    ]

    print("=" * 70)
    print("    ARABIYA ENGINE -- PIPELINE TEST")
    print("=" * 70)

    for text in tests:
        result = engine.process(text)
        print(f"\n{'_' * 60}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.diacritized}")
        print(f"  Coverage: {result.lexicon_coverage:.0%} lexicon | "
              f"{result.case_applied} case endings")
        for sent in result.sentences:
            for w in sent.words:
                cm = (w.case_diacritic.encode('unicode_escape').decode()
                      if w.case_diacritic else '-')
                print(
                    f"    {w.bare:>12} -> {w.final_diacritized:<18} "
                    f"[{w.pos:<5} {w.case_display:<8} {w.relation:<10} "
                    f"src={w.diac_source:<8} case={cm}]"
                )

    print(f"\n{'=' * 70}")
    print("    PIPELINE TEST COMPLETE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    quick_test()
