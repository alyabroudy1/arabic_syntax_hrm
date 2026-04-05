"""
Arabiya Engine -- Unit Tests
Run: python tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from arabiya.core import (
    strip_diacritics, decompose_word, recompose_word,
    replace_case_ending, has_definite_article,
    ends_with_taa_marbuta, case_to_arabic,
    FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN, SUKUN, SHADDA,
)
from arabiya.preprocessor import ArabicPreprocessor
from arabiya.engine import ArabiyaEngine


def test_strip_diacritics():
    assert strip_diacritics('كَتَبَ') == 'كتب'
    assert strip_diacritics('بِسْمِ') == 'بسم'
    assert strip_diacritics('hello') == 'hello'
    assert strip_diacritics('') == ''
    print("  OK strip_diacritics")


def test_decompose_recompose():
    for word in ['كَتَبَ', 'الطَّالِبُ', 'بِسْمِ']:
        assert recompose_word(decompose_word(word)) == word
    print("  OK decompose/recompose")


def test_replace_case_ending():
    r = replace_case_ending('الطالب', DAMMA)
    assert DAMMA in r, f"Expected damma in '{r}'"
    r2 = replace_case_ending('الطالبِ', DAMMA)
    assert DAMMA in r2 and KASRA not in r2
    print("  OK replace_case_ending")


def test_has_definite_article():
    assert has_definite_article('الكتاب') is True
    assert has_definite_article('كتاب') is False
    assert has_definite_article('ال') is False
    print("  OK has_definite_article")


def test_ends_with_taa_marbuta():
    assert ends_with_taa_marbuta('مدرسة') is True
    assert ends_with_taa_marbuta('كتاب') is False
    print("  OK ends_with_taa_marbuta")


def test_case_tags_are_ud():
    """Bug Fix #3 verification: all case tags are UD format."""
    assert case_to_arabic('Nom') == 'مرفوع'
    assert case_to_arabic('Acc') == 'منصوب'
    assert case_to_arabic('Gen') == 'مجرور'
    assert case_to_arabic(None) == 'مبني'
    print("  OK case tags are UD internally")


def test_preprocessor():
    prep = ArabicPreprocessor()
    r = prep.process_single('ذهب الطالب إلى المدرسة')
    assert r == ['ذهب', 'الطالب', 'إلى', 'المدرسة'], f"Got: {r}"
    r2 = prep.process('ذهب الطالب. ورجع.')
    assert len(r2) == 2, f"Expected 2 sentences, got {len(r2)}"
    print("  OK preprocessor")


def test_engine_known_sentence():
    """Test pipeline on a known sentence with hardcoded parse."""
    engine = ArabiyaEngine.create_with_mock()
    result = engine.process('ذهب الطالب إلى المدرسة')

    assert result.total_words == 4
    assert result.case_applied > 0
    assert len(result.sentences) == 1

    words = result.sentences[0].words

    # الطالب = nsubj = Nom -> damma
    talib = words[1]
    assert talib.case_tag == 'Nom', f"Expected Nom, got {talib.case_tag}"
    assert DAMMA in talib.final_diacritized, \
        f"Expected damma in '{talib.final_diacritized}'"

    # المدرسة = obl after إلى = Gen -> kasra
    madrasa = words[3]
    assert madrasa.case_tag == 'Gen', f"Expected Gen, got {madrasa.case_tag}"
    assert KASRA in madrasa.final_diacritized, \
        f"Expected kasra in '{madrasa.final_diacritized}'"

    print("  OK engine known sentence (ذهب الطالب)")


def test_engine_inn_construction():
    """إن العلم نور — tests accusative + nominative."""
    engine = ArabiyaEngine.create_with_mock()
    result = engine.process('إن العلم نور')
    words = result.sentences[0].words

    assert words[1].case_tag == 'Acc', f"Expected Acc, got {words[1].case_tag}"
    assert words[2].case_tag == 'Nom', f"Expected Nom, got {words[2].case_tag}"
    print("  OK inn construction")


def test_engine_transitive():
    engine = ArabiyaEngine.create_with_mock()
    result = engine.process('كتب الطالب الدرس')
    words = result.sentences[0].words

    assert words[1].case_tag == 'Nom'  # subject
    assert words[2].case_tag == 'Acc'  # object
    print("  OK transitive verb")


def test_word_info_has_features():
    """Bug Fix #1 verification: features dict exists."""
    engine = ArabiyaEngine.create_with_mock()
    result = engine.process('ذهب الطالب إلى المدرسة')
    for sent in result.sentences:
        for w in sent.words:
            assert hasattr(w, 'features'), "WordInfo missing features!"
            assert isinstance(w.features, dict)
    print("  OK WordInfo has features dict (Bug #1 fixed)")


def test_lexicon_coverage():
    engine = ArabiyaEngine.create_with_mock()
    result = engine.process('ذهب الطالب إلى المدرسة')
    assert result.lexicon_coverage > 0.5
    print("  OK lexicon coverage > 50%")


def test_result_summary():
    engine = ArabiyaEngine.create_with_mock()
    result = engine.process('ذهب الطالب إلى المدرسة')
    s = result.summary()
    assert 'Arabiya' in s
    d = result.detail()
    assert 'NOUN' in d or 'VERB' in d
    print("  OK result summary/detail")


def run_all():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')
    print("=" * 50)
    print("    ARABIYA ENGINE -- UNIT TESTS")
    print("=" * 50)

    tests = [
        test_strip_diacritics,
        test_decompose_recompose,
        test_replace_case_ending,
        test_has_definite_article,
        test_ends_with_taa_marbuta,
        test_case_tags_are_ud,
        test_preprocessor,
        test_engine_known_sentence,
        test_engine_inn_construction,
        test_engine_transitive,
        test_word_info_has_features,
        test_lexicon_coverage,
        test_result_summary,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'=' * 50}")
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
