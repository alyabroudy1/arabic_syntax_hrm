#!/usr/bin/env python3
"""
Tests for Arabic Syntax Grid Encoding
======================================

Verifies that the grid encoding produces linguistically correct grids
for known Arabic sentences.

Run: python -m pytest tests/test_grid_encoding.py -v
"""

import sys
import numpy as np
from pathlib import Path
import importlib.util

# ─────────────────────────────────────────────
# Import the numbered script via importlib
# (02_build_syntax_grids.py can't be imported normally)
# ─────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "build_syntax_grids",
    str(Path(__file__).parent.parent / "scripts" / "02_build_syntax_grids.py")
)
grid_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid_module)

sentence_to_grid = grid_module.sentence_to_grid
create_difficulty_variants = grid_module.create_difficulty_variants
parse_ud_features = grid_module.parse_ud_features
encode_agreement = grid_module.encode_agreement
word_to_bucket = grid_module.word_to_bucket
lemma_to_pattern = grid_module.lemma_to_pattern
print_grid = grid_module.print_grid
POS_TAGS = grid_module.POS_TAGS
CASE_TAGS = grid_module.CASE_TAGS
DEP_RELS = grid_module.DEP_RELS
MAX_WORDS = grid_module.MAX_WORDS
NUM_FEATURES = grid_module.NUM_FEATURES


# ─────────────────────────────────────────────
# Test Sentences (hand-crafted with known parses)
# ─────────────────────────────────────────────

# "ذهب الطالبُ إلى المدرسةِ" - The student went to the school
SENTENCE_1 = {
    "sent_id": "test-001",
    "text": "ذهب الطالبُ إلى المدرسةِ",
    "tokens": [
        {"id": 1, "form": "ذهب", "lemma": "ذَهَبَ", "upos": "VERB",
         "xpos": "_", "feats": "Aspect=Perf|Gender=Masc|Number=Sing|Person=3|Voice=Act",
         "head": 0, "deprel": "root", "deps": "_", "misc": "_"},
        {"id": 2, "form": "الطالبُ", "lemma": "طَالِب", "upos": "NOUN",
         "xpos": "_", "feats": "Case=Nom|Definite=Def|Gender=Masc|Number=Sing",
         "head": 1, "deprel": "nsubj", "deps": "_", "misc": "_"},
        {"id": 3, "form": "إلى", "lemma": "إِلَى", "upos": "ADP",
         "xpos": "_", "feats": "_",
         "head": 4, "deprel": "case", "deps": "_", "misc": "_"},
        {"id": 4, "form": "المدرسةِ", "lemma": "مَدْرَسَة", "upos": "NOUN",
         "xpos": "_", "feats": "Case=Gen|Definite=Def|Gender=Fem|Number=Sing",
         "head": 1, "deprel": "obl", "deps": "_", "misc": "_"},
    ],
    "num_tokens": 4,
}

# "إنَّ العلمَ نورٌ" - Indeed, knowledge is light
SENTENCE_2 = {
    "sent_id": "test-002",
    "text": "إنَّ العلمَ نورٌ",
    "tokens": [
        {"id": 1, "form": "إنَّ", "lemma": "إِنَّ", "upos": "PART",
         "xpos": "_", "feats": "_",
         "head": 3, "deprel": "mark", "deps": "_", "misc": "_"},
        {"id": 2, "form": "العلمَ", "lemma": "عِلْم", "upos": "NOUN",
         "xpos": "_", "feats": "Case=Acc|Definite=Def|Gender=Masc|Number=Sing",
         "head": 3, "deprel": "nsubj", "deps": "_", "misc": "_"},
        {"id": 3, "form": "نورٌ", "lemma": "نُور", "upos": "NOUN",
         "xpos": "_", "feats": "Case=Nom|Definite=Ind|Gender=Masc|Number=Sing",
         "head": 0, "deprel": "root", "deps": "_", "misc": "_"},
    ],
    "num_tokens": 3,
}


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

class TestFeatureExtraction:
    """Test individual feature extraction functions."""
    
    def test_parse_ud_features_nominal(self):
        feats = parse_ud_features("Case=Nom|Definite=Def|Gender=Masc|Number=Sing")
        assert feats['Case'] == 'Nom'
        assert feats['Definite'] == 'Def'
        assert feats['Gender'] == 'Masc'
        assert feats['Number'] == 'Sing'
    
    def test_parse_ud_features_empty(self):
        assert parse_ud_features("_") == {}
        assert parse_ud_features("") == {}
    
    def test_parse_ud_features_verb(self):
        feats = parse_ud_features("Aspect=Perf|Gender=Masc|Number=Sing|Person=3|Voice=Act")
        assert feats['Aspect'] == 'Perf'
        assert feats['Person'] == '3'
    
    def test_encode_agreement_masc_sing_3rd(self):
        feats = {'Gender': 'Masc', 'Number': 'Sing', 'Person': '3'}
        val = encode_agreement(feats)
        assert val > 0  # non-zero for real features
    
    def test_encode_agreement_unique(self):
        """Known gender×number×person combos should have unique codes.
        
        Note: combinations involving '_' (unknown) can collide with each other,
        which is acceptable — only *known* linguistic feature combos must be unique.
        """
        # Test that all KNOWN (non-underscore) combos are unique
        known_combos = {}
        for g in ['Masc', 'Fem']:
            for n in ['Sing', 'Dual', 'Plur']:
                for p in ['1', '2', '3']:
                    val = encode_agreement({'Gender': g, 'Number': n, 'Person': p})
                    key = f"Gender={g}, Number={n}, Person={p}"
                    assert val not in known_combos, \
                        f"Collision: {key} vs {known_combos[val]}"
                    known_combos[val] = key
        
        # All known combos should be > 0 (0 is reserved for fully unknown)
        for val in known_combos:
            assert val > 0
    
    def test_word_bucket_deterministic(self):
        """Same word should always hash to same bucket."""
        assert word_to_bucket("ذهب") == word_to_bucket("ذهب")
        assert word_to_bucket("الطالب") == word_to_bucket("الطالب")
    
    def test_word_bucket_range(self):
        """Bucket values should be in valid range."""
        for word in ["ذهب", "إلى", "المدرسة", "كتب", "علم"]:
            b = word_to_bucket(word)
            assert 1 <= b <= 255, f"Bucket {b} for '{word}' out of range"
    
    def test_word_bucket_not_zero(self):
        """Zero is reserved for padding, real words should never be 0."""
        for word in ["ا", "ب", "ت", "في", "من"]:
            assert word_to_bucket(word) > 0


class TestGridEncoding:
    """Test sentence-to-grid conversion."""
    
    def test_grid_shape(self):
        grid_obj = sentence_to_grid(SENTENCE_1)
        assert grid_obj.grid.shape == (MAX_WORDS, NUM_FEATURES)
        assert grid_obj.mask.shape == (MAX_WORDS,)
        assert grid_obj.solution.shape == (MAX_WORDS, NUM_FEATURES)
    
    def test_mask_correctness(self):
        grid_obj = sentence_to_grid(SENTENCE_1)
        # 4 real words
        assert grid_obj.mask[:4].sum() == 4
        # Rest is padding
        assert grid_obj.mask[4:].sum() == 0
    
    def test_num_words(self):
        grid_obj = sentence_to_grid(SENTENCE_1)
        assert grid_obj.num_words == 4
    
    def test_pos_tags_correct(self):
        grid_obj = sentence_to_grid(SENTENCE_1)
        # "ذهب" → VERB
        assert grid_obj.grid[0, 1] == POS_TAGS['VERB']
        assert grid_obj.solution[0, 1] == POS_TAGS['VERB']
        # "الطالبُ" → NOUN
        assert grid_obj.grid[1, 1] == POS_TAGS['NOUN']
        # "إلى" → ADP
        assert grid_obj.grid[2, 1] == POS_TAGS['ADP']
        # "المدرسةِ" → NOUN
        assert grid_obj.grid[3, 1] == POS_TAGS['NOUN']
    
    def test_case_solution_correct(self):
        """Verify case markings are correctly encoded in solution."""
        grid_obj = sentence_to_grid(SENTENCE_1)
        # "ذهب" → no case (verb, مبني)
        assert grid_obj.solution[0, 3] == CASE_TAGS['_']
        # "الطالبُ" → Nom (مرفوع — فاعل)
        assert grid_obj.solution[1, 3] == CASE_TAGS['Nom']
        # "إلى" → no case (preposition)
        assert grid_obj.solution[2, 3] == CASE_TAGS['_']
        # "المدرسةِ" → Gen (مجرور — بعد حرف جر)
        assert grid_obj.solution[3, 3] == CASE_TAGS['Gen']
    
    def test_case_masked_in_input(self):
        """Case column should be masked (0) in input grid."""
        grid_obj = sentence_to_grid(SENTENCE_1)
        assert grid_obj.grid[0, 3] == 0  # masked
        assert grid_obj.grid[1, 3] == 0  # masked
        assert grid_obj.grid[3, 3] == 0  # masked
    
    def test_dep_head_solution(self):
        """Verify dependency heads in solution."""
        grid_obj = sentence_to_grid(SENTENCE_1)
        # "ذهب" head=0 (root)
        assert grid_obj.solution[0, 4] == 0
        # "الطالبُ" head=1 (→ ذهب)
        assert grid_obj.solution[1, 4] == 1
        # "إلى" head=4 (→ المدرسة)
        assert grid_obj.solution[2, 4] == 4
        # "المدرسةِ" head=1 (→ ذهب)
        assert grid_obj.solution[3, 4] == 1
    
    def test_dep_rel_solution(self):
        """Verify dependency relations in solution."""
        grid_obj = sentence_to_grid(SENTENCE_1)
        assert grid_obj.solution[0, 5] == DEP_RELS['root']
        assert grid_obj.solution[1, 5] == DEP_RELS['nsubj']
        assert grid_obj.solution[2, 5] == DEP_RELS['case']
        assert grid_obj.solution[3, 5] == DEP_RELS['obl']
    
    def test_deps_masked_in_input(self):
        """Dependency columns should be masked in input."""
        grid_obj = sentence_to_grid(SENTENCE_1)
        for i in range(4):
            assert grid_obj.grid[i, 4] == 0  # head masked
            assert grid_obj.grid[i, 5] == 0  # rel masked
    
    def test_definiteness(self):
        grid_obj = sentence_to_grid(SENTENCE_1)
        # "الطالبُ" → definite
        assert grid_obj.grid[1, 7] == 1
        assert grid_obj.solution[1, 7] == 1
    
    def test_inna_sentence(self):
        """Test إنَّ + اسمها المنصوب pattern."""
        grid_obj = sentence_to_grid(SENTENCE_2)
        assert grid_obj.num_words == 3
        
        # "إنَّ" → PART, no case
        assert grid_obj.solution[0, 1] == POS_TAGS['PART']
        assert grid_obj.solution[0, 3] == CASE_TAGS['_']
        
        # "العلمَ" → NOUN, Acc (اسم إنّ منصوب)
        assert grid_obj.solution[1, 1] == POS_TAGS['NOUN']
        assert grid_obj.solution[1, 3] == CASE_TAGS['Acc']
        
        # "نورٌ" → NOUN, Nom (خبر إنّ مرفوع)
        assert grid_obj.solution[2, 1] == POS_TAGS['NOUN']
        assert grid_obj.solution[2, 3] == CASE_TAGS['Nom']
    
    def test_padding_is_zero(self):
        """Padding rows should be all zeros."""
        grid_obj = sentence_to_grid(SENTENCE_1)
        # Row 5 onwards should be zero (4 tokens, 0-indexed)
        assert np.all(grid_obj.grid[4:] == 0)
        assert np.all(grid_obj.solution[4:] == 0)
        assert np.all(grid_obj.mask[4:] == 0)
    
    def test_given_features_match_solution(self):
        """Given (unmasked) features in input should match solution."""
        grid_obj = sentence_to_grid(SENTENCE_1)
        for i in range(grid_obj.num_words):
            # Cols 0,1,2,6,7 are given (not masked)
            for col in [0, 1, 2, 6, 7]:
                assert grid_obj.grid[i, col] == grid_obj.solution[i, col], \
                    f"Mismatch at word {i}, col {col}"


class TestDifficultyVariants:
    """Test difficulty variant generation."""
    
    def test_creates_correct_number(self):
        base = sentence_to_grid(SENTENCE_1)
        variants = create_difficulty_variants(base, num_variants=3)
        assert len(variants) == 3
    
    def test_difficulty_0_all_masked(self):
        base = sentence_to_grid(SENTENCE_1)
        variants = create_difficulty_variants(base, num_variants=1)
        v = variants[0]
        # Cols 3,4,5 should be all zeros in input
        for col in [3, 4, 5]:
            assert np.all(v.grid[:, col] == 0)
    
    def test_difficulty_2_cases_revealed(self):
        base = sentence_to_grid(SENTENCE_1)
        variants = create_difficulty_variants(base, num_variants=3)
        v_easy = variants[2]
        # In easy mode, case (col 3) values should match solution for real words
        for i in range(base.num_words):
            assert v_easy.grid[i, 3] == base.solution[i, 3]
    
    def test_solutions_preserved(self):
        """All variants should have identical solutions."""
        base = sentence_to_grid(SENTENCE_1)
        variants = create_difficulty_variants(base, num_variants=3)
        for v in variants:
            np.testing.assert_array_equal(v.solution, base.solution)
    
    def test_masks_preserved(self):
        """All variants should have identical masks."""
        base = sentence_to_grid(SENTENCE_1)
        variants = create_difficulty_variants(base, num_variants=3)
        for v in variants:
            np.testing.assert_array_equal(v.mask, base.mask)
    
    def test_difficulty_label(self):
        base = sentence_to_grid(SENTENCE_1)
        variants = create_difficulty_variants(base, num_variants=3)
        assert variants[0].difficulty == 0
        assert variants[1].difficulty == 1
        assert variants[2].difficulty == 2


class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_single_word(self):
        """Minimal sentence (single word)."""
        sent = {
            "sent_id": "test-edge-1",
            "text": "اكتبْ",
            "tokens": [
                {"id": 1, "form": "اكتبْ", "lemma": "كَتَبَ", "upos": "VERB",
                 "xpos": "_", "feats": "Mood=Jus|Number=Sing|Person=2|Voice=Act",
                 "head": 0, "deprel": "root", "deps": "_", "misc": "_"},
            ],
            "num_tokens": 1,
        }
        grid_obj = sentence_to_grid(sent)
        assert grid_obj.num_words == 1
        assert grid_obj.mask[0] == 1
        assert grid_obj.mask[1:].sum() == 0
    
    def test_max_length_sentence(self):
        """Sentence at max length should not error."""
        tokens = []
        for i in range(MAX_WORDS):
            tokens.append({
                "id": i + 1, "form": f"كلمة{i}", "lemma": f"كلمة{i}", 
                "upos": "NOUN", "xpos": "_", 
                "feats": "Case=Nom|Gender=Masc|Number=Sing",
                "head": 0 if i == 0 else 1, 
                "deprel": "root" if i == 0 else "dep",
                "deps": "_", "misc": "_",
            })
        sent = {"sent_id": "test-max", "text": "جملة طويلة", 
                "tokens": tokens, "num_tokens": MAX_WORDS}
        grid_obj = sentence_to_grid(sent)
        assert grid_obj.num_words == MAX_WORDS
        assert grid_obj.mask.sum() == MAX_WORDS
    
    def test_over_max_length_truncated(self):
        """Sentences longer than MAX_WORDS should be truncated, not error."""
        tokens = []
        for i in range(MAX_WORDS + 10):
            tokens.append({
                "id": i + 1, "form": f"كلمة{i}", "lemma": f"كلمة{i}",
                "upos": "NOUN", "xpos": "_", "feats": "_",
                "head": 0 if i == 0 else 1,
                "deprel": "root" if i == 0 else "dep",
                "deps": "_", "misc": "_",
            })
        sent = {"sent_id": "test-trunc", "text": "جملة طويلة جداً",
                "tokens": tokens, "num_tokens": MAX_WORDS + 10}
        grid_obj = sentence_to_grid(sent)
        assert grid_obj.num_words == MAX_WORDS
        assert grid_obj.mask.sum() == MAX_WORDS
    
    def test_unknown_pos_tag(self):
        """Unknown POS tag should map to 0."""
        sent = {
            "sent_id": "test-unk",
            "text": "test",
            "tokens": [
                {"id": 1, "form": "test", "lemma": "test", "upos": "UNKNOWN_POS",
                 "xpos": "_", "feats": "_",
                 "head": 0, "deprel": "root", "deps": "_", "misc": "_"},
            ],
            "num_tokens": 1,
        }
        grid_obj = sentence_to_grid(sent)
        assert grid_obj.grid[0, 1] == 0  # unknown maps to 0
    
    def test_unknown_deprel(self):
        """Unknown dependency relation should use fallback."""
        sent = {
            "sent_id": "test-unk-dep",
            "text": "test",
            "tokens": [
                {"id": 1, "form": "test", "lemma": "test", "upos": "NOUN",
                 "xpos": "_", "feats": "_",
                 "head": 0, "deprel": "totally_unknown_rel", "deps": "_", "misc": "_"},
            ],
            "num_tokens": 1,
        }
        grid_obj = sentence_to_grid(sent)
        # Should either be 0 or dep (fallback)
        assert grid_obj.solution[0, 5] in (0, DEP_RELS.get('dep', 0))


# ─────────────────────────────────────────────
# Visual Verification (not a pytest test, run manually)
# ─────────────────────────────────────────────

def visual_verification():
    """Print grids for manual linguistic verification."""
    print("\n" + "=" * 60)
    print("VISUAL VERIFICATION — Check these manually!")
    print("=" * 60)
    
    sentences = [SENTENCE_1, SENTENCE_2]
    
    for sent in sentences:
        grid_obj = sentence_to_grid(sent)
        print_grid(grid_obj, show_solution=True)
        
        print("\n  Verification checklist:")
        print("  [ ] POS tags match the words?")
        print("  [ ] Case markings are linguistically correct?")
        print("  [ ] Dependency heads point to correct words?")
        print("  [ ] Dependency relations are correct?")
        print("  [ ] Agreement features are correct?")
        print("  [ ] Definiteness is correct?")
        print()


if __name__ == "__main__":
    # Run visual verification when called directly
    visual_verification()
    
    # Also run pytest
    import pytest
    pytest.main([__file__, "-v"])
