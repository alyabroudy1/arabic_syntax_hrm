"""
Parser interface — wraps the HRM parser with a clean API.

BUG FIX #3: Uses UD case tags (Nom/Acc/Gen) everywhere internally.
Arabic labels (مرفوع etc.) are display-only via core.case_to_arabic().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json
import os


@dataclass
class ParseResult:
    """Output from the parser for a single sentence."""
    words: List[str]
    pos_tags: List[str]
    heads: List[int]
    relations: List[str]
    case_tags: List[Optional[str]]      # UD format: "Nom"/"Acc"/"Gen"/"Jus"/None
    features: List[Dict[str, str]] = field(default_factory=list)
    case_confidences: List[float] = field(default_factory=list)
    head_confidences: List[float] = field(default_factory=list)


class ParserInterface(ABC):
    @abstractmethod
    def parse(self, words: List[str]) -> ParseResult:
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        ...


class MockParser(ParserInterface):
    """
    Mock parser for testing the pipeline without a trained model.
    Returns hardcoded analyses for known sentences and heuristic
    fallback for unknown ones. Uses UD case tags internally.
    """

    def __init__(self):
        self._loaded = True
        self._known = self._build_known()

    def is_loaded(self) -> bool:
        return True

    def parse(self, words: List[str]) -> ParseResult:
        key = ' '.join(words)
        if key in self._known:
            return self._known[key]
        return self._heuristic_parse(words)

    def _heuristic_parse(self, words: List[str]) -> ParseResult:
        n = len(words)
        pos_tags, heads, relations, case_tags, features = [], [], [], [], []

        for i, word in enumerate(words):
            pos, case, feat = self._guess(word, i, words)
            pos_tags.append(pos)
            case_tags.append(case)
            features.append(feat)
            if i == 0:
                heads.append(0)
                relations.append('root')
            else:
                heads.append(1)
                relations.append('dep')

        return ParseResult(
            words=words, pos_tags=pos_tags, heads=heads,
            relations=relations, case_tags=case_tags,
            features=features,
            case_confidences=[0.5] * n, head_confidences=[0.5] * n,
        )

    def _guess(self, word, pos_in_sent, all_words):
        feat = {
            'definite': 'yes' if word.startswith('ال') else 'no',
            'number': 'sing', 'gender': 'masc',
        }
        if word.endswith('ة'):
            feat['gender'] = 'fem'
        if word.endswith('ون') or word.endswith('ين'):
            feat['number'] = 'plur'
            feat['plural_type'] = 'sound_masc'
        if word.endswith('ات'):
            feat['number'] = 'plur'
            feat['gender'] = 'fem'
            feat['plural_type'] = 'sound_fem'

        PREPS = {'إلى', 'في', 'من', 'على', 'عن', 'حتى'}
        CONJ = {'و', 'أو', 'ثم', 'ف', 'لكن', 'بل'}
        PARTS = {'أن', 'إن', 'أنّ', 'إنّ', 'لأن', 'كأن', 'لكنّ',
                 'ليت', 'لعل', 'لا', 'لم', 'لن', 'ما', 'هل', 'أ', 'قد', 'يا'}
        PRONS = {'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم',
                 'الذي', 'التي', 'الذين', 'هذا', 'هذه', 'ذلك', 'تلك'}

        if word in PREPS:
            return 'ADP', None, feat
        if word in CONJ:
            return 'CCONJ', None, feat
        if word in PARTS:
            return 'PART', None, feat
        if word in PRONS:
            return 'PRON', None, feat
        if pos_in_sent == 0 and not word.startswith('ال'):
            feat['verb_form'] = 'past'
            return 'VERB', None, feat

        # Noun — determine case from context
        prev = all_words[pos_in_sent - 1] if pos_in_sent > 0 else ''
        if prev in PREPS:
            return 'NOUN', 'Gen', feat
        elif pos_in_sent == 1:
            return 'NOUN', 'Nom', feat
        else:
            return 'NOUN', 'Acc', feat

    def _build_known(self) -> Dict[str, ParseResult]:
        known = {}

        known['ذهب الطالب إلى المدرسة'] = ParseResult(
            words=['ذهب', 'الطالب', 'إلى', 'المدرسة'],
            pos_tags=['VERB', 'NOUN', 'ADP', 'NOUN'],
            heads=[0, 1, 4, 1],
            relations=['root', 'nsubj', 'case', 'obl'],
            case_tags=[None, 'Nom', None, 'Gen'],
            features=[
                {'verb_form': 'past', 'person': '3', 'gender': 'masc', 'number': 'sing'},
                {'definite': 'yes', 'gender': 'masc', 'number': 'sing'},
                {},
                {'definite': 'yes', 'gender': 'fem', 'number': 'sing'},
            ],
            case_confidences=[0.95, 0.92, 0.99, 0.91],
            head_confidences=[1.0, 0.88, 0.95, 0.85],
        )

        known['كتب الطالب الدرس'] = ParseResult(
            words=['كتب', 'الطالب', 'الدرس'],
            pos_tags=['VERB', 'NOUN', 'NOUN'],
            heads=[0, 1, 1],
            relations=['root', 'nsubj', 'obj'],
            case_tags=[None, 'Nom', 'Acc'],
            features=[
                {'verb_form': 'past', 'person': '3', 'gender': 'masc', 'number': 'sing'},
                {'definite': 'yes', 'gender': 'masc', 'number': 'sing'},
                {'definite': 'yes', 'gender': 'masc', 'number': 'sing'},
            ],
            case_confidences=[0.95, 0.93, 0.90],
            head_confidences=[1.0, 0.90, 0.87],
        )

        known['إن العلم نور'] = ParseResult(
            words=['إن', 'العلم', 'نور'],
            pos_tags=['PART', 'NOUN', 'NOUN'],
            heads=[0, 1, 1],
            relations=['root', 'nsubj', 'pred'],
            case_tags=[None, 'Acc', 'Nom'],
            features=[
                {},
                {'definite': 'yes', 'gender': 'masc', 'number': 'sing'},
                {'definite': 'no', 'gender': 'masc', 'number': 'sing'},
            ],
            case_confidences=[0.99, 0.93, 0.90],
            head_confidences=[1.0, 0.92, 0.88],
        )

        known['قرأ المعلمون كتبا جديدة'] = ParseResult(
            words=['قرأ', 'المعلمون', 'كتبا', 'جديدة'],
            pos_tags=['VERB', 'NOUN', 'NOUN', 'ADJ'],
            heads=[0, 1, 1, 3],
            relations=['root', 'nsubj', 'obj', 'amod'],
            case_tags=[None, 'Nom', 'Acc', 'Acc'],
            features=[
                {'verb_form': 'past'},
                {'definite': 'yes', 'gender': 'masc', 'number': 'plur',
                 'plural_type': 'sound_masc'},
                {'definite': 'no', 'gender': 'masc', 'number': 'plur',
                 'plural_type': 'broken'},
                {'definite': 'no', 'gender': 'fem', 'number': 'sing'},
            ],
            case_confidences=[0.95, 0.94, 0.88, 0.85],
            head_confidences=[1.0, 0.91, 0.86, 0.83],
        )

        known['ما أجمل السماء'] = ParseResult(
            words=['ما', 'أجمل', 'السماء'],
            pos_tags=['PART', 'VERB', 'NOUN'],
            heads=[2, 0, 2],
            relations=['dep', 'root', 'obj'],
            case_tags=[None, None, 'Acc'],
            features=[
                {},
                {'verb_form': 'past'},
                {'definite': 'yes', 'gender': 'fem', 'number': 'sing'},
            ],
            case_confidences=[0.99, 0.95, 0.92],
            head_confidences=[0.90, 1.0, 0.88],
        )

        known['يكتب المعلمون الدروس في المدارس الكبيرة'] = ParseResult(
            words=['يكتب', 'المعلمون', 'الدروس', 'في', 'المدارس', 'الكبيرة'],
            pos_tags=['VERB', 'NOUN', 'NOUN', 'ADP', 'NOUN', 'ADJ'],
            heads=[0, 1, 1, 5, 1, 5],
            relations=['root', 'nsubj', 'obj', 'case', 'obl', 'amod'],
            case_tags=['Nom', 'Nom', 'Acc', None, 'Gen', 'Gen'],
            features=[
                {'verb_form': 'pres', 'person': '3', 'gender': 'masc', 'number': 'sing'},
                {'definite': 'yes', 'gender': 'masc', 'number': 'plur',
                 'plural_type': 'sound_masc'},
                {'definite': 'yes', 'gender': 'masc', 'number': 'plur',
                 'plural_type': 'broken'},
                {},
                {'definite': 'yes', 'gender': 'fem', 'number': 'plur',
                 'plural_type': 'broken'},
                {'definite': 'yes', 'gender': 'fem', 'number': 'sing'},
            ],
            case_confidences=[0.91, 0.94, 0.90, 0.99, 0.88, 0.86],
            head_confidences=[1.0, 0.90, 0.87, 0.95, 0.84, 0.82],
        )

        return known
