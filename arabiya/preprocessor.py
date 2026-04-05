"""
Arabic text preprocessor — normalize, sentence-split, tokenize.
"""
import re
from typing import List, Tuple
from arabiya.core import strip_diacritics, strip_tatweel, DIACRITIC_PATTERN


class ArabicPreprocessor:
    SENT_END = re.compile(r'([.!؟?]\s*)')
    WHITESPACE = re.compile(r'\s+')
    PUNCT_SPLIT = re.compile(r'([.!؟?،؛:()«»\[\]"\'{}])')

    def __init__(self, strip_diacritics_from_input=True,
                 normalize_alef=False, split_punctuation=True):
        self.strip_diac = strip_diacritics_from_input
        self.norm_alef = normalize_alef
        self.split_punct = split_punctuation

    def process(self, text: str) -> List[List[str]]:
        text = self._normalize(text)
        raw_sents = self._split_sentences(text)
        result = []
        for sent in raw_sents:
            tokens = self._tokenize(sent)
            if tokens:
                result.append(tokens)
        return result

    def process_single(self, text: str) -> List[str]:
        return self._tokenize(self._normalize(text))

    def _normalize(self, text: str) -> str:
        text = strip_tatweel(text)
        text = re.sub('[\u200B-\u200F\u202A-\u202E\uFEFF]', '', text)
        text = self.WHITESPACE.sub(' ', text).strip()
        if self.norm_alef:
            text = text.replace('\u0623', '\u0627')
            text = text.replace('\u0625', '\u0627')
            text = text.replace('\u0671', '\u0627')
        return text

    def _split_sentences(self, text: str) -> List[str]:
        parts = self.SENT_END.split(text)
        sentences = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if self.SENT_END.fullmatch(part + ' ') or self.SENT_END.fullmatch(part):
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += " " + part if current else part
        if current.strip():
            sentences.append(current.strip())
        return sentences if sentences else [text]

    def _tokenize(self, sentence: str) -> List[str]:
        if self.split_punct:
            sentence = self.PUNCT_SPLIT.sub(r' \1 ', sentence)
        raw_tokens = self.WHITESPACE.split(sentence.strip())
        tokens = []
        for token in raw_tokens:
            if not token:
                continue
            if self.strip_diac:
                token = strip_diacritics(token)
            tokens.append(token)
        return tokens

    @staticmethod
    def is_punctuation(token: str) -> bool:
        return bool(re.fullmatch(r'[.!؟?،؛:()«»\[\]"\'{}…]+', token))

    @staticmethod
    def is_arabic(token: str) -> bool:
        return bool(re.search(r'[\u0621-\u064A]', token))
