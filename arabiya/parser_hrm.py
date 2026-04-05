"""
HRM Parser Adapter — Bridges ArabicHRMGridParserV2 to the Arabiya pipeline.

Loads the trained HRM parser model (33M params, 75% UAS / 68% LAS)
and translates its tensor outputs into the pipeline's ParseResult format.

Key Translation:
    model.pred_cases (int tensor, 4 classes) → UD case tags ("Nom"/"Acc"/"Gen"/None)
    model.pred_rels  (int tensor, 50 classes) → UD relation strings
    model.pred_heads (int tensor, 0-indexed)  → head indices

Usage:
    from arabiya.parser_hrm import HRMParserAdapter
    parser = HRMParserAdapter("models/v2_arabic_syntax/", device="cpu")
    result = parser.parse(["ذهب", "الطالب", "إلى", "المدرسة"])
"""

import os
import sys
import hashlib
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from arabiya.parser import ParserInterface, ParseResult


# ═══════════════════════════════════════════════════
#          VOCAB MAPPINGS (from 02_build_syntax_grids.py)
# ═══════════════════════════════════════════════════

POS_TAGS = {
    '_': 0, 'NOUN': 1, 'VERB': 2, 'ADJ': 3, 'ADV': 4, 'ADP': 5,
    'CONJ': 6, 'CCONJ': 6, 'SCONJ': 7, 'PART': 8, 'DET': 9,
    'PRON': 10, 'PROPN': 11, 'NUM': 12, 'PUNCT': 13, 'AUX': 14,
    'INTJ': 15, 'X': 16, 'SYM': 17,
}

POS_TAGS_REVERSE = {v: k for k, v in POS_TAGS.items() if k != 'CONJ'}
POS_TAGS_REVERSE[6] = 'CCONJ'  # prefer CCONJ over CONJ

DEP_RELS = {
    '_': 0, 'root': 1, 'nsubj': 2, 'obj': 3, 'iobj': 4, 'obl': 5,
    'advmod': 6, 'amod': 7, 'nmod': 8, 'det': 9, 'case': 10,
    'conj': 11, 'cc': 12, 'punct': 13, 'flat': 14, 'compound': 15,
    'appos': 16, 'acl': 17, 'advcl': 18, 'cop': 19, 'mark': 20,
    'dep': 21, 'parataxis': 22, 'fixed': 23, 'vocative': 24,
    'nummod': 25, 'flat:foreign': 26, 'nsubj:pass': 27, 'csubj': 28,
    'xcomp': 29, 'ccomp': 30, 'orphan': 31,
}

DEP_RELS_REVERSE = {v: k for k, v in DEP_RELS.items()}

# Case class IDs → UD case tags
CASE_CLASSES = {0: None, 1: 'Nom', 2: 'Acc', 3: 'Gen'}

DEFINITE_MAP = {'_': 0, 'Ind': 0, 'Def': 1, 'Com': 2, 'Cons': 2}

GENDER_MAP = {'_': 0, 'Masc': 1, 'Fem': 2}
NUMBER_MAP = {'_': 0, 'Sing': 1, 'Dual': 2, 'Plur': 3}
PERSON_MAP = {'_': 0, '1': 1, '2': 2, '3': 3}


# ═══════════════════════════════════════════════════
#               WORD ENCODING
# ═══════════════════════════════════════════════════

def word_to_bucket(word: str, num_buckets: int = 8192) -> int:
    """Same hash as used in grid builder."""
    h = 0
    for ch in word:
        h = (h * 31 + ord(ch)) % num_buckets
    return max(1, h)


def stable_hash(s: str, bins: int) -> int:
    if not s:
        return 0
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % bins


class ArabicRootExtractor:
    """Lightweight algorithmic Arabic root extractor (from train script)."""
    PREFIXES = [
        'وال', 'فال', 'بال', 'كال', 'لل',
        'وب', 'ول', 'وي', 'فب', 'فل', 'في',
        'ال', 'لي', 'لن', 'لا', 'ست', 'سي', 'سن', 'سأ', 'سـ',
        'و', 'ف', 'ب', 'ل', 'ك', 'ي', 'ت', 'ن', 'أ', 'س',
    ]
    SUFFIXES = [
        'كما', 'هما', 'كم', 'هم', 'هن', 'نا', 'ها', 'ون', 'ين', 'ان', 'ات', 'وا',
        'تم', 'تن', 'ية', 'ته', 'تك',
        'ك', 'ه', 'ي', 'ا', 'ة', 'ت', 'ن',
    ]
    VOWELS_AND_DIACRITICS = set(
        '\u064e\u064f\u0650\u0651\u0652\u064b\u064c\u064d\u0670اوىيءئؤإأآ'
    )

    def extract_root(self, word: str) -> str:
        if not word or len(word) < 2:
            return word
        w = word
        for prefix in self.PREFIXES:
            if w.startswith(prefix) and len(w) - len(prefix) >= 2:
                w = w[len(prefix):]
                break
        for suffix in self.SUFFIXES:
            if w.endswith(suffix) and len(w) - len(suffix) >= 2:
                w = w[:-len(suffix)]
                break
        consonants = [c for c in w if c not in self.VOWELS_AND_DIACRITICS]
        if len(consonants) >= 3:
            return consonants[0] + consonants[1] + consonants[2]
        elif len(consonants) == 2:
            return consonants[0] + consonants[1]
        elif consonants:
            return consonants[0]
        return w


# ═══════════════════════════════════════════════════
#               HRM PARSER ADAPTER
# ═══════════════════════════════════════════════════

class HRMParserAdapter(ParserInterface):
    """Wraps ArabicHRMGridParserV2 for the Arabiya pipeline."""

    MAX_LEN = 32
    MAX_CHARS = 16

    def __init__(self, model_dir: str, device: str = 'cpu',
                 checkpoint_name: str = 'best_model.pt'):
        self.device = torch.device(device)
        self.root_extractor = ArabicRootExtractor()
        self._loaded = False

        # Load model
        model_path = os.path.join(model_dir, checkpoint_name)
        if not os.path.exists(model_path):
            print(f"[WARNING] No checkpoint at {model_path}")
            return

        # Import parser model
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)
        from models.v2.parser import ArabicHRMGridParserV2, ParserConfig

        config = ParserConfig(
            word_dim=384, n_heads=6, n_transformer_layers=3,
            n_gnn_rounds=3, n_relations=50, n_cases=4,
        )
        self.model = ArabicHRMGridParserV2(config).to(self.device)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self._loaded = True

        epoch = ckpt.get('epoch', '?')
        uas = ckpt.get('uas', 0)
        las = ckpt.get('las', 0)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[HRMParser] Loaded: epoch={epoch}, UAS={uas:.1f}%, "
              f"LAS={las:.1f}%, params={n_params:,}")

    def is_loaded(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def parse(self, words: List[str]) -> ParseResult:
        if not self._loaded:
            raise RuntimeError("HRM model not loaded")

        batch = self._encode_words(words)
        output = self.model(batch, epoch=0, training=False)
        return self._decode_output(words, output, batch['mask'])

    # ─────────────────────────────────────────────
    #        ENCODE: words → model batch dict
    # ─────────────────────────────────────────────

    def _encode_words(self, words: List[str]) -> Dict[str, torch.Tensor]:
        n = min(len(words), self.MAX_LEN)
        words_padded = list(words[:n]) + [''] * (self.MAX_LEN - n)

        word_ids = []
        pos_tags = []
        char_ids_list = []
        bpe_ids_list = []
        root_ids = []
        pattern_ids = []
        proclitic_ids = []
        enclitic_ids = []
        diac_ids_list = []
        mask = []

        for i, w in enumerate(words_padded):
            if not w:
                word_ids.append(0)
                pos_tags.append(0)
                char_ids_list.append([0] * self.MAX_CHARS)
                bpe_ids_list.append([0] * 4)
                root_ids.append(0)
                pattern_ids.append(0)
                proclitic_ids.append(0)
                enclitic_ids.append(0)
                diac_ids_list.append([0] * self.MAX_CHARS)
                mask.append(0)
                continue

            # Word ID — same hash as grid builder
            wid = word_to_bucket(w, 8192) % 10000
            word_ids.append(wid)

            # POS — heuristic guess (model doesn't need perfect POS for input)
            pos = self._guess_pos(w)
            pos_tags.append(pos)

            # Char IDs
            chars = [ord(c) % 300 for c in w[:self.MAX_CHARS]]
            chars += [0] * (self.MAX_CHARS - len(chars))
            char_ids_list.append(chars)

            # BPE IDs
            bpe = stable_hash(w, 8000)
            bpe_ids_list.append([bpe] * 4)

            # Root/Pattern
            root = self.root_extractor.extract_root(w)
            root_id = stable_hash(root, 5000)
            pattern_id = stable_hash(w, 200)  # approximate
            root_ids.append(root_id)
            pattern_ids.append(pattern_id)

            # Proclitic/Enclitic (approximated from word_id)
            proclitic_ids.append(wid % 200)
            enclitic_ids.append(wid % 100)

            # Diac IDs
            diac = [c % 20 for c in chars]
            diac_ids_list.append(diac)

            mask.append(1)

        device = self.device
        return {
            'word_ids': torch.tensor([word_ids], dtype=torch.long, device=device),
            'pos_tags': torch.tensor([pos_tags], dtype=torch.long, device=device),
            'char_ids': torch.tensor([char_ids_list], dtype=torch.long, device=device),
            'bpe_ids': torch.tensor([bpe_ids_list], dtype=torch.long, device=device),
            'root_ids': torch.tensor([root_ids], dtype=torch.long, device=device),
            'pattern_ids': torch.tensor([pattern_ids], dtype=torch.long, device=device),
            'proclitic_ids': torch.tensor([proclitic_ids], dtype=torch.long, device=device),
            'enclitic_ids': torch.tensor([enclitic_ids], dtype=torch.long, device=device),
            'diac_ids': torch.tensor([diac_ids_list], dtype=torch.long, device=device),
            'mask': torch.tensor([mask], dtype=torch.long, device=device),
        }

    def _guess_pos(self, word: str) -> int:
        """Heuristic POS for input feature. The model's output is what matters."""
        PREPS = {'إلى', 'في', 'من', 'على', 'عن', 'حتى', 'منذ', 'خلال', 'بين', 'عبر'}
        CONJ = {'و', 'أو', 'ثم', 'ف', 'لكن', 'بل', 'أم'}
        PARTS = {'لا', 'لم', 'لن', 'ما', 'هل', 'أ', 'قد', 'إن', 'أن', 'سوف'}
        PRONS = {'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم',
                 'الذي', 'التي', 'الذين', 'هذا', 'هذه', 'ذلك', 'تلك'}

        if word in PREPS:
            return POS_TAGS['ADP']
        if word in CONJ:
            return POS_TAGS['CCONJ']
        if word in PARTS:
            return POS_TAGS['PART']
        if word in PRONS:
            return POS_TAGS['PRON']
        # Default: NOUN (most common in Arabic)
        return POS_TAGS['NOUN']

    # ─────────────────────────────────────────────
    #     DECODE: model output → ParseResult
    # ─────────────────────────────────────────────

    def _decode_output(self, words: List[str], output: dict,
                       mask: torch.Tensor) -> ParseResult:
        n = len(words)

        pred_heads = output['pred_heads'][0].cpu().tolist()[:n]
        pred_rels = output['pred_rels'][0].cpu().tolist()[:n]
        pred_cases = output['pred_cases'][0].cpu().tolist()[:n]
        arc_scores = output['arc_scores'][0].cpu()

        # Confidence from softmax of arc scores
        arc_probs = F.softmax(arc_scores[:n, :n], dim=-1)
        head_confs = [arc_probs[i, pred_heads[i]].item() for i in range(n)]

        # Decode to UD format
        heads = pred_heads
        relations = [DEP_RELS_REVERSE.get(r, 'dep') for r in pred_rels]
        case_tags = [CASE_CLASSES.get(c, None) for c in pred_cases]

        # Infer POS from relation + case + word form
        pos_tags = []
        for i, w in enumerate(words):
            pos = self._infer_pos_from_relation(
                w, relations[i], case_tags[i])
            pos_tags.append(pos)

        # Build features from model predictions
        features = []
        for i, w in enumerate(words):
            feat = {}
            if w.startswith('ال') and len(w) > 2:
                feat['definite'] = 'yes'
            else:
                feat['definite'] = 'no'
            if w.endswith('ة'):
                feat['gender'] = 'fem'
            else:
                feat['gender'] = 'masc'
            if w.endswith('ون') or w.endswith('ين'):
                feat['number'] = 'plur'
                feat['plural_type'] = 'sound_masc'
            elif w.endswith('ات'):
                feat['number'] = 'plur'
                feat['gender'] = 'fem'
                feat['plural_type'] = 'sound_fem'
            else:
                feat['number'] = 'sing'
            # Construct from deprel
            if relations[i] in ('nmod', 'flat', 'compound'):
                feat['construct'] = 'yes'
            features.append(feat)

        # Case confidences (from case logits if available)
        case_confs = [0.7] * n  # Default; can be refined

        return ParseResult(
            words=words,
            pos_tags=pos_tags,
            heads=heads,
            relations=relations,
            case_tags=case_tags,
            features=features,
            case_confidences=case_confs,
            head_confidences=head_confs,
        )

    # Common Arabic verb prefixes (imperfective)
    _VERB_PREFIXES = {'ي', 'ت', 'ن', 'أ'}  # ya, ta, na, a
    # Common past-tense verbs (3-letter, very frequent)
    _COMMON_VERBS = {
        'كان', 'قال', 'كتب', 'ذهب', 'جعل', 'علم', 'وجد', 'أخذ', 'ترك',
        'بدأ', 'رأى', 'جاء', 'أصبح', 'ظل', 'بات', 'صار', 'ليس', 'مازال',
        'وضع', 'أعلن', 'أكد', 'طلب', 'حقق', 'أضاف', 'أشار', 'عاد',
        'دعا', 'قرر', 'أوضح', 'اعتبر', 'قدم', 'حصل', 'نفى', 'أجرى',
        'وصل', 'خرج', 'دخل', 'فتح', 'أكل', 'شرب', 'درس', 'فهم', 'قرأ',
        'سمع', 'رجع', 'نزل', 'ركب', 'سأل', 'أراد', 'حمل', 'وقع',
    }
    _PARTICLES = {
        'إن', 'أن', 'لا', 'لم', 'لن', 'ما', 'هل', 'قد', 'سوف',
        'إذا', 'لو', 'كي', 'حتى', 'يا',
    }

    def _infer_pos_from_relation(self, word: str, rel: str,
                                 case_tag=None) -> str:
        """Infer POS from syntactic relation + case tag + word form.

        Key insight: case_tag=None from the model strongly signals
        indeclinable words (verbs, particles, prepositions).
        """
        # ── Closed-class words (always correct) ──
        if rel == 'case':
            return 'ADP'
        if rel == 'cc':
            return 'CCONJ'
        if rel == 'mark':
            return 'SCONJ'
        if rel == 'det':
            return 'DET'
        if rel == 'punct':
            return 'PUNCT'
        if rel == 'nummod':
            return 'NUM'

        # ── Particles ──
        if word in self._PARTICLES:
            return 'PART'

        # ── Root relation ──
        if rel == 'root':
            # If case=None, almost certainly a verb
            if case_tag is None:
                return 'VERB'
            # If case is set, might be a nominal sentence predicate
            return 'VERB'  # Still most likely

        # ── Verb detection from morphology ──
        if self._looks_like_verb(word):
            return 'VERB'

        # ── Subject/object ──
        if rel in ('nsubj', 'nsubj:pass', 'obj', 'iobj', 'obl', 'nmod'):
            return 'NOUN'
        if rel == 'amod':
            return 'ADJ'
        if rel == 'advmod':
            return 'ADV'
        if rel in ('conj', 'appos'):
            return 'NOUN'
        if rel in ('xcomp', 'ccomp', 'advcl', 'acl'):
            return 'VERB'

        # Fallback
        return 'NOUN'

    def _looks_like_verb(self, word: str) -> bool:
        """Heuristic: does this bare word look like a verb?"""
        if not word:
            return False
        # Known common verbs
        if word in self._COMMON_VERBS:
            return True
        # Imperfective prefix (يكتب، تكتب، نكتب، أكتب)
        if len(word) >= 3 and word[0] in self._VERB_PREFIXES and not word.startswith('ال'):
            return True
        # Past tense pattern: 3-letter no-prefix no-ال
        if len(word) == 3 and not word.startswith('ال'):
            return True  # Many 3-letter bare words are past verbs
        return False
