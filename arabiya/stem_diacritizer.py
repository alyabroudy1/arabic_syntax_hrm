"""
Stem diacritizer — lexicon-based internal vowel lookup.

BUG FIX #2: build_from_conllu reads Vform from MISC column,
not FORM (which is bare in PADT). Falls back to FORM only
if Vform is absent.
"""

import json
import os
from collections import defaultdict
from typing import Optional, Dict, List

from arabiya.core import strip_diacritics


class StemDiacritizer:
    def __init__(self):
        self._lexicon: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._fallback: Dict[str, str] = {}
        self._total_entries = 0

    @property
    def size(self) -> int:
        return self._total_entries

    def load_lexicon(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._lexicon = defaultdict(dict)
        for bare_word, pos_map in data.items():
            if isinstance(pos_map, dict):
                self._lexicon[bare_word] = pos_map
            elif isinstance(pos_map, str):
                self._lexicon[bare_word]['ANY'] = pos_map
        self._build_fallback()
        self._total_entries = sum(len(v) for v in self._lexicon.values())

    def _build_fallback(self):
        self._fallback = {}
        for bare, pos_map in self._lexicon.items():
            if len(pos_map) == 1:
                self._fallback[bare] = list(pos_map.values())[0]
            else:
                for pref in ['NOUN', 'VERB', 'ADJ', 'ANY']:
                    if pref in pos_map:
                        self._fallback[bare] = pos_map[pref]
                        break
                else:
                    self._fallback[bare] = list(pos_map.values())[0]

    def add_entry(self, bare_word: str, pos: str, diacritized: str):
        self._lexicon[bare_word][pos] = diacritized
        self._total_entries += 1
        if bare_word not in self._fallback:
            self._fallback[bare_word] = diacritized

    def lookup(self, bare_word: str, pos: str = '') -> Optional[str]:
        if bare_word in self._lexicon:
            pos_map = self._lexicon[bare_word]
            if pos in pos_map:
                return pos_map[pos]
            if 'ANY' in pos_map:
                return pos_map['ANY']
            if bare_word in self._fallback:
                return self._fallback[bare_word]
        return None

    def save_lexicon(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dict(self._lexicon), f, ensure_ascii=False, indent=2)

    def build_from_conllu(self, conllu_path: str):
        """
        Build lexicon from CoNLL-U file.

        BUG FIX #2: Reads Vform from MISC column (field[9]),
        NOT from FORM (field[1]) which is bare in PADT.
        """
        count = 0
        with open(conllu_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                fields = line.split('\t')
                if len(fields) < 10:
                    continue
                if '-' in fields[0] or '.' in fields[0]:
                    continue

                pos = fields[3]       # UPOS
                misc = fields[9]      # MISC column

                # ── BUG FIX: Extract Vform from MISC ──
                vform = ''
                for m in misc.split('|'):
                    if m.startswith('Vform='):
                        vform = m.split('=', 1)[1]
                        break

                # Fall back to FORM only if Vform absent
                if not vform or vform == '-':
                    form = fields[1]
                    bare = strip_diacritics(form)
                    if bare != form and bare:
                        vform = form
                    else:
                        continue  # No diacritized form available

                bare = strip_diacritics(vform)
                if bare and bare != vform:
                    self.add_entry(bare, pos, vform)
                    count += 1

        self._build_fallback()
        self._total_entries = sum(len(v) for v in self._lexicon.values())
        return count

    def build_from_diacritized_file(self, text_path: str):
        count = 0
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.strip().split():
                    bare = strip_diacritics(word)
                    if bare and bare != word:
                        self.add_entry(bare, 'ANY', word)
                        count += 1
        self._build_fallback()
        self._total_entries = sum(len(v) for v in self._lexicon.values())
        return count

    def build_from_inline_data(self, entries: Dict[str, Dict[str, str]]):
        for bare, pos_map in entries.items():
            for pos, diac in pos_map.items():
                self.add_entry(bare, pos, diac)
        self._build_fallback()
        self._total_entries = sum(len(v) for v in self._lexicon.values())

    def get_stats(self) -> Dict:
        total_forms = sum(len(v) for v in self._lexicon.values())
        ambiguous = sum(1 for v in self._lexicon.values() if len(v) > 1)
        return {
            'unique_words': len(self._lexicon),
            'total_forms': total_forms,
            'ambiguous_words': ambiguous,
        }
