#!/usr/bin/env python3
"""
Script 15: Full Arabic TTS Pipeline Demo
==========================================

Raw Arabic → Diacritize → Tajweed → TTS (Audio)

Uses:
- Standalone diacritizer (94.82% CharAcc)
- Tajweed engine (deterministic rules)
- Edge-TTS (Microsoft Arabic voices)
"""

import sys
import asyncio
import torch
import json
import tempfile
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.v2.diacritizer import ArabicDiacritizer
from models.v2.tajweed import TajweedEngine, TajweedRule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIAC_MODEL_PATH = PROJECT_ROOT / "models" / "v2_diacritizer" / "best_diacritizer.pt"

# ─── Diacritic Constants ───
ALL_DIACRITICS = set('\u064E\u064F\u0650\u0652\u0651\u064B\u064C\u064D\u0670\u0653\u0654\u0655')
ARABIC_CHARS = 'ءآأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهوي' \
               'ىةﻻﻷﻹﻵ' \
               'ٱ' \
               '٠١٢٣٤٥٦٧٨٩' \
               '0123456789' \
               ' .,:;!?()-/"\''
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(ARABIC_CHARS)}

DIAC_SYMBOLS = {
    0: '',       # NONE
    1: '\u064E',  # FATHA
    2: '\u064F',  # DAMMA
    3: '\u0650',  # KASRA
    4: '\u0652',  # SUKUN
    5: '\u0651',  # SHADDA
    6: '\u0651\u064E',  # SHADDA_FATHA
    7: '\u0651\u064F',  # SHADDA_DAMMA
    8: '\u0651\u0650',  # SHADDA_KASRA
    9: '\u064B',  # TANWEEN_FATH
    10: '\u064C', # TANWEEN_DAMM
    11: '\u064D', # TANWEEN_KASR
    12: '\u0651\u064B', # SHADDA_TANWEEN_FATH
    13: '\u0651\u064C', # SHADDA_TANWEEN_DAMM
    14: '\u0651\u064D', # SHADDA_TANWEEN_KASR
}

# Arabic TTS voices
ARABIC_VOICES = {
    'male': 'ar-SA-HamedNeural',
    'female': 'ar-SA-ZariyahNeural',
    'egyptian_male': 'ar-EG-ShakirNeural',
    'egyptian_female': 'ar-EG-SalmaNeural',
}


def strip_diacritics(text):
    return ''.join(c for c in text if c not in ALL_DIACRITICS)


def char_to_id(c):
    return CHAR_TO_ID.get(c, len(CHAR_TO_ID) + 1)


class ArabicTTSPipeline:
    """Full Arabic Text-to-Speech pipeline.
    
    Raw Arabic → Diacritize → Tajweed Analyze → TTS Audio
    """
    
    def __init__(self, device=DEVICE):
        self.device = device
        self.tajweed = TajweedEngine()
        self.diac_model = None
        self._load_diacritizer()
    
    def _load_diacritizer(self):
        """Load trained diacritizer model."""
        if not DIAC_MODEL_PATH.exists():
            print(f"⚠️  Diacritizer not found: {DIAC_MODEL_PATH}")
            return
        
        self.diac_model = ArabicDiacritizer(
            char_vocab=256, char_embed_dim=64, char_hidden=128,
            word_dim=256, n_heads=4, n_transformer_layers=3,
            n_diac_classes=15, max_chars=16, max_words=32, dropout=0.0,
        ).to(self.device)
        
        ckpt = torch.load(DIAC_MODEL_PATH, map_location=self.device, weights_only=False)
        self.diac_model.load_state_dict(ckpt['model_state_dict'])
        self.diac_model.eval()
        
        char_acc = ckpt.get('char_acc', '?')
        print(f"✅ Diacritizer loaded (CharAcc={char_acc:.2f}%)")
    
    def diacritize(self, text: str) -> str:
        """Add diacritics to undiacritized Arabic text."""
        if self.diac_model is None:
            return text
        
        # Strip existing diacritics first
        clean = strip_diacritics(text)
        words = clean.split()
        
        MAX_WORDS, MAX_CHARS = 32, 16
        char_ids = torch.zeros(1, MAX_WORDS, MAX_CHARS, dtype=torch.long, device=self.device)
        word_mask = torch.zeros(1, MAX_WORDS, dtype=torch.long, device=self.device)
        
        for wi, w in enumerate(words[:MAX_WORDS]):
            word_mask[0, wi] = 1
            for ci, c in enumerate(w[:MAX_CHARS]):
                char_ids[0, wi, ci] = char_to_id(c)
        
        with torch.no_grad():
            out = self.diac_model(char_ids, word_mask)
            pred_diacs = out['pred_diacs'][0]  # (W, C)
        
        # Reconstruct diacritized text
        result_words = []
        for wi, w in enumerate(words[:MAX_WORDS]):
            diac_word = ""
            for ci, c in enumerate(w[:MAX_CHARS]):
                diac_word += c
                label = pred_diacs[wi, ci].item()
                diac_word += DIAC_SYMBOLS.get(label, '')
            result_words.append(diac_word)
        
        # Append remaining words (beyond MAX_WORDS) undiacritized
        result_words.extend(words[MAX_WORDS:])
        
        return ' '.join(result_words)
    
    def analyze_tajweed(self, diacritized_text: str):
        """Apply tajweed rules to diacritized text."""
        return self.tajweed.analyze(diacritized_text)
    
    def format_tajweed_summary(self, analyzed_words) -> str:
        """Format tajweed analysis as readable text."""
        summary = self.tajweed.get_summary(analyzed_words)
        if not summary:
            return "  (no special tajweed rules detected)"
        lines = []
        for rule, count in sorted(summary.items(), key=lambda x: -x[1]):
            lines.append(f"  • {rule}: {count}")
        return '\n'.join(lines)
    
    async def synthesize_async(self, text: str, voice: str = 'male', 
                                output_path: str = None) -> str:
        """Generate speech from Arabic text using Edge-TTS."""
        import edge_tts
        
        voice_id = ARABIC_VOICES.get(voice, voice)
        
        if output_path is None:
            output_path = str(PROJECT_ROOT / "output" / "tts_output.mp3")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(output_path)
        
        return output_path
    
    def synthesize(self, text: str, voice: str = 'male', output_path: str = None) -> str:
        """Synchronous wrapper for TTS."""
        return asyncio.run(self.synthesize_async(text, voice, output_path))
    
    def full_pipeline(self, raw_text: str, voice: str = 'male', 
                       output_dir: str = None) -> dict:
        """
        Full pipeline: raw Arabic → diacritize → tajweed → TTS
        
        Returns dict with all intermediate outputs + audio path.
        """
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'─'*50}")
        print(f"📝 Input: {raw_text}")
        
        # Step 1: Diacritize
        diacritized = self.diacritize(raw_text)
        print(f"🔤 Diacritized: {diacritized}")
        
        # Step 2: Tajweed analysis
        analyzed = self.analyze_tajweed(diacritized)
        tajweed_summary = self.format_tajweed_summary(analyzed)
        print(f"📖 Tajweed rules:\n{tajweed_summary}")
        
        # Step 3: TTS
        audio_path = os.path.join(output_dir, "tts_output.mp3")
        self.synthesize(diacritized, voice, audio_path)
        
        file_size = os.path.getsize(audio_path) / 1024
        print(f"🔊 Audio: {audio_path} ({file_size:.1f} KB)")
        print(f"{'─'*50}")
        
        return {
            'input': raw_text,
            'diacritized': diacritized,
            'tajweed_analysis': analyzed,
            'tajweed_summary': tajweed_summary,
            'audio_path': audio_path,
        }


def main():
    print("="*60)
    print("Arabic TTS Pipeline — Full Demo")
    print("Raw Text → Diacritize → Tajweed → Speech")
    print("="*60)
    
    pipeline = ArabicTTSPipeline()
    
    test_texts = [
        # Quran - Al-Fatiha
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        # Quran - Al-Ikhlas
        "قل هو الله احد",
        "الله الصمد",
        # MSA
        "ذهب الطالب الى المدرسة",
        "كتب المعلم الدرس على السبورة",
        # With already diacritized input
        "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
    ]
    
    output_dir = str(PROJECT_ROOT / "output")
    
    for i, text in enumerate(test_texts):
        audio_path = os.path.join(output_dir, f"arabic_tts_{i+1}.mp3")
        # Remove old file if exists
        if os.path.exists(audio_path):
            os.remove(audio_path)
        result = pipeline.full_pipeline(
            text, 
            voice='male', 
            output_dir=output_dir
        )
        # Move to unique filename
        src = result['audio_path']
        if os.path.exists(src) and src != audio_path:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            os.rename(src, audio_path)
            print(f"  → Saved as: {audio_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ Generated {len(test_texts)} audio files in {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
