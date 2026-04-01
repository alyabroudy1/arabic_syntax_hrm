#!/usr/bin/env python3
"""
Arabic Prosody Annotator for Kokoro TTS
=========================================

Maps tajweed rules + sentence emotion → Kokoro-compatible prosody tags.

Kokoro controls prosody through:
1. Voice style tokens (speed, pitch, energy)
2. Punctuation-based pausing (commas, periods, ellipsis)
3. Phoneme-level emphasis markers

This module produces a ProsodyPlan that the Android Kokoro pipeline consumes
as a JSON manifest alongside the diacritized text.

Output format (consumed by Android KokoroTTS):
{
  "text": "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
  "emotion": "reverent",
  "style": {"speed": 0.85, "pitch": 0.95, "energy": 0.7},
  "segments": [
    {
      "word": "بِسْمِ",
      "speed_factor": 1.0,
      "pause_after_ms": 200,
      "emphasis": "none",
      "prosody_tags": ["laam_shamsiya"]
    },
    ...
  ]
}
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from enum import Enum

# Import tajweed (direct import to avoid loading full parser via __init__.py)
import sys
import importlib.util
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent

_tajweed_spec = importlib.util.spec_from_file_location(
    "tajweed", PROJECT_ROOT / "models" / "v2" / "tajweed.py"
)
_tajweed_mod = importlib.util.module_from_spec(_tajweed_spec)
_tajweed_spec.loader.exec_module(_tajweed_mod)
TajweedEngine = _tajweed_mod.TajweedEngine
TajweedRule = _tajweed_mod.TajweedRule
AnalyzedWord = _tajweed_mod.AnalyzedWord
strip_diacritics = _tajweed_mod.strip_diacritics


# ═══════════════════════════════════════════════════
# Emotion / Sentiment Detection
# ═══════════════════════════════════════════════════

class Emotion(Enum):
    """Sentence-level emotional tone for TTS voice control."""
    NEUTRAL = "neutral"
    REVERENT = "reverent"         # Quran, dua, worship
    ASSERTIVE = "assertive"       # news, formal speech
    CONVERSATIONAL = "conversational"  # chat, dialogue
    JOYFUL = "joyful"             # good news, celebration
    SAD = "sad"                   # lament, tragic news
    ANGRY = "angry"               # warning, admonition
    QUESTIONING = "questioning"    # questions


# Emotion detection keywords (Arabic)
EMOTION_KEYWORDS = {
    Emotion.REVERENT: {
        'بسم', 'الله', 'الرحمن', 'الرحيم', 'سبحان', 'الحمد', 'لله',
        'إن', 'شاء', 'ربنا', 'اللهم', 'صلى', 'سلم', 'عليه', 'وسلم',
        'تبارك', 'تعالى', 'جل', 'جلاله', 'قرآن', 'آية', 'سورة',
        'رب', 'العالمين', 'يوم', 'الدين', 'نعبد', 'نستعين',
        'اهدنا', 'الصراط', 'المستقيم', 'أعوذ', 'بالله',
    },
    Emotion.JOYFUL: {
        'مبروك', 'تهانينا', 'فرح', 'سعيد', 'سعادة', 'بشرى',
        'حمدا', 'شكرا', 'نجاح', 'فوز', 'احتفال', 'عيد',
    },
    Emotion.SAD: {
        'حزن', 'أسف', 'مصيبة', 'وفاة', 'رحمه', 'إنا', 'لله', 'راجعون',
        'فقد', 'مأساة', 'دموع', 'بكاء', 'ألم',
    },
    Emotion.ANGRY: {
        'ويل', 'عذاب', 'لعنة', 'حرام', 'ظلم', 'فساد', 'كفر',
        'نار', 'جهنم', 'عقاب', 'انتقام',
    },
    Emotion.QUESTIONING: set(),  # detected by punctuation
}

# Kokoro voice style mapping per emotion
EMOTION_STYLES = {
    Emotion.NEUTRAL: {
        "speed": 1.0, "pitch": 1.0, "energy": 0.8,
        "pause_scale": 1.0,
        "description": "Standard neutral delivery",
    },
    Emotion.REVERENT: {
        "speed": 0.82, "pitch": 0.92, "energy": 0.65,
        "pause_scale": 1.5,
        "description": "Slow, measured, reverent — suitable for Quran/dua",
    },
    Emotion.ASSERTIVE: {
        "speed": 0.95, "pitch": 1.05, "energy": 0.9,
        "pause_scale": 1.0,
        "description": "Clear, confident, authoritative",
    },
    Emotion.CONVERSATIONAL: {
        "speed": 1.05, "pitch": 1.0, "energy": 0.85,
        "pause_scale": 0.8,
        "description": "Natural, relaxed conversational tone",
    },
    Emotion.JOYFUL: {
        "speed": 1.1, "pitch": 1.1, "energy": 0.95,
        "pause_scale": 0.9,
        "description": "Upbeat, warm, celebratory",
    },
    Emotion.SAD: {
        "speed": 0.85, "pitch": 0.85, "energy": 0.5,
        "pause_scale": 1.4,
        "description": "Slow, soft, somber",
    },
    Emotion.ANGRY: {
        "speed": 1.0, "pitch": 1.15, "energy": 1.0,
        "pause_scale": 0.7,
        "description": "Intense, forceful, commanding",
    },
    Emotion.QUESTIONING: {
        "speed": 0.95, "pitch": 1.1, "energy": 0.8,
        "pause_scale": 1.1,
        "description": "Rising intonation, curious",
    },
}


# ═══════════════════════════════════════════════════
# Tajweed → Prosody Mapping
# ═══════════════════════════════════════════════════

# How each tajweed rule affects prosody
TAJWEED_PROSODY = {
    # ── Madd (elongation) → slow down ──
    TajweedRule.MADD_TABII:     {"speed_factor": 0.85, "emphasis": "slight"},
    TajweedRule.MADD_MUTTASIL:  {"speed_factor": 0.70, "emphasis": "moderate"},
    TajweedRule.MADD_MUNFASIL:  {"speed_factor": 0.70, "emphasis": "moderate"},
    TajweedRule.MADD_LAAZIM:    {"speed_factor": 0.55, "emphasis": "strong"},
    TajweedRule.MADD_AARID:     {"speed_factor": 0.80, "emphasis": "slight"},
    TajweedRule.MADD_LEEN:      {"speed_factor": 0.85, "emphasis": "slight"},
    TajweedRule.MADD_BADAL:     {"speed_factor": 0.85, "emphasis": "slight"},
    
    # ── Ghunna/nasalization → slight pause + nasal quality ──
    TajweedRule.IDGHAAM_GHUNNA:       {"speed_factor": 0.90, "pause_after_ms": 80},
    TajweedRule.IKHFAA_HAQIQI:        {"speed_factor": 0.90, "pause_after_ms": 60},
    TajweedRule.IKHFAA_SHAFAWI:       {"speed_factor": 0.90, "pause_after_ms": 60},
    TajweedRule.IDGHAAM_MUTAMATHILAIN: {"speed_factor": 0.90, "pause_after_ms": 50},
    TajweedRule.GHUNNA:               {"speed_factor": 0.90, "pause_after_ms": 80},
    
    # ── Idghaam without ghunna → smooth merge ──
    TajweedRule.IDGHAAM_NO_GHUNNA: {"speed_factor": 0.95},
    
    # ── Iqlaab → brief pause for noon→meem swap ──
    TajweedRule.IQLAAB: {"speed_factor": 0.90, "pause_after_ms": 60},
    
    # ── Qalqalah → emphasis/bounce ──
    TajweedRule.QALQALA_SUGHRA: {"emphasis": "slight", "speed_factor": 0.95},
    TajweedRule.QALQALA_KUBRA:  {"emphasis": "strong", "speed_factor": 0.90},
    
    # ── Tafkheem → heavier, deeper ──
    TajweedRule.TAFKHEEM:     {"emphasis": "moderate"},
    TajweedRule.RAA_TAFKHEEM: {"emphasis": "moderate"},
    TajweedRule.RAA_TARQEEQ:  {"emphasis": "slight"},
    
    # ── Ithaar → clear pronunciation ──
    TajweedRule.ITHAAR_HALQI:  {},
    TajweedRule.ITHAAR_SHAFAWI: {},
    
    # ── Laam rules → no prosody effect, just phonetic ──
    TajweedRule.LAAM_SHAMSIYA: {},
    TajweedRule.LAAM_QAMARIYA: {},
}


# ═══════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════

@dataclass
class WordProsody:
    """Prosody annotation for a single word."""
    word: str                        # diacritized text
    speed_factor: float = 1.0        # relative speed (0.5 = half speed, 2.0 = double)
    pause_after_ms: int = 0          # pause after word in milliseconds
    emphasis: str = "none"           # none | slight | moderate | strong
    prosody_tags: List[str] = field(default_factory=list)  # tajweed rule names
    
    def to_dict(self):
        return {
            "word": self.word,
            "speed_factor": round(self.speed_factor, 3),
            "pause_after_ms": self.pause_after_ms,
            "emphasis": self.emphasis,
            "prosody_tags": self.prosody_tags,
        }


@dataclass
class ProsodyPlan:
    """Complete prosody plan for TTS synthesis."""
    text: str                        # full diacritized text
    emotion: str = "neutral"         # detected emotion
    style: Dict = field(default_factory=dict)  # Kokoro style params
    segments: List[WordProsody] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "text": self.text,
            "emotion": self.emotion,
            "style": self.style,
            "segments": [s.to_dict() for s in self.segments],
        }
    
    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    def to_kokoro_text(self) -> str:
        """Convert to Kokoro-compatible text with punctuation prosody markers.
        
        Kokoro uses punctuation for prosody:
        - Commas (,) → short pause (~200ms)
        - Periods (.) → medium pause (~400ms)
        - Ellipsis (...) → long pause (~600ms)
        - Exclamation (!) → emphasis
        - Semicolons (;) → moderate pause
        """
        parts = []
        for seg in self.segments:
            word = seg.word
            
            # Add emphasis marker
            if seg.emphasis == "strong":
                word = word + "!"  # Kokoro reads ! as emphasis
            
            parts.append(word)
            
            # Add pause marker after word
            if seg.pause_after_ms >= 500:
                parts.append("...")
            elif seg.pause_after_ms >= 300:
                parts.append(".")
            elif seg.pause_after_ms >= 150:
                parts.append(",")
            elif seg.pause_after_ms >= 80:
                parts.append(";")
        
        return ' '.join(parts)


# ═══════════════════════════════════════════════════
# Prosody Annotator
# ═══════════════════════════════════════════════════

class ArabicProsodyAnnotator:
    """Maps tajweed analysis + emotion → Kokoro prosody plan.
    
    Usage:
        annotator = ArabicProsodyAnnotator()
        plan = annotator.annotate(diacritized_text)
        kokoro_text = plan.to_kokoro_text()
        android_json = plan.to_json()
    """
    
    def __init__(self):
        self.tajweed = TajweedEngine()
    
    def detect_emotion(self, text: str) -> Emotion:
        """Detect sentence-level emotion from text content."""
        # Strip diacritics for keyword matching
        # Strip diacritics for keyword matching
        clean = strip_diacritics(text)
        words = set(clean.split())
        
        # Question detection
        if '؟' in text or '?' in text:
            return Emotion.QUESTIONING
        
        # Keyword-based detection
        scores = {}
        for emotion, keywords in EMOTION_KEYWORDS.items():
            overlap = len(words & keywords)
            if overlap > 0:
                scores[emotion] = overlap
        
        if scores:
            return max(scores, key=scores.get)
        
        return Emotion.NEUTRAL
    
    def annotate(self, diacritized_text: str, 
                 emotion_override: Optional[Emotion] = None) -> ProsodyPlan:
        """Generate full prosody plan from diacritized Arabic text.
        
        Args:
            diacritized_text: Text with full tashkeel
            emotion_override: Force a specific emotion (optional)
        """
        # Step 1: Detect emotion
        emotion = emotion_override or self.detect_emotion(diacritized_text)
        style = EMOTION_STYLES[emotion].copy()
        
        # Step 2: Tajweed analysis
        analyzed_words = self.tajweed.analyze(diacritized_text)
        
        # Step 3: Build per-word prosody
        pause_scale = style.pop("pause_scale", 1.0)
        description = style.pop("description", "")
        base_speed = style.get("speed", 1.0)
        
        segments = []
        for wi, aw in enumerate(analyzed_words):
            word_prosody = WordProsody(word=aw.text)
            
            # Collect all tajweed effects on this word
            combined_speed = 1.0
            max_pause = 0
            emphasis_level = "none"
            emphasis_order = ["none", "slight", "moderate", "strong"]
            tags = []
            
            for ann in aw.annotations:
                rule_effect = TAJWEED_PROSODY.get(ann.rule, {})
                tags.append(self.tajweed.RULE_DESC_AR.get(ann.rule, ann.rule.name))
                
                # Speed: multiply factors
                if "speed_factor" in rule_effect:
                    combined_speed *= rule_effect["speed_factor"]
                
                # Pause: take maximum
                if "pause_after_ms" in rule_effect:
                    max_pause = max(max_pause, rule_effect["pause_after_ms"])
                
                # Emphasis: take strongest
                if "emphasis" in rule_effect:
                    new_level = rule_effect["emphasis"]
                    if emphasis_order.index(new_level) > emphasis_order.index(emphasis_level):
                        emphasis_level = new_level
            
            # Apply emotion-level speed modifier
            word_prosody.speed_factor = combined_speed * base_speed
            word_prosody.pause_after_ms = int(max_pause * pause_scale)
            word_prosody.emphasis = emphasis_level
            word_prosody.prosody_tags = tags
            
            segments.append(word_prosody)
        
        # Add natural pauses at sentence boundaries
        if segments:
            # Longer pause at end
            segments[-1].pause_after_ms = max(segments[-1].pause_after_ms, 
                                               int(400 * pause_scale))
        
        # Add inter-word pauses for reverent/slow emotions
        if emotion in (Emotion.REVERENT, Emotion.SAD):
            for seg in segments:
                if seg.pause_after_ms == 0:
                    seg.pause_after_ms = int(100 * pause_scale)
        
        return ProsodyPlan(
            text=diacritized_text,
            emotion=emotion.value,
            style={k: v for k, v in style.items() if k != "description"},
            segments=segments,
        )


# ═══════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════

def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("="*60)
    print("Arabic Prosody Annotator -- Demo")
    print("Tajweed + Emotion -> Kokoro Prosody Tags")
    print("="*60)
    
    annotator = ArabicProsodyAnnotator()
    
    test_cases = [
        # Quran -> REVERENT
        ("\u0628\u0650\u0633\u0652\u0645\u0650 \u0671\u0644\u0644\u0651\u064e\u0647\u0650 \u0671\u0644\u0631\u0651\u064e\u062d\u0652\u0645\u064e\u0670\u0646\u0650 \u0671\u0644\u0631\u0651\u064e\u062d\u0650\u064a\u0645\u0650", None),
        ("\u0671\u0644\u0652\u062d\u064e\u0645\u0652\u062f\u064f \u0644\u0650\u0644\u0651\u064e\u0647\u0650 \u0631\u064e\u0628\u0651\u0650 \u0671\u0644\u0652\u0639\u064e\u0670\u0644\u064e\u0645\u0650\u064a\u0646\u064e", None),
        ("\u0642\u064f\u0644\u0652 \u0647\u064f\u0648\u064e \u0671\u0644\u0644\u0651\u064e\u0647\u064f \u0623\u064e\u062d\u064e\u062f\u064c", None),
        
        # MSA -> NEUTRAL
        ("\u0630\u064e\u0647\u064e\u0628\u064e \u0627\u0644\u0637\u0651\u064e\u0627\u0644\u0650\u0628\u064f \u0625\u0650\u0644\u064e\u0649 \u0627\u0644\u0645\u064e\u062f\u0631\u064e\u0633\u064e\u0629\u0650", None),
        
        # Question -> QUESTIONING
        ("\u0647\u064e\u0644\u0652 \u0630\u064e\u0647\u064e\u0628\u0652\u062a\u064e \u0625\u0650\u0644\u064e\u0649 \u0627\u0644\u0645\u064e\u062f\u0631\u064e\u0633\u064e\u0629\u0650\u061f", None),
        
        # Override emotion
        ("\u0648\u064e\u0644\u064e\u0645\u0652 \u064a\u064e\u0644\u0650\u062f\u0652 \u0648\u064e\u0644\u064e\u0645\u0652 \u064a\u064f\u0648\u0644\u064e\u062f\u0652", Emotion.REVERENT),
    ]
    
    for text, override in test_cases:
        plan = annotator.annotate(text, emotion_override=override)
        
        print(f"\n{'_'*50}")
        print(f"[TEXT]    {text}")
        print(f"[EMOTION] {plan.emotion}")
        print(f"[STYLE]   speed={plan.style.get('speed',1.0):.2f}, "
              f"pitch={plan.style.get('pitch',1.0):.2f}, "
              f"energy={plan.style.get('energy',0.8):.2f}")
        
        print(f"[SEGMENTS]")
        for seg in plan.segments:
            tags_str = ', '.join(seg.prosody_tags) if seg.prosody_tags else '-'
            print(f"   {seg.word:20s} speed={seg.speed_factor:.2f} "
                  f"pause={seg.pause_after_ms:3d}ms "
                  f"emph={seg.emphasis:8s} [{tags_str}]")
        
        print(f"\n[KOKORO]  {plan.to_kokoro_text()}")
        
        # Show JSON (what Android consumes)
        j = json.loads(plan.to_json())
        print(f"[JSON]    emotion={j['emotion']}, segments={len(j['segments'])}")
    
    # Save a sample JSON for Android integration
    sample = annotator.annotate("\u0628\u0650\u0633\u0652\u0645\u0650 \u0671\u0644\u0644\u0651\u064e\u0647\u0650 \u0671\u0644\u0631\u0651\u064e\u062d\u0652\u0645\u064e\u0670\u0646\u0650 \u0671\u0644\u0631\u0651\u064e\u062d\u0650\u064a\u0645\u0650")
    output_path = PROJECT_ROOT / "output" / "prosody_sample.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample.to_json())
    print(f"\n{'_'*50}")
    print(f"[SAVED] {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

