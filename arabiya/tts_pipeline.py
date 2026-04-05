"""
Arabiya TTS Pipeline — Unified Diacritization → Rhetoric → Prosody → Kokoro
=============================================================================

Connects all three engines into a single pipeline:

    Raw Arabic Text
         ↓
    [1] Arabiya Engine (HRM parser + stem lexicon + Ibn Malik case engine)
         ↓  diacritized text + parse tree
    [2] Rhetoric Analyzer (Ibn Taymiyya emphasis + Ibn al-Qayyim particles)
         ↓  rhetorical devices + emphasis level + emotion
    [3] Prosody Annotator (tajweed rules + rhetoric → Kokoro markers)
         ↓  speed/pitch/pause/emphasis per word
    [4] Output: ProsodyPlan (JSON for Android Kokoro TTS)

Sources:
    • Ibn Malik (ألفية) — Case endings at 99.7%
    • Ibn Taymiyya (مجموع الفتاوى) — 5-level emphasis gradation
    • Ibn al-Qayyim (بدائع الفوائد) — Particle disambiguation
    • Al-Qazwini (الإيضاح) — Rhetorical device catalog

Usage:
    pipeline = ArabiyaTTSPipeline.load("models/v2")
    result = pipeline.process("إنّ العلمَ نورٌ")

    # Get Kokoro-ready text
    print(result.kokoro_text)

    # Get full JSON for Android
    print(result.to_json())
"""

import json
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path

# ═══════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent

# Core utilities
from arabiya.core import (
    WordInfo, SentenceInfo, ArabiyaResult,
    strip_diacritics, case_to_arabic,
)

# Rhetoric (Ibn Taymiyya + Ibn al-Qayyim + Al-Qazwini)
from arabiya.rhetoric import RhetoricAnalyzer, RhetoricResult, DeviceType

# Prosody (Tajweed → Kokoro)
sys.path.insert(0, str(PROJECT_ROOT))
from models.v2.prosody import (
    ArabicProsodyAnnotator, ProsodyPlan, WordProsody, Emotion,
    EMOTION_STYLES,
)


# ═══════════════════════════════════════════════════
# Rhetoric → Prosody Mapping
# ═══════════════════════════════════════════════════

# Map rhetoric emotion → prosody emotion
RHETORIC_TO_EMOTION = {
    "neutral": Emotion.NEUTRAL,
    "emphatic": Emotion.ASSERTIVE,
    "questioning": Emotion.QUESTIONING,
    "commanding": Emotion.ASSERTIVE,
    "wishing": Emotion.SAD,           # تمني has longing quality
    "warning": Emotion.ANGRY,
    "exclamatory": Emotion.JOYFUL,
    "reverent": Emotion.REVERENT,
    "oath": Emotion.ASSERTIVE,
    "conditional": Emotion.NEUTRAL,
}

# Map device type → prosodic impact on the TRIGGER word
DEVICE_PROSODY_MAP = {
    DeviceType.ISTIFHAM: {
        "pitch_contour": "rising",
        "speed_factor": 0.95,
        "emphasis": "moderate",
    },
    DeviceType.AMR: {
        "pitch_contour": "falling",
        "speed_factor": 0.90,
        "emphasis": "strong",
    },
    DeviceType.NAHI: {
        "pitch_contour": "falling",
        "speed_factor": 0.88,
        "emphasis": "strong",
        "pause_after_ms": 150,
    },
    DeviceType.NIDAA: {
        "pitch_contour": "peak",
        "speed_factor": 0.85,
        "emphasis": "strong",
        "pause_after_ms": 200,
    },
    DeviceType.TAMANNI: {
        "pitch_contour": "rising",
        "speed_factor": 0.80,
        "emphasis": "moderate",
    },
    DeviceType.TAAJJUB: {
        "pitch_contour": "peak",
        "speed_factor": 0.85,
        "emphasis": "strong",
    },
    DeviceType.TAWKEED: {
        "pitch_contour": "flat",
        "speed_factor": 0.92,
        "emphasis": "moderate",
    },
    DeviceType.NAFI: {
        "pitch_contour": "flat",
        "speed_factor": 0.95,
        "emphasis": "slight",
    },
    DeviceType.QASM: {
        "pitch_contour": "falling",
        "speed_factor": 0.85,
        "emphasis": "strong",
        "pause_after_ms": 250,
    },
    DeviceType.SHART: {
        "pitch_contour": "rising",
        "speed_factor": 0.90,
        "emphasis": "moderate",
        "pause_after_ms": 150,
    },
    DeviceType.HASR: {
        "pitch_contour": "peak",
        "speed_factor": 0.88,
        "emphasis": "strong",
    },
    DeviceType.TASHBEEH: {
        "pitch_contour": "flat",
        "speed_factor": 0.90,
        "emphasis": "slight",
    },
}


# ═══════════════════════════════════════════════════
# TTS Result
# ═══════════════════════════════════════════════════

@dataclass
class TTSResult:
    """Complete TTS-ready result from the unified pipeline."""

    # Input
    input_text: str

    # Stage 1: Diacritization
    diacritized: str = ""
    arabiya_result: Optional[ArabiyaResult] = None

    # Stage 2: Rhetoric
    rhetoric: Optional[RhetoricResult] = None

    # Stage 3: Prosody (final output)
    prosody_plan: Optional[ProsodyPlan] = None

    # Metadata
    pipeline_stages: List[str] = field(default_factory=list)

    @property
    def kokoro_text(self) -> str:
        """Get Kokoro-compatible text with prosody markers."""
        if self.prosody_plan:
            return self.prosody_plan.to_kokoro_text()
        return self.diacritized

    @property
    def emotion(self) -> str:
        """Detected emotion."""
        if self.prosody_plan:
            return self.prosody_plan.emotion
        return "neutral"

    @property
    def style(self) -> Dict:
        """Kokoro voice style parameters."""
        if self.prosody_plan:
            return self.prosody_plan.style
        return EMOTION_STYLES[Emotion.NEUTRAL].copy()

    def to_dict(self) -> Dict:
        """Full output as dictionary (for Android JSON)."""
        result = {
            "input": self.input_text,
            "diacritized": self.diacritized,
            "emotion": self.emotion,
            "style": self.style,
            "kokoro_text": self.kokoro_text,
            "pipeline_stages": self.pipeline_stages,
        }
        if self.prosody_plan:
            result["segments"] = [s.to_dict() for s in self.prosody_plan.segments]
        if self.rhetoric:
            result["rhetoric"] = self.rhetoric.to_prosody_dict()
        return result

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ═══════════════════════════════════════════════════
# Unified Pipeline
# ═══════════════════════════════════════════════════

class ArabiyaTTSPipeline:
    """Unified pipeline: Arabic text → TTS-ready prosody plan.

    Integrates three engines:
      1. Arabiya Engine (diacritization + case endings)
      2. Rhetoric Analyzer (Ibn Taymiyya + Ibn al-Qayyim)
      3. Prosody Annotator (tajweed → Kokoro)
    """

    def __init__(self, engine=None, rhetoric=None, prosody=None):
        self.engine = engine
        self.rhetoric = rhetoric or RhetoricAnalyzer()
        self.prosody = prosody or ArabicProsodyAnnotator()

    @classmethod
    def load(cls, model_dir: str = None, lexicon_path: str = None,
             device: str = 'cpu'):
        """Load the full pipeline with HRM parser."""
        from arabiya.engine import ArabiyaEngine

        model_dir = model_dir or str(PROJECT_ROOT / "models" / "v2")
        if lexicon_path is None:
            lexicon_path = str(PROJECT_ROOT / "arabiya" / "data" / "lexicon.json")

        # Load case engine
        from models.v2.case_engine import CaseEndingRuleEngine
        case_engine = CaseEndingRuleEngine()

        engine = ArabiyaEngine.load(
            model_dir, lexicon_path,
            external_case_engine=case_engine,
            device=device,
        )

        return cls(engine=engine)

    @classmethod
    def create_mock(cls):
        """Create pipeline with mock parser for testing."""
        from arabiya.engine import ArabiyaEngine
        engine = ArabiyaEngine.create_with_mock()
        return cls(engine=engine)

    def process(self, text: str) -> TTSResult:
        """Run the full pipeline.

        Pipeline:
            Text → [Diacritize] → [Rhetoric] → [Prosody] → TTSResult

        Each stage enriches the result with more information.
        """
        result = TTSResult(input_text=text)

        # ═══ Stage 1: Diacritization (Arabiya Engine) ═══
        if self.engine:
            arabiya_result = self.engine.process(text)
            result.diacritized = arabiya_result.diacritized
            result.arabiya_result = arabiya_result
            result.pipeline_stages.append("diacritization")
        else:
            result.diacritized = text

        # ═══ Stage 2: Rhetoric Analysis (Ibn Taymiyya + Ibn al-Qayyim) ═══
        rhetoric_result = self._analyze_rhetoric(result)
        result.rhetoric = rhetoric_result
        result.pipeline_stages.append("rhetoric")

        # ═══ Stage 3: Prosody Annotation (Tajweed + Emotion → Kokoro) ═══
        prosody_plan = self._generate_prosody(result)
        result.prosody_plan = prosody_plan
        result.pipeline_stages.append("prosody")

        return result

    def _analyze_rhetoric(self, result: TTSResult) -> RhetoricResult:
        """Stage 2: Analyze rhetorical devices from parse tree.

        Uses the parse tree from Stage 1 to detect devices per
        Al-Qazwini's catalog, then applies Ibn Taymiyya's emphasis
        gradation and Ibn al-Qayyim's particle disambiguation.
        """
        if result.arabiya_result and result.arabiya_result.sentences:
            # Extract parse info for the first sentence
            sent = result.arabiya_result.sentences[0]
            words = [w.bare for w in sent.words]
            pos_tags = [w.pos for w in sent.words]
            relations = [w.relation for w in sent.words]
            heads = [w.head for w in sent.words]
            case_tags = [w.case_tag for w in sent.words]

            return self.rhetoric.analyze_sentence(
                words=words,
                pos_tags=pos_tags,
                relations=relations,
                heads=heads,
                case_tags=case_tags,
            )

        # Fallback: analyze from raw text
        words = strip_diacritics(result.input_text).split()
        return self.rhetoric.analyze_sentence(
            words=words,
            pos_tags=["X"] * len(words),
            relations=["dep"] * len(words),
            heads=[-1] * len(words),
            case_tags=[None] * len(words),
        )

    def _generate_prosody(self, result: TTSResult) -> ProsodyPlan:
        """Stage 3: Generate Kokoro prosody from diacritized text + rhetoric.

        Combines:
        - Tajweed rules (from ArabicProsodyAnnotator)
        - Rhetoric-driven emotion (from RhetoricResult)
        - Device-specific prosody (emphasis, pitch, pauses)
        """
        # Determine emotion from rhetoric
        emotion = Emotion.NEUTRAL
        if result.rhetoric:
            rhetoric_emotion = result.rhetoric.dominant_emotion
            emotion = RHETORIC_TO_EMOTION.get(rhetoric_emotion, Emotion.NEUTRAL)

        # Generate base prosody plan from tajweed
        plan = self.prosody.annotate(
            result.diacritized,
            emotion_override=emotion,
        )

        # Overlay rhetoric-driven prosody on trigger words
        if result.rhetoric:
            self._apply_rhetoric_prosody(plan, result.rhetoric)

        return plan

    def _apply_rhetoric_prosody(self, plan: ProsodyPlan,
                                rhetoric: RhetoricResult):
        """Overlay rhetorical device prosody on the prosody plan.

        Per Ibn Taymiyya's emphasis gradation:
        - Higher emphasis = slower speed, more pauses, stronger articulation
        - The trigger word of each device gets special prosodic treatment
        """
        for device in rhetoric.devices:
            idx = device.trigger_word_idx
            if 0 <= idx < len(plan.segments):
                seg = plan.segments[idx]

                # Get device-specific prosody
                device_prosody = DEVICE_PROSODY_MAP.get(device.device_type, {})

                # Apply speed (multiply with existing)
                if "speed_factor" in device_prosody:
                    seg.speed_factor *= device_prosody["speed_factor"]

                # Apply emphasis (take strongest)
                emphasis_order = ["none", "slight", "moderate", "strong"]
                if "emphasis" in device_prosody:
                    new_emph = device_prosody["emphasis"]
                    if emphasis_order.index(new_emph) > emphasis_order.index(seg.emphasis):
                        seg.emphasis = new_emph

                # Apply pause
                if "pause_after_ms" in device_prosody:
                    seg.pause_after_ms = max(
                        seg.pause_after_ms,
                        device_prosody["pause_after_ms"]
                    )

        # Apply Ibn Taymiyya emphasis intensity to overall speed
        if rhetoric.deep and rhetoric.deep.emphasis:
            intensity = rhetoric.deep.emphasis.intensity
            # Higher intensity → slightly slower overall
            if intensity > 0.5:
                speed_mod = 1.0 - (intensity - 0.5) * 0.15  # max -7.5%
                for seg in plan.segments:
                    seg.speed_factor *= speed_mod


# ═══════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════

def demo():
    """Run the full TTS pipeline demo."""
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')

    print("=" * 70)
    print("    Arabiya TTS Pipeline — Full Demo")
    print("    Ibn Malik + Ibn Taymiyya + Ibn al-Qayyim → Kokoro TTS")
    print("=" * 70)

    # Use mock parser for demo
    pipeline = ArabiyaTTSPipeline.create_mock()

    test_sentences = [
        "إن العلم نور",
        "هل ذهب الطالب إلى المدرسة",
        "ما أجمل السماء",
        "بسم الله الرحمن الرحيم",
        "لا تذهب إلى المدرسة",
    ]

    for text in test_sentences:
        result = pipeline.process(text)

        print(f"\n{'─' * 60}")
        print(f"  INPUT:        {text}")
        print(f"  DIACRITIZED:  {result.diacritized}")
        print(f"  EMOTION:      {result.emotion}")
        print(f"  KOKORO TEXT:  {result.kokoro_text}")

        if result.rhetoric and result.rhetoric.devices:
            print(f"  DEVICES:")
            for d in result.rhetoric.devices:
                print(f"    • {d.device_type.value} ({d.description_ar})")

        if result.rhetoric and result.rhetoric.deep:
            deep = result.rhetoric.deep
            if deep.emphasis:
                print(f"  EMPHASIS:     {deep.emphasis.level.value} "
                      f"(intensity={deep.emphasis.intensity:.2f})")

        print(f"  SEGMENTS:")
        if result.prosody_plan:
            for seg in result.prosody_plan.segments:
                tags = ', '.join(seg.prosody_tags[:2]) if seg.prosody_tags else '-'
                print(f"    {seg.word:20s} speed={seg.speed_factor:.2f} "
                      f"pause={seg.pause_after_ms:3d}ms "
                      f"emph={seg.emphasis:8s} [{tags}]")

        print(f"  STAGES:       {' → '.join(result.pipeline_stages)}")

    # Save sample JSON
    sample = pipeline.process("إن العلم نور")
    output_path = PROJECT_ROOT / "output" / "tts_pipeline_sample.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample.to_json())
    print(f"\n{'─' * 60}")
    print(f"  [SAVED] {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    demo()
