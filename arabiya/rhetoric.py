"""
Module 5: Arabic Rhetoric & Emotion Detection
===============================================

Syntax-aware rhetoric analysis for high-fidelity Arabic TTS prosody.

    Sources:
      • Al-Qazwini — Device catalog (12 أساليب بلاغية)
      • Ibn Taymiyya (مجموع الفتاوى vols 7-9) — Emphasis gradation,
        interrogative classification, negation strength
      • Ibn al-Qayyim (بدائع الفوائد) — Particle disambiguation,
        conditional classification with emotional coloring

    Devices:
      INSHAA: استفهام, أمر, نهي, نداء, تمني, تعجب
      KHABAR: توكيد, نفي, قسم, شرط, حصر
      BAYAAN: تشبيه

    Deep Analysis (Ibn Taymiyya + Ibn al-Qayyim):
      • 5-level emphasis gradation (قد → إنّ → إنّ+لام → قسم+لام → قسم+لام+نون)
      • 5-way interrogative type (حقيقي / إنكاري / تقريري / تعجبي / تهكمي)
      • 5-level negation strength (ما → لا → لم → لن → ليس)
      • 4-way conditional type (محتمل / متوقع / امتناع / امتنان)
      • Particle disambiguation for ما، إن، لا، قد (Ibn al-Qayyim)

Usage:
    from arabiya.rhetoric import RhetoricAnalyzer
    analyzer = RhetoricAnalyzer()
    result = analyzer.analyze_sentence(
        words=["هل", "ذهب", "الطالب"],
        pos_tags=["PART", "VERB", "NOUN"],
        relations=["mark", "root", "nsubj"],
        heads=[1, -1, 1],
        case_tags=[None, None, "Nom"],
    )
    # result.devices → [RhetoricalDevice(ISTIFHAM, ...)]
    # result.deep    → DeepAnalysisResult (emphasis, interrogative, ...)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════
#          RHETORICAL DEVICE TYPES
# ═══════════════════════════════════════════════════

class DeviceCategory(Enum):
    """Top-level rhetorical category."""
    INSHAA = "إنشاء"       # Performative (questions, commands, wishes)
    KHABAR = "خبر"         # Informative (assertions, negations, oaths)
    BAYAAN = "بيان"        # Figurative (simile, metaphor)


class DeviceType(Enum):
    """Specific rhetorical device."""
    # ── إنشاء (Performative) ──
    ISTIFHAM = "استفهام"         # Question
    AMR = "أمر"                  # Command
    NAHI = "نهي"                 # Prohibition
    NIDAA = "نداء"               # Vocative
    TAMANNI = "تمني"             # Wish
    TAAJJUB = "تعجب"            # Exclamation

    # ── خبر (Informative) ──
    TAWKEED = "توكيد"           # Emphasis
    NAFI = "نفي"                 # Negation
    QASM = "قسم"                 # Oath
    SHART = "شرط"                # Conditional
    HASR = "حصر"                 # Restriction
    ILTIFAT = "التفات"          # Person/number shift

    # ── بيان (Figurative) ──
    TASHBEEH = "تشبيه"          # Simile
    ISTIAARA = "استعارة"        # Metaphor

    # ── Neutral ──
    NEUTRAL = "محايد"            # No rhetorical device


@dataclass
class RhetoricalDevice:
    """A detected rhetorical device in a sentence."""
    device_type: DeviceType
    category: DeviceCategory
    confidence: float            # 0.0 - 1.0
    trigger_word_idx: int        # Index of the word that triggers the device
    trigger_word: str            # The actual trigger word
    scope_start: int             # Start of the device's scope
    scope_end: int               # End of the device's scope
    description_ar: str          # Arabic grammatical description
    description_en: str          # English description

    # Prosody impact
    pitch_contour: str = "flat"   # rising | falling | flat | peak
    speed_factor: float = 1.0     # Relative to baseline
    emphasis: str = "none"        # none | slight | moderate | strong
    pause_before_ms: int = 0
    pause_after_ms: int = 0


# Import Ibn Taymiyya + Ibn al-Qayyim precision layer
from arabiya.deep_rhetoric import (
    deep_analyze, DeepAnalysisResult,
    EmphasisLevel, InterrogativeType, NegationStrength,
    ConditionalType, ParticleMeaning,
    measure_emphasis_strength, classify_interrogative,
    classify_negation, classify_conditional,
    disambiguate_ma, disambiguate_in, disambiguate_la, disambiguate_qad,
)


@dataclass
class RhetoricResult:
    """Analysis result for a sentence."""
    devices: List[RhetoricalDevice]
    dominant_emotion: str    # neutral | emphatic | questioning | commanding | etc.
    overall_intensity: float  # 0.0 (calm) - 1.0 (intense)
    deep: Optional[DeepAnalysisResult] = None  # Ibn Taymiyya + Ibn al-Qayyim

    def has_device(self, dtype: DeviceType) -> bool:
        return any(d.device_type == dtype for d in self.devices)

    def to_prosody_dict(self) -> Dict:
        """Convert to prosody-compatible format for TTS."""
        result = {
            "emotion": self.dominant_emotion,
            "intensity": self.overall_intensity,
            "devices": [
                {
                    "type": d.device_type.value,
                    "trigger": d.trigger_word,
                    "idx": d.trigger_word_idx,
                    "pitch": d.pitch_contour,
                    "speed": d.speed_factor,
                    "emphasis": d.emphasis,
                    "pause_before": d.pause_before_ms,
                    "pause_after": d.pause_after_ms,
                }
                for d in self.devices
            ],
        }
        if self.deep:
            result["deep"] = self.deep.to_dict()
        return result


# ═══════════════════════════════════════════════════
#          ARABIC PARTICLE LEXICONS
# ═══════════════════════════════════════════════════

# ── استفهام (Interrogation) ──
ISTIFHAM_PARTICLES = {
    'هل': 'هل التصديقية',     # Yes/no question
    'أ': 'همزة الاستفهام',     # Yes/no question (hamza)
    'ما': 'ما الاستفهامية',    # What?
    'ماذا': 'ماذا',            # What?
    'من': 'من الاستفهامية',    # Who?
    'أين': 'أين',              # Where?
    'متى': 'متى',              # When?
    'كيف': 'كيف',              # How?
    'كم': 'كم الاستفهامية',    # How many?
    'لماذا': 'لماذا',          # Why?
    'أي': 'أي الاستفهامية',    # Which?
    'أنى': 'أنى',              # How/Where?
}

# ── توكيد (Emphasis) ──
TAWKEED_PARTICLES = {
    'إن': 'إنّ وأخواتها',       # Indeed (إنّ)
    'أن': 'أنّ',                # That (indeed)
    'قد': 'قد التحقيقية',       # Certainly (with past verb)
    'لقد': 'لقد',               # Indeed (emphatic past)
    'إنما': 'إنما الحصرية',     # Only/Indeed (restriction+emphasis)
    'نعم': 'نعم',               # Yes (affirmation)
    'أجل': 'أجل',               # Yes indeed
    'بلى': 'بلى',               # Yes (after negative question)
}

# ── نفي (Negation) ──
NAFI_PARTICLES = {
    'لا': ('لا النافية', 'present'),       # No / not (present)
    'ما': ('ما النافية', 'past'),          # Not (past/present)
    'لم': ('لم الجازمة', 'past'),          # Did not (jussive+past)
    'لن': ('لن الناصبة', 'future'),        # Will not (subjunctive+future)
    'ليس': ('ليس', 'present'),             # Is not
    'لما': ('لم + ما', 'past'),            # Has not yet
}

# ── شرط (Conditional) ──
SHART_PARTICLES = {
    'إن': 'إن الشرطية',         # If (uncertain)
    'إذا': 'إذا الشرطية',       # When/If (certain future)
    'لو': 'لو الامتناعية',      # If (counterfactual)
    'من': 'من الشرطية',         # Whoever
    'ما': 'ما الشرطية',         # Whatever
    'مهما': 'مهما',             # Whatever
    'أينما': 'أينما',           # Wherever
    'حيثما': 'حيثما',           # Wherever
    'كلما': 'كلما',             # Whenever
    'لولا': 'لولا',             # If not for
}

# ── قسم (Oath) ──
QASM_PATTERNS = {
    'والله': 'والله',
    'بالله': 'بالله',
    'تالله': 'تالله',
    'وحياتك': 'وحياتك',
    'لعمري': 'لعمري',
    'أقسم': 'أقسم',
}

# ── نداء (Vocative) ──
NIDAA_PARTICLES = {'يا', 'أيها', 'أيتها', 'هيا', 'أي', 'وا'}

# ── تمني (Wish) ──
TAMANNI_PARTICLES = {'ليت', 'لعل', 'عسى', 'هلا', 'لو'}

# ── تعجب (Exclamation) ──
# Pattern: ما أفعل (ما أجمل, ما أحسن)
TAAJJUB_MA = 'ما'

# ── تشبيه (Simile) ──
TASHBEEH_PARTICLES = {'كـ', 'ك', 'مثل', 'كأن', 'كأنما', 'كما', 'شبيه', 'يشبه'}

# ── حصر (Restriction) ──
HASR_PARTICLES = {'إنما', 'إلا', 'فقط', 'فحسب', 'ليس', 'سوى', 'غير'}


# ═══════════════════════════════════════════════════
#          RHETORIC ANALYZER
# ═══════════════════════════════════════════════════

class RhetoricAnalyzer:
    """Detects Arabic rhetorical devices from syntax trees.
    
    Uses:
    - Word forms (particles, keywords)
    - POS tags (VERB, PART, etc.)
    - Dependency relations (nsubj, obj, vocative, etc.)
    - Case tags (Nom/Acc/Gen) for disambiguation
    """

    def analyze_sentence(
        self,
        words: List[str],
        pos_tags: List[str],
        relations: List[str],
        heads: List[int],
        case_tags: List[Optional[str]],
        features: Optional[List[Dict]] = None,
    ) -> RhetoricResult:
        """Analyze a sentence for rhetorical devices.
        
        Args:
            words: List of word forms
            pos_tags: UD POS tags
            relations: UD dependency relations
            heads: Head indices (0-indexed)
            case_tags: Case tags (Nom/Acc/Gen/None)
            features: Optional morphological features per word
        """
        n = len(words)
        devices = []

        # 1. Istifham (استفهام) — Question detection
        d = self._detect_istifham(words, pos_tags, relations, n)
        if d:
            devices.append(d)

        # 2. Amr (أمر) — Command detection
        d = self._detect_amr(words, pos_tags, relations, features, n)
        if d:
            devices.append(d)

        # 3. Nahi (نهي) — Prohibition detection
        d = self._detect_nahi(words, pos_tags, n)
        if d:
            devices.append(d)

        # 4. Nidaa (نداء) — Vocative
        d = self._detect_nidaa(words, relations, n)
        if d:
            devices.append(d)

        # 5. Tawkeed (توكيد) — Emphasis
        d = self._detect_tawkeed(words, pos_tags, relations, n)
        if d:
            devices.append(d)

        # 6. Nafi (نفي) — Negation
        d = self._detect_nafi(words, pos_tags, n)
        if d:
            devices.append(d)

        # 7. Qasm (قسم) — Oath
        d = self._detect_qasm(words, n)
        if d:
            devices.append(d)

        # 8. Shart (شرط) — Conditional
        d = self._detect_shart(words, pos_tags, relations, n)
        if d:
            devices.append(d)

        # 9. Hasr (حصر) — Restriction
        d = self._detect_hasr(words, n)
        if d:
            devices.append(d)

        # 10. Tashbeeh (تشبيه) — Simile
        d = self._detect_tashbeeh(words, n)
        if d:
            devices.append(d)

        # 11. Taajjub (تعجب) — Exclamation
        d = self._detect_taajjub(words, pos_tags, n)
        if d:
            devices.append(d)

        # 12. Tamanni (تمني) — Wish
        d = self._detect_tamanni(words, n)
        if d:
            devices.append(d)

        # Run Ibn Taymiyya + Ibn al-Qayyim deep analysis
        deep = deep_analyze(words, pos_tags, relations)

        # Determine dominant emotion and intensity
        emotion, intensity = self._classify_emotion(devices, words, deep)

        # Use deep analysis intensity if higher
        if deep.overall_intensity > intensity:
            intensity = deep.overall_intensity

        return RhetoricResult(
            devices=devices,
            dominant_emotion=emotion,
            overall_intensity=intensity,
            deep=deep,
        )

    # ─────────────────────────────────────────────
    #        DETECTION METHODS
    # ─────────────────────────────────────────────

    def _detect_istifham(self, words, pos_tags, relations, n):
        """Detect استفهام (interrogative).
        
        Disambiguation rules:
          - ما + أفعل → تعجب (not استفهام)
          - ما + noun...إلا → حصر (not استفهام)
          - من/ما at sentence start with ? → استفهام
        """
        has_question = any(w in ('?', '؟') for w in words)

        for i, w in enumerate(words):
            if w in ISTIFHAM_PARTICLES:
                # Disambiguate ما
                if w == 'ما':
                    # ما أفعل = تعجب
                    if i + 1 < n and words[i+1].startswith('أ') and len(words[i+1]) >= 3:
                        continue
                    # ما...إلا = حصر
                    if any(words[j] == 'إلا' for j in range(i+1, n)):
                        continue
                    # ما + past verb = نفي (ما ذهب = he didn't go)
                    if i + 1 < n and pos_tags[i+1] == 'VERB':
                        continue
                # Disambiguate من: من + noun = preposition, not question
                if w == 'من' and i + 1 < n and pos_tags[i+1] == 'NOUN':
                    if not has_question:
                        continue
                return RhetoricalDevice(
                    device_type=DeviceType.ISTIFHAM,
                    category=DeviceCategory.INSHAA,
                    confidence=0.95,
                    trigger_word_idx=i,
                    trigger_word=w,
                    scope_start=i,
                    scope_end=n - 1,
                    description_ar=f"استفهام بـ{ISTIFHAM_PARTICLES[w]}",
                    description_en=f"Interrogative with '{w}'",
                    pitch_contour="rising",
                    speed_factor=0.95,
                    emphasis="moderate",
                    pause_after_ms=300,
                )

        if has_question:
            return RhetoricalDevice(
                device_type=DeviceType.ISTIFHAM,
                category=DeviceCategory.INSHAA,
                confidence=0.8,
                trigger_word_idx=n - 1,
                trigger_word="؟",
                scope_start=0,
                scope_end=n - 1,
                description_ar="استفهام",
                description_en="Question (from punctuation)",
                pitch_contour="rising",
                speed_factor=0.95,
                emphasis="slight",
            )
        return None

    def _detect_amr(self, words, pos_tags, relations, features, n):
        """Detect أمر (command/imperative)."""
        for i, w in enumerate(words):
            # Imperative verb from features
            if pos_tags[i] == 'VERB':
                feats = features[i] if features else {}
                if isinstance(feats, dict) and feats.get('mood') == 'imperative':
                    return RhetoricalDevice(
                        device_type=DeviceType.AMR,
                        category=DeviceCategory.INSHAA,
                        confidence=0.9,
                        trigger_word_idx=i,
                        trigger_word=w,
                        scope_start=i,
                        scope_end=n - 1,
                        description_ar=f"أمر: {w}",
                        description_en=f"Command: {w}",
                        pitch_contour="falling",
                        speed_factor=1.05,
                        emphasis="strong",
                        pause_after_ms=200,
                    )
        return None

    def _detect_nahi(self, words, pos_tags, n):
        """Detect نهي (prohibition): لا + مضارع مجزوم."""
        for i in range(n - 1):
            if words[i] == 'لا' and pos_tags[i + 1] == 'VERB':
                return RhetoricalDevice(
                    device_type=DeviceType.NAHI,
                    category=DeviceCategory.INSHAA,
                    confidence=0.85,
                    trigger_word_idx=i,
                    trigger_word='لا',
                    scope_start=i,
                    scope_end=n - 1,
                    description_ar="نهي: لا الناهية",
                    description_en="Prohibition: لا + jussive verb",
                    pitch_contour="falling",
                    speed_factor=0.95,
                    emphasis="strong",
                    pause_before_ms=100,
                )
        return None

    def _detect_nidaa(self, words, relations, n):
        """Detect نداء (vocative): يا + منادى."""
        for i, w in enumerate(words):
            if w in NIDAA_PARTICLES:
                # Find the addressee (next noun)
                addressee = words[i + 1] if i + 1 < n else ''
                return RhetoricalDevice(
                    device_type=DeviceType.NIDAA,
                    category=DeviceCategory.INSHAA,
                    confidence=0.95,
                    trigger_word_idx=i,
                    trigger_word=w,
                    scope_start=i,
                    scope_end=min(i + 2, n - 1),
                    description_ar=f"نداء: {w} {addressee}",
                    description_en=f"Vocative: {w} {addressee}",
                    pitch_contour="peak",
                    speed_factor=0.9,
                    emphasis="strong",
                    pause_after_ms=250,
                )
            # Also detect from relation
            if i < len(relations) and relations[i] == 'vocative':
                return RhetoricalDevice(
                    device_type=DeviceType.NIDAA,
                    category=DeviceCategory.INSHAA,
                    confidence=0.9,
                    trigger_word_idx=i,
                    trigger_word=w,
                    scope_start=max(0, i - 1),
                    scope_end=i,
                    description_ar=f"نداء: {w}",
                    description_en=f"Vocative: {w} (from syntax)",
                    pitch_contour="peak",
                    speed_factor=0.9,
                    emphasis="moderate",
                )
        return None

    def _detect_tawkeed(self, words, pos_tags, relations, n):
        """Detect توكيد (emphasis): إنّ, قد, لام التوكيد."""
        for i, w in enumerate(words):
            if w in TAWKEED_PARTICLES:
                desc_ar = TAWKEED_PARTICLES[w]
                return RhetoricalDevice(
                    device_type=DeviceType.TAWKEED,
                    category=DeviceCategory.KHABAR,
                    confidence=0.9,
                    trigger_word_idx=i,
                    trigger_word=w,
                    scope_start=i,
                    scope_end=n - 1,
                    description_ar=f"توكيد بـ{desc_ar}",
                    description_en=f"Emphasis with '{w}'",
                    pitch_contour="peak",
                    speed_factor=0.9,
                    emphasis="strong",
                    pause_before_ms=150,
                )
            # لام التوكيد (prefix ل on verbs)
            if w.startswith('ل') and len(w) > 1 and pos_tags[i] == 'VERB':
                return RhetoricalDevice(
                    device_type=DeviceType.TAWKEED,
                    category=DeviceCategory.KHABAR,
                    confidence=0.7,
                    trigger_word_idx=i,
                    trigger_word=w,
                    scope_start=i,
                    scope_end=n - 1,
                    description_ar="توكيد بلام التوكيد",
                    description_en=f"Emphasis with laam on '{w}'",
                    pitch_contour="peak",
                    speed_factor=0.95,
                    emphasis="moderate",
                )
        return None

    def _detect_nafi(self, words, pos_tags, n):
        """Detect نفي (negation)."""
        for i, w in enumerate(words):
            if w in NAFI_PARTICLES:
                desc_ar, tense = NAFI_PARTICLES[w]
                return RhetoricalDevice(
                    device_type=DeviceType.NAFI,
                    category=DeviceCategory.KHABAR,
                    confidence=0.9,
                    trigger_word_idx=i,
                    trigger_word=w,
                    scope_start=i,
                    scope_end=n - 1,
                    description_ar=f"نفي بـ{desc_ar}",
                    description_en=f"Negation ({tense}) with '{w}'",
                    pitch_contour="falling",
                    speed_factor=0.95,
                    emphasis="moderate",
                )
        return None

    def _detect_qasm(self, words, n):
        """Detect قسم (oath): والله, بالله, تالله."""
        # Check multi-word oath
        text = ' '.join(words)
        for pattern, desc in QASM_PATTERNS.items():
            if pattern in text:
                idx = next((i for i, w in enumerate(words) if pattern.startswith(w)), 0)
                return RhetoricalDevice(
                    device_type=DeviceType.QASM,
                    category=DeviceCategory.KHABAR,
                    confidence=0.95,
                    trigger_word_idx=idx,
                    trigger_word=pattern,
                    scope_start=idx,
                    scope_end=n - 1,
                    description_ar=f"قسم: {desc}",
                    description_en=f"Oath: {pattern}",
                    pitch_contour="peak",
                    speed_factor=0.85,
                    emphasis="strong",
                    pause_before_ms=200,
                    pause_after_ms=300,
                )

        # و + noun (واو القسم)
        for i, w in enumerate(words):
            if w == 'و' and i + 1 < n:
                return RhetoricalDevice(
                    device_type=DeviceType.QASM,
                    category=DeviceCategory.KHABAR,
                    confidence=0.6,
                    trigger_word_idx=i,
                    trigger_word=f"و{words[i+1]}",
                    scope_start=i,
                    scope_end=n - 1,
                    description_ar="قسم بواو القسم",
                    description_en=f"Oath with waw al-qasam",
                    pitch_contour="peak",
                    speed_factor=0.85,
                    emphasis="moderate",
                    pause_after_ms=200,
                )
        return None

    def _detect_shart(self, words, pos_tags, relations, n):
        """Detect شرط (conditional): إن, إذا, لو."""
        for i, w in enumerate(words):
            if w in SHART_PARTICLES:
                # Disambiguate: إن can be شرطية or توكيدية
                # If followed by verb → conditional; if followed by noun → emphasis
                if w == 'إن' and i + 1 < n:
                    if pos_tags[i + 1] == 'NOUN':
                        continue  # This is إنّ التوكيدية, not شرطية
                desc = SHART_PARTICLES[w]
                return RhetoricalDevice(
                    device_type=DeviceType.SHART,
                    category=DeviceCategory.KHABAR,
                    confidence=0.85,
                    trigger_word_idx=i,
                    trigger_word=w,
                    scope_start=i,
                    scope_end=n - 1,
                    description_ar=f"شرط بـ{desc}",
                    description_en=f"Conditional with '{w}'",
                    pitch_contour="rising",
                    speed_factor=0.95,
                    emphasis="moderate",
                    pause_after_ms=200,
                )
        return None

    def _detect_hasr(self, words, n):
        """Detect حصر (restriction): إنما, ما...إلا."""
        for i, w in enumerate(words):
            if w == 'إنما':
                return RhetoricalDevice(
                    device_type=DeviceType.HASR,
                    category=DeviceCategory.KHABAR,
                    confidence=0.95,
                    trigger_word_idx=i,
                    trigger_word=w,
                    scope_start=i,
                    scope_end=n - 1,
                    description_ar="حصر بإنما",
                    description_en="Restriction with 'innama'",
                    pitch_contour="peak",
                    speed_factor=0.9,
                    emphasis="strong",
                )
        # ما...إلا pattern
        for i, w in enumerate(words):
            if w == 'إلا':
                # Check if there's a negation before
                has_negation = any(words[j] in ('ما', 'لا', 'ليس')
                                  for j in range(i))
                if has_negation:
                    return RhetoricalDevice(
                        device_type=DeviceType.HASR,
                        category=DeviceCategory.KHABAR,
                        confidence=0.9,
                        trigger_word_idx=i,
                        trigger_word='ما...إلا',
                        scope_start=0,
                        scope_end=n - 1,
                        description_ar="حصر بالنفي والاستثناء",
                        description_en="Restriction with negation + exception",
                        pitch_contour="peak",
                        speed_factor=0.9,
                        emphasis="strong",
                        pause_before_ms=150,
                    )
        return None

    def _detect_tashbeeh(self, words, n):
        """Detect تشبيه (simile): كـ, مثل, كأن."""
        for i, w in enumerate(words):
            if w in TASHBEEH_PARTICLES:
                return RhetoricalDevice(
                    device_type=DeviceType.TASHBEEH,
                    category=DeviceCategory.BAYAAN,
                    confidence=0.8,
                    trigger_word_idx=i,
                    trigger_word=w,
                    scope_start=max(0, i - 1),
                    scope_end=min(i + 3, n - 1),
                    description_ar=f"تشبيه بأداة '{w}'",
                    description_en=f"Simile with '{w}'",
                    pitch_contour="flat",
                    speed_factor=0.9,
                    emphasis="slight",
                )
        return None

    def _detect_taajjub(self, words, pos_tags, n):
        """Detect تعجب (exclamation): ما أفعل! (ما أجمل!, ما أحسن!)."""
        for i in range(n - 1):
            if words[i] == 'ما' and i + 1 < n:
                next_w = words[i + 1]
                # ما أفعل pattern: ما + word starting with أ
                if next_w.startswith('أ') and len(next_w) >= 3:
                    return RhetoricalDevice(
                        device_type=DeviceType.TAAJJUB,
                        category=DeviceCategory.INSHAA,
                        confidence=0.85,
                        trigger_word_idx=i,
                        trigger_word=f"ما {next_w}",
                        scope_start=i,
                        scope_end=min(i + 3, n - 1),
                        description_ar=f"تعجب: ما {next_w}!",
                        description_en=f"Exclamation: How {next_w}!",
                        pitch_contour="peak",
                        speed_factor=0.85,
                        emphasis="strong",
                        pause_after_ms=300,
                    )
        # Exclamation mark
        if any(w == '!' for w in words):
            return RhetoricalDevice(
                device_type=DeviceType.TAAJJUB,
                category=DeviceCategory.INSHAA,
                confidence=0.6,
                trigger_word_idx=n - 1,
                trigger_word='!',
                scope_start=0,
                scope_end=n - 1,
                description_ar="تعجب",
                description_en="Exclamation (from punctuation)",
                pitch_contour="peak",
                speed_factor=0.95,
                emphasis="moderate",
            )
        return None

    def _detect_tamanni(self, words, n):
        """Detect تمني (wish): ليت, لعل, عسى."""
        for i, w in enumerate(words):
            if w in TAMANNI_PARTICLES:
                return RhetoricalDevice(
                    device_type=DeviceType.TAMANNI,
                    category=DeviceCategory.INSHAA,
                    confidence=0.9,
                    trigger_word_idx=i,
                    trigger_word=w,
                    scope_start=i,
                    scope_end=n - 1,
                    description_ar=f"تمني بـ'{w}'",
                    description_en=f"Wish with '{w}'",
                    pitch_contour="rising",
                    speed_factor=0.9,
                    emphasis="moderate",
                    pause_before_ms=100,
                )
        return None

    # ─────────────────────────────────────────────
    #        EMOTION CLASSIFICATION
    # ─────────────────────────────────────────────

    def _classify_emotion(self, devices: List[RhetoricalDevice],
                          words: List[str],
                          deep: Optional[DeepAnalysisResult] = None,
                          ) -> Tuple[str, float]:
        """Classify overall emotional tone from detected devices.
        
        Enhanced with Ibn Taymiyya's intensity gradations:
          - Emphasis level refines tawkeed intensity
          - Interrogative type refines question emotion
          - Negation strength refines denial intensity
          - Conditional type adds emotional coloring
        """
        if not devices and (not deep or deep.emphasis.level == EmphasisLevel.NONE):
            return "neutral", 0.3

        types = {d.device_type for d in devices}

        # ── Use Ibn Taymiyya's deep classification when available ──
        if deep:
            # Interrogative sub-classification
            if DeviceType.ISTIFHAM in types and deep.interrogative:
                itype = deep.interrogative.q_type
                if itype == InterrogativeType.DENIAL:
                    return "rebuking", 0.9
                elif itype == InterrogativeType.CONFIRMATION:
                    return "affirming", 0.7
                elif itype == InterrogativeType.WONDER:
                    return "wondering", 0.85
                elif itype == InterrogativeType.SARCASM:
                    return "mocking", 0.8
                # Real question falls through

            # Conditional sub-classification
            if DeviceType.SHART in types and deep.conditional:
                ctype = deep.conditional.cond_type
                if ctype == ConditionalType.COUNTERFACTUAL:
                    if deep.conditional.law_subtype == 'تمني':
                        return "longing", 0.85
                    elif deep.conditional.law_subtype == 'برهان':
                        return "assertive", 0.9
                elif ctype == ConditionalType.BUT_FOR:
                    return "grateful", 0.7

            # Emphasis level upgrades tawkeed intensity
            if DeviceType.TAWKEED in types and deep.emphasis.level != EmphasisLevel.NONE:
                elevel = deep.emphasis.level
                return "emphatic", elevel.intensity

            # Negation strength upgrades nafi intensity
            if DeviceType.NAFI in types and deep.negation_strength:
                nstrength = deep.negation_strength[0]
                return "denying", nstrength.intensity

        # ── Standard priority fallback ──
        if DeviceType.QASM in types:
            return "solemn", 0.9
        if DeviceType.TAAJJUB in types:
            return "exclaiming", 0.8
        if DeviceType.AMR in types or DeviceType.NAHI in types:
            return "commanding", 0.8
        if DeviceType.NIDAA in types:
            return "addressing", 0.7
        if DeviceType.HASR in types:
            return "emphatic", 0.7
        if DeviceType.TAWKEED in types:
            return "emphatic", 0.7
        if DeviceType.ISTIFHAM in types:
            return "questioning", 0.6
        if DeviceType.TAMANNI in types:
            return "hopeful", 0.6
        if DeviceType.NAFI in types:
            return "denying", 0.5
        if DeviceType.SHART in types:
            return "conditional", 0.5
        if DeviceType.TASHBEEH in types:
            return "descriptive", 0.4

        # Deep analysis found emphasis but no device
        if deep and deep.emphasis.level != EmphasisLevel.NONE:
            return "emphatic", deep.emphasis.intensity

        return "neutral", 0.3
