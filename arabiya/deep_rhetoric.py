"""
Ibn Taymiyya & Ibn al-Qayyim Linguistic Precision Layer
=========================================================

Enhances Module 5 (rhetoric.py) with forensic-level particle disambiguation
and intensity gradation from two classical scholars:

1. Ibn al-Qayyim's بدائع الفوائد — Particle disambiguation rules
2. Ibn Taymiyya's مجموع الفتاوى (vols 7-9) — Classification trees

This module adds:
  • 5-level emphasis gradation scale (Ibn Taymiyya)
  • 5-way interrogative classification (Ibn Taymiyya)
  • 5-level negation strength scale (Ibn Taymiyya)
  • 4-way conditional classification with emotional coloring (Ibn al-Qayyim)
  • Particle disambiguation for ما، إن، لا، قد، لام (Ibn al-Qayyim)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════
#     IBN TAYMIYYA: EMPHASIS GRADATION SCALE
#     مراتب التوكيد — Majmū' al-Fatāwā, Vol. 7
# ═══════════════════════════════════════════════════

class EmphasisLevel(Enum):
    """Ibn Taymiyya's 5-level emphasis gradation."""
    NONE = ("لا_توكيد", 0.0)         # No emphasis
    WEAK = ("توكيد_ضعيف", 0.3)       # قد + past
    STANDARD = ("توكيد_متوسط", 0.6)  # إنّ/أنّ
    STRONG = ("توكيد_قوي", 0.8)      # إنّ + لام التوكيد
    VERY_STRONG = ("توكيد_شديد", 0.95)  # Oath + لام
    MAXIMUM = ("توكيد_مؤكد", 1.0)    # Oath + لام + نون التوكيد

    @property
    def arabic(self):
        return self.value[0]

    @property
    def intensity(self):
        return self.value[1]


@dataclass
class EmphasisResult:
    """Result of emphasis analysis."""
    level: EmphasisLevel
    intensity: float          # 0.0 - 1.0
    markers: List[str]        # Detected emphasis markers
    description_ar: str       # Arabic description


def measure_emphasis_strength(
    words: List[str],
    pos_tags: List[str],
    relations: List[str],
) -> EmphasisResult:
    """Ibn Taymiyya's emphasis gradation scale.

    Levels (from Majmū' al-Fatāwā, Vol. 7):
      1. قد + past verb = mild confirmation (0.3)
      2. إن/أن = standard emphasis (0.6)
      3. إن + لام التوكيد = strong emphasis (0.8)
      4. Oath + لام = very strong (0.95)
      5. Oath + لام + نون التوكيد = maximum (1.0)
    """
    score = 0.0
    markers = []
    n = len(words)

    # ── Check for قد + past verb ──
    has_qad = False
    for i in range(n - 1):
        if words[i] == 'قد' and pos_tags[i + 1] == 'VERB':
            has_qad = True
            score += 0.3
            markers.append('قد_التحقيقية')
            break

    # ── Check for إنّ / أنّ ──
    has_inna = False
    for i, w in enumerate(words):
        if w in ('إن', 'أن', 'إنّ', 'أنّ'):
            # Disambiguate: إن + noun = emphasis; إن + verb = conditional
            if i + 1 < n and pos_tags[i + 1] in ('NOUN', 'DET', 'ADJ'):
                has_inna = True
                score += 0.3
                markers.append('إنّ_التوكيدية')
                break

    # ── Check for لام التوكيد ──
    has_lam = False
    for i, w in enumerate(words):
        # لام as prefix on nouns/adjectives (لَالعلمُ نورٌ)
        # or standalone ل before verb
        if w == 'ل' and i + 1 < n and pos_tags[i + 1] == 'VERB':
            has_lam = True
            score += 0.2
            markers.append('لام_التوكيد')
            break
        # لام as prefix: لَيفعلنّ
        if w.startswith('ل') and len(w) > 1 and pos_tags[i] == 'VERB':
            has_lam = True
            score += 0.2
            markers.append('لام_التوكيد')
            break
        # لام as prefix on noun in إنّ sentence
        if has_inna and w.startswith('ل') and pos_tags[i] in ('NOUN', 'ADJ'):
            has_lam = True
            score += 0.2
            markers.append('لام_التوكيد')
            break

    # ── Check for oath (قسم) ──
    has_oath = False
    OATH_WORDS = {'والله', 'بالله', 'تالله', 'أقسم', 'لعمري', 'وحياتك'}
    for i, w in enumerate(words):
        if w in OATH_WORDS:
            has_oath = True
            score += 0.4
            markers.append(f'قسم:{w}')
            break
        # واو القسم + noun
        if w == 'و' and i + 1 < n and pos_tags[i + 1] in ('NOUN', 'PROPN'):
            has_oath = True
            score += 0.3
            markers.append('واو_القسم')
            break

    # ── Check for نون التوكيد ──
    has_noon = False
    for i, w in enumerate(words):
        if pos_tags[i] == 'VERB':
            # نون التوكيد الثقيلة (ـنّ) or الخفيفة (ـنْ/ـن)
            if w.endswith('نّ') or w.endswith('نَّ'):
                if has_oath or has_lam:
                    has_noon = True
                    score += 0.2
                    markers.append('نون_التوكيد_الثقيلة')
                    break
            # Also single ن at end of verb with oath/lam context
            elif (w.endswith('ن') and len(w) >= 3 and
                  (has_oath or has_lam)):
                has_noon = True
                score += 0.2
                markers.append('نون_التوكيد_الخفيفة')
                break

    # ── Calculate final intensity ──
    intensity = min(score, 1.0)

    # ── Map to level ──
    if intensity >= 0.95:
        level = EmphasisLevel.MAXIMUM
    elif intensity >= 0.8:
        level = EmphasisLevel.VERY_STRONG
    elif intensity >= 0.6:
        level = EmphasisLevel.STRONG
    elif intensity >= 0.3:
        level = EmphasisLevel.STANDARD if has_inna else EmphasisLevel.WEAK
    else:
        level = EmphasisLevel.NONE

    # ── Build Arabic description ──
    if markers:
        desc = "توكيد بـ" + " و".join(markers)
    else:
        desc = "لا توكيد"

    return EmphasisResult(
        level=level,
        intensity=intensity,
        markers=markers,
        description_ar=desc,
    )


# ═══════════════════════════════════════════════════
#     IBN TAYMIYYA: INTERROGATIVE CLASSIFICATION
#     أنواع الاستفهام — Majmū' al-Fatāwā, Vol. 9
# ═══════════════════════════════════════════════════

class InterrogativeType(Enum):
    """Ibn Taymiyya's 5-way interrogative classification."""
    REAL = "استفهام_حقيقي"            # Seeking information
    DENIAL = "استفهام_إنكاري"          # Rhetorical denial
    CONFIRMATION = "استفهام_تقريري"    # Rhetorical confirmation
    WONDER = "استفهام_تعجبي"           # Wonder/amazement
    SARCASM = "استفهام_تهكمي"          # Sarcasm/mockery


@dataclass
class InterrogativeResult:
    """Classified interrogative type with prosody."""
    q_type: InterrogativeType
    emotion: str
    intensity: float
    pitch_contour: str
    description_ar: str
    answer_expected: str   # "yes" / "no" / "unknown" / "none"


# Rhetorical denial markers: ألم, أولم, أفلم, ألا
_DENIAL_PREFIXES = {'ألم', 'أولم', 'أفلم', 'أفلا', 'ألا', 'أوَلم', 'أولا'}

# Confirmation markers: أليس, ألست
_CONFIRM_PREFIXES = {'أليس', 'ألست', 'أليست', 'ألسنا', 'ألسم'}


def classify_interrogative(
    words: List[str],
    pos_tags: List[str],
    relations: List[str],
    q_particle_idx: int,
) -> InterrogativeResult:
    """Ibn Taymiyya's 5-way interrogative classification.

    From Majmū' al-Fatāwā:
      TYPE 1: Real question — seeking unknown information
      TYPE 2: Rhetorical denial — ألم/أفلم constructions
      TYPE 3: Rhetorical confirmation — أليس constructions
      TYPE 4: Wonder question — كيف + amazing thing
      TYPE 5: Sarcastic question — mocking impossibility
    """
    n = len(words)
    q_word = words[q_particle_idx] if q_particle_idx < n else ''

    # ── TYPE 2: Rhetorical Denial (استفهام إنكاري) ──
    # Markers: ألم, أولم, أفلم — ALWAYS rhetorical denial
    if q_word in _DENIAL_PREFIXES:
        return InterrogativeResult(
            q_type=InterrogativeType.DENIAL,
            emotion="إنكار",
            intensity=0.9,
            pitch_contour="sharp_rise",
            description_ar=f"استفهام إنكاري بـ'{q_word}'",
            answer_expected="yes (implied)",
        )

    # Also check: أ + لم/لا/ليس (two-word patterns)
    if q_word == 'أ' and q_particle_idx + 1 < n:
        next_w = words[q_particle_idx + 1]
        if next_w in ('لم', 'لا', 'ليس', 'لن'):
            return InterrogativeResult(
                q_type=InterrogativeType.DENIAL,
                emotion="إنكار",
                intensity=0.85,
                pitch_contour="sharp_rise",
                description_ar=f"استفهام إنكاري: أ{next_w}",
                answer_expected="yes (implied)",
            )

    # ── TYPE 3: Rhetorical Confirmation (استفهام تقريري) ──
    if q_word in _CONFIRM_PREFIXES:
        return InterrogativeResult(
            q_type=InterrogativeType.CONFIRMATION,
            emotion="تقرير",
            intensity=0.7,
            pitch_contour="level_then_slight_rise",
            description_ar=f"استفهام تقريري بـ'{q_word}'",
            answer_expected="yes",
        )

    # ── TYPE 4: Wonder (استفهام تعجبي) ──
    if q_word in ('كيف', 'أنى'):
        # Check context for amazing/impossible things
        amazing_words = {'سبحان', 'عجب', 'غريب', 'عظيم', 'هائل'}
        has_amazing = any(w in amazing_words for w in words)
        has_exclamation = any(w in ('!',) for w in words)

        if has_amazing or has_exclamation:
            return InterrogativeResult(
                q_type=InterrogativeType.WONDER,
                emotion="إعجاب",
                intensity=0.85,
                pitch_contour="rising_exclamatory",
                description_ar=f"استفهام تعجبي بـ'{q_word}'",
                answer_expected="none (rhetorical)",
            )

    # ── TYPE 5: Sarcasm (استفهام تهكمي) ──
    # Check for impossible/absurd scenarios
    impossible_markers = {'أإذا', 'أئذا'}
    if q_word in impossible_markers:
        return InterrogativeResult(
            q_type=InterrogativeType.SARCASM,
            emotion="سخرية",
            intensity=0.8,
            pitch_contour="exaggerated_rise",
            description_ar="استفهام تهكمي",
            answer_expected="none (mocking)",
        )

    # ── TYPE 1: Default — Real Question (استفهام حقيقي) ──
    return InterrogativeResult(
        q_type=InterrogativeType.REAL,
        emotion="تساؤل",
        intensity=0.4,
        pitch_contour="standard_rising",
        description_ar=f"استفهام حقيقي بـ'{q_word}'",
        answer_expected="unknown",
    )


# ═══════════════════════════════════════════════════
#     IBN TAYMIYYA: NEGATION STRENGTH SCALE
#     درجات النفي — Majmū' al-Fatāwā, Vol. 8
# ═══════════════════════════════════════════════════

class NegationStrength(Enum):
    """Ibn Taymiyya's negation strength levels."""
    MILD = ("نفي_خفيف", 0.3)           # ما (can be countered)
    PRESENT = ("نفي_حاضر", 0.5)        # لا (prohibition/denial)
    PAST = ("نفي_ماض", 0.6)            # لم (factual past denial)
    FUTURE = ("نفي_مستقبل", 0.8)       # لن (emphatic never)
    EXISTENTIAL = ("نفي_وجودي", 0.9)   # ليس (absolute non-being)

    @property
    def arabic(self):
        return self.value[0]

    @property
    def intensity(self):
        return self.value[1]


def classify_negation(
    neg_particle: str,
    words: List[str],
    pos_tags: List[str],
    particle_idx: int,
) -> Tuple[NegationStrength, str]:
    """Classify negation strength per Ibn Taymiyya.

    Returns (strength, description).

    From Majmū' al-Fatāwā:
      ما → simple denial (can be countered with evidence)
      لا → prohibition/strong denial
      لم → past factual negation (it did NOT happen)
      لن → future emphatic negation (it will NEVER happen)
      ليس → existential negation (absolute non-being)
    """
    if neg_particle == 'لن':
        return NegationStrength.FUTURE, "نفي مؤبد للمستقبل — لن يحدث أبداً"
    elif neg_particle == 'ليس':
        return NegationStrength.EXISTENTIAL, "نفي وجودي — ليس كائناً"
    elif neg_particle == 'لم':
        return NegationStrength.PAST, "نفي جازم للماضي — لم يحدث"
    elif neg_particle == 'لا':
        # Distinguish: لا النافية vs لا الناهية
        n = len(words)
        if particle_idx + 1 < n and pos_tags[particle_idx + 1] == 'VERB':
            return NegationStrength.PRESENT, "نهي أو نفي حاضر"
        return NegationStrength.PRESENT, "لا النافية"
    elif neg_particle == 'ما':
        return NegationStrength.MILD, "نفي خفيف بما — يمكن دفعه"
    elif neg_particle == 'لما':
        return NegationStrength.PAST, "نفي مستمر — لم يحدث حتى الآن"
    else:
        return NegationStrength.MILD, f"نفي بـ{neg_particle}"


# ═══════════════════════════════════════════════════
#  IBN AL-QAYYIM: CONDITIONAL CLASSIFICATION
#  أنواع الشرط — بدائع الفوائد, باب حروف المعاني
# ═══════════════════════════════════════════════════

class ConditionalType(Enum):
    """Ibn al-Qayyim's 4-way conditional classification."""
    UNCERTAIN = "شرط_محتمل"         # إن — uncertain, open outcome
    EXPECTED = "شرط_متوقع"          # إذا — expected, confident
    COUNTERFACTUAL = "شرط_امتناع"   # لو — impossible, didn't happen
    BUT_FOR = "شرط_امتنان"          # لولا — thankful prevention


@dataclass
class ConditionalResult:
    """Classified conditional with emotional coloring."""
    cond_type: ConditionalType
    emotion: str
    intensity: float
    pitch_contour: str
    description_ar: str
    # لو sub-types from Ibn al-Qayyim
    law_subtype: str = ""  # تمني / برهان / فرضي


def classify_conditional(
    cond_particle: str,
    words: List[str],
    pos_tags: List[str],
    particle_idx: int,
) -> ConditionalResult:
    """Ibn al-Qayyim's conditional classification with emotional coloring.

    From بدائع الفوائد:
      إن → real condition, uncertain outcome (neutral/open)
      إذا → when-condition, expected outcome (confident)
      لو → counterfactual impossibility (regretful/assertive)
      لولا → but-for condition (thankful)
    """
    n = len(words)

    if cond_particle == 'إذا':
        return ConditionalResult(
            cond_type=ConditionalType.EXPECTED,
            emotion="واثق",
            intensity=0.5,
            pitch_contour="level_confident",
            description_ar="شرط متوقع — إذا الشرطية",
        )

    elif cond_particle == 'لولا':
        return ConditionalResult(
            cond_type=ConditionalType.BUT_FOR,
            emotion="امتنان",
            intensity=0.7,
            pitch_contour="gentle_fall",
            description_ar="شرط امتنان — لولا (لو لم يكن لحدث)",
        )

    elif cond_particle == 'لو':
        # Ibn al-Qayyim: لو has 3 sub-types
        # 1. لو + أن + 1st person → wish/regret (تمني)
        has_first_person = any(w in ('أنا', 'لي', 'نا', 'ي') for w in words)
        has_anna = any(w in ('أن', 'أنّ') for w in words[particle_idx+1:particle_idx+3])

        if has_anna and has_first_person:
            return ConditionalResult(
                cond_type=ConditionalType.COUNTERFACTUAL,
                emotion="حسرة",
                intensity=0.85,
                pitch_contour="descending_wistful",
                description_ar="لو التمنّي — حسرة على ما فات",
                law_subtype="تمني",
            )

        # 2. لو + كان — logical argument (برهان)
        has_kana = any(w in ('كان', 'كانت', 'كانوا') for w in words[particle_idx+1:particle_idx+3])
        if has_kana:
            return ConditionalResult(
                cond_type=ConditionalType.COUNTERFACTUAL,
                emotion="جازم",
                intensity=0.9,
                pitch_contour="level_emphatic",
                description_ar="لو البرهانية — دليل عقلي",
                law_subtype="برهان",
            )

        # 3. Default: hypothetical (فرضي)
        return ConditionalResult(
            cond_type=ConditionalType.COUNTERFACTUAL,
            emotion="افتراضي",
            intensity=0.6,
            pitch_contour="slightly_falling",
            description_ar="لو الفرضية — افتراض",
            law_subtype="فرضي",
        )

    elif cond_particle == 'إن':
        return ConditionalResult(
            cond_type=ConditionalType.UNCERTAIN,
            emotion="محايد",
            intensity=0.4,
            pitch_contour="level_open",
            description_ar="إن الشرطية — شرط محتمل",
        )

    # Fallback for other particles (من, ما, مهما, etc.)
    return ConditionalResult(
        cond_type=ConditionalType.UNCERTAIN,
        emotion="محايد",
        intensity=0.4,
        pitch_contour="level_open",
        description_ar=f"شرط بـ{cond_particle}",
    )


# ═══════════════════════════════════════════════════
#  IBN AL-QAYYIM: PARTICLE DISAMBIGUATION
#  حروف المعاني — بدائع الفوائد, باب الحروف
# ═══════════════════════════════════════════════════

class ParticleMeaning(Enum):
    """Disambiguated meanings of multi-sense particles."""
    # ما meanings
    MA_NAFI = "ما_النافية"           # Negation
    MA_ISTIFHAM = "ما_الاستفهامية"   # Question (what?)
    MA_TAAJJUB = "ما_التعجبية"       # Exclamation (how!)
    MA_MAWSUL = "ما_الموصولة"        # Relative pronoun (that which)
    MA_MASDAR = "ما_المصدرية"        # Complementizer

    # إن meanings
    IN_SHART = "إن_الشرطية"          # Conditional (if)
    IN_TAWKEED = "إن_التوكيدية"      # Emphasis (indeed)
    IN_NAFI = "إن_النافية"           # Negation (archaic)

    # لا meanings
    LA_NAFI = "لا_النافية"           # Negation
    LA_NAHI = "لا_الناهية"           # Prohibition
    LA_NAFY_JINS = "لا_نافية_للجنس"  # Generic negation (absolute)
    LA_ATIF = "لا_العاطفة"           # Coordinating (not...but)

    # قد meanings
    QAD_TAHQIQ = "قد_التحقيقية"      # Certainty (with past verb)
    QAD_TAQEEN = "قد_التقييد"        # Limitation (with imperfect)

    # لام meanings
    LAM_TAWKEED = "لام_التوكيد"      # Emphasis
    LAM_TALEEL = "لام_التعليل"       # Purpose (in order to)
    LAM_AMR = "لام_الأمر"            # Command (let him)
    LAM_JUHOOD = "لام_الجحود"        # Denial (preceded by كان)


def disambiguate_ma(
    words: List[str],
    pos_tags: List[str],
    ma_idx: int,
) -> ParticleMeaning:
    """Ibn al-Qayyim's disambiguation of ما (5 meanings).

    From بدائع الفوائد:
      1. ما + أفعل = تعجبية (exclamatory)
      2. ما + verb (past) = نافية (negation)
      3. ما + noun...إلا = نافية (negation for restriction)
      4. ما + noun (no verb) = specific context needed
      5. Sentence-initial ما + ? = استفهامية (question)
    """
    n = len(words)

    # 1. ما أفعل pattern → exclamation
    if ma_idx + 1 < n:
        next_w = words[ma_idx + 1]
        if next_w.startswith('أ') and len(next_w) >= 3:
            return ParticleMeaning.MA_TAAJJUB

    # 2. ما + verb → negation
    if ma_idx + 1 < n and pos_tags[ma_idx + 1] == 'VERB':
        return ParticleMeaning.MA_NAFI

    # 3. ما...إلا → negation (for restriction حصر)
    if any(w == 'إلا' for w in words[ma_idx + 1:]):
        return ParticleMeaning.MA_NAFI

    # 4. ما at sentence start + ? → question
    if ma_idx == 0 and any(w in ('?', '؟') for w in words):
        return ParticleMeaning.MA_ISTIFHAM

    # 5. Default: relative pronoun (ما الموصولة)
    return ParticleMeaning.MA_MAWSUL


def disambiguate_in(
    words: List[str],
    pos_tags: List[str],
    in_idx: int,
) -> ParticleMeaning:
    """Ibn al-Qayyim's disambiguation of إن (3 meanings).

    Key rule: إن + noun = توكيدية; إن + verb = شرطية
    """
    n = len(words)

    if in_idx + 1 < n:
        next_pos = pos_tags[in_idx + 1]
        # إن + noun/adj/det → emphasis (إنّ)
        if next_pos in ('NOUN', 'ADJ', 'DET', 'PROPN', 'PRON'):
            return ParticleMeaning.IN_TAWKEED
        # إن + verb → conditional
        if next_pos == 'VERB':
            return ParticleMeaning.IN_SHART

    # Default: conditional
    return ParticleMeaning.IN_SHART


def disambiguate_la(
    words: List[str],
    pos_tags: List[str],
    la_idx: int,
) -> ParticleMeaning:
    """Ibn al-Qayyim's disambiguation of لا (4 meanings)."""
    n = len(words)

    if la_idx + 1 < n:
        next_pos = pos_tags[la_idx + 1]
        # لا + imperfect verb (مضارع) → prohibition (نهي)
        if next_pos == 'VERB':
            next_w = words[la_idx + 1]
            # Imperfect verb starts with ي/ت/ن/أ
            if next_w and next_w[0] in 'يتنأ':
                return ParticleMeaning.LA_NAHI
            return ParticleMeaning.LA_NAFI
        # لا + indefinite noun → نافية للجنس
        if next_pos in ('NOUN', 'ADJ'):
            # If no ال → نافية للجنس (لا رجلَ في الدار)
            if not words[la_idx + 1].startswith('ال'):
                return ParticleMeaning.LA_NAFY_JINS
            return ParticleMeaning.LA_NAFI

    return ParticleMeaning.LA_NAFI


def disambiguate_qad(
    words: List[str],
    pos_tags: List[str],
    qad_idx: int,
) -> ParticleMeaning:
    """Ibn al-Qayyim's disambiguation of قد (2 meanings).

    قد + past verb = certainty (تحقيق): "قد فعل" = he certainly did
    قد + imperfect = limitation/probability (تقليل): "قد يفعل" = he might do
    """
    n = len(words)
    if qad_idx + 1 < n and pos_tags[qad_idx + 1] == 'VERB':
        next_w = words[qad_idx + 1]
        # Imperfect verb starts with ي/ت/ن/أ
        if next_w and next_w[0] in 'يتنأ':
            return ParticleMeaning.QAD_TAQEEN  # Probability
        return ParticleMeaning.QAD_TAHQIQ  # Certainty
    return ParticleMeaning.QAD_TAHQIQ


# ═══════════════════════════════════════════════════
#  UNIFIED DEEP ANALYSIS
# ═══════════════════════════════════════════════════

@dataclass
class DeepAnalysisResult:
    """Combined result from Ibn Taymiyya + Ibn al-Qayyim analysis."""
    emphasis: EmphasisResult
    interrogative: Optional[InterrogativeResult] = None
    negation_strength: Optional[Tuple[NegationStrength, str]] = None
    conditional: Optional[ConditionalResult] = None
    particle_meanings: Dict[int, ParticleMeaning] = None

    @property
    def overall_intensity(self) -> float:
        """Highest intensity across all analyses."""
        intensities = [self.emphasis.intensity]
        if self.interrogative:
            intensities.append(self.interrogative.intensity)
        if self.negation_strength:
            intensities.append(self.negation_strength[0].intensity)
        if self.conditional:
            intensities.append(self.conditional.intensity)
        return max(intensities)

    def to_dict(self) -> Dict:
        result = {
            "emphasis": {
                "level": self.emphasis.level.arabic,
                "intensity": self.emphasis.intensity,
                "markers": self.emphasis.markers,
            },
        }
        if self.interrogative:
            result["interrogative"] = {
                "type": self.interrogative.q_type.value,
                "emotion": self.interrogative.emotion,
                "intensity": self.interrogative.intensity,
            }
        if self.negation_strength:
            result["negation"] = {
                "strength": self.negation_strength[0].arabic,
                "intensity": self.negation_strength[0].intensity,
                "description": self.negation_strength[1],
            }
        if self.conditional:
            result["conditional"] = {
                "type": self.conditional.cond_type.value,
                "emotion": self.conditional.emotion,
                "intensity": self.conditional.intensity,
            }
        return result


def deep_analyze(
    words: List[str],
    pos_tags: List[str],
    relations: List[str],
) -> DeepAnalysisResult:
    """Full Ibn Taymiyya + Ibn al-Qayyim analysis pipeline.

    Runs all classifiers and returns unified result.
    """
    n = len(words)

    # 1. Emphasis analysis (always runs)
    emphasis = measure_emphasis_strength(words, pos_tags, relations)

    # 2. Interrogative classification
    interrogative = None
    ISTIFHAM = {'هل', 'أ', 'ما', 'ماذا', 'من', 'أين', 'متى', 'كيف',
                'كم', 'لماذا', 'أي', 'أنى', 'ألم', 'أفلم', 'أليس', 'ألست'}
    for i, w in enumerate(words):
        if w in ISTIFHAM or w in ('?', '؟'):
            interrogative = classify_interrogative(words, pos_tags, relations, i)
            break

    # 3. Negation classification
    neg_result = None
    NEG_PARTS = {'ما', 'لا', 'لم', 'لن', 'ليس', 'لما'}
    for i, w in enumerate(words):
        if w in NEG_PARTS:
            # Only classify as negation if particle meaning confirms it
            if w == 'ما':
                meaning = disambiguate_ma(words, pos_tags, i)
                if meaning != ParticleMeaning.MA_NAFI:
                    continue
            if w == 'لا':
                meaning = disambiguate_la(words, pos_tags, i)
                if meaning == ParticleMeaning.LA_ATIF:
                    continue
            neg_result = classify_negation(w, words, pos_tags, i)
            break

    # 4. Conditional classification
    cond_result = None
    COND_PARTS = {'إن', 'إذا', 'لو', 'لولا', 'من', 'ما', 'مهما'}
    for i, w in enumerate(words):
        if w in COND_PARTS:
            if w == 'إن':
                meaning = disambiguate_in(words, pos_tags, i)
                if meaning != ParticleMeaning.IN_SHART:
                    continue
            if w == 'ما':
                meaning = disambiguate_ma(words, pos_tags, i)
                if meaning != ParticleMeaning.MA_MAWSUL:
                    continue  # Only موصولة can be شرطية
            cond_result = classify_conditional(w, words, pos_tags, i)
            break

    # 5. Particle disambiguation
    particle_meanings = {}
    for i, w in enumerate(words):
        if w == 'ما':
            particle_meanings[i] = disambiguate_ma(words, pos_tags, i)
        elif w == 'إن':
            particle_meanings[i] = disambiguate_in(words, pos_tags, i)
        elif w == 'لا':
            particle_meanings[i] = disambiguate_la(words, pos_tags, i)
        elif w == 'قد':
            particle_meanings[i] = disambiguate_qad(words, pos_tags, i)

    return DeepAnalysisResult(
        emphasis=emphasis,
        interrogative=interrogative,
        negation_strength=neg_result,
        conditional=cond_result,
        particle_meanings=particle_meanings,
    )
