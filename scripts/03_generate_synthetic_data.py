#!/usr/bin/env python3
"""
Script 03: Generate Synthetic Arabic Syntax Training Data
==========================================================

Generates high-quality Arabic syntax instruction-response pairs
using a teacher model (Qwen2.5-3B or any available LLM).

This script requires a GPU with 12GB+ VRAM for the teacher model.
For CPU-only environments, it generates template-based data instead.

Usage:
    # Template-based (no GPU needed)
    python scripts/03_generate_synthetic_data.py --mode template --count 5000
    
    # LLM-based (requires GPU + transformers)
    python scripts/03_generate_synthetic_data.py --mode llm --count 20000
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ─────────────────────────────────────────────
# Arabic Grammar Knowledge Base
# ─────────────────────────────────────────────

# Seed sentences with full iʻrāb annotations
ANNOTATED_SENTENCES = [
    {
        "sentence": "ذهبَ الطالبُ إلى المدرسةِ",
        "type": "جملة فعلية",
        "analysis": [
            {"word": "ذهبَ", "type": "فعل ماضٍ", "case": "مبني على الفتح", "role": "فعل الجملة"},
            {"word": "الطالبُ", "type": "اسم", "case": "مرفوع بالضمة", "role": "فاعل"},
            {"word": "إلى", "type": "حرف جر", "case": "مبني على السكون", "role": "حرف جر"},
            {"word": "المدرسةِ", "type": "اسم", "case": "مجرور بالكسرة", "role": "اسم مجرور"},
        ],
    },
    {
        "sentence": "قرأَ الأستاذُ الكتابَ الجديدَ",
        "type": "جملة فعلية",
        "analysis": [
            {"word": "قرأَ", "type": "فعل ماضٍ", "case": "مبني على الفتح", "role": "فعل الجملة"},
            {"word": "الأستاذُ", "type": "اسم", "case": "مرفوع بالضمة", "role": "فاعل"},
            {"word": "الكتابَ", "type": "اسم", "case": "منصوب بالفتحة", "role": "مفعول به"},
            {"word": "الجديدَ", "type": "صفة", "case": "منصوب بالفتحة", "role": "نعت للمفعول به"},
        ],
    },
    {
        "sentence": "إنَّ العلمَ نورٌ",
        "type": "جملة اسمية (إنّ وأخواتها)",
        "analysis": [
            {"word": "إنَّ", "type": "حرف ناسخ", "case": "مبني على الفتح", "role": "حرف توكيد ونصب"},
            {"word": "العلمَ", "type": "اسم", "case": "منصوب بالفتحة", "role": "اسم إنّ"},
            {"word": "نورٌ", "type": "اسم", "case": "مرفوع بالضمة", "role": "خبر إنّ"},
        ],
    },
    {
        "sentence": "كانَ الجوُّ جميلاً",
        "type": "جملة اسمية (كان وأخواتها)",
        "analysis": [
            {"word": "كانَ", "type": "فعل ماضٍ ناقص", "case": "مبني على الفتح", "role": "فعل ناسخ"},
            {"word": "الجوُّ", "type": "اسم", "case": "مرفوع بالضمة", "role": "اسم كان"},
            {"word": "جميلاً", "type": "صفة", "case": "منصوب بالفتحة", "role": "خبر كان"},
        ],
    },
    {
        "sentence": "يدرسُ الطلابُ في المكتبةِ الكبيرةِ",
        "type": "جملة فعلية",
        "analysis": [
            {"word": "يدرسُ", "type": "فعل مضارع", "case": "مرفوع بالضمة", "role": "فعل الجملة"},
            {"word": "الطلابُ", "type": "اسم", "case": "مرفوع بالضمة", "role": "فاعل"},
            {"word": "في", "type": "حرف جر", "case": "مبني على السكون", "role": "حرف جر"},
            {"word": "المكتبةِ", "type": "اسم", "case": "مجرور بالكسرة", "role": "اسم مجرور"},
            {"word": "الكبيرةِ", "type": "صفة", "case": "مجرور بالكسرة", "role": "نعت"},
        ],
    },
    {
        "sentence": "أعطى المعلمُ الطالبَ جائزةً",
        "type": "جملة فعلية (فعل متعدٍّ لمفعولين)",
        "analysis": [
            {"word": "أعطى", "type": "فعل ماضٍ", "case": "مبني على الفتح المقدر", "role": "فعل الجملة"},
            {"word": "المعلمُ", "type": "اسم", "case": "مرفوع بالضمة", "role": "فاعل"},
            {"word": "الطالبَ", "type": "اسم", "case": "منصوب بالفتحة", "role": "مفعول به أول"},
            {"word": "جائزةً", "type": "اسم", "case": "منصوب بالفتحة", "role": "مفعول به ثانٍ"},
        ],
    },
    {
        "sentence": "لم يذهبْ أحمدُ إلى السوقِ",
        "type": "جملة فعلية (مضارع مجزوم)",
        "analysis": [
            {"word": "لم", "type": "حرف جزم", "case": "مبني على السكون", "role": "حرف نفي وجزم"},
            {"word": "يذهبْ", "type": "فعل مضارع", "case": "مجزوم بالسكون", "role": "فعل مضارع مجزوم"},
            {"word": "أحمدُ", "type": "اسم علم", "case": "مرفوع بالضمة", "role": "فاعل"},
            {"word": "إلى", "type": "حرف جر", "case": "مبني على السكون", "role": "حرف جر"},
            {"word": "السوقِ", "type": "اسم", "case": "مجرور بالكسرة", "role": "اسم مجرور"},
        ],
    },
    {
        "sentence": "الكتابُ مفيدٌ",
        "type": "جملة اسمية",
        "analysis": [
            {"word": "الكتابُ", "type": "اسم", "case": "مرفوع بالضمة", "role": "مبتدأ"},
            {"word": "مفيدٌ", "type": "صفة", "case": "مرفوع بالضمة", "role": "خبر"},
        ],
    },
    {
        "sentence": "رأيتُ طالباتٍ مجتهداتٍ",
        "type": "جملة فعلية (جمع مؤنث سالم)",
        "analysis": [
            {"word": "رأيتُ", "type": "فعل ماضٍ", "case": "مبني على السكون", "role": "فعل + فاعل (التاء)"},
            {"word": "طالباتٍ", "type": "اسم", "case": "منصوب بالكسرة (جمع مؤنث سالم)", "role": "مفعول به"},
            {"word": "مجتهداتٍ", "type": "صفة", "case": "منصوب بالكسرة (جمع مؤنث سالم)", "role": "نعت"},
        ],
    },
    {
        "sentence": "جاءَ المعلمونَ الجددُ",
        "type": "جملة فعلية (جمع مذكر سالم)",
        "analysis": [
            {"word": "جاءَ", "type": "فعل ماضٍ", "case": "مبني على الفتح", "role": "فعل الجملة"},
            {"word": "المعلمونَ", "type": "اسم", "case": "مرفوع بالواو (جمع مذكر سالم)", "role": "فاعل"},
            {"word": "الجددُ", "type": "صفة", "case": "مرفوع بالضمة", "role": "نعت"},
        ],
    },
]

# Instruction templates
TEMPLATES = {
    "full_irab": {
        "instruction": "أعرب الجملة التالية إعراباً كاملاً مع بيان نوع الجملة.",
        "format": "analysis",
    },
    "case_explain": {
        "instruction": "اشرح علامة الإعراب لكل كلمة في الجملة التالية وبيّن السبب.",
        "format": "case_focus",
    },
    "sentence_type": {
        "instruction": "حدد نوع الجملة التالية (اسمية أم فعلية) وبيّن أركانها.",
        "format": "type_focus",
    },
    "error_correction": {
        "instruction": "صحّح الخطأ النحوي في الجملة التالية إن وُجد، وبيّن سبب الخطأ والتصحيح.",
        "format": "correction",
    },
    "rule_extraction": {
        "instruction": "ما القاعدة النحوية المطبّقة في هذه الجملة؟ اشرحها مع أمثلة.",
        "format": "rule",
    },
}


# ─────────────────────────────────────────────
# Template-Based Generation
# ─────────────────────────────────────────────

def format_analysis(sent_data: Dict) -> str:
    """Format a full iʻrāb analysis as natural text."""
    lines = [f"نوع الجملة: {sent_data['type']}\n"]
    lines.append("الإعراب:")
    for word_data in sent_data['analysis']:
        lines.append(
            f"• {word_data['word']}: {word_data['type']}، "
            f"{word_data['case']}، {word_data['role']}."
        )
    return "\n".join(lines)


def format_case_focus(sent_data: Dict) -> str:
    """Format case-marking focused analysis."""
    lines = [f"تحليل علامات الإعراب في: \"{sent_data['sentence']}\"\n"]
    for word_data in sent_data['analysis']:
        lines.append(
            f"• {word_data['word']}: علامة إعرابه {word_data['case']} "
            f"لأنه {word_data['role']}."
        )
    return "\n".join(lines)


def format_type_focus(sent_data: Dict) -> str:
    """Format sentence-type focused analysis."""
    lines = [
        f"الجملة: \"{sent_data['sentence']}\"",
        f"نوعها: {sent_data['type']}",
        f"",
        f"أركان الجملة:",
    ]
    for word_data in sent_data['analysis']:
        lines.append(f"• {word_data['role']}: {word_data['word']}")
    return "\n".join(lines)


def introduce_error(sent_data: Dict) -> Tuple[str, str]:
    """Introduce a grammar error and explain the correction."""
    sentence = sent_data['sentence']
    # Simple diacritics swaps
    swaps = [
        ("ُ", "َ", "الضمة", "الفتحة", "رفع", "نصب"),
        ("َ", "ِ", "الفتحة", "الكسرة", "نصب", "جر"),
        ("ِ", "ُ", "الكسرة", "الضمة", "جر", "رفع"),
    ]
    for old, new, old_name, new_name, old_case, new_case in swaps:
        if old in sentence:
            corrupted = sentence.replace(old, new, 1)
            explanation = (
                f"الخطأ: تم وضع {new_name} بدلاً من {old_name}.\n"
                f"السبب: الكلمة يجب أن تكون في حالة {old_case} وليس {new_case}.\n"
                f"التصحيح: {sentence}"
            )
            return corrupted, explanation
    return sentence, "لا يوجد خطأ نحوي في هذه الجملة."


def generate_template_data(count: int, output_path: Path):
    """Generate training data using templates (no LLM needed)."""
    
    print(f"Generating {count} template-based examples...")
    
    examples = []
    formatters = {
        "analysis": format_analysis,
        "case_focus": format_case_focus,
        "type_focus": format_type_focus,
    }
    
    for i in range(count):
        sent_data = random.choice(ANNOTATED_SENTENCES)
        template_key = random.choice(list(TEMPLATES.keys()))
        template = TEMPLATES[template_key]
        
        if template_key == "error_correction":
            corrupted, explanation = introduce_error(sent_data)
            example = {
                "instruction": template["instruction"],
                "input": corrupted,
                "output": explanation,
                "grammar_pattern": sent_data["type"],
                "source": "template",
            }
        elif template_key == "rule_extraction":
            output = (
                f"القاعدة النحوية في الجملة \"{sent_data['sentence']}\":\n\n"
                f"نوع الجملة: {sent_data['type']}\n\n"
                f"القاعدة: "
            )
            # Add rule based on sentence type
            if "فعلية" in sent_data["type"]:
                output += "الجملة الفعلية تبدأ بفعل ويأتي بعده الفاعل مرفوعاً والمفعول به منصوباً."
            elif "إنّ" in sent_data["type"]:
                output += "إنّ وأخواتها تنصب المبتدأ ويُسمّى اسمها، وترفع الخبر ويُسمّى خبرها."
            elif "كان" in sent_data["type"]:
                output += "كان وأخواتها ترفع المبتدأ ويُسمّى اسمها، وتنصب الخبر ويُسمّى خبرها."
            else:
                output += "الجملة الاسمية تتكون من مبتدأ مرفوع وخبر مرفوع."
            
            example = {
                "instruction": template["instruction"],
                "input": sent_data["sentence"],
                "output": output,
                "grammar_pattern": sent_data["type"],
                "source": "template",
            }
        else:
            formatter = formatters.get(template["format"], format_analysis)
            output = formatter(sent_data)
            example = {
                "instruction": template["instruction"],
                "input": sent_data["sentence"],
                "output": output,
                "grammar_pattern": sent_data["type"],
                "source": "template",
            }
        
        examples.append(example)
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i+1}/{count}")
    
    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"  ✅ Saved {len(examples)} examples to {output_path}")
    
    # Show sample
    sample = random.choice(examples)
    print(f"\n  Sample:")
    print(f"    Instruction: {sample['instruction']}")
    print(f"    Input: {sample['input']}")
    print(f"    Output: {sample['output'][:200]}...")
    
    return examples


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Arabic syntax data")
    parser.add_argument("--mode", choices=["template", "llm"], default="template",
                         help="Generation mode: template (CPU) or llm (GPU)")
    parser.add_argument("--count", type=int, default=5000,
                         help="Number of examples to generate")
    parser.add_argument("--output", default=str(DATA_DIR / "synthetic_arabic_syntax.jsonl"),
                         help="Output file path")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "template":
        generate_template_data(args.count, output_path)
    elif args.mode == "llm":
        print("LLM-based generation requires GPU + transformers.")
        print("Install: pip install transformers bitsandbytes")
        print("Then run with a teacher model available.")
        # TODO: Implement LLM-based generation (requires GPU)
        pass


if __name__ == "__main__":
    main()
