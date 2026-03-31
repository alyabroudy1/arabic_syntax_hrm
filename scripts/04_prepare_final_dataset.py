#!/usr/bin/env python3
"""
Script 04: Prepare Final Combined Dataset
==========================================

Combines all data sources into unified training format for LLM fine-tuning.
Creates Alpaca-format JSONL files.

Usage:
    python scripts/04_prepare_final_dataset.py
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def create_chat_format(instruction: str, input_text: str, output_text: str) -> Dict:
    """Create chat-style training format for Qwen."""
    if input_text:
        return {
            "messages": [
                {"role": "system", "content": "أنت خبير في النحو العربي الفصيح. تقدم تحليلات نحوية دقيقة ومفصلة."},
                {"role": "user", "content": f"{instruction}\n\n{input_text}"},
                {"role": "assistant", "content": output_text},
            ]
        }
    else:
        return {
            "messages": [
                {"role": "system", "content": "أنت خبير في النحو العربي الفصيح. تقدم تحليلات نحوية دقيقة ومفصلة."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output_text},
            ]
        }


def create_alpaca_format(instruction: str, input_text: str, output_text: str) -> dict:
    """Standard Alpaca instruction format."""
    if input_text:
        return {
            "text": (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output_text}"
            )
        }
    else:
        return {
            "text": (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output_text}"
            )
        }


def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  Prepare Final Combined Dataset                 ║")
    print("╚══════════════════════════════════════════════════╝")
    
    all_examples = []
    
    # 1. Load synthetic data
    synthetic_path = DATA_DIR / "synthetic_arabic_syntax.jsonl"
    if synthetic_path.exists():
        count = 0
        with open(synthetic_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                all_examples.append(
                    create_alpaca_format(
                        item['instruction'],
                        item.get('input', ''),
                        item['output']
                    )
                )
                count += 1
        print(f"  ✅ Synthetic data: {count} examples")
    else:
        print(f"  ⚠️  Synthetic data not found: {synthetic_path}")
        print(f"      Run: python scripts/03_generate_synthetic_data.py")
    
    # 2. Load CIDAR (filter for grammar-relevant)
    cidar_path = DATA_DIR / "cidar_raw.json"
    if cidar_path.exists():
        with open(cidar_path, "r", encoding="utf-8") as f:
            cidar = json.load(f)
        count = 0
        for item in cidar:
            if item.get('output') and len(item['output']) > 20:
                all_examples.append(
                    create_alpaca_format(
                        item['instruction'],
                        item.get('input', ''),
                        item['output']
                    )
                )
                count += 1
        print(f"  ✅ CIDAR data: {count} examples")
    else:
        print(f"  ⚠️  CIDAR not found: {cidar_path}")
    
    if not all_examples:
        print("\n  ❌ No data found! Run scripts 01 and 03 first.")
        return
    
    # 3. Shuffle and split
    random.seed(42)
    random.shuffle(all_examples)
    
    split_idx = int(len(all_examples) * 0.95)
    train_data = all_examples[:split_idx]
    eval_data = all_examples[split_idx:]
    
    # Save
    train_path = DATA_DIR / "train_llm.jsonl"
    eval_path = DATA_DIR / "eval_llm.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n  📊 Final dataset:")
    print(f"     Train: {len(train_data)} examples → {train_path}")
    print(f"     Eval:  {len(eval_data)} examples → {eval_path}")
    
    # Show sample
    sample = random.choice(train_data)
    print(f"\n  Sample (truncated):")
    print(f"    {sample['text'][:300]}...")


if __name__ == "__main__":
    from typing import Dict
    main()
