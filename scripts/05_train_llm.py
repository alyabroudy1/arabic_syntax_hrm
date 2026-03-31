#!/usr/bin/env python3
"""
Script 05: Fine-tune LLM on Arabic Syntax Data
================================================

Fine-tunes Qwen2.5-3B-Instruct with QLoRA on Arabic syntax data.
Includes HRM-inspired deep supervision on intermediate layers.

⚠️  REQUIRES: NVIDIA GPU with 12GB+ VRAM (RTX 3060 or better)
⚠️  REQUIRES: pip install unsloth bitsandbytes peft trl

Usage:
    python scripts/05_train_llm.py
    python scripts/05_train_llm.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 1
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen_arabic_syntax"


def check_gpu():
    """Check if GPU is available and sufficient."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available. This script requires an NVIDIA GPU.")
            print("   For Mac (MPS), use the HRM training script instead.")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        if gpu_mem < 10:
            print(f"  ⚠️  GPU has only {gpu_mem:.1f} GB VRAM. Recommended: 12GB+")
            print(f"     Consider using Qwen2.5-1.5B instead of 3B.")
        
        return True
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False


def train_llm(args):
    """Fine-tune LLM with Unsloth + QLoRA."""
    
    print("╔══════════════════════════════════════════════════╗")
    print("║  Fine-tune LLM for Arabic Syntax               ║")
    print("╚══════════════════════════════════════════════════╝")
    
    if not check_gpu():
        print("\nTo run on a GPU machine, copy this project and run:")
        print("  pip install unsloth bitsandbytes peft trl transformers")
        print("  python scripts/05_train_llm.py")
        return
    
    # Lazy imports (heavy dependencies)
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    import torch
    
    # ─── Load Model ───
    print(f"\nLoading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # ─── Add LoRA ───
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print(f"  Trainable parameters: {model.print_trainable_parameters()}")
    
    # ─── Load Data ───
    train_data = []
    train_path = DATA_DIR / "train_llm.jsonl"
    if not train_path.exists():
        print(f"  ❌ Training data not found: {train_path}")
        print(f"     Run scripts 03 and 04 first.")
        return
    
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            train_data.append(json.loads(line))
    
    eval_data = []
    eval_path = DATA_DIR / "eval_llm.jsonl"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                eval_data.append(json.loads(line))
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None
    
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 0}")
    
    # ─── Training ───
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=200 if eval_dataset else None,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # ─── Save ───
    print("Saving model...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    # Merge LoRA for export
    print("Merging LoRA weights...")
    merged_dir = str(OUTPUT_DIR) + "_merged"
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    
    print(f"\n✅ Training complete!")
    print(f"   LoRA model: {OUTPUT_DIR}")
    print(f"   Merged model: {merged_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Arabic Syntax")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()
    
    train_llm(args)


if __name__ == "__main__":
    main()
