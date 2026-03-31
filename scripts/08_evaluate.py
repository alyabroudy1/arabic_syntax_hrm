#!/usr/bin/env python3
"""
Script 08: Evaluate Arabic Syntax Models
==========================================

Comprehensive evaluation of HRM and hybrid model performance.

Metrics:
    - Case accuracy (iʻrāb correctness)
    - UAS (Unlabeled Attachment Score — dependency heads)
    - LAS (Labeled Attachment Score — heads + relations)
    - Per-category breakdown by grammar pattern

Usage:
    python scripts/08_evaluate.py
    python scripts/08_evaluate.py --model-path models/hrm_arabic_syntax/best_hrm.pt
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "arabic_syntax_grid"
EVAL_DIR = PROJECT_ROOT / "eval"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def evaluate_hrm(model_path: str, data_dir: str = str(DATA_DIR)):
    """Evaluate HRM model on test set."""
    
    # Import model class
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_hrm",
        str(PROJECT_ROOT / "scripts" / "06_train_hrm.py")
    )
    hrm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hrm_module)
    
    data_path = Path(data_dir)
    
    # Load test data
    print("Loading test data...")
    test_grids = torch.from_numpy(np.load(data_path / "test_grids.npy")).to(DEVICE)
    test_masks = torch.from_numpy(np.load(data_path / "test_masks.npy")).to(DEVICE)
    test_solutions = torch.from_numpy(np.load(data_path / "test_solutions.npy")).to(DEVICE)
    
    with open(data_path / "test_texts.json", "r", encoding="utf-8") as f:
        test_texts = json.load(f)
    
    print(f"  Test set: {len(test_texts)} examples")
    
    # Load model config
    config_path = Path(model_path).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            'grid_rows': 32, 'grid_cols': 8, 'vocab_size': 512,
            'hidden_dim': 256, 'manager_dim': 128, 'worker_dim': 128,
            'embed_dim': 64, 'manager_steps': 8, 'worker_steps': 4,
        }
    
    # Load model
    model = hrm_module.ArabicSyntaxHRM(
        grid_rows=config['grid_rows'],
        grid_cols=config['grid_cols'],
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        manager_dim=config['manager_dim'],
        worker_dim=config['worker_dim'],
        embed_dim=config.get('embed_dim', 64),
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print(f"  Model loaded from: {model_path}")
    
    # Run inference
    BATCH = 64
    all_case_preds = []
    all_head_preds = []
    all_deprel_preds = []
    
    with torch.no_grad():
        for i in range(0, len(test_grids), BATCH):
            batch_g = test_grids[i:i+BATCH]
            batch_m = test_masks[i:i+BATCH]
            
            _, final_pred = model(
                batch_g, batch_m,
                config.get('manager_steps', 8),
                config.get('worker_steps', 4)
            )
            
            if 3 in final_pred:
                all_case_preds.append(final_pred[3].argmax(dim=-1).cpu())
            if 4 in final_pred:
                all_head_preds.append(final_pred[4].argmax(dim=-1).cpu())
            if 5 in final_pred:
                all_deprel_preds.append(final_pred[5].argmax(dim=-1).cpu())
    
    test_masks_cpu = test_masks.cpu()
    test_solutions_cpu = test_solutions.cpu()
    
    # Compute metrics
    metrics = {}
    
    if all_case_preds:
        case_preds = torch.cat(all_case_preds, dim=0)
        case_true = test_solutions_cpu[:, :, 3]
        case_correct = ((case_preds == case_true) * test_masks_cpu).sum().item()
        case_total = test_masks_cpu.sum().item()
        metrics['case_accuracy'] = case_correct / max(case_total, 1)
    
    if all_head_preds:
        head_preds = torch.cat(all_head_preds, dim=0)
        head_true = test_solutions_cpu[:, :, 4]
        head_correct = ((head_preds == head_true) * test_masks_cpu).sum().item()
        head_total = test_masks_cpu.sum().item()
        metrics['UAS'] = head_correct / max(head_total, 1)
    
    if all_deprel_preds:
        deprel_preds = torch.cat(all_deprel_preds, dim=0)
        deprel_true = test_solutions_cpu[:, :, 5]
        deprel_correct = ((deprel_preds == deprel_true) * test_masks_cpu).sum().item()
        deprel_total = test_masks_cpu.sum().item()
        metrics['deprel_accuracy'] = deprel_correct / max(deprel_total, 1)
    
    if all_head_preds and all_deprel_preds:
        head_preds = torch.cat(all_head_preds, dim=0)
        deprel_preds = torch.cat(all_deprel_preds, dim=0)
        both_correct = (
            (head_preds == test_solutions_cpu[:, :, 4]) &
            (deprel_preds == test_solutions_cpu[:, :, 5])
        ).float() * test_masks_cpu.float()
        metrics['LAS'] = both_correct.sum().item() / max(test_masks_cpu.sum().item(), 1)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"HRM EVALUATION RESULTS")
    print(f"{'='*50}")
    
    targets = {
        'case_accuracy': 0.90,
        'UAS': 0.85,
        'LAS': 0.80,
        'deprel_accuracy': 0.80,
    }
    
    for name, value in metrics.items():
        target = targets.get(name, 0)
        status = '✅' if value >= target else '❌'
        print(f"  {status} {name}: {value:.4f} ({value*100:.1f}%) "
              f"[target: {target*100:.0f}%]")
    
    # Save results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'model_path': str(model_path),
        'test_size': len(test_texts),
        'metrics': metrics,
        'config': config,
    }
    with open(EVAL_DIR / "hrm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {EVAL_DIR / 'hrm_results.json'}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Arabic Syntax Models")
    parser.add_argument("--model-path", 
                         default=str(PROJECT_ROOT / "models" / "hrm_arabic_syntax" / "best_hrm.pt"))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()
    
    evaluate_hrm(args.model_path, args.data_dir)


if __name__ == "__main__":
    main()
