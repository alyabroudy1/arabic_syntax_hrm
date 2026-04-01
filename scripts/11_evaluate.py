#!/usr/bin/env python3
"""
Evaluate HRM-Grid Parser V2 on the test set.
Reports UAS, LAS, Case accuracy with proper dep_mask.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.v2.parser import ArabicHRMGridParserV2, ParserConfig

DATA_DIR = PROJECT_ROOT / "data" / "arabic_syntax_grid"
MODEL_DIR = PROJECT_ROOT / "models" / "v2_arabic_syntax"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Reuse dataset class from training
from scripts import __path__ as _
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import the dataset directly
import importlib.util
spec = importlib.util.spec_from_file_location("train", PROJECT_ROOT / "scripts" / "10_train_v2.py")
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
RealArabicV2Dataset = train_mod.RealArabicV2Dataset


def evaluate_split(model, split_name, data_dir, device):
    """Evaluate on a data split, return metrics."""
    dataset = RealArabicV2Dataset(data_dir, split=split_name, max_len=32, device=device)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    total_head = 0
    total_las = 0
    total_case = 0
    total_dep = 0
    total_tok = 0
    total_diac_correct = 0
    total_diac_chars = 0
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch, epoch=0, training=False)
            
            pred_heads = out['pred_heads']
            pred_rels = out['pred_rels']
            pred_cases = out['pred_cases']
            pred_diacs = out.get('pred_diacs')
            
            gold_heads = batch['heads']
            gold_rels = batch['relations']
            gold_cases = batch['cases']
            m = batch['mask'].bool()
            dm = batch.get('dep_mask', batch['mask']).bool()
            
            total_head += ((pred_heads == gold_heads) & dm).sum().item()
            total_las += ((pred_heads == gold_heads) & (pred_rels == gold_rels) & dm).sum().item()
            total_case += ((pred_cases == gold_cases) & m).sum().item()
            total_dep += dm.sum().item()
            total_tok += m.sum().item()
            
            # Diac accuracy
            diac_mask = batch.get('diac_mask')
            diac_labels = batch.get('diac_labels')
            if pred_diacs is not None and diac_mask is not None and diac_labels is not None:
                dm2 = diac_mask.bool()
                total_diac_correct += ((pred_diacs == diac_labels) & dm2).sum().item()
                total_diac_chars += dm2.sum().item()
    
    uas = total_head / max(total_dep, 1) * 100
    las = total_las / max(total_dep, 1) * 100
    case = total_case / max(total_tok, 1) * 100
    diac = total_diac_correct / max(total_diac_chars, 1) * 100
    
    return {
        'uas': uas, 'las': las, 'case': case, 'diac': diac,
        'n_sentences': len(dataset), 'n_tokens': total_tok, 'n_deps': total_dep
    }


def main():
    print(f"\n{'='*60}")
    print(f"HRM-Grid Parser V2 — Final Evaluation")
    print(f"{'='*60}")
    
    # Load model
    ckpt_path = MODEL_DIR / "best_model.pt"
    print(f"\nLoading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    config = ParserConfig(
        word_dim=384, n_heads=6, n_transformer_layers=3,
        n_gnn_rounds=3, n_relations=50, n_cases=4,
    )
    model = ArabicHRMGridParserV2(config).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    
    train_uas = ckpt.get('uas', '?')
    train_las = ckpt.get('las', '?')
    train_epoch = ckpt.get('epoch', '?')
    print(f"Checkpoint: epoch={train_epoch}, dev UAS={train_uas}%, dev LAS={train_las}%")
    
    # Evaluate on all splits
    for split in ['dev', 'test']:
        print(f"\n--- {split.upper()} SET ---")
        metrics = evaluate_split(model, split, DATA_DIR, DEVICE)
        print(f"  Sentences: {metrics['n_sentences']}")
        print(f"  Tokens:    {metrics['n_tokens']} (deps: {metrics['n_deps']})")
        print(f"  UAS:       {metrics['uas']:.2f}%")
        print(f"  LAS:       {metrics['las']:.2f}%")
        print(f"  Case:      {metrics['case']:.2f}%")
        print(f"  Diac:      {metrics['diac']:.2f}%")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
