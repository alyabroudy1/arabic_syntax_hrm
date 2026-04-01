#!/usr/bin/env python3
"""
Script 09: Export HRM-Grid Parser V2 to ONNX for Android
=========================================================

Loads the best trained V2 model and exports it to ONNX format
suitable for ONNX Runtime on Android.

Usage:
    python scripts/09_export_android.py
    python scripts/09_export_android.py --test  # verify against PyTorch output
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.v2.parser import ArabicHRMGridParserV2, ParserConfig

OUTPUT_DIR = PROJECT_ROOT / "android" / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "v2_arabic_syntax"


class HRMv2ExportWrapper(nn.Module):
    """Thin wrapper around the parser for ONNX export.
    
    Flattens the input format to simple tensors (no dict) and
    returns only the prediction outputs needed at inference time.
    """
    
    def __init__(self, parser: ArabicHRMGridParserV2):
        super().__init__()
        self.parser = parser
    
    def forward(
        self,
        word_ids: torch.LongTensor,       # (B, W)
        pos_tags: torch.LongTensor,       # (B, W)
        char_ids: torch.LongTensor,       # (B, W, C)
        bpe_ids: torch.LongTensor,        # (B, W, S)
        root_ids: torch.LongTensor,       # (B, W)
        pattern_ids: torch.LongTensor,    # (B, W)
        proclitic_ids: torch.LongTensor,  # (B, W)
        enclitic_ids: torch.LongTensor,   # (B, W)
        diac_ids: torch.LongTensor,       # (B, W, C)
        mask: torch.LongTensor,           # (B, W)
    ):
        batch = {
            'word_ids': word_ids,
            'pos_tags': pos_tags,
            'char_ids': char_ids,
            'bpe_ids': bpe_ids,
            'root_ids': root_ids,
            'pattern_ids': pattern_ids,
            'proclitic_ids': proclitic_ids,
            'enclitic_ids': enclitic_ids,
            'diac_ids': diac_ids,
            'mask': mask,
        }
        
        out = self.parser(batch, epoch=0, training=False)
        
        # Return: pred_heads (B, W), pred_rels (B, W), pred_cases (B, W)
        return out['pred_heads'], out['pred_rels'], out['pred_cases']


def export_onnx(checkpoint_path: Path = None):
    """Export trained V2 parser to ONNX."""
    
    print(f"\n{'='*50}")
    print(f"Exporting HRM-Grid Parser V2 to ONNX")
    print(f"{'='*50}")
    
    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
    
    if not checkpoint_path.exists():
        print(f"  ❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"  📦 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Recreate config (must match training config)
    config = ParserConfig(
        word_dim=384,
        n_heads=6,
        n_transformer_layers=3,
        n_gnn_rounds=3,
        n_relations=50,
        n_cases=4,
    )
    
    model = ArabicHRMGridParserV2(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    uas = ckpt.get('uas', 'N/A')
    las = ckpt.get('las', 'N/A')
    epoch = ckpt.get('epoch', 'N/A')
    print(f"  📊 Checkpoint epoch={epoch}, UAS={uas}%, LAS={las}%")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  📐 Parameters: {num_params/1e6:.2f}M")
    
    # Wrap for ONNX
    wrapper = HRMv2ExportWrapper(model)
    wrapper.eval()
    
    # Create dummy inputs matching training shapes
    B, W, C, S = 1, 32, 8, 4  # batch, seq_len, char_len, bpe_len
    dummy_inputs = (
        torch.randint(1, 100, (B, W)),   # word_ids
        torch.randint(1, 20, (B, W)),    # pos_tags
        torch.randint(1, 100, (B, W, C)), # char_ids
        torch.randint(1, 100, (B, W, S)), # bpe_ids
        torch.randint(1, 100, (B, W)),   # root_ids
        torch.randint(1, 100, (B, W)),   # pattern_ids
        torch.randint(1, 100, (B, W)),   # proclitic_ids
        torch.randint(1, 50, (B, W)),    # enclitic_ids
        torch.randint(0, 20, (B, W, C)), # diac_ids
        torch.ones(B, W, dtype=torch.long),  # mask
    )
    
    # Test forward pass
    print("  🔄 Testing forward pass...")
    with torch.no_grad():
        heads, rels, cases = wrapper(*dummy_inputs)
    print(f"     Output shapes: heads={heads.shape}, rels={rels.shape}, cases={cases.shape}")
    
    # Export ONNX
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUTPUT_DIR / "arabic_syntax_hrm_v2.onnx"
    
    input_names = [
        "word_ids", "pos_tags", "char_ids", "bpe_ids",
        "root_ids", "pattern_ids", "proclitic_ids", "enclitic_ids",
        "diac_ids", "mask"
    ]
    output_names = ["pred_heads", "pred_rels", "pred_cases"]
    
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
    
    print(f"  📤 Exporting to ONNX (opset 14)...")
    try:
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )
        
        size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"  ✅ ONNX exported: {onnx_path}")
        print(f"     Size: {size_mb:.1f} MB")
        return onnx_path, size_mb
        
    except Exception as e:
        print(f"  ❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_onnx(onnx_path: Path):
    """Verify ONNX model output matches PyTorch."""
    
    print(f"\n{'='*50}")
    print(f"Verifying ONNX Model")
    print(f"{'='*50}")
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠️  pip install onnxruntime to verify")
        return
    
    # Load ONNX
    session = ort.InferenceSession(str(onnx_path))
    
    # Create test input
    B, W, C, S = 1, 32, 8, 4
    np_inputs = {
        "word_ids": np.random.randint(1, 100, (B, W)).astype(np.int64),
        "pos_tags": np.random.randint(1, 20, (B, W)).astype(np.int64),
        "char_ids": np.random.randint(1, 100, (B, W, C)).astype(np.int64),
        "bpe_ids": np.random.randint(1, 100, (B, W, S)).astype(np.int64),
        "root_ids": np.random.randint(1, 100, (B, W)).astype(np.int64),
        "pattern_ids": np.random.randint(1, 100, (B, W)).astype(np.int64),
        "proclitic_ids": np.random.randint(1, 100, (B, W)).astype(np.int64),
        "enclitic_ids": np.random.randint(1, 50, (B, W)).astype(np.int64),
        "diac_ids": np.random.randint(0, 20, (B, W, C)).astype(np.int64),
        "mask": np.ones((B, W), dtype=np.int64),
    }
    
    # Run ONNX
    onnx_out = session.run(None, np_inputs)
    print(f"  ✅ ONNX Runtime inference successful")
    print(f"     pred_heads shape: {onnx_out[0].shape}")
    print(f"     pred_rels shape:  {onnx_out[1].shape}")
    print(f"     pred_cases shape: {onnx_out[2].shape}")
    
    # Compare with PyTorch
    config = ParserConfig(
        word_dim=384, n_heads=6, n_transformer_layers=3,
        n_gnn_rounds=3, n_relations=50, n_cases=4,
    )
    model = ArabicHRMGridParserV2(config)
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    wrapper = HRMv2ExportWrapper(model)
    torch_inputs = tuple(torch.from_numpy(np_inputs[name]) for name in [
        "word_ids", "pos_tags", "char_ids", "bpe_ids",
        "root_ids", "pattern_ids", "proclitic_ids", "enclitic_ids",
        "diac_ids", "mask"
    ])
    
    with torch.no_grad():
        torch_out = wrapper(*torch_inputs)
    
    # Check match
    match_heads = np.array_equal(onnx_out[0], torch_out[0].numpy())
    match_rels = np.array_equal(onnx_out[1], torch_out[1].numpy())
    match_cases = np.array_equal(onnx_out[2], torch_out[2].numpy())
    
    print(f"  {'✅' if match_heads else '❌'} Heads match: {match_heads}")
    print(f"  {'✅' if match_rels else '❌'} Rels match:  {match_rels}")
    print(f"  {'✅' if match_cases else '❌'} Cases match: {match_cases}")


def main():
    parser = argparse.ArgumentParser(description="Export HRM-Grid V2 to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint (default: output/hrm_v2/best_model.pt)")
    parser.add_argument("--test", action="store_true",
                       help="Verify ONNX output matches PyTorch")
    args = parser.parse_args()
    
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    result = export_onnx(ckpt_path)
    
    if result and args.test:
        onnx_path, _ = result
        verify_onnx(onnx_path)
    
    if result:
        onnx_path, size_mb = result
        print(f"\n{'='*50}")
        print(f"DEPLOYMENT SUMMARY")
        print(f"{'='*50}")
        print(f"  Model: {onnx_path}")
        print(f"  Size:  {size_mb:.1f} MB")
        print(f"  Target: < 200 MB")
        print(f"  Status: {'✅ PASS' if size_mb < 200 else '⚠️  LARGE'}")


if __name__ == "__main__":
    main()
