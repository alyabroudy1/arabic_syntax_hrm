#!/usr/bin/env python3
"""
Script 09: Export Models for Android Deployment
=================================================

Exports:
1. LLM → GGUF via llama.cpp (Q4_K_M quantization, ~1.5GB)
2. HRM → ONNX (optimized, ~50MB)

⚠️  LLM export requires: llama.cpp (auto-cloned)
⚠️  HRM export works on CPU/MPS

Usage:
    # Export HRM only (works locally)
    python scripts/09_export_android.py --hrm-only
    
    # Export everything (needs LLM model)
    python scripts/09_export_android.py
"""

import argparse
import os
import subprocess
import torch
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "android" / "models"


def export_hrm_onnx(model_dir: str = "models/hrm_arabic_syntax"):
    """Export trained HRM to ONNX format."""
    
    print(f"\n{'='*50}")
    print(f"Exporting HRM to ONNX")
    print(f"{'='*50}")
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_hrm",
        str(PROJECT_ROOT / "scripts" / "06_train_hrm.py")
    )
    hrm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hrm_module)
    
    model_path = PROJECT_ROOT / model_dir
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        print(f"  ❌ Config not found: {config_path}")
        return None
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Load model
    model = hrm_module.ArabicSyntaxHRM(
        grid_rows=config['grid_rows'],
        grid_cols=config['grid_cols'],
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        manager_dim=config['manager_dim'],
        worker_dim=config['worker_dim'],
        embed_dim=config.get('embed_dim', 64),
    )
    
    best_path = model_path / "best_hrm.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
    else:
        print(f"  ❌ Model not found: {best_path}")
        return None
    
    model.eval()
    
    # Wrap model for ONNX export (fixed manager/worker steps)
    class HRMExportWrapper(torch.nn.Module):
        def __init__(self, hrm, manager_steps=8, worker_steps=4):
            super().__init__()
            self.hrm = hrm
            self.manager_steps = manager_steps
            self.worker_steps = worker_steps
        
        def forward(self, grid_input, mask):
            _, final_pred = self.hrm(
                grid_input, mask,
                self.manager_steps, self.worker_steps
            )
            # Stack predictions into single tensor
            results = []
            for col_idx in sorted(final_pred.keys()):
                results.append(final_pred[col_idx].argmax(dim=-1))
            return torch.stack(results, dim=-1)  # (B, R, num_pred_cols)
    
    wrapper = HRMExportWrapper(
        model,
        config.get('manager_steps', 8),
        config.get('worker_steps', 4)
    )
    
    # Create dummy inputs
    dummy_grid = torch.randint(0, 50, (1, config['grid_rows'], config['grid_cols']))
    dummy_mask = torch.ones(1, config['grid_rows'], dtype=torch.int32)
    
    # Export
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUTPUT_DIR / "arabic_syntax_hrm.onnx"
    
    try:
        torch.onnx.export(
            wrapper,
            (dummy_grid, dummy_mask),
            str(onnx_path),
            input_names=["grid_input", "mask"],
            output_names=["predictions"],
            dynamic_axes={
                "grid_input": {0: "batch"},
                "mask": {0: "batch"},
                "predictions": {0: "batch"},
            },
            opset_version=14,
        )
        
        size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"  ✅ ONNX exported: {onnx_path}")
        print(f"     Size: {size_mb:.1f} MB")
        return size_mb
        
    except Exception as e:
        print(f"  ❌ ONNX export failed: {e}")
        print(f"     This may require: pip install onnx")
        
        # Fallback: save as TorchScript
        ts_path = OUTPUT_DIR / "arabic_syntax_hrm.pt"
        traced = torch.jit.trace(wrapper, (dummy_grid, dummy_mask))
        traced.save(str(ts_path))
        size_mb = os.path.getsize(ts_path) / 1e6
        print(f"  ✅ TorchScript fallback: {ts_path} ({size_mb:.1f} MB)")
        return size_mb


def export_llm_gguf(model_dir: str = "models/qwen_arabic_syntax_merged"):
    """Export fine-tuned LLM to GGUF via llama.cpp."""
    
    print(f"\n{'='*50}")
    print(f"Exporting LLM to GGUF")
    print(f"{'='*50}")
    
    model_path = PROJECT_ROOT / model_dir
    if not model_path.exists():
        print(f"  ❌ Merged model not found: {model_path}")
        print(f"     Run scripts/05_train_llm.py first.")
        return None
    
    # Clone llama.cpp
    llama_dir = PROJECT_ROOT / "llama.cpp"
    if not llama_dir.exists():
        print("  📥 Cloning llama.cpp...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/ggerganov/llama.cpp.git",
            str(llama_dir)
        ], check=True)
    
    # Build
    print("  🔨 Building llama.cpp...")
    subprocess.run(["make", "-j", "-C", str(llama_dir)], check=True)
    
    # Convert
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    f16_path = OUTPUT_DIR / "arabic_syntax_llm.gguf"
    q4_path = OUTPUT_DIR / "arabic_syntax_llm_q4km.gguf"
    
    print("  📦 Converting to GGUF (F16)...")
    subprocess.run([
        "python", str(llama_dir / "convert_hf_to_gguf.py"),
        str(model_path),
        "--outfile", str(f16_path),
        "--outtype", "f16"
    ], check=True)
    
    print("  📦 Quantizing to Q4_K_M...")
    subprocess.run([
        str(llama_dir / "llama-quantize"),
        str(f16_path), str(q4_path), "Q4_K_M"
    ], check=True)
    
    f16_path.unlink(missing_ok=True)
    
    size_mb = os.path.getsize(q4_path) / 1e6
    print(f"  ✅ GGUF exported: {q4_path}")
    print(f"     Size: {size_mb:.0f} MB")
    return size_mb


def print_summary(hrm_size, llm_size):
    """Print deployment size summary."""
    print(f"\n{'='*50}")
    print(f"DEPLOYMENT SIZE SUMMARY")
    print(f"{'='*50}")
    
    if hrm_size:
        print(f"  HRM (ONNX):    {hrm_size:>8.1f} MB")
    if llm_size:
        print(f"  LLM (Q4_K_M):  {llm_size:>8.0f} MB")
    
    total = (hrm_size or 0) + (llm_size or 0)
    if total > 0:
        print(f"  {'─'*30}")
        print(f"  Total:         {total:>8.0f} MB")
        print(f"  Target:        < 2000 MB")
        print(f"  Status:        {'✅ PASS' if total < 2000 else '❌ OVER BUDGET'}")


def main():
    parser = argparse.ArgumentParser(description="Export models for Android")
    parser.add_argument("--hrm-only", action="store_true",
                         help="Only export HRM (no LLM)")
    parser.add_argument("--hrm-dir", default="models/hrm_arabic_syntax")
    parser.add_argument("--llm-dir", default="models/qwen_arabic_syntax_merged")
    args = parser.parse_args()
    
    hrm_size = export_hrm_onnx(args.hrm_dir)
    
    llm_size = None
    if not args.hrm_only:
        llm_size = export_llm_gguf(args.llm_dir)
    
    print_summary(hrm_size, llm_size)


if __name__ == "__main__":
    main()
