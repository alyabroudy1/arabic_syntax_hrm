#!/usr/bin/env python3
"""
Script 07: Hybrid Model — LLM + HRM Bridge
=============================================

Implements the hybrid inference pipeline:
1. User inputs Arabic sentence
2. LLM tokenizes and produces intermediate hidden states
3. Bridge module converts hidden states → HRM grid
4. HRM performs hierarchical reasoning on the grid
5. Bridge injects HRM output back into LLM
6. LLM generates final natural-language iʻrāb analysis

Architecture: Hard pipeline first (LLM → grid → HRM → grid → LLM)
Later upgrade to soft cross-attention if needed.

⚠️  REQUIRES: Fine-tuned LLM + trained HRM + GPU

Usage:
    python scripts/07_hybrid_model.py --sentence "ذهب الطالبُ إلى المدرسةِ"
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# ─────────────────────────────────────────────
# LLM ↔ HRM Bridge Module
# ─────────────────────────────────────────────
class LLMToHRMBridge(nn.Module):
    """
    Converts LLM hidden states to HRM grid format and back.
    
    Forward path (LLM → HRM):
        1. Pool LLM hidden states per word (subword → word aggregation)
        2. Project to grid cell values
    
    Backward path (HRM → LLM):
        1. Embed HRM grid predictions 
        2. Project to LLM hidden dimension
        3. Add as residual to LLM hidden states
    """
    
    def __init__(self, llm_hidden_dim: int, grid_rows: int = 32, 
                 grid_cols: int = 8, bridge_dim: int = 256):
        super().__init__()
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # LLM → Grid: per-word feature extraction
        self.to_grid_features = nn.Sequential(
            nn.Linear(llm_hidden_dim, bridge_dim),
            nn.GELU(),
            nn.LayerNorm(bridge_dim),
            nn.Linear(bridge_dim, bridge_dim),
            nn.GELU(),
        )
        
        # Per-column classifiers for grid cells
        self.col_classifiers = nn.ModuleDict({
            '0': nn.Linear(bridge_dim, 256),   # word bucket
            '1': nn.Linear(bridge_dim, 18),    # POS
            '2': nn.Linear(bridge_dim, 64),    # morph pattern
            '3': nn.Linear(bridge_dim, 5),     # case (empty for input)
            '4': nn.Linear(bridge_dim, 33),    # dep head (empty for input)
            '5': nn.Linear(bridge_dim, 32),    # dep rel (empty for input)
            '6': nn.Linear(bridge_dim, 36),    # agreement
            '7': nn.Linear(bridge_dim, 3),     # definiteness
        })
        
        # Grid → LLM: project HRM output back
        self.from_grid = nn.Sequential(
            nn.Linear(5 + 33 + 32, bridge_dim),  # case + head + deprel predictions
            nn.GELU(),
            nn.Linear(bridge_dim, llm_hidden_dim),
        )
    
    def llm_to_grid(self, hidden_states: torch.Tensor, 
                     word_starts: torch.Tensor) -> torch.Tensor:
        """
        Convert LLM hidden states to HRM grid.
        
        Args:
            hidden_states: (B, seq_len, llm_hidden_dim)
            word_starts: (B, seq_len) — 1 at word boundary positions
        
        Returns:
            grid: (B, grid_rows, grid_cols) int tensor
        """
        B, S, D = hidden_states.shape
        
        # Extract word-level representations
        word_reprs = torch.zeros(B, self.grid_rows, D, device=hidden_states.device)
        
        for b in range(B):
            word_idx = 0
            for s in range(S):
                if word_starts[b, s] > 0 and word_idx < self.grid_rows:
                    word_reprs[b, word_idx] = hidden_states[b, s]
                    word_idx += 1
        
        # Project to grid features
        features = self.to_grid_features(word_reprs)  # (B, R, bridge_dim)
        
        # Classify each column
        grid = torch.zeros(B, self.grid_rows, self.grid_cols, 
                          dtype=torch.long, device=hidden_states.device)
        
        for col_str, classifier in self.col_classifiers.items():
            col_idx = int(col_str)
            col_logits = classifier(features)  # (B, R, V)
            
            if col_idx in [3, 4, 5]:
                # Masked columns — set to 0 (HRM will predict these)
                grid[:, :, col_idx] = 0
            else:
                # Given columns — use predicted values
                grid[:, :, col_idx] = col_logits.argmax(dim=-1)
        
        return grid
    
    def grid_to_llm(self, hrm_predictions: Dict[int, torch.Tensor],
                     word_starts: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Convert HRM predictions back to LLM hidden-state injection.
        
        Args:
            hrm_predictions: dict {col_idx: (B, R, V)} — HRM output logits
            word_starts: (B, seq_len) — word boundaries
            seq_len: original sequence length
        
        Returns:
            injection: (B, seq_len, llm_hidden_dim) — add to LLM states
        """
        B = word_starts.shape[0]
        
        # Concatenate predicted column logits as soft features
        pred_features = []
        for col_idx in [3, 4, 5]:
            if col_idx in hrm_predictions:
                # Use softmax probabilities (soft)
                probs = torch.softmax(hrm_predictions[col_idx], dim=-1)
                pred_features.append(probs)
        
        if not pred_features:
            return torch.zeros(B, seq_len, self.from_grid[-1].out_features,
                             device=word_starts.device)
        
        combined = torch.cat(pred_features, dim=-1)  # (B, R, 5+33+32)
        word_features = self.from_grid(combined)  # (B, R, llm_hidden_dim)
        
        # Scatter back to token positions
        D_out = word_features.shape[-1]
        injection = torch.zeros(B, seq_len, D_out, device=word_starts.device)
        
        for b in range(B):
            word_idx = 0
            for s in range(seq_len):
                if word_starts[b, s] > 0 and word_idx < self.grid_rows:
                    injection[b, s] = word_features[b, word_idx]
                    word_idx += 1
        
        return injection


# ─────────────────────────────────────────────
# Hybrid Inference (Standalone / No LLM needed)
# ─────────────────────────────────────────────
class StandaloneHRMAnalyzer:
    """
    Standalone Arabic syntax analyzer using only HRM.
    Does not require the LLM — uses the grid encoding directly.
    
    Useful for testing HRM independently before hybrid integration.
    """
    
    CASE_NAMES_AR = {0: '—', 1: 'مرفوع', 2: 'منصوب', 3: 'مجرور', 4: 'مجزوم'}
    DEP_NAMES_AR = {
        0: '—', 1: 'جذر', 2: 'فاعل', 3: 'مفعول به', 4: 'مفعول غير مباشر',
        5: 'متعلق', 6: 'حال', 7: 'نعت', 8: 'مضاف إليه', 9: 'أداة تعريف',
        10: 'حرف جر', 11: 'عطف', 12: 'حرف عطف', 13: 'ترقيم', 14: 'بدل',
        15: 'مركب', 16: 'بدل', 17: 'صلة الموصول', 18: 'ظرف', 19: 'مبتدأ/خبر',
        20: 'أداة', 21: 'تابع', 22: 'اعتراض', 23: 'ثابت', 24: 'نداء',
    }
    POS_NAMES_AR = {
        0: '—', 1: 'اسم', 2: 'فعل', 3: 'صفة', 4: 'ظرف', 5: 'حرف جر',
        6: 'حرف عطف', 7: 'حرف شرط/مصدري', 8: 'حرف', 9: 'أداة تعريف',
        10: 'ضمير', 11: 'اسم علم', 12: 'عدد', 13: 'ترقيم', 14: 'فعل مساعد',
        15: 'تعجب', 16: 'أخرى',
    }
    
    def __init__(self, hrm_model, config):
        self.model = hrm_model
        self.config = config
        self.model.eval()
    
    @classmethod
    def load(cls, model_dir: str):
        """Load from saved model directory."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_hrm",
            str(PROJECT_ROOT / "scripts" / "06_train_hrm.py")
        )
        hrm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hrm_module)
        
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        
        with open(config_path) as f:
            config = json.load(f)
        
        model = hrm_module.ArabicSyntaxHRM(
            grid_rows=config['grid_rows'],
            grid_cols=config['grid_cols'],
            vocab_size=config['vocab_size'],
            hidden_dim=config['hidden_dim'],
            manager_dim=config['manager_dim'],
            worker_dim=config['worker_dim'],
            embed_dim=config.get('embed_dim', 64),
        ).to(DEVICE)
        
        model.load_state_dict(torch.load(
            model_dir / "best_hrm.pt", map_location=DEVICE
        ))
        
        return cls(model, config)
    
    def analyze_grid(self, grid: np.ndarray, mask: np.ndarray) -> List[Dict]:
        """
        Analyze a pre-encoded grid.
        
        Args:
            grid: (R, C) numpy array — input grid
            mask: (R,) numpy array — word mask
        
        Returns:
            List of per-word analysis dicts
        """
        grid_t = torch.from_numpy(grid).unsqueeze(0).to(DEVICE)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            _, final_pred = self.model(
                grid_t, mask_t,
                self.config.get('manager_steps', 8),
                self.config.get('worker_steps', 4)
            )
        
        results = []
        num_words = int(mask.sum())
        
        for i in range(num_words):
            word_info = {
                'index': i,
                'pos': self.POS_NAMES_AR.get(int(grid[i, 1]), '?'),
            }
            
            if 3 in final_pred:
                case_pred = final_pred[3][0, i].argmax().item()
                word_info['case'] = self.CASE_NAMES_AR.get(case_pred, '?')
            
            if 4 in final_pred:
                head_pred = final_pred[4][0, i].argmax().item()
                word_info['head'] = head_pred
            
            if 5 in final_pred:
                deprel_pred = final_pred[5][0, i].argmax().item()
                word_info['relation'] = self.DEP_NAMES_AR.get(deprel_pred, '?')
            
            results.append(word_info)
        
        return results


def demo_standalone():
    """Demo: analyze a test grid from the dataset."""
    
    data_dir = PROJECT_ROOT / "data" / "arabic_syntax_grid"
    model_dir = PROJECT_ROOT / "models" / "hrm_arabic_syntax"
    
    if not (model_dir / "best_hrm.pt").exists():
        print("❌ HRM model not found. Run scripts/06_train_hrm.py first.")
        return
    
    # Load analyzer
    analyzer = StandaloneHRMAnalyzer.load(str(model_dir))
    
    # Load test examples
    test_grids = np.load(data_dir / "test_grids.npy")
    test_masks = np.load(data_dir / "test_masks.npy")
    test_solutions = np.load(data_dir / "test_solutions.npy")
    
    with open(data_dir / "test_texts.json", "r", encoding="utf-8") as f:
        test_texts = json.load(f)
    
    # Analyze first 5 test sentences
    print(f"\n{'='*60}")
    print(f"HRM Standalone Analysis Demo")
    print(f"{'='*60}")
    
    for i in range(min(5, len(test_texts))):
        print(f"\n  Input: {test_texts[i]}")
        
        results = analyzer.analyze_grid(test_grids[i], test_masks[i])
        
        print(f"  {'#':>3} {'POS':>10} {'Case':>10} {'Head':>5} {'Relation':>15}")
        print(f"  {'─'*3} {'─'*10} {'─'*10} {'─'*5} {'─'*15}")
        
        for r in results:
            print(f"  {r['index']:>3} {r.get('pos', '?'):>10} "
                  f"{r.get('case', '?'):>10} {r.get('head', '?'):>5} "
                  f"{r.get('relation', '?'):>15}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid LLM+HRM Arabic Syntax")
    parser.add_argument("--demo", action="store_true", help="Run standalone HRM demo")
    parser.add_argument("--sentence", type=str, help="Arabic sentence to analyze")
    args = parser.parse_args()
    
    if args.demo:
        demo_standalone()
    else:
        print("Usage:")
        print("  python scripts/07_hybrid_model.py --demo")
        print("  python scripts/07_hybrid_model.py --sentence 'ذهب الطالبُ إلى المدرسةِ'")


if __name__ == "__main__":
    main()
