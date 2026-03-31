#!/usr/bin/env python3
"""
Script 06: Train HRM on Arabic Syntax Grids
============================================

Priority 3 — Feasibility check: Can HRM learn Arabic syntax at all?

Architecture (adapted from sapientinc/HRM):
    - Grid Encoder: embeds each cell of the input grid
    - Manager (slow clock): processes global sentence structure every K steps
    - Worker (fast clock): processes word-level features every step
    - Grid Decoder: predicts solution grid from final states
    - Deep Supervision: loss at every recurrent step

Usage:
    # Quick feasibility check (100 sentences, small model)
    python scripts/06_train_hrm.py --quick-test
    
    # Full training
    python scripts/06_train_hrm.py --epochs 5000
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "arabic_syntax_grid"
OUTPUT_DIR = PROJECT_ROOT / "models" / "hrm_arabic_syntax"

GRID_ROWS = 32         # max words
GRID_COLS = 8           # features per word
VOCAB_SIZE = 512        # max cell value
MASKED_COLS = [3, 4, 5]  # case, head, deprel — columns to predict

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# ─────────────────────────────────────────────
# Arabic Syntax HRM Model
# ─────────────────────────────────────────────
class ArabicSyntaxHRM(nn.Module):
    """
    Hierarchical Recurrent Model adapted for Arabic syntax.
    
    Key properties (from HRM paper):
    1. HIERARCHICAL RECURRENCE: Manager (slow) + Worker (fast)
    2. DEEP SUPERVISION: Loss at every recurrent step
    3. ONE-STEP GRADIENTS: Truncated BPTT to single step (stable training)
    4. ITERATIVE REFINEMENT: Output of each step feeds back as input
    """
    
    def __init__(self, grid_rows=32, grid_cols=8, vocab_size=512,
                 hidden_dim=256, manager_dim=128, worker_dim=128,
                 embed_dim=64):
        super().__init__()
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Cell embedding: each grid cell value → dense vector
        self.cell_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        
        # Grid encoder: flatten embedded grid → global representation
        grid_flat_dim = grid_rows * grid_cols * embed_dim
        self.grid_encoder = nn.Sequential(
            nn.Linear(grid_flat_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Row encoder: per-word (row) features → local representation
        row_flat_dim = grid_cols * embed_dim
        self.row_encoder = nn.Sequential(
            nn.Linear(row_flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Manager RNN (slow clock — sentence-level structure)
        self.manager_rnn = nn.GRUCell(hidden_dim, manager_dim)
        self.manager_goal = nn.Linear(manager_dim, worker_dim)
        
        # Worker RNN (fast clock — word-level features)
        self.worker_rnn = nn.GRUCell(
            hidden_dim + worker_dim,  # global context + manager goal
            worker_dim
        )
        
        # Grid decoder: predict solution at each step
        # Produces per-row predictions using worker state + row features
        self.grid_decoder = nn.Sequential(
            nn.Linear(worker_dim + manager_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Per-column prediction heads (only for masked columns)
        self.col_heads = nn.ModuleDict()
        col_vocab_sizes = {
            3: 5,    # case: 0-4 (none, Nom, Acc, Gen, Jus)
            4: grid_rows + 1,  # dep head: 0-32 (pointer to word)
            5: 32,   # dep rel: 0-31
        }
        for col_idx, col_vocab in col_vocab_sizes.items():
            self.col_heads[str(col_idx)] = nn.Linear(hidden_dim, col_vocab)
    
    def forward(self, grid_input, mask, num_manager_steps=8, num_worker_steps=4):
        """
        Args:
            grid_input: (B, R, C) int tensor — input grid with masked cells
            mask: (B, R) int tensor — 1 for real words, 0 for padding
            num_manager_steps: slow clock iterations
            num_worker_steps: fast clock ticks per manager step
            
        Returns:
            all_predictions: list of dicts {col_idx: (B, R, V)} at each step
            final_prediction: dict {col_idx: (B, R, V)} — final solution logits
        """
        B = grid_input.shape[0]
        
        # Embed grid cells: (B, R, C) → (B, R, C, embed_dim)
        cell_embeds = self.cell_embed(grid_input.clamp(0, self.vocab_size))
        
        # Global grid representation: (B, R*C*embed_dim) → (B, hidden)
        flat_embeds = cell_embeds.reshape(B, -1)
        grid_repr = self.grid_encoder(flat_embeds)
        
        # Per-row representations: (B, R, C*embed_dim) → (B, R, hidden)
        row_embeds = cell_embeds.reshape(B, self.grid_rows, -1)
        row_repr = self.row_encoder(row_embeds)
        
        # Initialize RNN states
        manager_state = torch.zeros(B, self.manager_rnn.hidden_size,
                                     device=grid_input.device)
        worker_state = torch.zeros(B, self.worker_rnn.hidden_size,
                                    device=grid_input.device)
        
        all_predictions = []
        
        for m_step in range(num_manager_steps):
            # Manager update (slow clock — one step per outer loop)
            manager_state = self.manager_rnn(grid_repr, manager_state)
            manager_goal = self.manager_goal(manager_state)
            
            for w_step in range(num_worker_steps):
                # Worker update (fast clock)
                worker_input = torch.cat([grid_repr, manager_goal], dim=-1)
                worker_state = self.worker_rnn(worker_input, worker_state)
                
                # Decode: combine worker + manager + row features for per-row prediction
                # Broadcast worker & manager states to each row
                worker_exp = worker_state.unsqueeze(1).expand(-1, self.grid_rows, -1)
                manager_exp = manager_state.unsqueeze(1).expand(-1, self.grid_rows, -1)
                
                combined = torch.cat([worker_exp, manager_exp, row_repr], dim=-1)
                decoded = self.grid_decoder(combined)  # (B, R, hidden)
                
                # Per-column predictions
                step_preds = {}
                for col_str, head in self.col_heads.items():
                    col_logits = head(decoded)  # (B, R, col_vocab)
                    step_preds[int(col_str)] = col_logits
                
                all_predictions.append(step_preds)
                
                # Iterative refinement: update grid with current predictions
                # This is key to HRM — each step refines the solution
                pred_grid = grid_input.clone()
                for col_idx, logits in step_preds.items():
                    pred_vals = logits.argmax(dim=-1)  # (B, R)
                    # Only fill masked positions (where input is 0)
                    is_masked = (grid_input[:, :, col_idx] == 0)
                    pred_grid[:, :, col_idx] = torch.where(
                        is_masked, pred_vals, grid_input[:, :, col_idx]
                    )
                
                # Re-encode with updated grid
                cell_embeds = self.cell_embed(pred_grid.clamp(0, self.vocab_size))
                flat_embeds = cell_embeds.reshape(B, -1)
                grid_repr = self.grid_encoder(flat_embeds)
                row_embeds = cell_embeds.reshape(B, self.grid_rows, -1)
                row_repr = self.row_encoder(row_embeds)
        
        return all_predictions, all_predictions[-1]


# ─────────────────────────────────────────────
# Deep Supervision Loss
# ─────────────────────────────────────────────
def deep_supervision_loss(all_predictions, solution, mask, masked_cols=MASKED_COLS):
    """
    Weighted sum of losses over all recurrent steps.
    Later steps get higher weight (they should be more accurate).
    
    Args:
        all_predictions: list of dicts {col_idx: (B, R, V)} 
        solution: (B, R, C) int tensor — ground truth
        mask: (B, R) int tensor — 1 for real words
        masked_cols: which columns to compute loss on
    """
    total_loss = 0.0
    num_steps = len(all_predictions)
    
    for step_idx, step_preds in enumerate(all_predictions):
        # Linear weight: later steps count more
        weight = (step_idx + 1) / num_steps
        
        step_loss = 0.0
        for col_idx in masked_cols:
            if col_idx not in step_preds:
                continue
            
            col_logits = step_preds[col_idx]  # (B, R, V)
            col_target = solution[:, :, col_idx].long()  # (B, R)
            
            # Flatten
            B, R, V = col_logits.shape
            logits_flat = col_logits.reshape(-1, V)
            target_flat = col_target.reshape(-1)
            mask_flat = mask.reshape(-1).float()
            
            # CE loss with word mask
            ce_loss = F.cross_entropy(logits_flat, target_flat, reduction='none')
            masked_loss = (ce_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            step_loss += masked_loss
        
        total_loss += weight * step_loss
    
    return total_loss / num_steps


# ─────────────────────────────────────────────
# Accuracy Metrics
# ─────────────────────────────────────────────
def compute_accuracy(predictions, solution, mask, masked_cols=MASKED_COLS):
    """Compute per-column and overall accuracy."""
    metrics = {}
    total_correct = 0
    total_count = 0
    
    for col_idx in masked_cols:
        if col_idx not in predictions:
            continue
        
        pred_vals = predictions[col_idx].argmax(dim=-1)  # (B, R)
        true_vals = solution[:, :, col_idx]
        
        correct = ((pred_vals == true_vals) * mask).sum().item()
        count = mask.sum().item()
        
        col_names = {3: 'case', 4: 'head', 5: 'deprel'}
        metrics[col_names.get(col_idx, f'col{col_idx}')] = correct / max(count, 1)
        
        total_correct += correct
        total_count += count
    
    metrics['overall'] = total_correct / max(total_count, 1)
    return metrics


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def train_hrm(args):
    print(f"\n{'='*60}")
    print(f"Training HRM on Arabic Syntax Grids")
    print(f"{'='*60}")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Manager steps: {args.manager_steps}")
    print(f"  Worker steps: {args.worker_steps}")
    print(f"  Hidden dim: {args.hidden_dim}")
    
    # Load data
    print(f"\nLoading data from {DATA_DIR}...")
    train_grids = torch.from_numpy(np.load(DATA_DIR / "train_grids.npy")).to(DEVICE)
    train_masks = torch.from_numpy(np.load(DATA_DIR / "train_masks.npy")).to(DEVICE)
    train_solutions = torch.from_numpy(np.load(DATA_DIR / "train_solutions.npy")).to(DEVICE)
    
    dev_grids = torch.from_numpy(np.load(DATA_DIR / "dev_grids.npy")).to(DEVICE)
    dev_masks = torch.from_numpy(np.load(DATA_DIR / "dev_masks.npy")).to(DEVICE)
    dev_solutions = torch.from_numpy(np.load(DATA_DIR / "dev_solutions.npy")).to(DEVICE)
    
    print(f"  Train: {train_grids.shape[0]} examples, shape {train_grids.shape}")
    print(f"  Dev: {dev_grids.shape[0]} examples, shape {dev_grids.shape}")
    
    # Create model
    model = ArabicSyntaxHRM(
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        vocab_size=VOCAB_SIZE,
        hidden_dim=args.hidden_dim,
        manager_dim=args.hidden_dim // 2,
        worker_dim=args.hidden_dim // 2,
        embed_dim=args.embed_dim,
    ).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Training
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_dev_acc = 0.0
    best_dev_loss = float('inf')
    
    print(f"\nStarting training...\n")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t_start = time.time()
        
        # Shuffle
        perm = torch.randperm(train_grids.shape[0], device=DEVICE)
        
        for start in range(0, train_grids.shape[0], args.batch_size):
            end = min(start + args.batch_size, train_grids.shape[0])
            idx = perm[start:end]
            
            batch_grids = train_grids[idx]
            batch_masks = train_masks[idx]
            batch_solutions = train_solutions[idx]
            
            optimizer.zero_grad()
            
            all_preds, final_pred = model(
                batch_grids, batch_masks,
                args.manager_steps, args.worker_steps
            )
            
            loss = deep_supervision_loss(
                all_preds, batch_solutions, batch_masks
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t_start
        
        # Evaluate periodically
        eval_freq = max(1, args.epochs // 50)  # ~50 eval points
        if (epoch + 1) % eval_freq == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on dev set (or subset if large)
                eval_size = min(512, dev_grids.shape[0])
                all_preds, final_pred = model(
                    dev_grids[:eval_size], dev_masks[:eval_size],
                    args.manager_steps, args.worker_steps
                )
                dev_loss = deep_supervision_loss(
                    all_preds, dev_solutions[:eval_size], dev_masks[:eval_size]
                ).item()
                
                metrics = compute_accuracy(
                    final_pred, dev_solutions[:eval_size], dev_masks[:eval_size]
                )
            
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:>5}/{args.epochs} │ "
                  f"Loss: {avg_loss:.4f} │ "
                  f"Dev Loss: {dev_loss:.4f} │ "
                  f"Case: {metrics.get('case', 0):.3f} │ "
                  f"Head: {metrics.get('head', 0):.3f} │ "
                  f"DepRel: {metrics.get('deprel', 0):.3f} │ "
                  f"Overall: {metrics['overall']:.3f} │ "
                  f"LR: {lr:.2e} │ "
                  f"{elapsed:.1f}s")
            
            # Save best model
            if metrics['overall'] > best_dev_acc:
                best_dev_acc = metrics['overall']
                torch.save(model.state_dict(), OUTPUT_DIR / "best_hrm.pt")
                print(f"           → New best! (overall acc: {best_dev_acc:.4f})")
            
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
    
    # Save final model
    torch.save(model.state_dict(), OUTPUT_DIR / "final_hrm.pt")
    
    # Save training config
    config = {
        'grid_rows': GRID_ROWS, 'grid_cols': GRID_COLS,
        'vocab_size': VOCAB_SIZE, 'hidden_dim': args.hidden_dim,
        'manager_dim': args.hidden_dim // 2, 'worker_dim': args.hidden_dim // 2,
        'embed_dim': args.embed_dim,
        'manager_steps': args.manager_steps, 'worker_steps': args.worker_steps,
        'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
        'best_dev_acc': best_dev_acc, 'best_dev_loss': best_dev_loss,
        'num_params': num_params, 'device': str(DEVICE),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"  Best dev accuracy: {best_dev_acc:.4f} ({best_dev_acc*100:.1f}%)")
    print(f"  Best dev loss: {best_dev_loss:.4f}")
    print(f"  Model saved to: {OUTPUT_DIR}")
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Target metrics
    print(f"\n  Target Metrics:")
    print(f"    Case accuracy > 90%: {'✅' if metrics.get('case', 0) > 0.9 else '❌'} ({metrics.get('case', 0)*100:.1f}%)")
    print(f"    Head accuracy > 85%: {'✅' if metrics.get('head', 0) > 0.85 else '❌'} ({metrics.get('head', 0)*100:.1f}%)")
    print(f"    DepRel accuracy > 80%: {'✅' if metrics.get('deprel', 0) > 0.8 else '❌'} ({metrics.get('deprel', 0)*100:.1f}%)")
    
    return model, config


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train HRM on Arabic Syntax Grids")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--manager-steps", type=int, default=8)
    parser.add_argument("--worker-steps", type=int, default=4)
    parser.add_argument("--quick-test", action="store_true",
                         help="Quick feasibility check (200 epochs, tiny model)")
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("🚀 Quick test mode: 200 epochs, smaller model")
        args.epochs = 200
        args.hidden_dim = 128
        args.embed_dim = 32
        args.batch_size = 64
        args.manager_steps = 4
        args.worker_steps = 2
    
    train_hrm(args)


if __name__ == "__main__":
    main()
