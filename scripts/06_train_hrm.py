#!/usr/bin/env python3
"""
Script 06: Train HRM on Arabic Syntax Grids
============================================

Architecture Overhaul: Biaffine Dependency Parsing
    - Grid Encoder: Per-column embedding -> word-level encoding
    - Cross-word self-attention (Transformer layer)
    - Biaffine Scorer for Head (UAS) and Relation (LAS)
    - Deep Supervision across manager/worker steps.

Usage:
    # Quick test (256 hidden dim)
    python scripts/06_train_hrm.py --quick-test
    
    # Full training
    python scripts/06_train_hrm.py --epochs 200
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

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class ColumnDropout(nn.Module):
    """Randomly zero out entire feature columns during training."""
    def __init__(self, num_cols=8, drop_prob=0.15):
        super().__init__()
        self.num_cols = num_cols
        self.drop_prob = drop_prob
        # Never drop column 1 (POS). We drop: 0 (word bucket), 2 (morph), 6 (agr), 7 (def)
        self.droppable_cols = [0, 2, 6, 7]
    
    def forward(self, grid):
        if not self.training:
            return grid
        B, R, C = grid.shape
        grid = grid.clone()
        for col in self.droppable_cols:
            if torch.rand(1).item() < self.drop_prob:
                grid[:, :, col] = 0
        return grid


class ArabicSyntaxHRM(nn.Module):
    """
    Improved HRM with Biaffine Parsing and Cross-Word Attention.
    Replaces earlier independent linear projection bottleneck.
    """
    def __init__(
        self,
        grid_rows: int = 32,
        grid_cols: int = 8,
        vocab_size: int = 512,
        embed_dim: int = 32,             # cell_embed_dim
        hidden_dim: int = 256,           # word_dim
        manager_dim: int = 256,
        worker_dim: int = 256,
        arc_dim: int = 128,
        rel_dim: int = 64,
        num_rels: int = 33,
        num_cases: int = 5,
        sa_heads: int = 4,
        sa_layers: int = 1,
        dropout: float = 0.33,
    ):
        super().__init__()
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.vocab_size = vocab_size
        
        # ── Embeddings ──
        self.col_embeds = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for _ in range(grid_cols)
        ])
        self.pos_embed = nn.Embedding(grid_rows, hidden_dim)
        
        # ── Row Encoder ──
        row_input_dim = grid_cols * embed_dim
        self.row_encoder = nn.Sequential(
            nn.Linear(row_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # ── Global Pooling ──
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        
        # ── Manager & Worker ──
        self.manager_gru = nn.GRUCell(hidden_dim, manager_dim)
        self.manager_goal_proj = nn.Linear(manager_dim, worker_dim)
        self.worker_gru = nn.GRUCell(hidden_dim + worker_dim, worker_dim)
        
        # ── Context Fusion ──
        fuse_input_dim = hidden_dim + manager_dim + worker_dim
        self.context_fuse = nn.Sequential(
            nn.Linear(fuse_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # ── Self Attention ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=sa_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.cross_word_attn = nn.TransformerEncoder(encoder_layer, num_layers=sa_layers)
        
        # ── Biaffine Head Scorer ──
        self.root_emb = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        
        self.mlp_arc_dep = nn.Sequential(
            nn.Linear(hidden_dim, arc_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.mlp_arc_head = nn.Sequential(
            nn.Linear(hidden_dim, arc_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        
        self.W_arc = nn.Parameter(torch.empty(arc_dim, arc_dim))
        self.u_arc = nn.Linear(arc_dim, 1, bias=False)
        self.v_arc = nn.Linear(arc_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.W_arc)
        
        max_rel_dist = grid_rows + 1
        self.dist_bias = nn.Embedding(2 * max_rel_dist + 1, 1)
        nn.init.zeros_(self.dist_bias.weight)
        
        # ── Conditioned Relation Scorer ──
        self.mlp_rel_dep = nn.Sequential(
            nn.Linear(hidden_dim, rel_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.mlp_rel_head = nn.Sequential(
            nn.Linear(hidden_dim, rel_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        
        self.W_rel = nn.Parameter(torch.empty(num_rels, rel_dim, rel_dim))
        self.u_rel = nn.Linear(rel_dim, num_rels, bias=False)
        self.v_rel = nn.Linear(rel_dim, num_rels, bias=False)
        self.b_rel = nn.Parameter(torch.zeros(num_rels))
        nn.init.xavier_uniform_(self.W_rel)
        
        # ── Case Prediction ──
        self.case_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_cases),
        )
        
        # ── Regularization ──
        self.embed_drop = nn.Dropout(0.2)
        self.gru_drop = nn.Dropout(0.25)
        self.word_drop_rate = 0.1

    def encode_grid_to_words(self, grid, mask):
        B, R, C = grid.shape
        col_embs = []
        for c in range(C):
            col_vals = grid[:, :, c].clamp(0, self.vocab_size)
            col_embs.append(self.col_embeds[c](col_vals))
            
        row_features = torch.cat(col_embs, dim=-1)
        row_features = self.embed_drop(row_features)
        
        h_words = self.row_encoder(row_features)
        positions = torch.arange(R, device=grid.device)
        h_words = h_words + self.pos_embed(positions)
        
        if self.training and self.word_drop_rate > 0:
            word_mask = torch.bernoulli(
                torch.full((B, R, 1), 1.0 - self.word_drop_rate, device=grid.device)
            )
            h_words = h_words * word_mask / (1.0 - self.word_drop_rate)
        return h_words

    def score_heads(self, h_enriched, mask):
        B, R, _ = h_enriched.shape
        root = self.root_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        h_candidates = torch.cat([root, h_enriched], dim=1)
        
        dep = self.mlp_arc_dep(h_enriched)
        head = self.mlp_arc_head(h_candidates)
        
        scores = torch.bmm(dep @ self.W_arc, head.transpose(1, 2))
        scores += self.u_arc(dep)
        scores += self.v_arc(head).transpose(1, 2)
        
        dep_pos = torch.arange(R, device=scores.device)
        head_pos = torch.arange(-1, R, device=scores.device)
        rel_dist = head_pos.unsqueeze(0) - dep_pos.unsqueeze(1)
        
        max_d = self.dist_bias.num_embeddings // 2
        dist_idx = (rel_dist + max_d).clamp(0, 2 * max_d)
        d_bias = self.dist_bias(dist_idx).squeeze(-1)
        scores = scores + d_bias.unsqueeze(0)
        
        cand_mask = torch.cat([mask.new_ones(B, 1), mask], dim=1).bool()
        scores = scores.masked_fill(~cand_mask.unsqueeze(1), -1e4)
        
        self_loop = torch.zeros(R, R + 1, dtype=torch.bool, device=scores.device)
        for i in range(R):
            self_loop[i, i + 1] = True
        scores = scores.masked_fill(self_loop.unsqueeze(0), -1e4)
        
        return scores

    def score_rels(self, h_enriched, head_indices, mask):
        B, R, _ = h_enriched.shape
        root = self.root_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        h_with_root = torch.cat([root, h_enriched], dim=1)
        
        dep = self.mlp_rel_dep(h_enriched)
        head_all = self.mlp_rel_head(h_with_root)
        
        idx = head_indices.clamp(0, R)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, head_all.shape[-1])
        head_sel = torch.gather(head_all, dim=1, index=idx_exp)
        
        scores = torch.einsum('bnd,rde,bne->bnr', dep, self.W_rel, head_sel)
        scores = scores + self.u_rel(dep) + self.v_rel(head_sel) + self.b_rel
        return scores

    def forward(self, grid, mask, num_manager_steps=8, num_worker_steps=4, gold_heads=None):
        B = grid.shape[0]
        R = self.grid_rows
        device = grid.device
        
        manager_h = torch.zeros(B, self.manager_gru.hidden_size, device=device)
        worker_h  = torch.zeros(B, self.worker_gru.hidden_size, device=device)
        
        current_grid = grid.clone()
        step_outputs = []
        
        for m_step in range(num_manager_steps):
            h_words = self.encode_grid_to_words(current_grid, mask)
            
            mask_f = mask.unsqueeze(-1).float()
            h_global = (h_words * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            h_global = self.global_proj(h_global)
            
            manager_h = self.manager_gru(h_global, manager_h)
            manager_h = self.gru_drop(manager_h)
            goal = self.manager_goal_proj(manager_h)
            
            for w_step in range(num_worker_steps):
                w_in = torch.cat([h_global, goal], dim=-1)
                worker_h = self.worker_gru(w_in, worker_h)
                worker_h = self.gru_drop(worker_h)
                
                m_exp = manager_h.unsqueeze(1).expand(-1, R, -1)
                w_exp = worker_h.unsqueeze(1).expand(-1, R, -1)
                fused = torch.cat([h_words, m_exp, w_exp], dim=-1)
                h_enriched = self.context_fuse(fused)
                
                sa_pad_mask = ~mask.bool()
                h_enriched = self.cross_word_attn(h_enriched, src_key_padding_mask=sa_pad_mask)
                
                arc_scores = self.score_heads(h_enriched, mask)
                case_logits = self.case_head(h_enriched)
                
                if self.training and gold_heads is not None:
                    rel_head_input = gold_heads
                else:
                    rel_head_input = arc_scores.argmax(dim=-1)
                
                rel_scores = self.score_rels(h_enriched, rel_head_input, mask)
                
                step_preds = {
                    4: arc_scores,    # head
                    5: rel_scores,    # deprel
                    3: case_logits,   # case
                }
                step_outputs.append(step_preds)
                
                with torch.no_grad():
                    pred_heads = arc_scores.argmax(dim=-1)
                    pred_rels  = rel_scores.argmax(dim=-1)
                    pred_cases = case_logits.argmax(dim=-1)
                    
                    new_grid = current_grid.clone()
                    new_grid[:, :, 3] = torch.where(grid[:, :, 3] == 0, pred_cases, new_grid[:, :, 3])
                    new_grid[:, :, 4] = torch.where(grid[:, :, 4] == 0, pred_heads, new_grid[:, :, 4])
                    new_grid[:, :, 5] = torch.where(grid[:, :, 5] == 0, pred_rels, new_grid[:, :, 5])
                    current_grid = new_grid
                    
        return step_outputs, step_outputs[-1]


def compute_loss(step_outputs, solution, mask, arc_weight=0.5, rel_weight=0.3, case_weight=0.2):
    total_loss = 0.0
    num_steps = len(step_outputs)
    flat_mask = mask.reshape(-1).float()
    num_real = flat_mask.sum().clamp(min=1)
    
    gold_cases = solution[:, :, 3].long()
    gold_heads = solution[:, :, 4].long()
    gold_rels = solution[:, :, 5].long()
    
    for t, step in enumerate(step_outputs):
        w_t = (t + 1) / num_steps
        
        # Arc loss
        arc = step[4]
        arc_loss = F.cross_entropy(
            arc.reshape(-1, arc.shape[-1]), gold_heads.reshape(-1), 
            reduction='none', label_smoothing=0.03
        )
        arc_loss = (arc_loss * flat_mask).sum() / num_real
        
        # Rel loss
        rel = step[5]
        rel_loss = F.cross_entropy(
            rel.reshape(-1, rel.shape[-1]), gold_rels.reshape(-1), 
            reduction='none', label_smoothing=0.1
        )
        rel_loss = (rel_loss * flat_mask).sum() / num_real
        
        # Case loss
        cas = step[3]
        case_loss = F.cross_entropy(
            cas.reshape(-1, cas.shape[-1]), gold_cases.reshape(-1), 
            reduction='none', label_smoothing=0.1
        )
        case_loss = (case_loss * flat_mask).sum() / num_real
        
        step_loss = (arc_weight * arc_loss + rel_weight * rel_loss + case_weight * case_loss)
        total_loss += w_t * step_loss

    return total_loss

def compute_accuracy(final_pred, solution, mask):
    gold_cases = solution[:, :, 3]
    gold_heads = solution[:, :, 4]
    gold_rels = solution[:, :, 5]
    
    m = mask.bool()
    
    pred_cases = final_pred[3].argmax(dim=-1)
    pred_heads = final_pred[4].argmax(dim=-1)
    pred_rels = final_pred[5].argmax(dim=-1)
    
    case_correct = ((pred_cases == gold_cases) & m).sum().item()
    head_correct = ((pred_heads == gold_heads) & m).sum().item()
    rel_correct = ((pred_rels == gold_rels) & m).sum().item()
    
    total = m.sum().item()
    
    return {
        'case': case_correct / max(total, 1),
        'head': head_correct / max(total, 1),
        'deprel': rel_correct / max(total, 1),
        'overall': (case_correct + head_correct + rel_correct) / (3 * max(total, 1))
    }

def train_hrm(args):
    print(f"\n{'='*60}")
    print(f"Training HRM (BIAFFINE OVERHAUL) on Arabic Syntax Grids")
    print(f"{'='*60}")
    
    train_grids = torch.from_numpy(np.load(DATA_DIR / "train_grids.npy")).long()
    train_masks = torch.from_numpy(np.load(DATA_DIR / "train_masks.npy")).long()
    train_solutions = torch.from_numpy(np.load(DATA_DIR / "train_solutions.npy")).long()
    
    dev_grids = torch.from_numpy(np.load(DATA_DIR / "dev_grids.npy")).long()
    dev_masks = torch.from_numpy(np.load(DATA_DIR / "dev_masks.npy")).long()
    dev_solutions = torch.from_numpy(np.load(DATA_DIR / "dev_solutions.npy")).long()

    model = ArabicSyntaxHRM(
        hidden_dim=args.hidden_dim, 
        manager_dim=args.hidden_dim, 
        worker_dim=args.hidden_dim, 
        embed_dim=args.embed_dim
    ).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params/1e6:.2f}M")
    
    col_dropout = ColumnDropout(drop_prob=0.15)
    
    biaffine_params, other_params = [], []
    for name, p in model.named_parameters():
        if 'W_arc' in name or 'W_rel' in name or 'dist_bias' in name:
            biaffine_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': other_params,    'lr': 2e-3, 'weight_decay': 0.02},
        {'params': biaffine_params, 'lr': 1e-3, 'weight_decay': 0.0},  
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_dev_acc = 0.0
    best_dev_loss = float('inf')
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t_start = time.time()
        
        perm = torch.randperm(train_grids.shape[0])
        
        for start in range(0, train_grids.shape[0], args.batch_size):
            end = min(start + args.batch_size, train_grids.shape[0])
            idx = perm[start:end]
            
            b_grids = train_grids[idx].to(DEVICE)
            b_masks = train_masks[idx].to(DEVICE)
            b_sols = train_solutions[idx].to(DEVICE)
            b_gold_heads = b_sols[:, :, 4]
            
            b_grids_dropped = col_dropout(b_grids)
            
            optimizer.zero_grad()
            all_preds, final_pred = model(
                b_grids_dropped, b_masks, 
                args.manager_steps, args.worker_steps, 
                gold_heads=b_gold_heads
            )
            
            loss = compute_loss(all_preds, b_sols, b_masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t_start
        
        eval_freq = max(1, args.epochs // 20)
        if (epoch + 1) % eval_freq == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                eval_size = min(512, dev_grids.shape[0])
                b_grids = dev_grids[:eval_size].to(DEVICE)
                b_masks = dev_masks[:eval_size].to(DEVICE)
                b_sols = dev_solutions[:eval_size].to(DEVICE)
                
                all_preds, final_pred = model(
                    b_grids, b_masks, 
                    args.manager_steps, args.worker_steps
                )
                dev_loss = compute_loss(all_preds, b_sols, b_masks).item()
                metrics = compute_accuracy(final_pred, b_sols, b_masks)
                
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:>3}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | Dev Loss: {dev_loss:.4f} | "
                  f"Case: {metrics['case']:.3f} | UAS/Head: {metrics['head']:.3f} | "
                  f"LAS/Rel: {metrics['deprel']:.3f} | LR: {lr:.2e} | {elapsed:.1f}s")
            
            if metrics['head'] > best_dev_acc:
                best_dev_acc = metrics['head']
                torch.save(model.state_dict(), OUTPUT_DIR / "best_hrm.pt")
            
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss

    torch.save(model.state_dict(), OUTPUT_DIR / "final_hrm.pt")
    
    config = {
        'grid_rows': GRID_ROWS, 'grid_cols': GRID_COLS,
        'vocab_size': VOCAB_SIZE, 'hidden_dim': args.hidden_dim,
        'manager_dim': args.hidden_dim, 'worker_dim': args.hidden_dim,
        'embed_dim': args.embed_dim,
        'manager_steps': args.manager_steps, 'worker_steps': args.worker_steps,
        'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': 2e-3,
        'best_dev_acc': best_dev_acc, 'best_dev_loss': best_dev_loss,
        'num_params': num_params, 'device': str(DEVICE),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Train HRM on Arabic Syntax Grids")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--manager-steps", type=int, default=4)
    parser.add_argument("--worker-steps", type=int, default=2)
    parser.add_argument("--quick-test", action="store_true")
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("🚀 Quick test mode: Biaffine Overhaul -> 256 dim")
        args.epochs = 50 
        args.hidden_dim = 256
        args.embed_dim = 32
        args.batch_size = 64
        args.manager_steps = 2
        args.worker_steps = 1
    
    train_hrm(args)

if __name__ == "__main__":
    main()
