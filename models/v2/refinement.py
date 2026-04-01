import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeMessagePassingLayer(nn.Module):
    """Single round of tree-structured message passing."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        # Upward message: child → parent
        self.head_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Downward message: parent → children (aggregated)
        self.child_transform = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Sibling message
        self.sibling_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model),  # self + head + child + sibling
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, H, pred_heads, mask):
        B, N, D = H.shape
        device = H.device
        
        # ── 1. Upward (head) message ──
        head_idx = pred_heads.clamp(0, N-1).unsqueeze(-1).expand(-1, -1, D)
        head_repr = H.gather(1, head_idx)
        
        head_msg, _ = self.head_attention(H, head_repr, head_repr, need_weights=False)
        
        # ── 2. Downward (children) message ──
        child_sum = torch.zeros_like(H)
        child_count = torch.zeros(B, N, 1, device=device)
        
        head_idx_flat = pred_heads.clamp(0, N-1)
        for b in range(B):
            length = mask[b].sum().item()
            for d in range(1, length):
                h = head_idx_flat[b, d].item()
                child_sum[b, h] += H[b, d]
                child_count[b, h] += 1
        
        child_avg = child_sum / child_count.clamp(min=1)
        child_msg = self.child_transform(torch.cat([H, child_avg], dim=-1))
        
        # ── 3. Sibling message ──
        heads_exp = pred_heads.unsqueeze(2).expand(-1, -1, N)
        sibling_mask = (heads_exp == heads_exp.transpose(1, 2))
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        sibling_mask = sibling_mask & ~eye & mask.bool().unsqueeze(1) & mask.bool().unsqueeze(2)
        
        sib_attn_mask = ~sibling_mask
        
        # Safe self-attention for siblings without enormous rep_interleave
        # Since nn.MultiheadAttention expects 2D mask (L, S) or 3D mask (B*num_heads, L, S)
        # we reshape
        sib_attn_mask_expanded = sib_attn_mask.unsqueeze(1).repeat(1, self.sibling_attention.num_heads, 1, 1)
        sib_attn_mask_expanded = sib_attn_mask_expanded.view(B * self.sibling_attention.num_heads, N, N)

        if N <= 128:
            sib_msg, _ = self.sibling_attention(
                H, H, H, 
                attn_mask=sib_attn_mask_expanded,
                need_weights=False
            )
        else:
            sib_msg = torch.zeros_like(H)
        
        # ── 4. Gated fusion ──
        combined = torch.cat([H, head_msg, child_msg, sib_msg], dim=-1)
        gate = self.gate(combined)
        update = self.transform(combined)
        
        H_updated = H + self.dropout(gate * update)
        return self.norm(H_updated)

class TreeGNNRefinement(nn.Module):
    """
    Given an initial predicted tree (from biaffine + MST or soft scores), constructs 
    the tree as a directed graph and runs K rounds of message passing.
    """
    def __init__(self, d_model: int = 256, n_rounds: int = 3, 
                 n_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.n_rounds = n_rounds
        
        self.message_layers = nn.ModuleList([
            TreeMessagePassingLayer(d_model, n_heads, dropout)
            for _ in range(n_rounds)
        ])
        
        self.refined_arc_head = nn.Linear(d_model, d_model)
        self.refined_arc_dep = nn.Linear(d_model, d_model)
        self.refined_arc_bilinear = nn.Parameter(
            torch.randn(d_model, d_model) * (2.0 / d_model)**0.5
        )
        self.refined_arc_bias_h = nn.Linear(d_model, 1)
        self.refined_arc_bias_d = nn.Linear(d_model, 1)
    
    def forward(self, H, predicted_heads, mask):
        H_refined = H.clone()
        for layer in self.message_layers:
            H_refined = layer(H_refined, predicted_heads, mask)
        
        h_head = self.refined_arc_head(H_refined)
        h_dep = self.refined_arc_dep(H_refined)
        
        refined_arc_scores = torch.einsum(
            'bnd, de, bme -> bnm', h_head, self.refined_arc_bilinear, h_dep
        )
        refined_arc_scores += self.refined_arc_bias_h(h_head).transpose(1, 2)
        refined_arc_scores += self.refined_arc_bias_d(h_dep)
        
        return H_refined, refined_arc_scores

class SecondOrderScorer(nn.Module):
    """
    Scores PAIRS of arcs jointly (Siblings and Grandparents).
    """
    def __init__(self, input_dim: int = 256, scorer_dim: int = 128):
        super().__init__()
        # ── Sibling projections ──
        self.sib_head_mlp = nn.Sequential(nn.Linear(input_dim, scorer_dim), nn.LeakyReLU(0.1))
        self.sib_dep_mlp = nn.Sequential(nn.Linear(input_dim, scorer_dim), nn.LeakyReLU(0.1))
        
        self.trilinear_rank = 64
        self.sib_U = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
        self.sib_V = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
        self.sib_W = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
        
        # ── Grandparent projections ──
        self.gp_grandparent_mlp = nn.Sequential(nn.Linear(input_dim, scorer_dim), nn.LeakyReLU(0.1))
        self.gp_head_mlp = nn.Sequential(nn.Linear(input_dim, scorer_dim), nn.LeakyReLU(0.1))
        self.gp_dep_mlp = nn.Sequential(nn.Linear(input_dim, scorer_dim), nn.LeakyReLU(0.1))
        
        self.gp_U = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
        self.gp_V = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
        self.gp_W = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
    
    def score_siblings(self, H, i_indices, j_indices, h_indices):
        B, N, D = H.shape
        # Gather representations
        h_repr = self.sib_head_mlp(H.gather(1, h_indices.unsqueeze(-1).expand(-1, -1, D)))
        i_repr = self.sib_dep_mlp(H.gather(1, i_indices.unsqueeze(-1).expand(-1, -1, D)))
        j_repr = self.sib_dep_mlp(H.gather(1, j_indices.unsqueeze(-1).expand(-1, -1, D)))
        
        h_proj = torch.einsum('bkd, rd -> bkr', h_repr, self.sib_U)
        i_proj = torch.einsum('bkd, rd -> bkr', i_repr, self.sib_V)
        j_proj = torch.einsum('bkd, rd -> bkr', j_repr, self.sib_W)
        
        scores = (h_proj * i_proj * j_proj).sum(dim=-1)
        return scores
    
    def score_grandparents(self, H, g_indices, h_indices, d_indices):
        B, N, D = H.shape
        g_repr = self.gp_grandparent_mlp(H.gather(1, g_indices.unsqueeze(-1).expand(-1, -1, D)))
        h_repr = self.gp_head_mlp(H.gather(1, h_indices.unsqueeze(-1).expand(-1, -1, D)))
        d_repr = self.gp_dep_mlp(H.gather(1, d_indices.unsqueeze(-1).expand(-1, -1, D)))
        
        g_proj = torch.einsum('bkd, rd -> bkr', g_repr, self.gp_U)
        h_proj = torch.einsum('bkd, rd -> bkr', h_repr, self.gp_V)
        d_proj = torch.einsum('bkd, rd -> bkr', d_repr, self.gp_W)
        
        return (g_proj * h_proj * d_proj).sum(dim=-1)
    
    def extract_gold_pairs(self, gold_heads, mask):
        B, N = gold_heads.shape
        device = gold_heads.device
        all_sib_h, all_sib_i, all_sib_j = [], [], []
        all_gp_g, all_gp_h, all_gp_d = [], [], []
        
        from collections import defaultdict
        
        for b in range(B):
            length = mask[b].sum().item()
            heads = gold_heads[b, :length]
            
            children = defaultdict(list)
            for d in range(1, length):
                h = heads[d].item()
                children[h].append(d)
            
            sib_h, sib_i, sib_j = [], [], []
            for h, deps in children.items():
                for idx_a in range(len(deps)):
                    for idx_b in range(idx_a + 1, len(deps)):
                        sib_h.append(h)
                        sib_i.append(deps[idx_a])
                        sib_j.append(deps[idx_b])
            
            all_sib_h.append(sib_h)
            all_sib_i.append(sib_i)
            all_sib_j.append(sib_j)
            
            gp_g, gp_h, gp_d = [], [], []
            for d in range(1, length):
                h = heads[d].item()
                if 0 < h < length:  # non root and valid index
                    g = heads[h].item()
                    gp_g.append(g)
                    gp_h.append(h)
                    gp_d.append(d)
            
            all_gp_g.append(gp_g)
            all_gp_h.append(gp_h)
            all_gp_d.append(gp_d)
        
        def pad_lists(list_of_lists, pad_val=0):
            max_k = max(len(l) for l in list_of_lists) if list_of_lists else 1
            max_k = max(max_k, 1)
            padded = torch.full((B, max_k), pad_val, dtype=torch.long, device=device)
            pair_mask = torch.zeros(B, max_k, dtype=torch.bool, device=device)
            for b, l in enumerate(list_of_lists):
                if l:
                    padded[b, :len(l)] = torch.tensor(l, device=device)
                    pair_mask[b, :len(l)] = True
            return padded, pair_mask
        
        sib_h_t, sib_mask = pad_lists(all_sib_h)
        sib_i_t, _ = pad_lists(all_sib_i)
        sib_j_t, _ = pad_lists(all_sib_j)
        
        gp_g_t, gp_mask = pad_lists(all_gp_g)
        gp_h_t, _ = pad_lists(all_gp_h)
        gp_d_t, _ = pad_lists(all_gp_d)
        
        return {
            'sib': (sib_h_t, sib_i_t, sib_j_t, sib_mask),
            'gp': (gp_g_t, gp_h_t, gp_d_t, gp_mask),
        }
    
    def second_order_loss(self, H, gold_heads, pred_heads, mask):
        margin = 1.0
        gold_pairs = self.extract_gold_pairs(gold_heads, mask)
        pred_pairs = self.extract_gold_pairs(pred_heads.detach(), mask)
        
        loss = torch.tensor(0.0, device=H.device)
        n_terms = 0
        
        g_sh, g_si, g_sj, g_smask = gold_pairs['sib']
        p_sh, p_si, p_sj, p_smask = pred_pairs['sib']
        
        if g_smask.any() and p_smask.any():
            gold_sib_scores = self.score_siblings(H, g_si, g_sj, g_sh)
            gold_sib_scores = (gold_sib_scores * g_smask.float()).sum() / g_smask.float().sum().clamp(1)
            
            pred_sib_scores = self.score_siblings(H, p_si, p_sj, p_sh)
            pred_sib_scores = (pred_sib_scores * p_smask.float()).sum() / p_smask.float().sum().clamp(1)
            
            loss = loss + F.relu(margin - gold_sib_scores + pred_sib_scores)
            n_terms += 1
        
        g_gg, g_gh, g_gd, g_gmask = gold_pairs['gp']
        p_gg, p_gh, p_gd, p_gmask = pred_pairs['gp']
        
        if g_gmask.any() and p_gmask.any():
            gold_gp_scores = self.score_grandparents(H, g_gg, g_gh, g_gd)
            gold_gp_scores = (gold_gp_scores * g_gmask.float()).sum() / g_gmask.float().sum().clamp(1)
            
            pred_gp_scores = self.score_grandparents(H, p_gg, p_gh, p_gd)
            pred_gp_scores = (pred_gp_scores * p_gmask.float()).sum() / p_gmask.float().sum().clamp(1)
            
            loss = loss + F.relu(margin - gold_gp_scores + pred_gp_scores)
            n_terms += 1
        
        return loss / max(n_terms, 1)
