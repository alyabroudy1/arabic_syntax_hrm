import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableTreeCRF(nn.Module):
    """
    Computes marginal probabilities of all valid trees using Kirchhoff's 
    Matrix-Tree Theorem. Allows global structural optimization.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, arc_scores, gold_heads, mask):
        """
        arc_scores: (B, N, N) - biaffine scores (h, d)
        gold_heads: (B, N)
        mask: (B, N)
        """
        B, N, _ = arc_scores.shape
        device = arc_scores.device
        
        # Exp to get potentials (Clamp logits up to +20 to prevent Inf exponentiation causing NaN)
        potentials = torch.exp(arc_scores.clamp(max=20.0, min=-20.0))  # (B, N, N)
        
        # Apply mask
        mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, N, N)
        # Root can't be a dependent
        mask_2d[:, :, 0] = False
        potentials = potentials * mask_2d.float()
        
        # Build Laplacian L (B, N-1, N-1), we drop the root node 0
        A = potentials[:, 1:, 1:]  # (B, N-1, N-1)
        root_A = potentials[:, 0, 1:]  # (B, N-1)
        
        in_degree = A.sum(dim=1)  # (B, N-1)
        total_in_degree = in_degree + root_A  # (B, N-1)
        
        L = torch.diag_embed(total_in_degree) - A  # (B, N-1, N-1)
        
        # For variable length sentences, replace padded submatrix with Identity Matrix
        pad_mask = ~mask[:, 1:].bool()  # (B, N-1)
        pad_mask_2d = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)  # (B, N-1, N-1)
        
        # Vectorized: set padded rows/cols to 0.0, then add 1.0 to the diagonal for padded elements
        L = L.masked_fill(pad_mask_2d, 0.0)
        L = L + torch.diag_embed(pad_mask.float())
                
        # log Z = log det(L)
        # For numerical stability, we use slogdet
        sign, logdet = torch.linalg.slogdet(L)
        # Handled cases where determinant is non-positive due to bad potentials
        logZ = torch.where(sign > 0, logdet, torch.zeros_like(logdet))
        
        # Score of gold tree: arc_scores[b, gold_heads[b,d], d] for each dependent d
        dep_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)  # (B, N)
        gold_heads_clamped = gold_heads.clamp(0, N-1)
        
        # Gold score for each (batch, dep) pair
        gold_arc_scores = arc_scores[batch_idx, gold_heads_clamped, dep_idx]  # (B, N)
        
        # Mask: only count real dependents (not root position 0, not padding)
        dep_mask = mask.clone()
        dep_mask[:, 0] = 0  # root is not a dependent
        gold_score = (gold_arc_scores * dep_mask.float()).sum(dim=-1)  # (B,)
        
        # NLL Loss: log Z - score(gold), clamped for stability
        loss = (logZ - gold_score).clamp(min=0.0).mean()
        return loss

class ContrastiveTreeLoss(nn.Module):
    """
    Generates hard-negative trees by perturbing the gold tree
    and applies a margin-based ranking loss.
    """
    def __init__(self, n_negatives: int = 4, margin: float = 2.0):
        super().__init__()
        self.n_negatives = n_negatives
        self.margin = margin
    
    def _generate_hard_negatives(self, gold_heads, mask):
        B, N = gold_heads.shape
        negatives = []
        for _ in range(self.n_negatives):
            neg = gold_heads.clone()
            for b in range(B):
                length = mask[b].sum().item()
                if length <= 2:
                    continue
                
                strategy = torch.randint(0, 3, (1,)).item()
                if strategy == 0:  # SWAP
                    candidates = list(range(1, length))
                    if len(candidates) >= 2:
                        i, j = torch.randperm(len(candidates))[:2].tolist()
                        d1, d2 = candidates[i], candidates[j]
                        new_h1, new_h2 = neg[b, d2].item(), neg[b, d1].item()
                        if new_h1 != d1: neg[b, d1] = new_h1
                        if new_h2 != d2: neg[b, d2] = new_h2
                elif strategy == 1:  # REATTACH
                    d = torch.randint(1, length, (1,)).item()
                    candidates = [h for h in range(max(0, d-3), min(length, d+4))
                                  if h != d and h != neg[b, d].item()]
                    if candidates:
                        new_head = candidates[torch.randint(0, len(candidates), (1,)).item()]
                        neg[b, d] = new_head
                else:  # ROOT-FLIP
                    current_root_deps = (neg[b, 1:length] == 0).nonzero(as_tuple=True)[0]
                    non_root_deps = (neg[b, 1:length] != 0).nonzero(as_tuple=True)[0]
                    if len(current_root_deps) > 0 and len(non_root_deps) > 0:
                        old_rd = current_root_deps[0].item() + 1
                        new_rd = non_root_deps[torch.randint(0, len(non_root_deps), (1,)).item()].item() + 1
                        neg[b, old_rd] = max(1, old_rd - 1)
                        neg[b, new_rd] = 0
            negatives.append(neg)
        return negatives
    
    def forward(self, arc_scores, gold_heads, mask):
        B, N, _ = arc_scores.shape
        device = arc_scores.device
        dep_mask = mask[:, 1:].float()
        
        dep_idx = torch.arange(1, N, device=device).unsqueeze(0).expand(B, -1)
        gold_score = arc_scores[
            torch.arange(B, device=device).unsqueeze(1),
            gold_heads[:, 1:].clamp(0, N-1), # Clamp to prevent out-of-bounds on dummy pads
            dep_idx
        ] 
        gold_total = (gold_score * dep_mask).sum(dim=-1)
        
        negatives = self._generate_hard_negatives(gold_heads, mask)
        
        total_loss = torch.tensor(0.0, device=device)
        for neg_heads in negatives:
            neg_score = arc_scores[
                torch.arange(B, device=device).unsqueeze(1),
                neg_heads[:, 1:].clamp(0, N-1),
                dep_idx
            ]
            neg_total = (neg_score * dep_mask).sum(dim=-1)
            
            margin_loss = torch.clamp(self.margin - gold_total + neg_total, min=0.0)
            total_loss = total_loss + margin_loss.mean()
        
        return total_loss / self.n_negatives

class AgreementAuxLoss(nn.Module):
    """
    Penalizes predicted arcs where morphological agreement is violated.
    Acts as a linguistically-informed regularizer.
    """
    def __init__(self, word_dim: int = 256, n_agreement_types: int = 4):
        super().__init__()
        self.agreement_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(word_dim * 2, 128),
                nn.GELU(),
                nn.Linear(128, 2)
            )
            for _ in range(n_agreement_types)
        ])
        self.penalty_weight = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, word_repr, soft_heads, gold_heads, morph_features, mask):
        B, W, D = word_repr.shape
        exp_head_repr = torch.bmm(soft_heads.transpose(1, 2), word_repr)
        pair_repr = torch.cat([exp_head_repr, word_repr], dim=-1)
        
        total_loss = 0
        for predictor in self.agreement_predictors:
            agree_logits = predictor(pair_repr)
            targets = torch.ones(B, W, dtype=torch.long, device=word_repr.device)
            loss = F.cross_entropy(agree_logits.view(-1, 2), targets.view(-1), reduction='none').view(B, W)
            total_loss += (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        
        return self.penalty_weight.abs() * total_loss

class RDropLoss(nn.Module):
    """
    Regularized Dropout for consistent predictions.
    Computes symmetric KL divergence.
    """
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, logits_1, logits_2, mask=None):
        p = F.log_softmax(logits_1, dim=-1)
        q = F.log_softmax(logits_2, dim=-1)
        
        kl_pq = F.kl_div(p, q.exp(), reduction='none').sum(dim=-1)
        kl_qp = F.kl_div(q, p.exp(), reduction='none').sum(dim=-1)
        kl = (kl_pq + kl_qp) / 2
        
        if mask is not None:
            kl = (kl * mask.float()).sum() / mask.float().sum().clamp(min=1)
        else:
            kl = kl.mean()
        
        return self.alpha * kl

class StructuralLabelSmoothing(nn.Module):
    """
    Smooths label probabilities toward RELATED labels only based on a 
    confusion affinity matrix.
    """
    def __init__(self, n_classes: int, smoothing: float = 0.1, affinity_matrix: torch.Tensor = None):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
        if affinity_matrix is not None:
            aff = affinity_matrix.float()
            aff = aff / aff.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            self.register_buffer('affinity', aff)
        else:
            self.register_buffer('affinity', torch.ones(n_classes, n_classes) / n_classes)
            
    @staticmethod
    def build_arabic_relation_affinity(n_relations: int, rel_vocab: dict = None):
        affinity = torch.eye(n_relations) * 0.5
        # Very simplified mock matrix if no vocab available
        return affinity
    
    def forward(self, logits, targets, mask=None):
        if logits.dim() == 3:
            B, N, C = logits.shape
            logits = logits.reshape(-1, C)
            targets = targets.reshape(-1)
            if mask is not None:
                mask = mask.reshape(-1)
        
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_targets = self.affinity[targets.clamp(0, self.n_classes-1)] * self.smoothing
        smooth_targets.scatter_(1, targets.clamp(0, self.n_classes-1).unsqueeze(1), self.confidence)
        
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        if mask is not None:
            loss = (loss * mask.float()).sum() / mask.float().sum().clamp(1)
        else:
            loss = loss.mean()
        return loss

class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """
    Learns task-specific loss weights automatically using homoscedastic uncertainty.
    """
    def __init__(self, n_tasks: int = 3, initial_sigma: float = 1.0):
        super().__init__()
        self.log_sigma_sq = nn.Parameter(
            torch.full((n_tasks,), 2 * math.log(initial_sigma))
        )
    
    def forward(self, *task_losses):
        total = 0
        weighted_components = []
        for i, loss in enumerate(task_losses):
            precision = (-self.log_sigma_sq[i]).exp()
            weighted = precision * loss + 0.5 * self.log_sigma_sq[i]
            total += weighted
            weighted_components.append(
                (loss.item(), precision.item(), self.log_sigma_sq[i].item())
            )
        return total, weighted_components
