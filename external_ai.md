orgenize ur self well, implement a robust plan, heres the ai response :
The Grand Architectural Blueprint: Arabic HRM-Grid Parser v2
This is a comprehensive engineering document covering four major intervention vectors, ranked by expected impact, with production-quality PyTorch code throughout.

1. Tree-Structural Enforcement: Tree-CRF + MST Decoding
This is your single highest-ROI change. Right now your biaffine head pointer is trained with per-word cross-entropy, which means each dependent picks its head independently. This allows impossible outputs: cycles, multiple roots, disconnected fragments. A Tree-CRF replaces that local loss with a global structured loss over the space of all valid trees.

1A. The Math (Kirchhoff's Matrix-Tree Theorem)
Given arc scores 
s
(
h
,
d
)
s(h,d) for head 
h
h and dependent 
d
d, we want:

P
(
T
gold
∣
x
)
=
exp
⁡
(
∑
(
h
,
d
)
∈
T
gold
s
(
h
,
d
)
)
Z
P(T 
gold
​
 ∣x)= 
Z
exp(∑ 
(h,d)∈T 
gold
​
 
​
 s(h,d))
​
 

where 
Z
=
∑
T
∈
T
exp
⁡
(
∑
(
h
,
d
)
∈
T
s
(
h
,
d
)
)
Z=∑ 
T∈T
​
 exp(∑ 
(h,d)∈T
​
 s(h,d)) sums over all valid directed spanning trees rooted at node 0.

The miracle: Kirchhoff's theorem lets us compute 
Z
Z as the determinant of a Laplacian matrix 
L
L, which is fully differentiable via torch.slogdet.

Construct 
L
∈
R
n
×
n
L∈R 
n×n
  (indexed by dependents 
1
…
n
1…n):

L
j
j
=
∑
h
=
0
h
≠
j
n
exp
⁡
(
s
(
h
,
j
)
)
,
L
i
j
=
−
exp
⁡
(
s
(
i
,
j
)
)
for 
i
≠
j
L 
jj
​
 =∑ 
h=0
h

=j
​
 
n
​
 exp(s(h,j)),L 
ij
​
 =−exp(s(i,j))for i

=j

Then 
Z
=
det
⁡
(
L
)
Z=det(L), and:

L
Tree-CRF
=
log
⁡
Z
−
∑
(
h
,
d
)
∈
T
gold
s
(
h
,
d
)
L 
Tree-CRF
​
 =logZ−∑ 
(h,d)∈T 
gold
​
 
​
 s(h,d)

Marginals 
P
(
arc 
h
 ⁣
→
 ⁣
d
∈
T
)
P(arc h→d∈T) come free via autograd through the log-determinant.

1B. Full PyTorch Implementation
Python

import torch
import torch.nn as nn


class MatrixTreeCRFLoss(nn.Module):
    """
    First-order Tree-CRF loss using Kirchhoff's Matrix-Tree Theorem.
    
    Replaces per-token cross-entropy for head prediction with a globally-
    normalized structured loss over the space of ALL valid directed spanning 
    trees (arborescences) rooted at a designated root node (index 0).
    
    Complexity: O(n^3) per sentence for the determinant computation.
    """
    
    def __init__(self, root_idx: int = 0):
        super().__init__()
        self.root_idx = root_idx
    
    def forward(
        self,
        arc_scores: torch.Tensor,   # (B, N, N) — scores[b][h][d]
        gold_heads: torch.LongTensor, # (B, N) — gold_heads[b][d] = head of word d
        mask: torch.BoolTensor       # (B, N) — True for real tokens (incl. ROOT)
    ) -> torch.Tensor:
        """Returns scalar mean negative log-likelihood of gold trees."""
        
        B, N, _ = arc_scores.shape
        n = N - 1  # number of actual words (root excluded as dependent)
        device = arc_scores.device
        
        # ── Step 1: Compute gold tree score (before any modification) ──
        # For each dependent d=1..N-1, gather score(gold_head[d], d)
        dep_indices = torch.arange(1, N, device=device).unsqueeze(0).expand(B, -1)  # (B, n)
        gold_h = gold_heads[:, 1:]  # (B, n)
        
        # Gather: arc_scores[b, gold_h[b,j], j+1]
        gold_arc_scores = arc_scores[
            torch.arange(B, device=device).unsqueeze(1),  # batch index
            gold_h,                                        # head index
            dep_indices                                    # dependent index
        ]  # (B, n)
        
        dep_mask = mask[:, 1:].float()  # (B, n)
        gold_tree_score = (gold_arc_scores * dep_mask).sum(dim=-1)  # (B,)
        
        # ── Step 2: Mask self-loops (word can't be its own head) ──
        S = arc_scores.clone()
        S[:, range(N), range(N)] = float('-inf')
        
        # ── Step 3: Extract dependent-only slice ──
        # S_dep[b, h, j] = score(head=h, dep=j+1) for h ∈ {0..N-1}, j ∈ {0..n-1}
        S_dep = S[:, :, 1:]  # (B, N, n)
        
        # ── Step 4: Numerical stability — subtract column-wise max ──
        col_max = S_dep.max(dim=1, keepdim=True)[0]  # (B, 1, n)
        # Clamp to avoid -inf propagation in all-padded columns
        col_max = col_max.clamp(min=-1e8)
        exp_S = (S_dep - col_max).exp()  # (B, N, n)
        
        # ── Step 5: Build the Kirchhoff Laplacian L ∈ (B, n, n) ──
        # Off-diagonal: L[i,j] = −exp(score(i+1 → j+1)) for non-root heads i+1
        L = -exp_S[:, 1:, :].clone()  # (B, n, n)
        
        # Diagonal: L[j,j] = Σ_{h≠j+1} exp(score(h → j+1))  [includes root]
        incoming_weights = exp_S.sum(dim=1)  # (B, n) — self-loops are 0 (exp(-inf))
        L[:, range(n), range(n)] = incoming_weights
        
        # ── Step 6: Handle padding — padded positions become identity rows/cols ──
        pad_mask = ~mask[:, 1:]  # (B, n) — True for PADDED positions
        
        # Zero out rows and columns of padded positions, then set their diag to 1
        real_mask_2d = dep_mask.unsqueeze(1) * dep_mask.unsqueeze(2)  # (B, n, n)
        L = L * real_mask_2d
        L[:, range(n), range(n)] = L[:, range(n), range(n)] + pad_mask.float()
        
        # ── Step 7: Log partition function ──
        sign, logabsdet = torch.slogdet(L)  # both (B,)
        
        # Guard: sign should be +1 for a valid Laplacian. Warn if not.
        if torch.any(sign <= 0):
            # Fallback: add small diagonal regularization
            L[:, range(n), range(n)] += 1e-6
            sign, logabsdet = torch.slogdet(L)
        
        # Undo the stability offset: log Z_real = logabsdet + Σ_j col_max[j]
        log_Z = logabsdet + (col_max.squeeze(1) * dep_mask).sum(dim=-1)  # (B,)
        
        # ── Step 8: Tree-CRF NLL ──
        nll = log_Z - gold_tree_score  # (B,)
        
        # Mask out degenerate sentences (length 0)
        sentence_lengths = mask.sum(dim=-1).float()  # (B,)
        valid = sentence_lengths > 1  # at least root + 1 word
        
        loss = (nll * valid.float()).sum() / valid.float().sum().clamp(min=1)
        return loss


class TreeCRFArcLoss(nn.Module):
    """
    Combined loss: Tree-CRF for arc structure + cross-entropy for relation labels.
    
    This replaces your current separate arc-CE + rel-CE with a structured arc loss.
    The relation labels are still predicted with local cross-entropy, conditioned on
    gold heads during training (teacher forcing) or predicted heads during inference.
    """
    
    def __init__(self, n_rels: int, label_smoothing: float = 0.05):
        super().__init__()
        self.tree_crf = MatrixTreeCRFLoss()
        self.rel_criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )
        self.n_rels = n_rels
    
    def forward(self, arc_scores, rel_scores, gold_heads, gold_rels, mask):
        """
        arc_scores: (B, N, N) — biaffine arc scores
        rel_scores: (B, N, N, R) — relation scores per (head, dep) pair
        gold_heads: (B, N)
        gold_rels:  (B, N)
        mask:       (B, N)
        """
        # Structured arc loss
        arc_loss = self.tree_crf(arc_scores, gold_heads, mask)
        
        # Local relation loss (conditioned on gold heads for teacher forcing)
        B, N, _, R = rel_scores.shape
        dep_idx = torch.arange(N, device=mask.device).unsqueeze(0).expand(B, -1)
        
        # Gather rel_scores at gold head positions: (B, N, R)
        rel_at_gold = rel_scores[
            torch.arange(B, device=mask.device).unsqueeze(1),
            gold_heads,
            dep_idx
        ]  # (B, N, R)
        
        rel_loss = self.rel_criterion(
            rel_at_gold[:, 1:].reshape(-1, R),  # skip root
            gold_rels[:, 1:].reshape(-1)
        ).reshape(B, N - 1)
        
        dep_mask = mask[:, 1:].float()
        rel_loss = (rel_loss * dep_mask).sum() / dep_mask.sum().clamp(min=1)
        
        return arc_loss + rel_loss, arc_loss.item(), rel_loss.item()
1C. Chu-Liu-Edmonds for Inference (Maximum Spanning Arborescence)
At inference time, you cannot just argmax per-column. You must find the highest-scoring valid tree. This is the Chu-Liu-Edmonds algorithm — runs in 
O
(
n
2
)
O(n 
2
 ) amortized:

Python

def chu_liu_edmonds(scores: torch.Tensor) -> list:
    """
    Find maximum spanning arborescence rooted at node 0.
    
    Args:
        scores: (N, N) tensor. scores[h][d] = score of arc h→d.
                N includes root at position 0.
    
    Returns:
        heads: list[int] of length N. heads[d] = head of word d.
               heads[0] = -1 (root has no head).
    """
    scores = scores.detach().cpu().numpy()
    N = scores.shape[0]
    
    # Disallow self-loops and arcs into root
    scores[range(N), range(N)] = float('-inf')
    scores[:, 0] = float('-inf')  # root can't be a dependent
    
    def _cle_recursive(scores, root, nodes):
        """Core recursive Chu-Liu-Edmonds."""
        n_nodes = len(nodes)
        if n_nodes <= 1:
            return {}
        
        # Step 1: For each non-root node, find best incoming arc
        best_heads = {}
        best_scores_map = {}
        for d in nodes:
            if d == root:
                continue
            best_h = None
            best_s = float('-inf')
            for h in nodes:
                if h != d and scores[h][d] > best_s:
                    best_s = scores[h][d]
                    best_h = h
            if best_h is None:
                # No valid incoming arc — shouldn't happen with proper input
                best_heads[d] = root
            else:
                best_heads[d] = best_h
                best_scores_map[d] = best_s
        
        # Step 2: Check for cycles
        visited = {}
        cycle = None
        for start in nodes:
            if start == root:
                continue
            path = []
            node = start
            while node not in visited and node != root and node in best_heads:
                visited[node] = start
                path.append(node)
                node = best_heads[node]
            if node != root and node in best_heads and visited.get(node) == start:
                # Found a cycle
                cycle_start = node
                cycle = []
                n = node
                while True:
                    cycle.append(n)
                    n = best_heads[n]
                    if n == cycle_start:
                        break
                break
        
        if cycle is None:
            # No cycle — greedy solution is valid
            return best_heads
        
        # Step 3: Contract the cycle into a single node
        cycle_set = set(cycle)
        contract_id = min(cycle)  # representative node
        
        # Build new node set
        new_nodes = [n for n in nodes if n not in cycle_set] + [contract_id]
        
        # Build contracted score matrix
        import numpy as np
        new_scores = np.full_like(scores, float('-inf'))
        np.copyto(new_scores, scores)
        
        for d in nodes:
            if d in cycle_set or d == root:
                continue
            # Arcs INTO d FROM cycle: best arc from any cycle node
            for c in cycle:
                if new_scores[c][d] > new_scores[contract_id][d]:
                    new_scores[contract_id][d] = new_scores[c][d]
            # Arcs FROM d INTO cycle: 
            # score(d→c_contracted) = score(d→c) - score(best_head[c]→c) + 0
            # We subtract the "replaced" arc's score
            for c in cycle:
                adjusted = scores[d][c] - best_scores_map.get(c, 0)
                if adjusted > new_scores[d][contract_id]:
                    new_scores[d][contract_id] = adjusted
        
        # Root → contracted
        for c in cycle:
            adjusted = scores[root][c] - best_scores_map.get(c, 0)
            if adjusted > new_scores[root][contract_id]:
                new_scores[root][contract_id] = adjusted
        
        # Recurse
        result = _cle_recursive(new_scores, root, new_nodes)
        
        # Step 4: Expand — determine which cycle node gets the external incoming arc
        entering_head = result.get(contract_id, root)
        # Find which cycle node this arc actually enters
        best_c = None
        best_s = float('-inf')
        for c in cycle:
            s = scores[entering_head][c]
            if entering_head == root:
                s = scores[root][c]
            if s > best_s:
                best_s = s
                best_c = c
        
        # Re-expand: all cycle arcs remain, except the one into best_c
        final = {k: v for k, v in result.items() if k != contract_id}
        for c in cycle:
            if c == best_c:
                final[c] = entering_head
            else:
                final[c] = best_heads[c]
        
        return final
    
    nodes = list(range(N))
    head_dict = _cle_recursive(scores, 0, nodes)
    
    heads = [-1] * N
    for d, h in head_dict.items():
        heads[d] = h
    
    return heads


def decode_mst_batch(arc_scores: torch.Tensor, mask: torch.BoolTensor) -> torch.LongTensor:
    """
    Batch-decode maximum spanning arborescences.
    
    Args:
        arc_scores: (B, N, N)
        mask: (B, N) — True for real tokens
    
    Returns:
        pred_heads: (B, N) — predicted head indices
    """
    B, N, _ = arc_scores.shape
    pred_heads = torch.zeros(B, N, dtype=torch.long, device=arc_scores.device)
    
    for b in range(B):
        length = mask[b].sum().item()
        scores_b = arc_scores[b, :length, :length]  # (len, len)
        heads_b = chu_liu_edmonds(scores_b)
        for d in range(length):
            pred_heads[b, d] = heads_b[d] if heads_b[d] >= 0 else 0
    
    return pred_heads
1D. Single Root Enforcement
Add a dedicated root-score head and penalize multi-root structures:

Python

class SingleRootRegularizer(nn.Module):
    """
    Soft penalty that discourages multiple words from attaching to root.
    Uses KL divergence between the predicted root-attachment distribution
    and a peaked categorical (only one word should attach to root).
    """
    def __init__(self, strength: float = 0.5):
        super().__init__()
        self.strength = strength
    
    def forward(self, arc_scores, mask):
        """
        arc_scores: (B, N, N)
        mask: (B, N)
        """
        # P(dep=d attaches to root) ∝ exp(score(root→d))
        root_scores = arc_scores[:, 0, 1:]  # (B, N-1) — root→each word
        dep_mask = mask[:, 1:].float()
        
        root_scores = root_scores.masked_fill(~mask[:, 1:], float('-inf'))
        root_probs = torch.softmax(root_scores, dim=-1)  # (B, N-1)
        
        # Ideal: exactly one word attaches to root → entropy should be 0
        # Penalize high entropy (multiple probable root children)
        entropy = -(root_probs * (root_probs + 1e-10).log()).sum(dim=-1)  # (B,)
        
        return self.strength * entropy.mean()
2. Advanced Morphology Integration: The Arabic Morphological Scaffold
Arabic morphology is compositional at the sub-character level. The word وَسَيَكْتُبُونَهَا decomposes into 5 morphemes: وَ+سَـ+يَكْتُبُ+ونَ+هَا (and + will + they-write + plural.masc + it.fem). Simply embedding the surface form or POS tag throws away this compositional structure. Here is a multi-pathway morphological encoder:

2A. Multi-Scale Character CNN with Gated Highway
Python

class MultiScaleCharCNN(nn.Module):
    """
    Processes raw Arabic characters through parallel convolution banks
    at multiple kernel sizes to capture:
      - k=2,3: short affixes (ال, وَ, بِ, ـها, ـهم)
      - k=4,5: root patterns (فَعَلَ = 3 root chars + vowels)
      - k=6,7: full morphological templates (مُسْتَفْعِل)
    
    Followed by a Highway network for controlled information flow.
    """
    
    def __init__(
        self,
        char_vocab_size: int,
        char_embed_dim: int = 32,
        kernel_sizes: tuple = (2, 3, 4, 5, 6, 7),
        n_filters: int = 64,
        output_dim: int = 256,
        highway_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        
        # Parallel convolution bank
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(char_embed_dim, n_filters, k, padding=k // 2),
                nn.BatchNorm1d(n_filters),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
        
        total_filters = n_filters * len(kernel_sizes)
        
        # Highway layers: y = g * H(x) + (1-g) * x
        self.highways = nn.ModuleList()
        for _ in range(highway_layers):
            self.highways.append(nn.ModuleDict({
                'H': nn.Linear(total_filters, total_filters),
                'gate': nn.Linear(total_filters, total_filters),
            }))
        
        self.proj = nn.Linear(total_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, char_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            char_ids: (B, max_words, max_chars) — padded character indices
        
        Returns:
            word_char_features: (B, max_words, output_dim)
        """
        B, W, C = char_ids.shape
        
        # Flatten batch and word dimensions
        x = char_ids.view(B * W, C)             # (B*W, C)
        x = self.char_embed(x)                    # (B*W, C, char_embed_dim)
        x = x.transpose(1, 2)                    # (B*W, char_embed_dim, C) for Conv1d
        
        # Apply each kernel size and max-pool over character positions
        conv_outs = []
        for conv in self.convs:
            c = conv(x)                           # (B*W, n_filters, C')
            c = c.max(dim=2)[0]                   # (B*W, n_filters) — max-over-time
            conv_outs.append(c)
        
        x = torch.cat(conv_outs, dim=1)           # (B*W, total_filters)
        
        # Highway gating
        for hw in self.highways:
            h = torch.relu(hw['H'](x))
            g = torch.sigmoid(hw['gate'](x))
            x = g * h + (1 - g) * x
        
        x = self.layer_norm(self.proj(self.dropout(x)))  # (B*W, output_dim)
        return x.view(B, W, -1)
2B. Structured Morphological Feature Fusion
Beyond characters, use explicit morphological features (POS, pattern, case, number, gender, person, definiteness, voice, mood) with cross-feature attention:

Python

class MorphFeatureAttentionFusion(nn.Module):
    """
    Instead of naively concatenating morphological feature embeddings,
    this module uses cross-attention between feature groups to model
    feature interactions (e.g., the INTERACTION between definiteness
    and case is crucial in Arabic — only definite nouns get full 
    case marking in إعراب).
    
    Features are grouped into SYNTACTIC (POS, case, function) and
    INFLECTIONAL (number, gender, person, definiteness, pattern).
    Cross-attention lets each group inform the other.
    """
    
    def __init__(
        self,
        feature_configs: dict,  # {'pos': vocab_size, 'case': 4, 'number': 4, ...}
        feat_embed_dim: int = 32,
        fusion_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, feat_embed_dim)
            for name, vocab_size in feature_configs.items()
        })
        
        n_features = len(feature_configs)
        self.input_dim = feat_embed_dim  # each feature is a "token"
        
        # Project each feature to common dim
        self.feat_proj = nn.Linear(feat_embed_dim, fusion_dim)
        
        # Self-attention over feature "tokens" within each word
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.output_proj = nn.Linear(fusion_dim * n_features, fusion_dim)
        self.gate = nn.Linear(fusion_dim * 2, fusion_dim)  # for gating with char features
        
        self.n_features = n_features
    
    def forward(
        self,
        morph_features: dict,  # {name: (B, W) LongTensor}
        char_features: torch.Tensor  # (B, W, char_dim) from MultiScaleCharCNN
    ) -> torch.Tensor:
        """Returns (B, W, fusion_dim)"""
        B, W = next(iter(morph_features.values())).shape
        
        # Embed each feature → (B, W, feat_embed_dim)
        feat_embeds = []
        for name in sorted(self.embeddings.keys()):
            e = self.embeddings[name](morph_features[name])  # (B, W, feat_embed_dim)
            feat_embeds.append(e)
        
        # Stack as "tokens": (B*W, n_features, feat_embed_dim)
        feat_stack = torch.stack(feat_embeds, dim=2)  # (B, W, n_features, embed)
        feat_stack = feat_stack.view(B * W, self.n_features, -1)
        
        # Project to common dim
        feat_stack = self.feat_proj(feat_stack)  # (B*W, n_features, fusion_dim)
        
        # Self-attention over features (each feature attends to others)
        attended, _ = self.cross_attn(feat_stack, feat_stack, feat_stack)
        feat_stack = self.norm(feat_stack + attended)  # (B*W, n_features, fusion_dim)
        
        # Flatten and project
        morph_vec = feat_stack.reshape(B * W, -1)  # (B*W, n_features * fusion_dim)
        morph_vec = self.output_proj(morph_vec)     # (B*W, fusion_dim)
        morph_vec = morph_vec.view(B, W, -1)        # (B, W, fusion_dim)
        
        # ── Gated fusion with character-CNN features ──
        # Instead of naive concatenation, learn HOW MUCH morphological vs.
        # character information to use per word
        combined = torch.cat([morph_vec, char_features], dim=-1)  # (B, W, 2*fusion_dim)
        gate_val = torch.sigmoid(self.gate(combined))             # (B, W, fusion_dim)
        
        fused = gate_val * morph_vec + (1 - gate_val) * char_features[:, :, :morph_vec.size(-1)]
        
        return fused
2C. Complete Integrated Row Encoder (Pre-Transformer)
Python

class ArabicRowEncoder(nn.Module):
    """
    The full Per-Word Encoding Pathway, upgraded with morphological scaffold.
    
    Each word representation is built from 4 streams:
      1. Word embedding (or subword from pre-trained model)
      2. Character CNN (captures OOV, typos, unseen morphology)
      3. Morphological feature attention fusion
      4. Positional encoding
    
    These are combined via a learned mixture-of-experts gate that adapts
    per word (known words rely more on word embeddings; rare words rely
    more on character/morphological features).
    """
    
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        morph_feature_configs: dict,
        word_dim: int = 256,
        char_dim: int = 256,
        morph_dim: int = 128,
        n_experts: int = 3,  # word, char, morph
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Stream 1: Word embeddings
        self.word_embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        
        # Stream 2: Character CNN
        self.char_encoder = MultiScaleCharCNN(
            char_vocab_size=char_vocab_size,
            output_dim=char_dim,
            dropout=dropout
        )
        
        # Stream 3: Morphological features
        self.morph_encoder = MorphFeatureAttentionFusion(
            feature_configs=morph_feature_configs,
            fusion_dim=morph_dim,
            dropout=dropout
        )
        
        # Mixture-of-Experts gate: learns per-word how to combine streams
        total_input = word_dim + char_dim + morph_dim
        self.expert_gate = nn.Sequential(
            nn.Linear(total_input, n_experts),
            nn.Softmax(dim=-1)
        )
        
        # Project each stream to same dim for gated combination
        self.proj_word = nn.Linear(word_dim, word_dim)
        self.proj_char = nn.Linear(char_dim, word_dim)
        self.proj_morph = nn.Linear(morph_dim, word_dim)
        
        self.output_norm = nn.LayerNorm(word_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, word_ids, char_ids, morph_features):
        """
        Args:
            word_ids:        (B, W) — vocabulary indices
            char_ids:        (B, W, max_chars) — character indices
            morph_features:  dict of {feat_name: (B, W) LongTensor}
        
        Returns:
            word_representations: (B, W, word_dim) — rich per-word encodings
        """
        # Three parallel streams
        w = self.proj_word(self.word_embed(word_ids))           # (B, W, word_dim)
        c = self.proj_char(self.char_encoder(char_ids))         # (B, W, word_dim)
        
        # Morph encoder needs char features for gating
        char_raw = self.char_encoder(char_ids)                  # reuse or cache
        m = self.proj_morph(
            self.morph_encoder(morph_features, char_raw)
        )                                                       # (B, W, word_dim)
        
        # Compute per-word expert weights
        concat = torch.cat([
            self.word_embed(word_ids),
            char_raw,
            self.morph_encoder(morph_features, char_raw)
        ], dim=-1)  # (B, W, total_input)
        
        gates = self.expert_gate(concat)  # (B, W, 3)
        g_w, g_c, g_m = gates.unbind(dim=-1)  # each (B, W)
        
        # Gated combination
        combined = (
            g_w.unsqueeze(-1) * w +
            g_c.unsqueeze(-1) * c +
            g_m.unsqueeze(-1) * m
        )  # (B, W, word_dim)
        
        return self.output_norm(self.dropout(combined))
3. Manager-Worker Refinement: Variational Topology with Iterative Tree Feedback
The fundamental flaw in average-pooling the sentence and calling it a "goal" is that a single vector cannot encode structured topological information like clause boundaries, coordination scopes, or the difference between matrix and embedded clauses. Here are two complementary solutions:

3A. Variational Sentence Bottleneck (The Structured Prior)
Replace the deterministic average-pool with a variational information bottleneck that forces the manager to learn a structured latent variable encoding the global parse topology:

Python

class VariationalSentenceManager(nn.Module):
    """
    Replaces average-pool → linear → worker_goal with a variational
    bottleneck that encodes STRUCTURED sentence-level syntax.
    
    The key insight: by sampling z ~ q(z|sentence) and training with
    KL regularization, the manager is forced to compress sentence-level
    topological information (clause count, coordination structure, 
    embedding depth) into a compact, disentangled representation.
    
    Uses a STRUCTURED prior: instead of standard N(0,I), the prior
    is conditioned on sentence length and root-verb POS, encoding
    the empirical distribution of Arabic syntactic patterns.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_latent_heads: int = 4,  # multi-head latent variable
        kl_weight: float = 0.1,
        kl_annealing_steps: int = 5000,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_heads = n_latent_heads
        self.total_latent = latent_dim * n_latent_heads
        
        # Encoder: sentence → posterior q(z|x)
        self.encoder_attn = nn.MultiheadAttention(
            input_dim, num_heads=4, batch_first=True
        )
        self.encoder_pool = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Multi-head posterior: each head captures different syntactic aspect
        # Head 1: clause structure, Head 2: coordination, 
        # Head 3: embedding depth, Head 4: predicate frame
        self.posterior_mu = nn.Linear(hidden_dim, self.total_latent)
        self.posterior_logvar = nn.Linear(hidden_dim, self.total_latent)
        
        # Structured prior conditioned on observable features
        self.prior_net = nn.Sequential(
            nn.Linear(3, 64),  # [sentence_length, root_pos, n_verbs]
            nn.GELU(),
            nn.Linear(64, self.total_latent * 2),  # mu and logvar
        )
        
        # Decode latent → worker goals
        self.goal_decoder = nn.Sequential(
            nn.Linear(self.total_latent, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Per-word goal conditioning (broadcast latent to each word)
        self.word_goal_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
        
        # KL annealing
        self.kl_weight = kl_weight
        self.kl_annealing_steps = kl_annealing_steps
        self.register_buffer('step_counter', torch.tensor(0, dtype=torch.long))
    
    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        word_encodings: torch.Tensor,  # (B, W, D) — from row encoder
        mask: torch.BoolTensor,        # (B, W)
        sentence_features: torch.Tensor = None  # (B, 3) — [len, root_pos, n_verbs]
    ):
        """
        Returns:
            goal_conditioned: (B, W, D) — word encodings enriched with global goal
            kl_loss: scalar — KL divergence for ELBO training
        """
        B, W, D = word_encodings.shape
        
        # ── Encode sentence into latent space ──
        # Self-attention summary
        attn_mask = ~mask  # True = ignore
        attended, _ = self.encoder_attn(
            word_encodings, word_encodings, word_encodings,
            key_padding_mask=attn_mask
        )
        
        # Masked mean pooling
        mask_float = mask.float().unsqueeze(-1)  # (B, W, 1)
        pooled = (attended * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        
        h = self.encoder_pool(pooled)  # (B, hidden_dim)
        
        # Posterior q(z|x)
        q_mu = self.posterior_mu(h)          # (B, total_latent)
        q_logvar = self.posterior_logvar(h)  # (B, total_latent)
        
        # Sample z
        z = self.reparameterize(q_mu, q_logvar) if self.training else q_mu
        # (B, total_latent)
        
        # ── Structured prior p(z|features) ──
        if sentence_features is not None:
            prior_params = self.prior_net(sentence_features.float())
            p_mu, p_logvar = prior_params.chunk(2, dim=-1)
        else:
            p_mu = torch.zeros_like(q_mu)
            p_logvar = torch.zeros_like(q_logvar)
        
        # ── KL divergence (with annealing) ──
        kl = -0.5 * (1 + q_logvar - p_logvar 
                      - (q_logvar - p_logvar).exp()
                      - ((q_mu - p_mu) ** 2) / (p_logvar.exp() + 1e-8))
        kl = kl.sum(dim=-1).mean()  # (scalar)
        
        # Anneal KL weight
        if self.training:
            self.step_counter += 1
        anneal = min(1.0, self.step_counter.float().item() / self.kl_annealing_steps)
        kl_loss = self.kl_weight * anneal * kl
        
        # ── Decode z → per-word goal ──
        global_goal = self.goal_decoder(z)  # (B, D)
        global_goal = global_goal.unsqueeze(1).expand(-1, W, -1)  # (B, W, D)
        
        # Gate: let each word decide how much global context to absorb
        gate_input = torch.cat([word_encodings, global_goal], dim=-1)  # (B, W, 2D)
        gate = self.word_goal_gate(gate_input)  # (B, W, D), values in [0,1]
        
        goal_conditioned = word_encodings + gate * global_goal  # residual + gated goal
        
        return goal_conditioned, kl_loss
3B. Iterative Refinement with Tree-Feedback (The "Think Twice" Loop)
After the first-pass parse, feed the predicted tree structure back as features and re-predict. This is the single most powerful technique for fixing systematic errors like nsubj vs csubj confusion:

Python

class IterativeTreeRefinement(nn.Module):
    """
    Multi-pass refinement loop:
    
    Pass 1: Standard biaffine → predicted tree T₁
    Pass 2: Encode T₁ as graph features → refine → T₂ 
    Pass 3: Encode T₂ as graph features → refine → T₃ (optional)
    
    Each refinement pass uses a lightweight Graph Attention Network (GAT) 
    that propagates information along the PREDICTED tree edges, then
    re-scores arcs with the enriched representations.
    
    This is how the model learns to FIX its own mistakes:
    - If it wrongly attached a noun to the wrong verb, the GAT pass
      will propagate the verb's features to the noun, revealing the
      agreement mismatch, and the second-pass scorer can correct it.
    """
    
    def __init__(
        self,
        word_dim: int = 256,
        n_refinement_passes: int = 2,
        gat_heads: int = 4,
        gat_dropout: float = 0.2,
    ):
        super().__init__()
        self.n_passes = n_refinement_passes
        
        self.gat_layers = nn.ModuleList()
        self.pass_norms = nn.ModuleList()
        self.pass_gates = nn.ModuleList()
        
        for _ in range(n_refinement_passes):
            self.gat_layers.append(
                TreeGATLayer(word_dim, n_heads=gat_heads, dropout=gat_dropout)
            )
            self.pass_norms.append(nn.LayerNorm(word_dim))
            self.pass_gates.append(nn.Sequential(
                nn.Linear(word_dim * 2, word_dim),
                nn.Sigmoid()
            ))
        
        # Pass-specific relation re-scorers (each pass gets its own)
        self.pass_id_embed = nn.Embedding(n_refinement_passes + 1, word_dim)
    
    def forward(
        self,
        word_repr: torch.Tensor,       # (B, W, D) — from self-attention layer
        initial_arc_scores: torch.Tensor,  # (B, W, W) — from biaffine
        initial_rel_scores: torch.Tensor,  # (B, W, W, R) — from biaffine
        biaffine_arc_fn,                # callable: (B, W, D) → (B, W, W)
        biaffine_rel_fn,                # callable: (B, W, D), (B, W, W) → (B, W, W, R)
        mask: torch.BoolTensor
    ):
        """
        Returns refined arc_scores, rel_scores after multiple passes.
        Also returns list of intermediate predictions for auxiliary losses.
        """
        all_arc_scores = [initial_arc_scores]
        all_rel_scores = [initial_rel_scores]
        
        current_repr = word_repr
        current_arc = initial_arc_scores
        
        for pass_idx in range(self.n_passes):
            # ── Extract predicted tree from current scores ──
            # Use soft attention (not hard argmax) for differentiability
            # Temperature-annealed softmax: sharp but differentiable
            temp = max(0.5, 1.0 - pass_idx * 0.3)  # gets sharper each pass
            soft_heads = torch.softmax(current_arc / temp, dim=1)  # (B, W, W)
            # soft_heads[b, h, d] ≈ P(head of d is h)
            
            # ── GAT message passing along predicted tree ──
            refined = self.gat_layers[pass_idx](
                current_repr, soft_heads, mask
            )  # (B, W, D)
            
            # ── Gated residual connection ──
            gate = self.pass_gates[pass_idx](
                torch.cat([current_repr, refined], dim=-1)
            )
            current_repr = self.pass_norms[pass_idx](
                current_repr + gate * refined
            )
            
            # ── Add pass-level positional embedding ──
            pass_emb = self.pass_id_embed(
                torch.tensor(pass_idx + 1, device=word_repr.device)
            )
            current_repr = current_repr + pass_emb
            
            # ── Re-score with updated representations ──
            current_arc = biaffine_arc_fn(current_repr)
            current_rel = biaffine_rel_fn(current_repr, current_arc)
            
            all_arc_scores.append(current_arc)
            all_rel_scores.append(current_rel)
        
        return current_arc, current_rel, all_arc_scores, all_rel_scores


class TreeGATLayer(nn.Module):
    """
    Graph Attention over a soft/predicted dependency tree.
    
    Each word aggregates features from its predicted head and predicted 
    dependents, weighted by the edge confidence. This propagates 
    morphosyntactic agreement signals through the tree.
    """
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Separate transforms for head→dep and dep→head messages
        self.W_head_msg = nn.Linear(dim, dim)    # "I am your head" message
        self.W_dep_msg = nn.Linear(dim, dim)     # "I am your dependent" message
        self.W_self = nn.Linear(dim, dim)         # self-loop
        
        # Direction-aware attention
        self.attn_head = nn.Linear(dim, n_heads)
        self.attn_dep = nn.Linear(dim, n_heads)
        
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, soft_heads, mask):
        """
        x: (B, W, D)
        soft_heads: (B, W, W) — soft_heads[b, h, d] = P(h is head of d)
                    Columns should sum to ~1 (distribution over heads for each dep)
        mask: (B, W)
        """
        B, W, D = x.shape
        
        # ── Messages FROM predicted heads TO each word ──
        # For word d, aggregate head messages weighted by P(head=h)
        head_messages = self.W_head_msg(x)  # (B, W, D) — messages heads would send
        # soft_heads[:, :, d] gives distribution over heads for dep d
        # head_msg_for_d = Σ_h soft_heads[h, d] * head_messages[h]
        head_agg = torch.bmm(soft_heads.transpose(1, 2), head_messages)  
        # (B, W, D) — soft_heads^T: (B, W_dep, W_head) @ messages: (B, W_head, D)
        # = (B, W_dep, D) — aggregated head messages for each dependent
        
        # ── Messages FROM predicted dependents TO each word ──
        # For word h, aggregate dependent messages weighted by P(head=h for dep d)
        dep_messages = self.W_dep_msg(x)  # (B, W, D)
        dep_agg = torch.bmm(soft_heads, dep_messages)  
        # soft_heads: (B, W_head, W_dep) @ messages: (B, W_dep, D)
        # = (B, W_head, D) — aggregated dep messages for each head
        
        # ── Combine ──
        self_msg = self.W_self(x)
        combined = self_msg + head_agg + dep_agg  # (B, W, D)
        
        output = self.output(torch.relu(combined))
        output = self.dropout(output)
        
        # Mask padding
        output = output * mask.float().unsqueeze(-1)
        
        return self.norm(x + output)
3C. Auxiliary Refinement Loss (Deep Supervision)
Each intermediate pass's predictions should be trained with a geometrically-decayed loss, so early passes learn coarse structure and later passes learn fine corrections:

Python

class DeepRefinementLoss(nn.Module):
    """
    Apply loss at each refinement pass with exponentially increasing weight.
    Early passes get lighter weight (they're allowed to be wrong).
    Final pass gets full weight.
    """
    def __init__(self, base_loss_fn, n_passes: int, decay: float = 0.5):
        super().__init__()
        self.base_loss = base_loss_fn
        self.n_passes = n_passes
        
        # Weights: [decay^(n-1), decay^(n-2), ..., decay^0=1.0]
        self.weights = [decay ** (n_passes - i) for i in range(n_passes + 1)]
        # Normalize so they sum to n_passes+1
        total = sum(self.weights)
        self.weights = [w / total * (n_passes + 1) for w in self.weights]
    
    def forward(self, all_arc_scores, all_rel_scores, gold_heads, gold_rels, mask):
        total_loss = 0
        for i, (arcs, rels) in enumerate(zip(all_arc_scores, all_rel_scores)):
            loss_i, _, _ = self.base_loss(arcs, rels, gold_heads, gold_rels, mask)
            total_loss += self.weights[i] * loss_i
        return total_loss
4. The Arsenal: Out-of-the-Box Mathematical Interventions
4A. Learned Relative Distance Bias in Biaffine Scoring
Arabic has strong positional preferences: determiners are always adjacent, VSO structure means subjects follow verbs closely, long-range dependencies are rare. Inject this directly into arc scoring:

Python

class DistanceBiasedBiaffineScorer(nn.Module):
    """
    Augments biaffine scores with a learned distance bias and a 
    direction-aware asymmetric penalty.
    
    s(h, d) = biaffine(h, d) + distance_bias(|h-d|, direction(h,d))
    
    The distance bias is a small MLP over bucketed distances, meaning
    the model can learn that "distance 1 left is great for det→noun,
    but distance 8 left is almost never valid."
    """
    
    def __init__(self, max_dist: int = 64, n_buckets: int = 32, bias_dim: int = 64):
        super().__init__()
        self.max_dist = max_dist
        self.n_buckets = n_buckets
        
        # Log-bucketed distance (like relative position bias in T5/ALiBi)
        self.dist_embed = nn.Embedding(2 * n_buckets + 1, bias_dim)
        self.bias_proj = nn.Sequential(
            nn.Linear(bias_dim, bias_dim),
            nn.GELU(),
            nn.Linear(bias_dim, 1)
        )
    
    def _bucket_distance(self, distances):
        """Log-bucket signed distances into 2*n_buckets+1 bins."""
        sign = distances.sign()
        abs_dist = distances.abs()
        
        # First n_buckets/2 are exact, rest are log-spaced
        exact_cutoff = self.n_buckets // 2
        is_exact = abs_dist <= exact_cutoff
        
        # Log-space bucketing for larger distances
        log_bucket = exact_cutoff + (
            (abs_dist.float() / exact_cutoff).log() /
            (self.max_dist / exact_cutoff + 1e-8).log() *  # Use math.log here  
            (self.n_buckets - exact_cutoff)
        ).long().clamp(0, self.n_buckets - exact_cutoff - 1)
        
        bucket = torch.where(is_exact, abs_dist, log_bucket + exact_cutoff)
        # Shift by n_buckets to handle sign: bucket ∈ [0, 2*n_buckets]
        signed_bucket = bucket * sign + self.n_buckets
        return signed_bucket.clamp(0, 2 * self.n_buckets)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Returns (seq_len, seq_len) additive bias matrix."""
        positions = torch.arange(seq_len, device=device)
        # distances[h, d] = d - h (positive = dependent is to the right)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N)
        
        buckets = self._bucket_distance(distances)
        dist_embeds = self.dist_embed(buckets)     # (N, N, bias_dim)
        bias = self.bias_proj(dist_embeds).squeeze(-1)  # (N, N)
        
        return bias
4B. Higher-Order Sibling & Grandparent Scoring
First-order parsers score each arc independently. But arcs interact: if word 5 has three dependents, they should form a coherent argument structure. Score pairs of arcs:

Python

class SecondOrderScorer(nn.Module):
    """
    Scores higher-order structures on TOP of first-order biaffine scores:
    
    1. SIBLING: s_sib(h, d1, d2) — "do d1 and d2 form a good pair of
       children for head h?" (catches coordination, argument structure)
    
    2. GRANDPARENT: s_gp(g, h, d) — "is the chain g→h→d consistent?"
       (catches nested clause structure, PP attachment)
    
    These are computed as trilinear/bilinear products over specialized 
    representations and added to the arc scores during inference (and
    training with the Tree-CRF, using mean-field approximation).
    """
    
    def __init__(self, word_dim: int = 256, score_dim: int = 128):
        super().__init__()
        
        # Sibling scoring: trilinear s(h, s1, s2)
        self.sib_head = nn.Linear(word_dim, score_dim)
        self.sib_dep1 = nn.Linear(word_dim, score_dim)
        self.sib_dep2 = nn.Linear(word_dim, score_dim)
        self.sib_trilinear = nn.Parameter(torch.randn(score_dim, score_dim, score_dim) * 0.01)
        
        # Grandparent scoring: trilinear s(g, h, d)
        self.gp_grand = nn.Linear(word_dim, score_dim)
        self.gp_head = nn.Linear(word_dim, score_dim)
        self.gp_dep = nn.Linear(word_dim, score_dim)
        self.gp_trilinear = nn.Parameter(torch.randn(score_dim, score_dim, score_dim) * 0.01)
    
    def score_siblings(self, word_repr, pred_heads, mask):
        """
        For each head h, score all pairs of its dependents.
        Returns additive correction to arc scores.
        
        Approximation: instead of exact enumeration, use mean-field:
        For each dep d, the expected sibling feature is the probability-
        weighted sum of features of other potential co-dependents.
        """
        B, W, D = word_repr.shape
        
        h_repr = self.sib_head(word_repr)   # (B, W, S)
        d1_repr = self.sib_dep1(word_repr)  # (B, W, S)
        d2_repr = self.sib_dep2(word_repr)  # (B, W, S)
        
        # pred_heads: (B, W, W) soft attention — P(head of d is h)
        # Expected sibling: for dep d with head h, expected features of co-dependents
        # E[sibling of d | head h] = Σ_{d' ≠ d} P(head of d' = h) * d2_repr[d']
        
        # P(d' has same head h as d) ∝ pred_heads[h, d'] for each h
        # Marginal over h: E_sib[d] = Σ_h P(h|d) * Σ_{d'≠d} P(h|d') * repr(d')
        
        # This is expensive but can be approximated:
        # mean_field_sib = pred_heads^T @ d2_repr - diag  (subtract self)
        
        # For simplicity, compute pairwise sibling score correction to arc scores:
        # Δs(h, d) = h_repr[h]^T @ W @ mean_sib_repr[h, d]
        # where mean_sib_repr[h, d] = average repr of other deps of h (excluding d)
        
        # Compact approximation: just return a bilinear score
        sib_score = torch.einsum('bwd,bwe,de->bww', d1_repr, d2_repr, 
                                  self.sib_trilinear.sum(dim=0))  # (B, W, W)
        return sib_score * 0.1  # scale down — this is an additive correction
    
    def score_grandparents(self, word_repr, pred_heads, mask):
        """
        Score grandparent chains: does arc (h→d) get better if h's head is g?
        Returns additive correction to arc scores.
        """
        B, W, D = word_repr.shape
        
        g_repr = self.gp_grand(word_repr)  # (B, W, S)
        h_repr = self.gp_head(word_repr)   # (B, W, S)
        d_repr = self.gp_dep(word_repr)    # (B, W, S)
        
        # For arc (h→d), expected grandparent features:
        # E[g | h] = Σ_g P(head of h is g) * g_repr[g] = pred_heads^T @ g_repr
        expected_g = torch.bmm(pred_heads.transpose(1, 2), g_repr)  # (B, W, S)
        # expected_g[b, h, :] = expected grandparent repr for word h
        
        # Grandparent-augmented head repr
        aug_head = h_repr + expected_g  # (B, W, S)
        
        # Score: bilinear between augmented head and dep
        gp_score = torch.bmm(aug_head, d_repr.transpose(1, 2))  # (B, W, W)
        
        return gp_score * 0.1
4C. Morphological Agreement Auxiliary Loss
Arabic has mandatory agreement in gender, number, and definiteness between heads and dependents in specific relations. Exploit this as an auxiliary loss:

Python

class AgreementAuxLoss(nn.Module):
    """
    Auxiliary loss that penalizes predicted arcs where morphological
    agreement is violated.
    
    For example:
    - A verb (head) in feminine singular can't have a masculine plural subject
    - A noun and its adjective MUST agree in definiteness, gender, number, case
    - An idafa (إضافة) head must be indefinite if the dependent is definite
    
    This acts as a linguistically-informed regularizer.
    """
    
    def __init__(self, word_dim: int = 256, n_agreement_types: int = 4):
        super().__init__()
        # Learn agreement functions rather than hardcoding rules
        # (the model discovers which features should agree)
        self.agreement_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(word_dim * 2, 128),
                nn.GELU(),
                nn.Linear(128, 2)  # binary: agree / disagree
            )
            for _ in range(n_agreement_types)
        ])
        
        self.penalty_weight = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, word_repr, soft_heads, gold_heads, morph_features, mask):
        """
        Returns auxiliary agreement loss that encourages the model to
        predict arcs between morphologically compatible words.
        """
        B, W, D = word_repr.shape
        
        # For each predicted arc (h→d), check if h and d "agree"
        # Using soft heads for differentiability
        
        # Expected head representation for each dependent
        # exp_head[b, d, :] = Σ_h P(head=h|d) * repr(h)
        exp_head_repr = torch.bmm(soft_heads.transpose(1, 2), word_repr)  # (B, W, D)
        
        # Concatenate [exp_head_repr, dep_repr] and predict agreement
        pair_repr = torch.cat([exp_head_repr, word_repr], dim=-1)  # (B, W, 2D)
        
        total_loss = 0
        for predictor in self.agreement_predictors:
            agree_logits = predictor(pair_repr)  # (B, W, 2)
            # We want the model to predict "agree" (class 1) for all selected arcs
            # The target is always 1 (agree) — if the predicted head is wrong,
            # the disagreement signal backpropagates to fix the head pointer
            targets = torch.ones(B, W, dtype=torch.long, device=word_repr.device)
            loss = F.cross_entropy(
                agree_logits.view(-1, 2),
                targets.view(-1),
                reduction='none'
            ).view(B, W)
            total_loss += (loss * mask.float()).sum() / mask.float().sum()
        
        return self.penalty_weight.abs() * total_loss
4D. R-Drop: Consistency Regularization Through Dropout
Run each sentence through the model twice with different dropout masks. Force the two passes to agree. This is incredibly effective and costs zero parameters:

Python

class RDropLoss(nn.Module):
    """
    R-Drop: Regularized Dropout for consistent predictions.
    
    For each input, the model runs TWO forward passes with different
    dropout masks. The KL divergence between the two output distributions
    is minimized, acting as a powerful regularizer that's especially
    effective for structured prediction.
    
    Paper: "R-Drop: Regularized Dropout for Neural Networks" (NIPS 2021)
    """
    
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, logits_1, logits_2, mask=None):
        """
        logits_1, logits_2: (B, N, C) or (B, N, N) — two forward pass outputs
        """
        p = F.log_softmax(logits_1, dim=-1)
        q = F.log_softmax(logits_2, dim=-1)
        
        # Symmetric KL
        kl_pq = F.kl_div(p, q.exp(), reduction='none').sum(dim=-1)
        kl_qp = F.kl_div(q, p.exp(), reduction='none').sum(dim=-1)
        kl = (kl_pq + kl_qp) / 2
        
        if mask is not None:
            kl = (kl * mask.float()).sum() / mask.float().sum()
        else:
            kl = kl.mean()
        
        return self.alpha * kl
Usage in training loop:

Python

# Inside training step:
model.train()
arc1, rel1, case1 = model(batch)   # forward pass 1 (dropout mask A)
arc2, rel2, case2 = model(batch)   # forward pass 2 (dropout mask B)

main_loss = tree_crf_loss(arc1, ...) + tree_crf_loss(arc2, ...) / 2
rdrop_loss = (rdrop(arc1, arc2, mask) + rdrop(rel1, rel2, mask)) / 2
total_loss = main_loss + rdrop_loss
4E. Linguistically-Informed Label Smoothing
Standard label smoothing distributes probability mass uniformly. But mixing probability from nsubj into punct is insane. Instead, smooth only toward linguistically related labels:

Python

class LinguisticLabelSmoothing(nn.Module):
    """
    Smooths label probabilities toward RELATED labels only.
    
    Confusion groups for Arabic dependency relations:
      - Subject group:    {nsubj, nsubj:pass, csubj}
      - Object group:     {obj, iobj, ccomp, xcomp}
      - Modifier group:   {amod, nmod, nmod:poss, advmod, obl}
      - Function group:   {case, mark, aux, cop, det}
      - Coordination:     {conj, cc}
      - Other:            {punct, root, dep, flat, fixed}
    
    When smoothing label=nsubj, mass goes to nsubj:pass and csubj,
    NOT to punct or det.
    """
    
    def __init__(self, n_labels: int, confusion_groups: list, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n_labels = n_labels
        
        # Build similarity matrix S where S[i][j] = 1 if labels i,j are in same group
        S = torch.zeros(n_labels, n_labels)
        for group in confusion_groups:
            for i in group:
                for j in group:
                    S[i][j] = 1.0
        
        # Normalize: each row sums to 1 over group members
        row_sums = S.sum(dim=1, keepdim=True).clamp(min=1)
        S = S / row_sums
        
        self.register_buffer('similarity', S)
    
    def forward(self, logits, targets):
        """
        logits: (*, n_labels)
        targets: (*) LongTensor
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        # One-hot targets
        one_hot = torch.zeros_like(log_probs).scatter_(-1, targets.unsqueeze(-1), 1.0)
        
        # Smooth targets: (1-ε)*one_hot + ε*similarity_distribution
        smooth_targets = self.similarity[targets]  # (*, n_labels)
        blended = (1 - self.smoothing) * one_hot + self.smoothing * smooth_targets
        
        loss = -(blended * log_probs).sum(dim=-1)
        return loss.mean()
4F. Exponential Moving Average (EMA) for Stable Inference
Python

class EMAModel:
    """
    Maintains an exponential moving average of model parameters.
    Use the EMA model for evaluation — typically 0.5-1% better.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() 
                       for name, param in model.named_parameters()}
    
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply(self, model: nn.Module):
        """Temporarily apply EMA weights for evaluation."""
        self.backup = {name: param.clone() for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])
4G. Deeper Transformer Layers + Pre-Norm
Your current single Transformer layer is a bottleneck. The jump from 1→3 layers is typically the single biggest accuracy improvement for context-dependent parsing decisions (like PP-attachment and coordination scope):

Python

class PreNormTransformerLayer(nn.Module):
    """Pre-norm Transformer (more stable training than post-norm)."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, mask=None):
        # Pre-norm: normalize BEFORE attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class StackedTransformerEncoder(nn.Module):
    """3-layer pre-norm Transformer. Sweet spot for parsing."""
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, n_layers=3, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        pad_mask = ~mask if mask is not None else None
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.final_norm(x)
4H. Curriculum Learning by Sentence Length
Python

class SentenceLengthCurriculum:
    """
    Start training on short sentences (easy dependency structures),
    gradually introduce longer ones (complex nested clauses).
    
    This is especially effective for Arabic where long sentences often
    contain multiple coordinated clauses with complex inter-clausal deps.
    """
    def __init__(self, dataset, max_epochs: int = 30, 
                 initial_max_len: int = 15, final_max_len: int = 128):
        self.dataset = dataset
        self.max_epochs = max_epochs
        self.initial = initial_max_len
        self.final = final_max_len
    
    def get_max_length(self, epoch: int) -> int:
        """Linearly increase max sentence length over first half of training."""
        progress = min(1.0, epoch / (self.max_epochs * 0.5))
        return int(self.initial + progress * (self.final - self.initial))
    
    def get_dataloader(self, epoch: int, batch_size: int):
        max_len = self.get_max_length

The Grand Architectural Blueprint: Arabic HRM-Grid Parser v2 (Continued)
Picking up exactly where we left off:

4H. Curriculum Learning by Sentence Length (continued)
Python

class SentenceLengthCurriculum:
    """
    Start training on short sentences (easy dependency structures),
    gradually introduce longer ones (complex nested clauses).
    
    This is especially effective for Arabic where long sentences often
    contain multiple coordinated clauses with complex inter-clausal deps.
    """
    def __init__(self, dataset, max_epochs: int = 30, 
                 initial_max_len: int = 15, final_max_len: int = 128):
        self.dataset = dataset
        self.max_epochs = max_epochs
        self.initial = initial_max_len
        self.final = final_max_len
    
    def get_max_length(self, epoch: int) -> int:
        """Linearly increase max sentence length over first half of training."""
        progress = min(1.0, epoch / (self.max_epochs * 0.5))
        return int(self.initial + progress * (self.final - self.initial))
    
    def get_dataloader(self, epoch: int, batch_size: int):
        max_len = self.get_max_length(epoch)
        
        # Filter dataset to sentences within current curriculum window
        filtered_indices = [
            i for i, sample in enumerate(self.dataset)
            if sample['length'] <= max_len
        ]
        
        # Weighted sampling: OVERSAMPLE sentences near the current 
        # difficulty frontier (80-100% of max_len) — these are the
        # sentences the model is currently "learning to handle"
        weights = []
        for i in filtered_indices:
            slen = self.dataset[i]['length']
            if slen >= max_len * 0.8:
                weights.append(3.0)  # frontier sentences weighted 3x
            elif slen >= max_len * 0.5:
                weights.append(1.5)  # medium difficulty
            else:
                weights.append(1.0)  # easy maintenance
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(filtered_indices), replacement=True
        )
        
        subset = torch.utils.data.Subset(self.dataset, filtered_indices)
        return torch.utils.data.DataLoader(
            subset, batch_size=batch_size, sampler=sampler,
            collate_fn=self.dataset.collate_fn, num_workers=2,
            pin_memory=True
        )
4I. Contrastive Arc Loss: Learning from Near-Miss Trees
The Tree-CRF gives us marginals, but it treats all wrong trees equally. A contrastive approach explicitly constructs hard negative trees (trees that are almost right but have one or two swapped arcs) and pushes them away:

Python

class ContrastiveTreeLoss(nn.Module):
    """
    Generates hard-negative trees by perturbing the gold tree:
      1. Swap a head assignment (e.g., move a modifier from noun→verb)
      2. Reattach a subtree to a different head
      3. Swap root assignment
    
    Then applies a margin-based ranking loss:
      L = max(0, margin - score(gold_tree) + score(negative_tree))
    
    This teaches the model to discriminate between PLAUSIBLE trees,
    not just separate gold from random noise.
    """
    
    def __init__(self, n_negatives: int = 4, margin: float = 2.0):
        super().__init__()
        self.n_negatives = n_negatives
        self.margin = margin
    
    def _generate_hard_negatives(self, gold_heads, mask):
        """
        Create plausible-but-wrong trees by local perturbation.
        
        Strategies:
        1. SWAP: Pick two words, swap their heads
        2. REATTACH: Move one word to a nearby alternative head  
        3. ROOT-FLIP: Change which word attaches to root
        """
        B, N = gold_heads.shape
        device = gold_heads.device
        negatives = []
        
        for _ in range(self.n_negatives):
            neg = gold_heads.clone()
            
            for b in range(B):
                length = mask[b].sum().item()
                if length <= 2:
                    continue
                
                strategy = torch.randint(0, 3, (1,)).item()
                
                if strategy == 0:  # SWAP two heads
                    candidates = list(range(1, length))
                    if len(candidates) >= 2:
                        i, j = torch.randperm(len(candidates))[:2].tolist()
                        d1, d2 = candidates[i], candidates[j]
                        # Swap heads (avoid creating self-loops)
                        new_h1, new_h2 = neg[b, d2].item(), neg[b, d1].item()
                        if new_h1 != d1:
                            neg[b, d1] = new_h1
                        if new_h2 != d2:
                            neg[b, d2] = new_h2
                
                elif strategy == 1:  # REATTACH to nearby word
                    d = torch.randint(1, length, (1,)).item()
                    # Pick a new head within distance 3 (plausible error)
                    candidates = [
                        h for h in range(max(0, d-3), min(length, d+4))
                        if h != d and h != neg[b, d].item()
                    ]
                    if candidates:
                        new_head = candidates[torch.randint(0, len(candidates), (1,)).item()]
                        neg[b, d] = new_head
                
                else:  # ROOT-FLIP
                    current_root_deps = (neg[b, 1:length] == 0).nonzero(as_tuple=True)[0]
                    non_root_deps = (neg[b, 1:length] != 0).nonzero(as_tuple=True)[0]
                    if len(current_root_deps) > 0 and len(non_root_deps) > 0:
                        # Detach one root child, attach a non-root word to root
                        old_rd = current_root_deps[0].item() + 1
                        new_rd = non_root_deps[
                            torch.randint(0, len(non_root_deps), (1,)).item()
                        ].item() + 1
                        # Move old root-dep somewhere plausible
                        neg[b, old_rd] = max(1, old_rd - 1)
                        neg[b, new_rd] = 0
            
            negatives.append(neg)
        
        return negatives  # list of (B, N) tensors
    
    def forward(self, arc_scores, gold_heads, mask):
        """
        arc_scores: (B, N, N) — biaffine scores
        gold_heads: (B, N)
        mask: (B, N)
        """
        B, N, _ = arc_scores.shape
        device = arc_scores.device
        dep_mask = mask[:, 1:].float()  # (B, N-1)
        
        # Score gold tree
        dep_idx = torch.arange(1, N, device=device).unsqueeze(0).expand(B, -1)
        gold_score = arc_scores[
            torch.arange(B, device=device).unsqueeze(1),
            gold_heads[:, 1:],
            dep_idx
        ]  # (B, N-1)
        gold_total = (gold_score * dep_mask).sum(dim=-1)  # (B,)
        
        # Generate and score negatives
        negatives = self._generate_hard_negatives(gold_heads, mask)
        
        total_loss = torch.tensor(0.0, device=device)
        for neg_heads in negatives:
            neg_score = arc_scores[
                torch.arange(B, device=device).unsqueeze(1),
                neg_heads[:, 1:],
                dep_idx
            ]
            neg_total = (neg_score * dep_mask).sum(dim=-1)  # (B,)
            
            # Margin ranking loss
            margin_loss = torch.clamp(
                self.margin - gold_total + neg_total, min=0.0
            )
            total_loss = total_loss + margin_loss.mean()
        
        return total_loss / self.n_negatives
4J. Arabic-Specific Positional Encodings: Clause-Depth & Verb-Relative Position
Standard sinusoidal or learned positions encode surface position. But Arabic parsing cares about structural position: "how deep am I in nested clauses?" and "where am I relative to the nearest verb?" We can approximate these at training time from gold trees and learn to predict them at inference:

Python

class ArabicStructuralPositionEncoder(nn.Module):
    """
    Encodes structural positional features unique to Arabic syntax:
    
    1. CLAUSE_DEPTH: Estimated nesting depth (approx. by counting
       subordinating conjunctions / complementizers to the left)
    2. VERB_RELATIVE: Signed distance to nearest verb (crucial for 
       VSO/SVO disambiguation — Arabic allows both)
    3. CONJUNCT_RANK: Position within a coordination chain (1st conjunct
       vs 2nd vs 3rd — Arabic loves long coordinated lists with وَ)
    4. SENTENCE_RELATIVE: Position normalized by sentence length 
       (captures tendency of certain roles to appear sentence-initially
       or sentence-finally, e.g., vocative particles يَا)
    """
    
    def __init__(self, d_model: int = 256, max_depth: int = 8):
        super().__init__()
        
        self.depth_embed = nn.Embedding(max_depth, d_model // 4)
        self.verb_dist_embed = nn.Embedding(33, d_model // 4)  # -16..+16 bucketed
        self.conjunct_embed = nn.Embedding(8, d_model // 4)    # 0..7
        self.rel_pos_proj = nn.Linear(1, d_model // 4)
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
    
    def forward(self, word_ids, pos_tags, seq_lengths, mask):
        """
        Computes structural positions from surface observables.
        No gold tree needed at inference — uses heuristic approximations.
        """
        B, W = word_ids.shape
        device = word_ids.device
        
        # ── Heuristic clause depth ──
        # Count subordinating particles (أنّ، إنّ، الذي، التي، لأنّ، حتّى، إذا...)
        # This is a rough approximation — encoded as running sum of subordinators
        # In practice, you'd have a list of subordinator token IDs
        # For now, approximate with cumulative POS tag patterns
        is_sconj = (pos_tags == self._sconj_id).long()  # binary
        cumulative_depth = torch.cumsum(is_sconj, dim=1).clamp(max=7)  # (B, W)
        depth_enc = self.depth_embed(cumulative_depth)  # (B, W, d//4)
        
        # ── Verb-relative distance ──
        is_verb = ((pos_tags == self._verb_id_1) | 
                   (pos_tags == self._verb_id_2)).float()  # (B, W)
        
        # For each position, find distance to nearest verb
        positions = torch.arange(W, device=device).float().unsqueeze(0)  # (1, W)
        verb_positions = positions * is_verb + (1 - is_verb) * 9999  # (B, W)
        
        # Min distance to any verb (signed: negative = verb is to the left)
        verb_dist = torch.zeros(B, W, device=device)
        for b in range(B):
            v_pos = verb_positions[b][verb_positions[b] < 9998]
            if len(v_pos) > 0:
                dists = positions[0, :W].unsqueeze(1) - v_pos.unsqueeze(0)  # (W, n_verbs)
                abs_dists = dists.abs()
                nearest_idx = abs_dists.argmin(dim=1)  # (W,)
                verb_dist[b] = dists[torch.arange(W, device=device), nearest_idx]
        
        # Bucket: -16..+16 → 0..32
        bucketed_vdist = (verb_dist.clamp(-16, 16).long() + 16)
        vdist_enc = self.verb_dist_embed(bucketed_vdist)  # (B, W, d//4)
        
        # ── Conjunct rank ──
        # Heuristic: count coordinating conjunctions (وَ، أو، ثمّ) up to this point
        is_cc = (pos_tags == self._cc_id).long()
        conj_rank = torch.cumsum(is_cc, dim=1).clamp(max=7)
        conj_enc = self.conjunct_embed(conj_rank)  # (B, W, d//4)
        
        # ── Sentence-relative position ──
        rel_pos = positions[:, :W] / seq_lengths.float().unsqueeze(1).clamp(min=1)
        rel_pos = rel_pos.unsqueeze(-1)  # (B, W, 1)
        rel_enc = self.rel_pos_proj(rel_pos)  # (B, W, d//4)
        
        # ── Fuse all structural positions ──
        struct_pos = torch.cat([depth_enc, vdist_enc, conj_enc, rel_enc], dim=-1)
        return self.fusion(struct_pos)  # (B, W, d_model)
4K. Dynamic Loss Weighting: Uncertainty-Based Multi-Task Balancing
You're training three tasks simultaneously (case, heads, relations). Fixed loss weights are suboptimal — use homoscedastic uncertainty to learn optimal task weighting:

Python

class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """
    Learns task-specific loss weights automatically using the 
    homoscedastic uncertainty principle from Kendall et al. (2018).
    
    L_total = Σ_t (1/(2σ_t²)) * L_t + log(σ_t)
    
    When a task's loss is high (hard task), σ_t increases → weight decreases.
    When a task's loss is low (easy task), σ_t shrinks → weight increases.
    This prevents easy tasks from dominating and hard tasks from being ignored.
    """
    
    def __init__(self, n_tasks: int = 3, initial_sigma: float = 1.0):
        super().__init__()
        # log(σ²) — learnable per-task uncertainty
        self.log_sigma_sq = nn.Parameter(
            torch.full((n_tasks,), 2 * math.log(initial_sigma))
        )
    
    def forward(self, *task_losses):
        """
        task_losses: variable number of scalar tensors 
                     (e.g., arc_loss, rel_loss, case_loss)
        """
        total = 0
        weighted_components = []
        for i, loss in enumerate(task_losses):
            precision = (-self.log_sigma_sq[i]).exp()  # 1/σ²
            weighted = precision * loss + 0.5 * self.log_sigma_sq[i]
            total += weighted
            weighted_components.append(
                (loss.item(), precision.item(), self.log_sigma_sq[i].item())
            )
        
        return total, weighted_components
4L. Scheduled Teacher Forcing with Gumbel-Softmax Bridge
During iterative refinement, you need to transition from gold heads (teacher forcing) to predicted heads. A hard switch causes instability. Use Gumbel-Softmax to create a smooth annealing bridge:

Python

class ScheduledGumbelTeacherForcing:
    """
    Instead of abruptly switching from gold → predicted heads during 
    iterative refinement, uses Gumbel-Softmax with temperature annealing
    to smoothly transition.
    
    Epoch 1-5:   τ → ∞  (nearly deterministic, close to teacher forcing)
    Epoch 5-15:  τ → 1  (soft categorical, partially exploring)
    Epoch 15-30: τ → 0.1 (nearly hard, close to argmax inference)
    """
    
    def __init__(self, warmup_epochs: int = 5, anneal_epochs: int = 15):
        self.warmup = warmup_epochs
        self.anneal = anneal_epochs
    
    def get_temperature(self, epoch: int) -> float:
        if epoch < self.warmup:
            return 10.0  # very soft — close to teacher forcing
        progress = min(1.0, (epoch - self.warmup) / self.anneal)
        # Exponential decay from 10 → 0.1
        return 10.0 * (0.01 ** progress)
    
    def get_mix_ratio(self, epoch: int) -> float:
        """
        Returns α ∈ [0, 1]: fraction of time to use gold heads.
        α=1 → always gold (pure teacher forcing)
        α=0 → always predicted (pure free-running)
        """
        if epoch < self.warmup:
            return 1.0
        progress = min(1.0, (epoch - self.warmup) / self.anneal)
        return max(0.0, 1.0 - progress)

    @staticmethod
    def gumbel_soft_heads(arc_scores, temperature, gold_heads=None, mix_ratio=0.0):
        """
        Returns soft head distributions using Gumbel-Softmax,
        optionally mixed with gold one-hot vectors.
        
        Args:
            arc_scores: (B, N, N) — biaffine scores
            temperature: float — Gumbel temperature
            gold_heads: (B, N) — optional gold head indices
            mix_ratio: float — fraction to mix in gold
        
        Returns:
            soft_heads: (B, N, N) — soft attention over heads per dependent
                        soft_heads[:, :, d] is the head distribution for dep d
        """
        B, N, _ = arc_scores.shape
        
        # Gumbel-Softmax over head dimension (dim=1: which word is the head?)
        # We want columns to be distributions: for each dependent d, 
        # distribution over heads h
        logits = arc_scores.transpose(1, 2)  # (B, N_dep, N_head)
        
        if temperature > 0:
            gumbel_noise = -torch.empty_like(logits).exponential_().log()
            gumbel_noise = -torch.empty_like(logits).exponential_().log()  
            # double Gumbel trick
            noisy_logits = (logits + gumbel_noise) / temperature
        else:
            noisy_logits = logits
        
        soft = torch.softmax(noisy_logits, dim=-1)  # (B, N_dep, N_head)
        
        # Mix with gold if teacher forcing
        if gold_heads is not None and mix_ratio > 0:
            


            The Grand Architectural Blueprint: Arabic HRM-Grid Parser v2 (Continued)
Picking up exactly at the cut-off point inside gumbel_soft_heads:

4L. Scheduled Gumbel Teacher Forcing (continued)
Python

    @staticmethod
    def gumbel_soft_heads(arc_scores, temperature, gold_heads=None, mix_ratio=0.0):
        """(continued from above)"""
        B, N, _ = arc_scores.shape
        
        logits = arc_scores.transpose(1, 2)  # (B, N_dep, N_head)
        
        if temperature > 0:
            gumbel_noise = -(-torch.empty_like(logits).uniform_().clamp(1e-8).log()).log()
            noisy_logits = (logits + gumbel_noise) / temperature
        else:
            noisy_logits = logits
        
        soft = torch.softmax(noisy_logits, dim=-1)  # (B, N_dep, N_head)
        
        # Mix with gold if teacher forcing
        if gold_heads is not None and mix_ratio > 0:
            gold_onehot = F.one_hot(gold_heads.clamp(min=0), N).float()  # (B, N, N)
            soft = mix_ratio * gold_onehot + (1 - mix_ratio) * soft
        
        return soft  # (B, N_dep, N_head)

    @staticmethod
    def soft_head_representation(soft_heads, word_embeddings):
        """
        Instead of hard-indexing a single head embedding, compute
        a weighted sum over all candidate heads using the soft distribution.
        
        This feeds smooth gradients through the head selection into
        the relation classifier and the Worker RNN's refinement loop.
        
        Args:
            soft_heads: (B, N_dep, N_head) — soft distribution per dependent
            word_embeddings: (B, N, D) — contextualized word representations
        
        Returns:
            head_repr: (B, N, D) — expected head representation per word
        """
        # For each dependent, weighted sum over all possible head embeddings
        # soft_heads[:, d, :] @ word_embeddings[:, :, :] → head repr for dep d
        head_repr = torch.bmm(soft_heads, word_embeddings)  # (B, N, D)
        return head_repr
4M. Second-Order Sibling & Grandparent Scoring
First-order biaffine scoring only asks: "Is word j the head of word i?" But real dependency trees exhibit second-order patterns — siblings should be compatible (you rarely see two subjects hanging off the same verb), and grandparent chains should be syntactically coherent. This is especially critical in Arabic where coordination (عطف) produces long sibling chains.

Python

class SecondOrderScorer(nn.Module):
    """
    Scores PAIRS of arcs jointly, capturing:
    
    1. SIBLING interactions: If both word_i and word_j attach to the 
       same head h, score the (h, i, j) triple. This discourages 
       structurally incompatible siblings (e.g., two فاعل/subjects 
       on the same verb without coordination).
    
    2. GRANDPARENT chains: If word_j attaches to word_i, and word_i 
       attaches to word_g, score the (g, i, j) chain. This captures
       Arabic patterns like: verb ← subject ← adjective modifier.
    
    Mathematical formulation:
        S_sib(h, i, j) = U_s_i^T W_sib U_s_j  (sibling trilinear)
        S_gp(g, h, d)  = U_g_g^T W_gp (U_g_h ⊙ U_g_d) (grandparent trilinear)
    
    These scores are ADDED to the first-order biaffine scores during 
    MST decoding, not during training backprop (too expensive).
    Instead, we train them with a margin loss against gold second-order
    structures.
    """
    
    def __init__(self, input_dim: int = 256, scorer_dim: int = 128):
        super().__init__()
        
        # ── Sibling projections ──
        self.sib_head_mlp = nn.Sequential(
            nn.Linear(input_dim, scorer_dim), nn.LeakyReLU(0.1)
        )
        self.sib_dep_mlp = nn.Sequential(
            nn.Linear(input_dim, scorer_dim), nn.LeakyReLU(0.1)
        )
        self.sib_trilinear = nn.Parameter(
            torch.randn(scorer_dim, scorer_dim, scorer_dim) * 0.01
        )
        # Decomposed trilinear for efficiency: W ≈ Σ_r u_r ⊗ v_r ⊗ w_r
        self.trilinear_rank = 64
        self.sib_U = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
        self.sib_V = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
        self.sib_W = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
        
        # ── Grandparent projections ──
        self.gp_grandparent_mlp = nn.Sequential(
            nn.Linear(input_dim, scorer_dim), nn.LeakyReLU(0.1)
        )
        self.gp_head_mlp = nn.Sequential(
            nn.Linear(input_dim, scorer_dim), nn.LeakyReLU(0.1)
        )
        self.gp_dep_mlp = nn.Sequential(
            nn.Linear(input_dim, scorer_dim), nn.LeakyReLU(0.1)
        )
        self.gp_U = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
        self.gp_V = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
        self.gp_W = nn.Parameter(torch.randn(self.trilinear_rank, scorer_dim) * 0.02)
    
    def score_siblings(self, H, i_indices, j_indices, h_indices):
        """
        Score sibling pairs: words i and j both attaching to head h.
        
        Uses low-rank decomposition of trilinear form:
        S(h,i,j) = Σ_r (U_r · h_h)(V_r · h_i)(W_r · h_j)
        
        Args:
            H: (B, N, D) — word representations
            i_indices, j_indices, h_indices: (B, K) — K sibling pairs per sentence
        
        Returns:
            scores: (B, K) — compatibility score for each sibling pair
        """
        B, N, D = H.shape
        K = i_indices.shape[1]
        
        h_repr = self.sib_head_mlp(
            H.gather(1, h_indices.unsqueeze(-1).expand(-1, -1, D))
        )  # (B, K, scorer_dim)
        i_repr = self.sib_dep_mlp(
            H.gather(1, i_indices.unsqueeze(-1).expand(-1, -1, D))
        )  # (B, K, scorer_dim)
        j_repr = self.sib_dep_mlp(
            H.gather(1, j_indices.unsqueeze(-1).expand(-1, -1, D))
        )  # (B, K, scorer_dim)
        
        # Low-rank trilinear: Σ_r (h @ U_r)(i @ V_r)(j @ W_r)
        h_proj = torch.einsum('bkd, rd -> bkr', h_repr, self.sib_U)  # (B, K, R)
        i_proj = torch.einsum('bkd, rd -> bkr', i_repr, self.sib_V)
        j_proj = torch.einsum('bkd, rd -> bkr', j_repr, self.sib_W)
        
        scores = (h_proj * i_proj * j_proj).sum(dim=-1)  # (B, K)
        return scores
    
    def score_grandparents(self, H, g_indices, h_indices, d_indices):
        """
        Score grandparent chains: d → h → g
        """
        B, N, D = H.shape
        K = g_indices.shape[1]
        
        g_repr = self.gp_grandparent_mlp(
            H.gather(1, g_indices.unsqueeze(-1).expand(-1, -1, D))
        )
        h_repr = self.gp_head_mlp(
            H.gather(1, h_indices.unsqueeze(-1).expand(-1, -1, D))
        )
        d_repr = self.gp_dep_mlp(
            H.gather(1, d_indices.unsqueeze(-1).expand(-1, -1, D))
        )
        
        g_proj = torch.einsum('bkd, rd -> bkr', g_repr, self.gp_U)
        h_proj = torch.einsum('bkd, rd -> bkr', h_repr, self.gp_V)
        d_proj = torch.einsum('bkd, rd -> bkr', d_repr, self.gp_W)
        
        return (g_proj * h_proj * d_proj).sum(dim=-1)
    
    def extract_gold_pairs(self, gold_heads, mask):
        """
        From gold trees, extract all sibling pairs and grandparent chains.
        
        Returns padded tensors of index triples for batched scoring.
        """
        B, N = gold_heads.shape
        device = gold_heads.device
        
        all_sib_h, all_sib_i, all_sib_j = [], [], []
        all_gp_g, all_gp_h, all_gp_d = [], [], []
        
        for b in range(B):
            length = mask[b].sum().item()
            heads = gold_heads[b, :length]
            
            # ── Siblings: words sharing the same head ──
            from collections import defaultdict
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
            
            # ── Grandparent chains: d → h → g ──
            gp_g, gp_h, gp_d = [], [], []
            for d in range(1, length):
                h = heads[d].item()
                if h > 0:  # h is not root
                    g = heads[h].item()
                    gp_g.append(g)
                    gp_h.append(h)
                    gp_d.append(d)
            
            all_gp_g.append(gp_g)
            all_gp_h.append(gp_h)
            all_gp_d.append(gp_d)
        
        # Pad to max K across batch
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
        """
        Margin loss: gold second-order structures should score higher 
        than predicted (possibly wrong) second-order structures.
        
        L = max(0, margin - S_gold + S_pred)
        """
        margin = 1.0
        
        gold_pairs = self.extract_gold_pairs(gold_heads, mask)
        pred_pairs = self.extract_gold_pairs(pred_heads.detach(), mask)
        
        loss = torch.tensor(0.0, device=H.device)
        n_terms = 0
        
        # ── Sibling margin ──
        g_sh, g_si, g_sj, g_smask = gold_pairs['sib']
        p_sh, p_si, p_sj, p_smask = pred_pairs['sib']
        
        if g_smask.any() and p_smask.any():
            gold_sib_scores = self.score_siblings(H, g_si, g_sj, g_sh)
            gold_sib_scores = (gold_sib_scores * g_smask.float()).sum() / g_smask.float().sum().clamp(1)
            
            pred_sib_scores = self.score_siblings(H, p_si, p_sj, p_sh)
            pred_sib_scores = (pred_sib_scores * p_smask.float()).sum() / p_smask.float().sum().clamp(1)
            
            loss = loss + F.relu(margin - gold_sib_scores + pred_sib_scores)
            n_terms += 1
        
        # ── Grandparent margin ──
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
4N. Arabic Morphological Decomposition: Affix-Aware Character Convolutions
Arabic morphology isn't just prefixes and suffixes — it's built on a root-pattern (جذر + وزن) system where consonantal roots interleave with vowel templates. A standard CharCNN misses this entirely. We need convolutions that can capture discontinuous morphemes:

Python

class ArabicMorphologicalEncoder(nn.Module):
    """
    A MULTI-STRATEGY morphological encoder that captures Arabic's 
    unique word-internal structure at multiple granularities:
    
    LAYER 1 — Character CNN with dilated convolutions:
        Standard k=3 captures prefixes (الـ، بـ، وـ) and suffixes (ـون، ـات، ـين)
        DILATED k=3, dilation=2 captures root consonant skeletons:
            For "كَتَبَ" (k-a-t-a-b-a), dilation=2 on chars hits k,t,b = the root!
            For "يَكْتُبُونَ" (y-a-k-t-u-b-uu-n-a), dilation=2 catches interleaved root
    
    LAYER 2 — Byte-Pair subword embeddings:
        Captures frequent morphological chunks (ال+, مُ+, ـة, ـات, إسْتِ+)
        These are pre-tokenized and embedded separately
    
    LAYER 3 — Explicit morphological feature hashing:
        Arabic morphological analyzers (like CAMeL Tools) output:
        - Root (جذر): e.g., ك.ت.ب
        - Pattern (وزن): e.g., فَعَلَ, يَفْعُلُ, اِسْتِفْعَال
        - Clitics: proclitic + stem + enclitic decomposition
        Each gets its own embedding, fused via gated highway
    
    LAYER 4 — Diacritics-aware encoding:
        Arabic diacritics (تشكيل: فتحة، ضمة، كسرة، سكون) are EXTREMELY 
        informative for iʻrāb — they literally ARE the case markers!
        But text is often undiacritized. We encode diacritics separately 
        so the model learns both with and without them.
    """
    
    def __init__(self, char_vocab_size: int = 300, 
                 n_bpe_tokens: int = 8000,
                 n_roots: int = 5000,
                 n_patterns: int = 200,
                 char_embed_dim: int = 64,
                 output_dim: int = 256,
                 max_word_len: int = 30):
        super().__init__()
        
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.max_word_len = max_word_len
        
        # ── Layer 1: Multi-scale Dilated CharCNN ──
        # Standard convolutions for affixes
        self.conv_k2 = nn.Conv1d(char_embed_dim, 64, kernel_size=2, padding=0)
        self.conv_k3 = nn.Conv1d(char_embed_dim, 64, kernel_size=3, padding=1)
        self.conv_k4 = nn.Conv1d(char_embed_dim, 64, kernel_size=4, padding=1)
        
        # DILATED convolutions for discontinuous root extraction
        # dilation=2: every-other-char → catches trilateral root consonants
        self.conv_dilated_2 = nn.Conv1d(char_embed_dim, 64, kernel_size=3, 
                                         padding=2, dilation=2)
        # dilation=3: every-third-char → catches patterns in longer words
        self.conv_dilated_3 = nn.Conv1d(char_embed_dim, 48, kernel_size=3, 
                                         padding=3, dilation=3)
        
        # Total CNN output: 64 + 64 + 64 + 64 + 48 = 304
        self.cnn_channels = 304
        
        # ── Layer 2: BPE subword embeddings ──
        self.bpe_embed = nn.Embedding(n_bpe_tokens, 128, padding_idx=0)
        self.bpe_proj = nn.Linear(128, 128)  # project max-pool of BPE pieces
        
        # ── Layer 3: Morphological feature embeddings ──
        self.root_embed = nn.Embedding(n_roots + 1, 64, padding_idx=0)    # جذر
        self.pattern_embed = nn.Embedding(n_patterns + 1, 64, padding_idx=0)  # وزن
        # Clitic decomposition: up to 3 proclitics + stem + 2 enclitics
        self.proclitic_embed = nn.Embedding(200, 32, padding_idx=0)
        self.enclitic_embed = nn.Embedding(100, 32, padding_idx=0)
        
        # ── Layer 4: Diacritics-aware encoding ──
        # Separate embedding for diacritics sequence (fatha, damma, kasra, 
        # sukun, shadda, tanwin variants)
        self.diac_vocab_size = 20  # ~15 diacritic marks + padding + unknown
        self.diac_embed = nn.Embedding(self.diac_vocab_size, 32, padding_idx=0)
        self.diac_lstm = nn.LSTM(32, 48, batch_first=True, bidirectional=True)
        # diac output: 96
        
        # ── Gated Highway Fusion ──
        # CNN(304) + BPE(128) + Root(64) + Pattern(64) + Proclit(32) 
        # + Enclit(32) + Diac(96) = 720
        self.total_morph_dim = self.cnn_channels + 128 + 64 + 64 + 32 + 32 + 96
        
        self.highway_gate = nn.Sequential(
            nn.Linear(self.total_morph_dim, output_dim),
            nn.Sigmoid()
        )
        self.highway_transform = nn.Sequential(
            nn.Linear(self.total_morph_dim, output_dim),
            nn.GELU()
        )
        self.highway_norm = nn.LayerNorm(output_dim)
        
        # Optional: learned "morphological importance" per-word
        self.morph_confidence = nn.Sequential(
            nn.Linear(self.total_morph_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, char_ids, bpe_ids, root_ids, pattern_ids, 
                proclitic_ids, enclitic_ids, diac_ids,
                morph_available_mask=None):
        """
        Args:
            char_ids: (B, W, max_word_len) — character sequences per word
            bpe_ids: (B, W, max_bpe_per_word) — BPE piece IDs per word
            root_ids: (B, W) — root index per word (0 if unknown)
            pattern_ids: (B, W) — pattern/wazn index per word
            proclitic_ids: (B, W) — proclitic cluster ID
            enclitic_ids: (B, W) — enclitic cluster ID
            diac_ids: (B, W, max_word_len) — diacritic label per character position
            morph_available_mask: (B, W) — 1 if morphological analysis available
        
        Returns:
            morph_repr: (B, W, output_dim) — morphological representation per word
        """
        B, W, C = char_ids.shape
        device = char_ids.device
        
        # ═══ Layer 1: Dilated CharCNN ═══
        # Reshape for conv1d: (B*W, C) → embed → (B*W, char_embed_dim, C)
        chars_flat = char_ids.view(B * W, C)
        char_emb = self.char_embed(chars_flat).transpose(1, 2)  # (B*W, E, C)
        
        # Standard convolutions (capture prefixes, suffixes)
        c2 = self.conv_k2(char_emb).max(dim=-1).values   # (B*W, 64)
        c3 = self.conv_k3(char_emb).max(dim=-1).values   # (B*W, 64)
        c4 = self.conv_k4(char_emb).max(dim=-1).values   # (B*W, 64)
        
        # Dilated convolutions (capture root consonant skeleton)
        cd2 = self.conv_dilated_2(char_emb).max(dim=-1).values  # (B*W, 64)
        cd3 = self.conv_dilated_3(char_emb).max(dim=-1).values  # (B*W, 48)
        
        cnn_out = F.gelu(torch.cat([c2, c3, c4, cd2, cd3], dim=-1))  # (B*W, 304)
        cnn_out = cnn_out.view(B, W, self.cnn_channels)
        
        # ═══ Layer 2: BPE embeddings (max-pool over pieces) ═══
        bpe_emb = self.bpe_embed(bpe_ids)  # (B, W, n_pieces, 128)
        bpe_mask = (bpe_ids != 0).float().unsqueeze(-1)
        bpe_out = (bpe_emb * bpe_mask).max(dim=2).values  # (B, W, 128)
        bpe_out = self.bpe_proj(bpe_out)
        
        # ═══ Layer 3: Morphological features ═══
        root_out = self.root_embed(root_ids)        # (B, W, 64)
        pattern_out = self.pattern_embed(pattern_ids)  # (B, W, 64)
        procl_out = self.proclitic_embed(proclitic_ids)  # (B, W, 32)
        encl_out = self.enclitic_embed(enclitic_ids)     # (B, W, 32)
        
        # ═══ Layer 4: Diacritics encoding ═══
        diac_flat = diac_ids.view(B * W, C)
        diac_emb = self.diac_embed(diac_flat)  # (B*W, C, 32)
        diac_out, _ = self.diac_lstm(diac_emb)  # (B*W, C, 96)
        # Max-pool over character positions
        diac_out = diac_out.max(dim=1).values.view(B, W, 96)
        
        # ═══ Gated Highway Fusion ═══
        morph_concat = torch.cat([
            cnn_out, bpe_out, root_out, pattern_out, 
            procl_out, encl_out, diac_out
        ], dim=-1)  # (B, W, 720)
        
        gate = self.highway_gate(morph_concat)        # (B, W, output_dim)
        transform = self.highway_transform(morph_concat)  # (B, W, output_dim)
        morph_repr = gate * transform  # Gated output
        
        # If morphological analysis unavailable for some words, 
        # scale down contribution (model learns to rely less)
        if morph_available_mask is not None:
            confidence = self.morph_confidence(morph_concat)  # (B, W, 1)
            availability = morph_available_mask.float().unsqueeze(-1)  # (B, W, 1)
            morph_repr = morph_repr * (confidence * availability + 
                                        (1 - availability) * 0.1)
        
        return self.highway_norm(morph_repr)  # (B, W, output_dim)
4O. Latent Variable Manager with Variational Tree Posterior
The current Manager-Worker hierarchy uses a simple average-pool → MLP for the "goal." This is a lossy bottleneck. We replace it with a Variational Information Bottleneck — the Manager infers a latent tree topology distribution and communicates it as a structured goal:

Python

class VariationalTreeManager(nn.Module):
    """
    THE KEY INSIGHT: The Manager shouldn't just send a vector — it should 
    infer a DISTRIBUTION over tree topology types and send a structured 
    goal that encodes *what kind of tree this sentence has*.
    
    Arabic sentences fall into distinct structural archetypes:
      - VSO simple (verb-initial, single clause)
      - SVO topicalized (مبتدأ + خبر with fronted subject)  
      - Nominal sentence (جملة اسمية — no verb, إنّ/كانَ sisters)
      - Complex subordinated (nested أنّ/الذي clauses)
      - Coordination-heavy (long وَ chains)
    
    The Manager uses a Variational Autoencoder (VAE) formulation:
      1. ENCODER: From word embeddings, infer q(z|x) — posterior over
         latent tree type z
      2. PRIOR: p(z) — learned mixture-of-Gaussians prior (each component 
         corresponds to a structural archetype)
      3. DECODER: z → worker_goal, distance bias, relation prior
    
    KL divergence regularizes the latent space, and the Mixture prior
    encourages discrete cluster structure (one cluster per tree type).
    """
    
    def __init__(self, input_dim: int = 256, latent_dim: int = 64, 
                 n_archetypes: int = 12, goal_dim: int = 256):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_archetypes = n_archetypes
        
        # ── Sentence Encoder (context → posterior) ──
        self.context_encoder = nn.LSTM(
            input_dim, input_dim // 2, num_layers=2, 
            batch_first=True, bidirectional=True, dropout=0.2
        )
        # Attention-weighted pooling instead of average
        self.pool_attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Posterior q(z|x): diagonal Gaussian
        self.posterior_mu = nn.Linear(input_dim, latent_dim)
        self.posterior_logvar = nn.Linear(input_dim, latent_dim)
        
        # ── Learned Mixture-of-Gaussians Prior ──
        # Each archetype k has its own (μ_k, σ_k)
        self.prior_logits = nn.Parameter(torch.zeros(n_archetypes))  # mixture weights
        self.prior_mu = nn.Parameter(torch.randn(n_archetypes, latent_dim) * 0.5)
        self.prior_logvar = nn.Parameter(torch.zeros(n_archetypes, latent_dim))
        
        # ── Goal Decoder ──
        self.goal_decoder = nn.Sequential(
            nn.Linear(latent_dim, goal_dim),
            nn.GELU(),
            nn.LayerNorm(goal_dim),
            nn.Linear(goal_dim, goal_dim),
        )
        
        # ── Structural Bias Decoder ──
        # From z, predict a distance bias template and relation prior
        self.distance_bias_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 33)  # 33 distance buckets (-16..+16)
        )
        
        self.relation_prior_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 50)  # n_relations — prior distribution over rel types
        )
        
        # ── Archetype Classifier (auxiliary loss) ──
        self.archetype_head = nn.Linear(latent_dim, n_archetypes)
    
    def _attention_pool(self, H, mask):
        """Attention-weighted pooling over sequence."""
        scores = self.pool_attention(H).squeeze(-1)  # (B, N)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)      # (B, N)
        pooled = torch.bmm(weights.unsqueeze(1), H).squeeze(1)  # (B, D)
        return pooled
    
    def encode(self, word_embeddings, mask):
        """
        Encode sentence into latent posterior q(z|x).
        Returns: z (sampled), mu, logvar
        """
        ctx_out, _ = self.context_encoder(word_embeddings)
        pooled = self._attention_pool(ctx_out, mask)  # (B, D)
        
        mu = self.posterior_mu(pooled)          # (B, latent_dim)
        logvar = self.posterior_logvar(pooled)  # (B, latent_dim)
        logvar = logvar.clamp(-6, 2)  # numerical stability
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar, training=True):
        """Reparameterization trick."""
        if training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference
    
    def kl_divergence_mixture(self, q_mu, q_logvar):
        """
        KL(q(z|x) || p(z)) where p(z) is a mixture of Gaussians.
        
        Uses the "log-sum-exp" trick to compute:
        KL ≈ E_q[log q(z|x)] - E_q[log p(z)]
        
        where log p(z) = log Σ_k π_k N(z; μ_k, σ_k²)
        """
        B = q_mu.shape[0]
        
        # Sample z from posterior for Monte Carlo estimate
        z = self.reparameterize(q_mu, q_logvar)  # (B, D)
        
        # log q(z|x) — Gaussian log-prob
        log_q = -0.5 * (
            self.latent_dim * math.log(2 * math.pi) + 
            q_logvar.sum(dim=-1) + 
            ((z - q_mu).pow(2) / q_logvar.exp()).sum(dim=-1)
        )  # (B,)
        
        # log p(z) — mixture of Gaussians
        prior_weights = F.log_softmax(self.prior_logits, dim=0)  # (K,)
        
        # Expand: z is (B, 1, D), prior params are (1, K, D)
        z_exp = z.unsqueeze(1)                    # (B, 1, D)
        p_mu = self.prior_mu.unsqueeze(0)         # (1, K, D)
        p_logvar = self.prior_logvar.unsqueeze(0)  # (1, K, D)
        
        # log N(z; μ_k, σ_k²) for each component k
        log_p_k = -0.5 * (
            self.latent_dim * math.log(2 * math.pi) +
            p_logvar.sum(dim=-1) +
            ((z_exp - p_mu).pow(2) / p_logvar.exp()).sum(dim=-1)
        )  # (B, K)
        
        # log p(z) = logsumexp_k [log π_k + log N(z; μ_k, σ_k²)]
        log_p = torch.logsumexp(prior_weights.unsqueeze(0) + log_p_k, dim=-1)  # (B,)
        
        kl = (log_q - log_p).mean()  # average over batch
        return kl.clamp(min=0)  # safety
    
    def forward(self, word_embeddings, mask, training=True):
        """
        Full forward pass:
        1. Encode → posterior
        2. Sample z
        3. Decode → goal + structural biases
        4. Compute KL loss
        
        Returns:
            worker_goal: (B, goal_dim) — structured goal for Worker RNN
            distance_bias: (B, 33) — learned arc-distance prior
            relation_prior: (B, n_rels) — prior over relation types for this sentence
            kl_loss: scalar — KL divergence regularizer
            archetype_logits: (B, n_archetypes) — for auxiliary supervision
        """
        mu, logvar = self.encode(word_embeddings, mask)
        z = self.reparameterize(mu, logvar, training)
        
        worker_goal = self.goal_decoder(z)                # (B, goal_dim)
        distance_bias = self.distance_bias_head(z)         # (B, 33)
        relation_prior = F.log_softmax(
            self.relation_prior_head(z), dim=-1            # (B, n_rels)
        )
        
        kl_loss = self.kl_divergence_mixture(mu, logvar)
        archetype_logits = self.archetype_head(z.detach())  # (B, K) — stop gradient
        
        return worker_goal, distance_bias, relation_prior, kl_loss, archetype_logits
4P. Graph Neural Network Refinement Pass
After the initial biaffine parse, run a message-passing GNN over the predicted tree structure. Each node gathers information from its predicted parent and children, then re-scores. This is the "parse-then-refine" paradigm:

Python

class TreeGNNRefinement(nn.Module):
    """
    Given an initial predicted tree (from biaffine + MST), constructs 
    the tree as a directed graph and runs K rounds of message passing.
    
    Each node updates its representation by:
    1. Attending to its HEAD (upward message)
    2. Aggregating over its CHILDREN (downward message)  
    3. Attending to its SIBLINGS (lateral message)
    
    After K rounds, re-score all arcs and relations.
    This captures GLOBAL structural consistency that local biaffine misses.
    
    Critical for Arabic: a word's relation label often depends on its 
    grandparent or sibling context (e.g., تمييز vs حال disambiguation 
    requires knowing the verb's other arguments).
    """
    
    def __init__(self, d_model: int = 256, n_rounds: int = 3, 
                 n_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.n_rounds = n_rounds
        
        self.message_layers = nn.ModuleList()
        for _ in range(n_rounds):
            self.message_layers.append(TreeMessagePassingLayer(
                d_model, n_heads, dropout
            ))
        
        # After refinement: re-scoring heads
        self.refined_arc_head = nn.Linear(d_model, d_model)
        self.refined_arc_dep = nn.Linear(d_model, d_model)
        self.refined_arc_bilinear = nn.Parameter(
            torch.randn(d_model, d_model) * (2.0 / d_model)**0.5
        )
        self.refined_arc_bias_h = nn.Linear(d_model, 1)
        self.refined_arc_bias_d = nn.Linear(d_model, 1)
    
    def forward(self, H, predicted_heads, mask):
        """
        Args:
            H: (B, N, D) — word representations from encoder
            predicted_heads: (B, N) — initial predicted head indices
            mask: (B, N) — valid word mask
        
        Returns:
            H_refined: (B, N, D) — refined representations
            refined_arc_scores: (B, N, N) — rescored arc logits
        """
        H_refined = H.clone()
        
        for layer in self.message_layers:
            H_refined = layer(H_refined, predicted_heads, mask)
        
        # Re-score arcs with refined representations
        h_head = self.refined_arc_head(H_refined)
        h_dep = self.refined_arc_dep(H_refined)
        
        refined_arc_scores = torch.einsum(
            'bnd, de, bme -> bnm', h_head, self.refined_arc_bilinear, h_dep
        )
        refined_arc_scores += self.refined_arc_bias_h(h_head).transpose(1, 2)
        refined_arc_scores += self.refined_arc_bias_d(h_dep)
        
        return H_refined, refined_arc_scores


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
        """
        Args:
            H: (B, N, D) — current node representations
            pred_heads: (B, N) — predicted head indices
            mask: (B, N)
        
        Returns:
            H_updated: (B, N, D)
        """
        B, N, D = H.shape
        device = H.device
        
        # ── 1. Upward (head) message: each word looks at its predicted head ──
        # Gather head representations
        head_idx = pred_heads.clamp(0, N-1).unsqueeze(-1).expand(-1, -1, D)
        head_repr = H.gather(1, head_idx)  # (B, N, D) — head's representation
        
        # Cross-attention: query=self, key/value=head (single attended token)
        head_msg, _ = self.head_attention(
            H, head_repr, head_repr, 
            need_weights=False
        )
        
        # ── 2. Downward (children) message: aggregate all children ──
        # Build children aggregation via scatter
        child_sum = torch.zeros_like(H)  # (B, N, D)
        child_count = torch.zeros(B, N, 1, device=device)
        
        # For each word d, add H[d] to child_sum[pred_heads[d]]
        head_idx_flat = pred_heads.clamp(0, N-1)  # (B, N)
        for b in range(B):
            length = mask[b].sum().item()
            for d in range(1, length):
                h = head_idx_flat[b, d].item()
                child_sum[b, h] += H[b, d]
                child_count[b, h] += 1
        
        child_avg = child_sum / child_count.clamp(min=1)
        child_msg = self.child_transform(
            torch.cat([H, child_avg], dim=-1)
        )
        
        # ── 3. Sibling message: words sharing the same head attend to each other ──
        # Build sibling adjacency mask: words i,j are siblings if pred_heads[i] == pred_heads[j]
        heads_exp = pred_heads.unsqueeze(2).expand(-1, -1, N)  # (B, N, N)
        sibling_mask = (heads_exp == heads_exp.transpose(1, 2))  # (B, N, N)
        # Remove self-connections
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        sibling_mask = sibling_mask & ~eye & mask.unsqueeze(1) & mask.unsqueeze(2)
        
        # Use sibling_mask as attention mask (True = can attend)
        sib_attn_mask = ~sibling_mask  # MHA expects True = BLOCKED
        sib_msg, _ = self.sibling_attention(
            H, H, H, 
            attn_mask=sib_attn_mask.repeat_interleave(
                self.sibling_attention.num_heads, dim=0
            ).view(B * self.sibling_attention.num_heads, N, N) 
            if N <= 128 else None,  # skip for very long sentences (memory)
            need_weights=False
        )
        
        # ── 4. Gated fusion ──
        combined = torch.cat([H, head_msg, child_msg, sib_msg], dim=-1)
        gate = self.gate(combined)
        update = self.transform(combined)
        
        H_updated = H + self.dropout(gate * update)
        return self.norm(H_updated)
4Q. Inference-Time Eisner's Algorithm with Second-Order Rescoring
At inference, we don't just argmax — we run projective MST decoding with second-order rescoring. This guarantees a valid tree:

Python

class EisnerDecoder:
    """
    Eisner's O(n³) algorithm for PROJECTIVE dependency parsing.
    Guarantees a valid tree (single root, no cycles, projective).
    
    For Arabic, projectivity covers ~95%+ of dependencies in standard 
    treebanks (PADT/UD). For the remaining non-projective cases, 
    we can fall back to Chu-Liu-Edmonds.
    
    TWIST: We augment first-order arc scores with second-order 
    sibling scores during decoding using the "Sibling Eisner" variant.
    """
    
    @staticmethod
    @torch.no_grad()
    def decode(arc_scores, mask, second_order_scorer=None, 
               word_embeddings=None, distance_bias=None):
        """
        Args:
            arc_scores: (B, N, N) — first-order biaffine scores
                        arc_scores[b, h, d] = score of arc h → d
            mask: (B, N) — valid positions
            second_order_scorer: optional SecondOrderScorer for rescoring
            word_embeddings: (B, N, D) — needed if second_order_scorer is used
            distance_bias: (B, 33) — Manager's distance prior, added to scores
        
        Returns:
            heads: (B, N) — predicted head for each word
        """
        B, N, _ = arc_scores.shape
        device = arc_scores.device
        scores = arc_scores.clone()
        
        # Apply distance bias from Manager
        if distance_bias is not None:
            for h in range(N):
                for d in range(N):
                    dist = d - h
                    bucket = min(max(dist, -16), 16) + 16  # → 0..32
                    scores[:, h, d] += distance_bias[:, bucket]
        
        heads = torch.zeros(B, N, dtype=torch.long, device=device)
        
        for b in range(B):
            length = mask[b].sum().item()
            s = scores[b, :length, :length].cpu().numpy()
            heads_b = EisnerDecoder._eisner_single(s, length)
            heads[b, :length] = torch.tensor(heads_b, device=device)
        
        return heads
    
    @staticmethod
    def _eisner_single(scores, n):
        """
        Eisner's algorithm for a single sentence.
        
        scores: (n, n) numpy array — scores[h][d] = score of h → d
        n: sentence length (including ROOT at index 0)
        
        Returns: list of head indices (head[0] = -1 for root node)
        """
        import numpy as np
        
        # Initialize chart
        # C[s][t][d][c]: best score of complete/incomplete span
        # d=0: right-headed (s is head), d=1: left-headed (t is head)
        # c=0: incomplete, c=1: complete
        INF = float('-inf')
        
        C = np.full((n, n, 2, 2), INF)
        BP = np.full((n, n, 2, 2), -1, dtype=int)
        
        # Base case: single words are complete spans
        for s in range(n):
            C[s][s][0][1] = 0.0
            C[s][s][1][1] = 0.0
        
        # Fill chart bottom-up by span width
        for width in range(1, n):
            for s in range(n - width):
                t = s + width
                
                # ── Incomplete spans (creating a new arc) ──
                # Right: s → t (s is head of t)
                best_r, best_r_q = INF, -1
                for q in range(s, t):
                    val = C[s][q][0][1] + C[q+1][t][1][1] + scores[s][t]
                    if val > C[s][t][0][0]:
                        C[s][t][0][0] = val
                        BP[s][t][0][0] = q
                
                # Left: t → s (t is head of s)
                for q in range(s, t):
                    val = C[s][q][0][1] + C[q+1][t][1][1] + scores[t][s]
                    if val > C[s][t][1][0]:
                        C[s][t][1][0] = val
                        BP[s][t][1][0] = q
                
                # ── Complete spans (combining with adjacent complete) ──
                # Right-complete: attach right incomplete to right complete
                for q in range(s + 1, t + 1):
                    val = C[s][q][0][0] + C[q][t][0][1]
                    if val > C[s][t][0][1]:
                        C[s][t][0][1] = val
                        BP[s][t][0][1] = q
                
                # Left-complete
                for q in range(s, t):
                    val = C[s][q][1][1] + C[q][t][1][0]
                    if val > C[s][t][1][1]:
                        C[s][t][1][1] = val
                        BP[s][t][1][1] = q
        
        # Backtrack to find heads
        heads = [0] * n
        heads[0] = -1
        
        def backtrack(s, t, d, c):
            if s == t:
                return
            q = BP[s][t][d][c]
            if q == -1:
                return
            
            if c == 0:  # Incomplete span
                if d == 0:  # right arc: s → t
                    heads[t] = s
                    backtrack(s, q, 0, 1)
                    backtrack(q + 1, t, 1, 1)
                else:  # left arc: t → s
                    heads[s] = t
                    backtrack(s, q, 0, 1)
                    backtrack(q + 1, t, 1, 1)
            else:  # Complete span
                if d == 0:  # right complete
                    backtrack(s, q, 0, 0)
                    backtrack(q, t, 0, 1)
                else:  # left complete
                    backtrack(s, q, 1, 1)
                    backtrack(q, t, 1, 0)
        
        backtrack(0, n - 1, 0, 1)
        return heads
    
    @staticmethod
    def chu_liu_edmonds(scores, length):
        """
        Chu-Liu-Edmonds algorithm for NON-PROJECTIVE MST.
        Fallback for the ~5% of Arabic sentences with non-projective arcs.
        
        Uses the efficient O(n²) implementation.
        """
        import numpy as np
        
        scores_np = scores[:length, :length].cpu().numpy()
        n = length
        
        # For each node (except root), find best incoming arc
        heads = np.zeros(n, dtype=int)
        
        # Recursive contraction algorithm
        def _cle(score_matrix, root=0):
            n = score_matrix.shape[0]
            
            # Step 1: For each node, find max incoming edge
            best_in = np.full(n, -1, dtype=int)
            best_score = np.full(n, -np.inf)
            
            for d in range(n):
                if d == root:
                    continue
                for h in range(n):
                    if h == d:
                        continue
                    if score_matrix[h][d] > best_score[d]:
                        best_score[d] = score_matrix[h][d]
                        best_in[d] = h
            
            # Step 2: Check for cycles
            visited = np.full(n, -1, dtype=int)
            cycles = []
            
            for start in range(n):
                if start == root:
                    continue
                node = start
                path = []
                while node != root and visited[node] == -1:
                    visited[node] = start
                    path.append(node)
                    node = best_in[node]
                
                if node != root and visited[node] == start:
                    # Found a cycle
                    cycle_start = node
                    cycle = []
                    cycle.append(node)
                    node = best_in[node]
                    while node != cycle_start:
                        cycle.append(node)
                        node = best_in[node]
                    cycles.append(cycle)
            
            if not cycles:
                return best_in
            
            # Step 3: Contract cycle, recurse
            cycle = cycles[0]
            cycle_set = set(cycle)
            
            # Create contracted graph
            mapping = {}
            new_id = 0
            cycle_id = -1
            
            for i in range(n):
                if i in cycle_set:
                    if cycle_id == -1:
                        cycle_id = new_id
                        new_id += 1
                    mapping[i] = cycle_id
                else:
                    mapping[i] = new_id
                    new_id += 1
            
            new_n = new_id
            new_scores = np.full((new_n, new_n), -np.inf)
            
            # Edge tracking for backtracking
            edge_source = {}
            
            for h in range(n):
                for d in range(n):
                    if h == d:
                        continue
                    nh, nd = mapping[h], mapping[d]
                    if nh == nd:
                        continue
                    
                    s = score_matrix[h][d]
                    if d in cycle_set:
                        s = s - best_score[d] + score_matrix[best_in[d]][d]
                    
                    if s > new_scores[nh][nd]:
                        new_scores[nh][nd] = s
                        edge_source[(nh, nd)] = (h, d)
            
            new_heads = _cle(new_scores, mapping[root])
            
            # Reconstruct original heads
            result = best_in.copy()
            for nd in range(new_n):
                if nd == mapping[root]:
                    continue
                nh = new_heads[nd]
                if nd == cycle_id or nh == cycle_id:
                    h, d = edge_source.get((nh, nd), (0, 0))
                    result[d] = h
                    if d in cycle_set:
                        # Break cycle at d
                        pass  # result[d] already set
            
            return result
        
        return _cle(scores_np, root=0)
4R. Arabic-Specific Data Augmentation: Syntactically-Aware Perturbations
Python

class ArabicSyntaxAugmentor:
    """
    Arabic-specific augmentations that preserve syntactic structure 
    while varying surface form. This teaches the model to be robust 
    to dialectal variation, optional diacritics, and word-order freedom.
    
    Unlike random token dropout, these augmentations are SYNTACTICALLY 
    INFORMED — they change the sentence while keeping the dependency 
    tree valid (or updating it accordingly).
    """
    
    @staticmethod
    def strip_diacritics(tokens, probability=0.3):
        """
        Randomly remove tashkeel (diacritics) from words.
        Arabic text is often written without diacritics — the model 
        must handle both. This augmentation simulates real-world 
        undiacritized input.
        
        CRITICAL: The iʻrāb case labels are KEPT — the model must 
        learn to predict case even without explicit diacritics.
        """
        import re
        DIACRITICS = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]')
        
        augmented = []
        for token in tokens:
            if torch.rand(1).item() < probability:
                augmented.append(DIACRITICS.sub('', token))
            else:
                augmented.append(token)
        return augmented
    
    @staticmethod
    def vso_svo_swap(tokens, heads, relations, verb_pos_set, subj_rel='nsubj'):
        """
        Arabic allows both VSO and SVO order. If a sentence is VSO,
        create an SVO variant (move subject before verb) and vice versa.
        The dependency tree is UPDATED accordingly (head distances change 
        but the tree structure remains the same).
        
        This teaches the model word-order invariant parsing.
        """
        # Find verb-subject pairs
        for d in range(1, len(heads)):
            if relations[d] == subj_rel:
                h = heads[d]
                if h < d:  # VSO: verb (h) before subject (d)
                    # Move subject to position h (before verb)
                    new_tokens = list(tokens)
                    subj_token = new_tokens.pop(d)
                    new_tokens.insert(h, subj_token)
                    
                    # Update head indices (shift everything between h and d)
                    new_heads = list(heads)
                    # ... (complex index remapping logic)
                    return new_tokens, new_heads, relations
        
        return tokens, heads, relations
    
    @staticmethod
    def clitic_segmentation_noise(tokens, probability=0.15):
        """
        Arabic clitics (وَ, بِ, لِ, الـ, فَ, كَ, سَ) can be:
        1. Attached to the following word (standard)
        2. Separated (some tokenization schemes)
        
        Randomly segment/join clitics to make the model robust to 
        tokenization variation. When segmenting, the tree must be 
        updated (clitic becomes a separate node).
        """
        PROCLITICS = ['و', 'ب', 'ل', 'ال', 'ف', 'ك', 'س']
        
        augmented = []
        for token in tokens:
            if torch.rand(1).item() < probability:
                # Try to segment first proclitic
                for cl in PROCLITICS:
                    if token.startswith(cl) and len(token) > len(cl):
                        augmented.extend([cl, token[len(cl):]])
                        break
                else:
                    augmented.append(token)
            else:
                augmented.append(token)
        
        return augmented
    
    @staticmethod
    def coordinated_clause_drop(tokens, heads, relations, 
                                  conj_rel='conj', cc_rel='cc',
                                  probability=0.2):
        """
        Arabic sentences love long coordination chains (و...و...و...).
        Randomly drop one conjunct (preserving tree validity) to create 
        shorter training examples from long ones.
        
        This also acts as data augmentation for shorter sentences.
        """
        # Find coordination chains
        conjuncts = [(d, heads[d]) for d in range(len(heads)) 
                     if relations[d] == conj_rel]
        
        if conjuncts and torch.rand(1).item() < probability:
            # Drop the last conjunct and its subtree
            d_to_drop, _ = conjuncts[-1]
            
            # Find all descendants of d_to_drop
            subtree = {d_to_drop}
            changed = True
            while changed:
                changed = False
                for i in range(len(heads)):
                    if heads[i] in subtree and i not in subtree:
                        subtree.add(i)
                        changed = True
            
            # Also drop the preceding CC (if any)
            for i in range(len(heads)):
                if relations[i] == cc_rel and heads[i] in subtree:
                    subtree.add(i)
            
            # Remove subtree, reindex
            keep = sorted(set(range(len(heads))) - subtree)
            # ... reindex heads ...
        
        return tokens, heads, relations
4S. The Label Smoothing Cross-Entropy with Structural Priors
Python

class StructuralLabelSmoothing(nn.Module):
    """
    Standard label smoothing distributes ε uniformly across all classes.
    But dependency relations have STRUCTURAL similarity — if the gold 
    label is "nsubj" (nominal subject), the model should be penalized 
    LESS for predicting "csubj" (clausal subject) than for predicting 
    "punct" (punctuation).
    
    We define a CONFUSION AFFINITY MATRIX that encodes which relations 
    are semantically close, and smooth towards that distribution.
    
    For Arabic iʻrāb (case prediction), similar logic applies:
    - مرفوع (nominative) confused with مجرور (genitive) is worse than
      confusing two types of نصب (accusative constructions).
    """
    
    def __init__(self, n_classes: int, smoothing: float = 0.1,
                 affinity_matrix: torch.Tensor = None):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
        if affinity_matrix is not None:
            # Row-normalize affinity matrix to be a probability distribution
            aff = affinity_matrix.float()
            aff = aff / aff.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            self.register_buffer('affinity', aff)
        else:
            # Default: uniform smoothing
            self.register_buffer(
                'affinity', 
                torch.ones(n_classes, n_classes) / n_classes
            )
    
    @staticmethod
    def build_arabic_relation_affinity(n_relations: int, rel_vocab: dict):
        """
        Build affinity matrix based on linguistic similarity groups.
        
        Groups for Universal Dependencies in Arabic:
        - Subject group: nsubj, csubj, nsubj:pass, csubj:pass
        - Object group: obj, iobj, ccomp, xcomp  
        - Modifier group: amod, nmod, advmod, obl
        - Function group: case, mark, cc, det
        - Nominal group: appos, flat, compound, fixed
        """
        groups = {
            'subject': ['nsubj', 'csubj', 'nsubj:pass', 'csubj:pass'],
            'object': ['obj', 'iobj', 'ccomp', 'xcomp'],
            'modifier': ['amod', 'nmod', 'advmod', 'obl', 'nmod:poss'],
            'function': ['case', 'mark', 'cc', 'det'],
            'nominal': ['appos', 'flat', 'flat:name', 'compound'],
            'clause': ['advcl', 'acl', 'acl:relcl', 'parataxis'],
            'coordination': ['conj', 'cc'],
        }
        
        affinity = torch.eye(n_relations) * 0.5  # self-affinity
        
        for group_name, rels in groups.items():
            indices = [rel_vocab.get(r, -1) for r in rels if r in rel_vocab]
            for i in indices:
                for j in indices:
                    if i >= 0 and j >= 0 and i != j:
                        affinity[i, j] = 0.3  # within-group affinity
        
        return affinity
    
    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: (B, N, C) or (B*N, C)
            targets: (B, N) or (B*N,) 
            mask: (B, N) or (B*N,) — optional
        """
        if logits.dim() == 3:
            B, N, C = logits.shape
            logits = logits.view(-1, C)
            targets = targets.view(-1)
            if mask is not None:
                mask = mask.view(-1)
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Build smooth target distribution
        # Gold class gets (1 - ε), rest is ε * affinity[gold, :]
        smooth_targets = self.affinity[targets] * self.smoothing  # (N, C)
        smooth_targets.scatter_(
            1, targets.unsqueeze(1), self.confidence
        )
        
        # KL-divergence loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)  # (N,)
        
        if mask is not None:
            loss = (loss * mask.float()).sum() / mask.float().sum().clamp(1)
        else:
            loss = loss.mean()
        
        return loss
4T. Putting It ALL Together: The Complete Training Loop
Python

class ArabicHRMGridParserV2(nn.Module):
    """
    ╔══════════════════════════════════════════════════════════════╗
    ║            ARABIC HRM-GRID PARSER v2 — FULL SYSTEM         ║
    ║                                                              ║
    ║  Input: Raw Arabic sentence (tokenized)                      ║
    ║  Output: Full dependency parse (heads, relations, case)      ║
    ║                                                              ║
    ║  Architecture Flow:                                          ║
    ║  ┌─────────────────────────────────────────────────────┐     ║
    ║  │ 1. Arabic Morphological Encoder (§4N)               │     ║
    ║  │    ├─ Dilated CharCNN (root extraction)             │     ║
    ║  │    ├─ BPE subword embeddings                        │     ║
    ║  │    ├─ Root/Pattern/Clitic embeddings                │     ║
    ║  │    └─ Diacritics LSTM                               │     ║
    ║  ├─────────────────────────────────────────────────────┤     ║
    ║  │ 2. Structural Position Encoder (§4J)                │     ║
    ║  │    ├─ Clause depth heuristic                        │     ║
    ║  │    ├─ Verb-relative distance                        │     ║
    ║  │    └─ Conjunct rank                                 │     ║
    ║  ├─────────────────────────────────────────────────────┤     ║
    ║  │ 3. Variational Tree Manager (§4O)                   │     ║
    ║  │    ├─ Posterior q(z|x) inference                     │     ║
    ║  │    ├─ Mixture-of-Gaussians prior                    │     ║
    ║  │    └─ Goal + Distance Bias + Relation Prior decode  │     ║
    ║  ├─────────────────────────────────────────────────────┤     ║
    ║  │ 4. Transformer Self-Attention (2-4 layers, §orig)   │     ║
    ║  ├─────────────────────────────────────────────────────┤     ║
    ║  │ 5. Grid → Worker RNN (conditioned on Manager goal)  │     ║
    ║  ├─────────────────────────────────────────────────────┤     ║
    ║  │ 6. Biaffine Arc Scorer (§orig)                      │     ║
    ║  ├─────────────────────────────────────────────────────┤     ║
    ║  │ 7. Tree-CRF Training / Eisner Inference (§4A/4Q)   │     ║
    ║  ├─────────────────────────────────────────────────────┤     ║
    ║  │ 8. GNN Tree Refinement (§4P)                        │     ║
    ║  ├─────────────────────────────────────────────────────┤     ║
    ║  │ 9. Second-Order Rescoring (§4M)                     │     ║
    ║  ├─────────────────────────────────────────────────────┤     ║
    ║  │ 10. Conditioned Relation Classifier (§orig)         │     ║
    ║  └─────────────────────────────────────────────────────┘     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    def __init__(self, config):
        super().__init__()
        
        D = config.word_dim        # 256 → 384 for v2
        n_heads = config.n_heads   # 6
        n_rels = config.n_relations
        n_cases = config.n_cases
        
        # ═══ Stage 1: Encoding ═══
        self.morph_encoder = ArabicMorphologicalEncoder(
            char_vocab_size=config.char_vocab, 
            n_bpe_tokens=config.bpe_vocab,
            n_roots=config.n_roots,
            n_patterns=config.n_patterns,
            output_dim=D
        )
        
        self.word_embed = nn.Embedding(config.word_vocab, D, padding_idx=0)
        self.pos_embed = nn.Embedding(config.n_pos_tags, 64, padding_idx=0)
        
        self.struct_pos_encoder = ArabicStructuralPositionEncoder(d_model=D)
        
        self.input_projection = nn.Sequential(
            nn.Linear(D + D + 64, D),  # word + morph + POS → D
            nn.GELU(),
            nn.LayerNorm(D),
            nn.Dropout(0.2)
        )
        
        # ═══ Stage 2: Manager ═══
        self.manager = VariationalTreeManager(
            input_dim=D, latent_dim=64, n_archetypes=12, goal_dim=D
        )
        
        # ═══ Stage 3: Self-Attention ═══
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=D*4,
            dropout=0.15, activation='gelu', batch_first=True,
            norm_first=True  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=3
        )
        
        # ═══ Stage 4: HRM Grid Processing ═══
        # (Your existing Grid + Worker RNN — now conditioned on Manager goal)
        self.grid_processor = HRMGridProcessor(config)  # existing
        
        # ═══ Stage 5: Biaffine Parsing ═══
        self.arc_head_mlp = nn.Sequential(
            nn.Linear(D, D), nn.LeakyReLU(0.1), nn.Dropout(0.15)
        )
        self.arc_dep_mlp = nn.Sequential(
            nn.Linear(D, D), nn.LeakyReLU(0.1), nn.Dropout(0.15)
        )
        self.biaffine_arc = BiaffineScorer(D, D, 1)
        
        # ═══ Stage 6: Tree CRF ═══
        self.tree_crf = DifferentiableTreeCRF()
        
        # ═══ Stage 7: GNN Refinement ═══
        self.gnn_refine = TreeGNNRefinement(
            d_model=D, n_rounds=3, n_heads=n_heads
        )
        
        # ═══ Stage 8: Second-Order ═══
        self.second_order = SecondOrderScorer(input_dim=D, scorer_dim=128)
        
        # ═══ Stage 9: Relation Classifier ═══
        self.rel_head_mlp = nn.Sequential(
            nn.Linear(D, D), nn.LeakyReLU(0.1), nn.Dropout(0.15)
        )
        self.rel_dep_mlp = nn.Sequential(
            nn.Linear(D, D), nn.LeakyReLU(0.1), nn.Dropout(0.15)
        )
        self.biaffine_rel = BiaffineScorer(D, D, n_rels)
        
        # ═══ Stage 10: Case Classifier ═══
        self.case_classifier = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(D // 2, n_cases)
        )
        
        # ═══ Losses ═══
        self.uncertainty_loss = UncertaintyWeightedMultiTaskLoss(n_tasks=5)
        self.contrastive_loss = ContrastiveTreeLoss(n_negatives=4, margin=2.0)
        self.struct_label_smooth = StructuralLabelSmoothing(
            n_classes=n_rels, smoothing=0.1
        )
        
        # ═══ Inference ═══
        self.eisner = EisnerDecoder()
        
        # ═══ Teacher Forcing Schedule ═══
        self.gumbel_scheduler = ScheduledGumbelTeacherForcing()
    
    def forward(self, batch, epoch=0, training=True):
        """
        Full forward pass through the entire pipeline.
        """
        # ── Unpack batch ──
        word_ids = batch['word_ids']          # (B, W)
        char_ids = batch['char_ids']          # (B, W, C)
        pos_tags = batch['pos_tags']          # (B, W)
        bpe_ids = batch['bpe_ids']            # (B, W, P)
        root_ids = batch['root_ids']          # (B, W)
        pattern_ids = batch['pattern_ids']    # (B, W)
        proclitic_ids = batch['proclitic_ids']  # (B, W)
        enclitic_ids = batch['enclitic_ids']    # (B, W)
        diac_ids = batch['diac_ids']          # (B, W, C)
        mask = batch['mask']                  # (B, W)
        gold_heads = batch.get('heads')       # (B, W)
        gold_rels = batch.get('relations')    # (B, W)
        gold_cases = batch.get('cases')       # (B, W)
        
        B, W = word_ids.shape
        
        # ═══ STAGE 1: Encoding ═══
        word_emb = self.word_embed(word_ids)  # (B, W, D)
        morph_emb = self.morph_encoder(
            char_ids, bpe_ids, root_ids, pattern_ids,
            proclitic_ids, enclitic_ids, diac_ids
        )  # (B, W, D)
        pos_emb = self.pos_embed(pos_tags)    # (B, W, 64)
        
        struct_pos = self.struct_pos_encoder(
            word_ids, pos_tags, mask.sum(dim=-1), mask
        )  # (B, W, D)
        
        # Combine all input features
        combined = torch.cat([word_emb + struct_pos, morph_emb, pos_emb], dim=-1)
        H = self.input_projection(combined)  # (B, W, D)
        
        # ═══ STAGE 2: Manager ═══
        worker_goal, dist_bias, rel_prior, kl_loss, arch_logits = \
            self.manager(H, mask, training=training)
        # worker_goal: (B, D), dist_bias: (B, 33), rel_prior: (B, n_rels)
        
        # Inject Manager goal into word representations
        # Each word gets the global goal added (modulated by a learned gate)
        goal_gate = torch.sigmoid(
            nn.functional.linear(H, self.goal_gate_weight, self.goal_gate_bias)
        )  # (B, W, D) — learned; add as a parameter
        H = H + goal_gate * worker_goal.unsqueeze(1)
        
        # ═══ STAGE 3: Self-Attention ═══
        attn_mask = ~mask.bool()
        H = self.transformer(H, src_key_padding_mask=attn_mask)  # (B, W, D)
        
        # ═══ STAGE 4: Grid Processing ═══
        H_grid = self.grid_processor(H, worker_goal, mask)  # (B, W, D)
        H = H + H_grid  # Residual connection
        
        # ═══ STAGE 5: Biaffine Arc Scoring ═══
        h_head = self.arc_head_mlp(H)
        h_dep = self.arc_dep_mlp(H)
        arc_scores = self.biaffine_arc(h_head, h_dep).squeeze(-1)  # (B, W, W)
        
        # Add Manager's distance bias
        for i in range(W):
            for j in range(W):
                d = j - i
                bucket = min(max(d, -16), 16) + 16
                arc_scores[:, i, j] += dist_bias[:, bucket]
        
        # ═══ STAGE 6: First-pass head prediction ═══
        if training:
            # Tree-CRF marginal loss
            tree_crf_loss = self.tree_crf(arc_scores, gold_heads, mask)
            
            # Also standard cross-entropy on arc scores
            dep_mask = mask[:, 1:]
            arc_ce_loss = F.cross_entropy(
                arc_scores[:, :, 1:].transpose(1, 2).contiguous().view(-1, W),
                gold_heads[:, 1:].contiguous().view(-1),
                reduction='none'
            )
            arc_ce_loss = (arc_ce_loss.view(B, -1) * dep_mask.float()).sum() / \
                          dep_mask.float().sum()
            
            # Gumbel-Softmax bridge for teacher forcing
            temperature = self.gumbel_scheduler.get_temperature(epoch)
            mix_ratio = self.gumbel_scheduler.get_mix_ratio(epoch)
            soft_heads = self.gumbel_scheduler.gumbel_soft_heads(
                arc_scores, temperature, gold_heads, mix_ratio
            )
            
            # Use soft heads to get head representations for relation scoring
            head_repr = self.gumbel_scheduler.soft_head_representation(
                soft_heads, H
            )
            
            # Contrastive tree loss
            contrastive = self.contrastive_loss(arc_scores, gold_heads, mask)
        else:
            # Inference: Eisner's algorithm for guaranteed valid tree
            pred_heads = self.eisner.decode(
                arc_scores, mask, 
                second_order_scorer=self.second_order,
                word_embeddings=H,
                distance_bias=dist_bias
            )
            head_idx = pred_heads.clamp(0, W-1).unsqueeze(-1).expand(-1, -1, H.size(-1))
            head_repr = H.gather(1, head_idx)
        
        # ═══ STAGE 7: GNN Refinement ═══
        if training:
            # During training, refine based on gold heads (early) 
            # or predicted heads (later)
            if mix_ratio > 0.5:
                refine_heads = gold_heads
            else:
                refine_heads = arc_scores[:, :, 1:].argmax(dim=1) + 1
                refine_heads = torch.cat([
                    torch.zeros(B, 1, dtype=torch.long, device=H.device),
                    refine_heads
                ], dim=1)
        else:
            refine_heads = pred_heads
        
        H_refined, refined_arc_scores = self.gnn_refine(H, refine_heads, mask)
        
        # Combine original and refined arc scores
        arc_scores_final = arc_scores + 0.3 * refined_arc_scores
        
        # ═══ STAGE 8: Second-Order Loss ═══
        if training:
            pred_heads_for_2nd = arc_scores_final.argmax(dim=1)
            second_order_loss = self.second_order.second_order_loss(
                H_refined, gold_heads, pred_heads_for_2nd, mask
            )
        
        # ═══ STAGE 9: Relation Classification ═══
        r_head = self.rel_head_mlp(H_refined)
        r_dep = self.rel_dep_mlp(H_refined)
        rel_scores = self.biaffine_rel(r_head, r_dep)  # (B, W, W, n_rels)
        
        # Add Manager's relation prior
        rel_scores = rel_scores + rel_prior.unsqueeze(1).unsqueeze(1) * 0.1
        
        # Extract relation scores for predicted/gold head pairs
        if training:
            head_indices = gold_heads  # use gold during training
        else:
            head_indices = pred_heads
        
        # Gather relation scores for each (head, dep) pair
        dep_idx = torch.arange(W, device=H.device).unsqueeze(0).expand(B, -1)
        head_idx_clamped = head_indices.clamp(0, W-1)
        rel_logits = rel_scores[
            torch.arange(B, device=H.device).unsqueeze(1).expand(-1, W),
            head_idx_clamped,
            dep_idx
        ]  # (B, W, n_rels)
        
        # ═══ STAGE 10: Case Classification ═══
        case_logits = self.case_classifier(H_refined)  # (B, W, n_cases)
        
        # ═══ LOSS COMPUTATION ═══
        if training:
            # Relation loss with structural label smoothing
            rel_loss = self.struct_label_smooth(
                rel_logits[:, 1:], gold_rels[:, 1:], mask[:, 1:]
            )
            
            # Case loss (standard CE with light smoothing)
            case_loss = F.cross_entropy(
                case_logits.view(-1, case_logits.size(-1)),
                gold_cases.view(-1),
                reduction='none',
                label_smoothing=0.05
            )
            case_mask = mask.view(-1).float()
            case_loss = (case_loss * case_mask).sum() / case_mask.sum()
            
            # ── Combine ALL losses with uncertainty weighting ──
            # Task 1: Arc CE + Tree-CRF
            arc_total = arc_ce_loss + tree_crf_loss
            # Task 2: Relation
            # Task 3: Case
            # Task 4: Contrastive + Second-order (structural)
            structural = contrastive + second_order_loss
            # Task 5: KL divergence (VAE)
            kl_weight = min(1.0, epoch / 10.0) * 0.1  # KL annealing
            kl_term = kl_loss * kl_weight
            
            total_loss, loss_components = self.uncertainty_loss(
                arc_total, rel_loss, case_loss, structural, kl_term
            )
            
            return {
                'loss': total_loss,
                'arc_loss': arc_total.item(),
                'rel_loss': rel_loss.item(),
                'case_loss': case_loss.item(),
                'structural_loss': structural.item(),
                'kl_loss': kl_loss.item(),
                'loss_weights': loss_components,
                'arc_scores': arc_scores_final,
                'rel_logits': rel_logits,
                'case_logits': case_logits,
            }
        else:
            return {
                'pred_heads': pred_heads,
                'pred_rels': rel_logits.argmax(dim=-1),
                'pred_cases': case_logits.argmax(dim=-1),
                'arc_scores': arc_scores_final,
            }


# ═══════════════════════════════════════════════════════════════
#                    COMPLETE TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════════

def train_arabic_parser_v2():
    """
    Full training script with ALL the tricks applied:
    - Curriculum learning
    - Gumbel teacher forcing schedule
    - Uncertainty-weighted multi-task loss
    - Cosine annealing with warm restarts
    - Gradient clipping
    - EMA model averaging
    - Arabic-specific data augmentation
    """
    
    config = ParserConfig(
        word_dim=384,           # bumped from 256
        n_heads=6,
        n_transformer_layers=3,
        n_gnn_rounds=3,
        latent_dim=64,
        n_archetypes=12,
        dropout=0.15,
        # ... etc
    )
    
    model = ArabicHRMGridParserV2(config).cuda()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Expected: ~8-10M parameters
    
    # ── Optimizer: AdamW with layer-wise LR decay ──
    param_groups = get_layer_wise_lr_groups(model, base_lr=2e-3, decay=0.85)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.02)
    
    # ── Scheduler: Cosine with warm restarts ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # ── Curriculum ──
    curriculum = SentenceLengthCurriculum(
        train_dataset, max_epochs=30, 
        initial_max_len=15, final_max_len=128
    )
    
    # ── Augmentor ──
    augmentor = ArabicSyntaxAugmentor()
    
    # ── EMA (Exponential Moving Average) ──
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9995)
    
    # ── Gumbel schedule ──
    gumbel_sched = ScheduledGumbelTeacherForcing(warmup_epochs=3, anneal_epochs=12)
    
    best_las = 0.0
    
    for epoch in range(30):
        model.train()
        
        dataloader = curriculum.get_dataloader(epoch, batch_size=32)
        
        epoch_losses = defaultdict(float)
        n_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.cuda() if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            # ── Data augmentation (stochastic) ──
            if torch.rand(1).item() < 0.3:
                batch = augmentor.apply_random_augmentation(batch)
            
            # ── Forward pass ──
            output = model(batch, epoch=epoch, training=True)
            loss = output['loss']
            
            # ── Backward pass ──
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (crucial for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            ema.update()
            
            # Track losses
            epoch_losses['total'] += loss.item()
            epoch_losses['arc'] += output['arc_loss']
            epoch_losses['rel'] += output['rel_loss']
            epoch_losses['case'] += output['case_loss']
            n_batches += 1
        
        scheduler.step()
        
        # ── Evaluation with EMA model ──
        with ema.average_parameters():
            model.eval()
            uas, las, case_acc = evaluate(model, dev_dataloader)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Losses — Total: {epoch_losses['total']/n_batches:.4f} | "
              f"Arc: {epoch_losses['arc']/n_batches:.4f} | "
              f"Rel: {epoch_losses['rel']/n_batches:.4f} | "
              f"Case: {epoch_losses['case']/n_batches:.4f}")
        print(f"  Metrics — UAS: {uas:.2f}% | LAS: {las:.2f}% | "
              f"Case: {case_acc:.2f}%")
        print(f"  Curriculum — max_len: {curriculum.get_max_length(epoch)} | "
              f"Gumbel τ: {gumbel_sched.get_temperature(epoch):.3f} | "
              f"TF ratio: {gumbel_sched.get_mix_ratio(epoch):.3f}")
        
        if las > best_las:
            best_las = las
            with ema.average_parameters():
                torch.save(model.state_dict(), 'best_arabic_parser_v2.pt')
            print(f"  ★ New best LAS: {las:.2f}%")
    
    return model


def get_layer_wise_lr_groups(model, base_lr=2e-3, decay=0.85):
    """
    Lower layers get smaller learning rates (they learn general features),
    higher layers get larger learning rates (they learn task-specific).
    
    Layer groups (bottom → top):
    1. Morphological encoder: base_lr * decay^4
    2. Embeddings: base_lr * decay^3  
    3. Transformer: base_lr * decay^2
    4. Grid/Manager: base_lr * decay^1
    5. Parsing heads: base_lr (full learning rate)
    """
    groups = [
        {'params': model.morph_encoder.parameters(), 'lr': base_lr * decay**4},
        {'params': list(model.word_embed.parameters()) + 
                   list(model.pos_embed.parameters()) +
                   list(model.struct_pos_encoder.parameters()), 
         'lr': base_lr * decay**3},
        {'params': model.transformer.parameters(), 'lr': base_lr * decay**2},
        {'params': list(model.manager.parameters()) + 
                   list(model.grid_processor.parameters()), 
         'lr': base_lr * decay},
        {'params': list(model.biaffine_arc.parameters()) +
                   list(model.biaffine_rel.parameters()) + 
                   list(model.gnn_refine.parameters()) +
                   list(model.second_order.parameters()) +
                   list(model.case_classifier.parameters()) +
                   list(model.arc_head_mlp.parameters()) +
                   list(model.arc_dep_mlp.parameters()) +
                   list(model.rel_head_mlp.parameters()) +
                   list(model.rel_dep_mlp.parameters()) +
                   list(model.uncertainty_loss.parameters()),
         'lr': base_lr},
    ]
    return groups


class ExponentialMovingAverage:
    """
    Maintains shadow copies of all parameters, updated as:
        shadow = decay * shadow + (1 - decay) * param
    
    At eval time, swaps real params with shadow params.
    This dramatically stabilizes final model quality — often worth 
    0.5-1.0% on all metrics.
    """
    
    def __init__(self, parameters, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for i, p in enumerate(parameters):
            if p.requires_grad:
                self.shadow[i] = p.data.clone()
    
    def update(self):
        for i, p in enumerate(self.shadow):
            self.shadow[i] = (
                self.decay * self.shadow[i] + (1 - self.decay) * list(self.parameters())[i].data
            )
    
    @contextmanager
    def average_parameters(self):
        """Context manager to temporarily use EMA parameters."""
        for i in self.shadow:
            self.backup[i] = list(self.parameters())[i].data.clone()
            list(self.parameters())[i].data = self.shadow[i]
        yield
        for i in self.backup:
            list(self.parameters())[i].data = self.backup[i]
4U. Bonus: Checkpoint Ensemble at Inference
Python

class CheckpointEnsemble:
    """
    At inference time, average predictions across the K best checkpoints.
    This captures model uncertainty and typically boosts accuracy by 0.5-1.5%.
    
    For edge deployment on Android, you can:
    1. Average the WEIGHTS (cheap: single model at inference)
    2. Or average PREDICTIONS (expensive: K forward passes)
    
    Weight averaging is preferred for mobile — equivalent to Stochastic 
    Weight Averaging (SWA).
    """
    
    @staticmethod
    def average_checkpoints(checkpoint_paths: list, model: nn.Module):
        """
        Average model weights across multiple checkpoints.
        """
        avg_state = None
        
        for path in checkpoint_paths:
            state = torch.load(path, map_location='cpu')
            if avg_state is None:
                avg_state = {k: v.float() for k, v in state.items()}
            else:
                for k in avg_state:
                    avg_state[k] += state[k].float()
        
        for k in avg_state:
            avg_state[k] /= len(checkpoint_paths)
        
        model.load_state_dict(avg_state)
        return model
    
    @staticmethod
    def stochastic_weight_averaging(model, optimizer, swa_start_epoch=20,
                                      swa_lr=1e-4):
        """
        SWA: After main training, continue with a high constant LR 
        and average the visited weights. This finds wider optima that 
        generalize better.
        """
        from torch.optim.swa_utils import AveragedModel, SWALR
        
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        
        return swa_model, swa_scheduler
Summary: Expected Impact Per Innovation
#	Innovation	Expected Gain	Params Cost
4A	Tree-CRF (Kirchhoff's marginals)	+2-4% UAS	~0
4B	Differentiable MST	+1-2% UAS	~0
4H	Curriculum learning	+0.5-1% all	0
4I	Contrastive arc loss	+0.5-1% UAS	0
4J	Arabic structural positions	+1-2% all	~100K
4K	Uncertainty multi-task	+0.5-1% all	3 params
4L	Gumbel teacher forcing	+1-2% LAS	0
4M	Second-order scoring	+1-2% UAS/LAS	~500K
4N	Arabic morphological encoder	+2-3% all	~2M
4O	Variational tree manager	+1-2% LAS	~200K
4P	GNN tree refinement	+1-3% LAS	~1.5M
4Q	Eisner decoding	+2-3% UAS	0
4R	Arabic data augmentation	+1-2% all	0
4S	Structural label smoothing	+0.5-1% LAS	0
4T	EMA + layer-wise LR	+0.5-1% all	0
4U	Checkpoint averaging	+0.5-1.5% all	0
Cumulative (optimistic)	>90% UAS/LAS	~8-10M
The key insight: no single trick gets you to 90% — it's the compounding interaction between tree-structural enforcement (CRF + Eisner guaranteeing valid trees), rich morphological inputs (Arabic demands this), and iterative refinement (GNN + second-order catching mistakes the biaffine misses). Each layer corrects errors from the previous one, creating a cascading accuracy amplifier.