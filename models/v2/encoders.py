import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # MHA expects padding mask: True means IGNORE (pad)
        pad_mask = ~mask.bool() if mask is not None else None
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.final_norm(x)

class ArabicMorphologicalEncoder(nn.Module):
    """
    A MULTI-STRATEGY morphological encoder that captures Arabic's 
    unique word-internal structure at multiple granularities.
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
        self.conv_k2 = nn.Conv1d(char_embed_dim, 64, kernel_size=2, padding=0)
        self.conv_k3 = nn.Conv1d(char_embed_dim, 64, kernel_size=3, padding=1)
        self.conv_k4 = nn.Conv1d(char_embed_dim, 64, kernel_size=4, padding=1)
        
        self.conv_dilated_2 = nn.Conv1d(char_embed_dim, 64, kernel_size=3, padding=2, dilation=2)
        self.conv_dilated_3 = nn.Conv1d(char_embed_dim, 48, kernel_size=3, padding=3, dilation=3)
        
        self.cnn_channels = 64 * 4 + 48 # 304
        
        # ── Layer 2: BPE subword embeddings ──
        self.bpe_embed = nn.Embedding(n_bpe_tokens, 128, padding_idx=0)
        self.bpe_proj = nn.Linear(128, 128)
        
        # ── Layer 3: Morphological feature embeddings ──
        self.root_embed = nn.Embedding(n_roots + 1, 64, padding_idx=0)
        self.pattern_embed = nn.Embedding(n_patterns + 1, 64, padding_idx=0)
        self.proclitic_embed = nn.Embedding(200, 32, padding_idx=0)
        self.enclitic_embed = nn.Embedding(100, 32, padding_idx=0)
        
        # ── Layer 4: Diacritics encoding ──
        self.diac_vocab_size = 20
        self.diac_embed = nn.Embedding(self.diac_vocab_size, 32, padding_idx=0)
        self.diac_lstm = nn.LSTM(32, 48, batch_first=True, bidirectional=True)
        
        # ── Gated Highway Fusion ──
        self.total_morph_dim = self.cnn_channels + 128 + 64 + 64 + 32 + 32 + 96 # 720
        
        self.highway_gate = nn.Sequential(
            nn.Linear(self.total_morph_dim, output_dim),
            nn.Sigmoid()
        )
        self.highway_transform = nn.Sequential(
            nn.Linear(self.total_morph_dim, output_dim),
            nn.GELU()
        )
        self.highway_norm = nn.LayerNorm(output_dim)
        
        self.morph_confidence = nn.Sequential(
            nn.Linear(self.total_morph_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, char_ids, bpe_ids, root_ids, pattern_ids, 
                proclitic_ids, enclitic_ids, diac_ids,
                morph_available_mask=None):
        B, W, C = char_ids.shape
        # ═══ Layer 1: Dilated CharCNN ═══
        chars_flat = char_ids.view(B * W, C)
        char_emb = self.char_embed(chars_flat).transpose(1, 2)  # (B*W, E, C)
        
        # Handle k2 padding manually if size is 1
        if C >= 2:
            c2 = self.conv_k2(char_emb).max(dim=-1).values
        else:
            c2 = torch.zeros(B*W, 64, device=char_ids.device)

        c3 = self.conv_k3(char_emb).max(dim=-1).values
        c4 = self.conv_k4(char_emb).max(dim=-1).values
        cd2 = self.conv_dilated_2(char_emb).max(dim=-1).values
        cd3 = self.conv_dilated_3(char_emb).max(dim=-1).values
        
        cnn_out = F.gelu(torch.cat([c2, c3, c4, cd2, cd3], dim=-1))
        cnn_out = cnn_out.view(B, W, self.cnn_channels)
        
        # ═══ Layer 2: BPE embeddings ═══
        bpe_emb = self.bpe_embed(bpe_ids)  # (B, W, n_pieces, 128)
        bpe_mask = (bpe_ids != 0).float().unsqueeze(-1)
        # Avoid max over empty seq by clamping mask to very small
        bpe_out = (bpe_emb * bpe_mask + (1 - bpe_mask) * -1e9).max(dim=2).values
        bpe_out = self.bpe_proj(bpe_out)
        bpe_out = bpe_out.masked_fill((bpe_ids.sum(dim=-1) == 0).unsqueeze(-1), 0.0)
        
        # ═══ Layer 3: Morphological features ═══
        root_out = self.root_embed(root_ids)
        pattern_out = self.pattern_embed(pattern_ids)
        procl_out = self.proclitic_embed(proclitic_ids)
        encl_out = self.enclitic_embed(enclitic_ids)
        
        # ═══ Layer 4: Diacritics encoding ═══
        diac_flat = diac_ids.view(B * W, C)
        diac_emb = self.diac_embed(diac_flat)
        diac_out, _ = self.diac_lstm(diac_emb) # (B*W, C, 96)
        diac_out = diac_out.max(dim=1).values.view(B, W, 96)
        
        # ═══ Gated Highway Fusion ═══
        morph_concat = torch.cat([
            cnn_out, bpe_out, root_out, pattern_out, 
            procl_out, encl_out, diac_out
        ], dim=-1)
        
        gate = self.highway_gate(morph_concat)
        transform = self.highway_transform(morph_concat)
        morph_repr = gate * transform
        
        if morph_available_mask is not None:
            confidence = self.morph_confidence(morph_concat)
            availability = morph_available_mask.float().unsqueeze(-1)
            morph_repr = morph_repr * (confidence * availability + (1 - availability) * 0.1)
        
        return self.highway_norm(morph_repr)

class ArabicStructuralPositionEncoder(nn.Module):
    """Encodes structural positional features unique to Arabic syntax."""
    def __init__(self, d_model: int = 256, max_depth: int = 8):
        super().__init__()
        # Ensure dimensions match up exactly: d_model / 4
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        d_quarter = d_model // 4
        
        self.depth_embed = nn.Embedding(max_depth, d_quarter)
        self.verb_dist_embed = nn.Embedding(33, d_quarter)  # -16..+16 bucketed
        self.conjunct_embed = nn.Embedding(8, d_quarter)    # 0..7
        self.rel_pos_proj = nn.Linear(1, d_quarter)
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Mock POS tags assumptions. The exact POS tags for Verb, SCONJ, CC
        # should ideally be configurable, or we provide a way to set them.
        self._verb_ids = {10, 11} # Example ids, needs updating via config
        self._sconj_id = 15
        self._cc_id = 9
    
    def forward(self, word_ids, pos_tags, seq_lengths, mask):
        B, W = word_ids.shape
        device = word_ids.device
        
        # ── Heuristic clause depth ──
        is_sconj = (pos_tags == self._sconj_id).long()
        cumulative_depth = torch.cumsum(is_sconj, dim=1).clamp(max=7)
        depth_enc = self.depth_embed(cumulative_depth)
        
        # ── Verb-relative distance ──
        # Handle multiple verb pos tag ids dynamically
        is_verb = torch.zeros_like(pos_tags, dtype=torch.float)
        for v_id in self._verb_ids:
            is_verb += (pos_tags == v_id).float()
        is_verb = is_verb.clamp(max=1.0)
        
        positions = torch.arange(W, device=device).float().unsqueeze(0)
        verb_positions = positions * is_verb + (1 - is_verb) * 9999
        
        verb_dist = torch.zeros(B, W, device=device)
        for b in range(B):
            v_pos = verb_positions[b][verb_positions[b] < 9998]
            if len(v_pos) > 0:
                dists = positions[0, :W].unsqueeze(1) - v_pos.unsqueeze(0)
                abs_dists = dists.abs()
                nearest_idx = abs_dists.argmin(dim=1)
                verb_dist[b] = dists[torch.arange(W, device=device), nearest_idx]
        
        bucketed_vdist = (verb_dist.clamp(-16, 16).long() + 16)
        vdist_enc = self.verb_dist_embed(bucketed_vdist)
        
        # ── Conjunct rank ──
        is_cc = (pos_tags == self._cc_id).long()
        conj_rank = torch.cumsum(is_cc, dim=1).clamp(max=7)
        conj_enc = self.conjunct_embed(conj_rank)
        
        # ── Sentence-relative position ──
        rel_pos = positions[:, :W] / seq_lengths.float().unsqueeze(1).clamp(min=1)
        rel_pos = rel_pos.unsqueeze(-1)
        rel_enc = self.rel_pos_proj(rel_pos)
        
        # ── Fuse ──
        struct_pos = torch.cat([depth_enc, vdist_enc, conj_enc, rel_enc], dim=-1)
        return self.fusion(struct_pos)
