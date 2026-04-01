#!/usr/bin/env python3
"""
Arabic Diacritization Model (تشكيل تلقائي)
============================================

Character-level diacritizer using:
1. Character BiLSTM per word (local context)
2. Word-level Transformer (sentence context)
3. Combined classifier → per-character diacritic

Can be trained standalone on PADT + Quran data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ArabicDiacritizer(nn.Module):
    """
    Arabic Auto-Diacritization Model.
    
    Input: sequence of Arabic words (as character IDs)
    Output: per-character diacritic predictions (15 classes)
    
    Architecture:
        Char Embed → Char BiLSTM (per-word) → Word Repr
        Word Repr → Transformer (sentence) → Contextual Word Repr
        Contextual + Char features → Per-char classifier
    """
    
    def __init__(
        self,
        char_vocab: int = 256,
        char_embed_dim: int = 64,
        char_hidden: int = 128,
        word_dim: int = 256,
        n_heads: int = 4,
        n_transformer_layers: int = 3,
        n_diac_classes: int = 15,
        max_chars: int = 16,
        max_words: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.max_chars = max_chars
        self.max_words = max_words
        self.n_diac_classes = n_diac_classes
        
        # ── Stage 1: Character Encoding ──
        self.char_embed = nn.Embedding(char_vocab, char_embed_dim, padding_idx=0)
        self.char_bilstm = nn.LSTM(
            char_embed_dim, char_hidden,
            batch_first=True, bidirectional=True,
            num_layers=2, dropout=dropout
        )
        self.char_to_word = nn.Linear(char_hidden * 2, word_dim)
        
        # ── Stage 2: Word-Level Transformer ──
        self.word_pos = PositionalEncoding(word_dim, max_len=max_words)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=word_dim, nhead=n_heads,
            dim_feedforward=word_dim * 4,
            dropout=dropout, batch_first=True,
            activation='gelu'
        )
        self.word_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )
        self.word_dropout = nn.Dropout(dropout)
        
        # ── Stage 3: Per-Character Classification ──
        # Combine: word context (word_dim) + char features (char_hidden*2) → classifier
        self.context_proj = nn.Linear(word_dim, char_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(char_hidden * 4, char_hidden * 2),  # char + context
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(char_hidden * 2, n_diac_classes),
        )
    
    def forward(self, char_ids, word_mask=None, diac_labels=None, diac_mask=None):
        """
        Args:
            char_ids: (B, W, C) - character IDs per word
            word_mask: (B, W) - 1 for valid words, 0 for padding
            diac_labels: (B, W, C) - gold diacritic labels (for training)
            diac_mask: (B, W, C) - 1 for valid characters (for loss)
        
        Returns:
            dict with 'diac_logits' (B, W, C, n_classes) and optionally 'loss'
        """
        B, W, C = char_ids.shape
        
        if word_mask is None:
            word_mask = (char_ids.sum(dim=-1) > 0).long()  # (B, W)
        
        # ── Stage 1: Char BiLSTM per word ──
        # Reshape to (B*W, C) for LSTM
        chars_flat = char_ids.view(B * W, C)
        char_emb = self.char_embed(chars_flat)  # (B*W, C, embed_dim)
        
        char_out, _ = self.char_bilstm(char_emb)  # (B*W, C, hidden*2)
        char_features = char_out.view(B, W, C, -1)  # (B, W, C, hidden*2)
        
        # Word representation = average of char outputs (masked)
        char_mask_flat = (chars_flat > 0).unsqueeze(-1).float()  # (B*W, C, 1)
        word_repr = (char_out * char_mask_flat).sum(dim=1) / char_mask_flat.sum(dim=1).clamp(min=1)
        word_repr = word_repr.view(B, W, -1)  # (B, W, hidden*2)
        word_repr = self.char_to_word(word_repr)  # (B, W, word_dim)
        
        # ── Stage 2: Word Transformer ──
        word_repr = self.word_pos(word_repr)
        word_repr = self.word_dropout(word_repr)
        
        # Create attention mask
        attn_mask = (word_mask == 0)  # True = masked position
        word_context = self.word_transformer(
            word_repr,
            src_key_padding_mask=attn_mask
        )  # (B, W, word_dim)
        
        # ── Stage 3: Per-char classification ──
        # Broadcast word context to character level
        context_proj = self.context_proj(word_context)  # (B, W, hidden*2)
        context_expanded = context_proj.unsqueeze(2).expand(-1, -1, C, -1)  # (B, W, C, hidden*2)
        
        # Concatenate char features + word context
        combined = torch.cat([char_features, context_expanded], dim=-1)  # (B, W, C, hidden*4)
        diac_logits = self.classifier(combined)  # (B, W, C, n_classes)
        
        output = {
            'diac_logits': diac_logits,
            'pred_diacs': diac_logits.argmax(dim=-1),  # (B, W, C)
        }
        
        # ── Loss ──
        if diac_labels is not None:
            if diac_mask is None:
                diac_mask = (char_ids > 0).long()
            
            # Flatten for cross-entropy
            logits_flat = diac_logits.view(-1, self.n_diac_classes)
            labels_flat = diac_labels.view(-1)
            mask_flat = diac_mask.view(-1).bool()
            
            # Masked cross-entropy
            loss = F.cross_entropy(
                logits_flat[mask_flat],
                labels_flat[mask_flat],
                reduction='mean'
            )
            output['loss'] = loss
        
        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = ArabicDiacritizer()
    print(f"Parameters: {model.count_parameters()/1e6:.2f}M")
    
    # Dummy input
    B, W, C = 2, 32, 16
    char_ids = torch.randint(1, 100, (B, W, C))
    word_mask = torch.ones(B, W, dtype=torch.long)
    diac_labels = torch.randint(0, 15, (B, W, C))
    diac_mask = (char_ids > 0).long()
    
    out = model(char_ids, word_mask, diac_labels, diac_mask)
    print(f"Logits: {out['diac_logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"Preds: {out['pred_diacs'].shape}")
    print("✅ Quick test passed!")
