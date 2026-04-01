import torch
import torch.nn as nn
import torch.nn.functional as F

# Modules from v2 package
from .encoders import ArabicMorphologicalEncoder, ArabicStructuralPositionEncoder, StackedTransformerEncoder
from .manager import VariationalTreeManager
from .grid_processor import HRMGridProcessor
from .refinement import TreeGNNRefinement, SecondOrderScorer
from .losses import (
    DifferentiableTreeCRF, 
    ContrastiveTreeLoss, 
    AgreementAuxLoss, 
    StructuralLabelSmoothing, 
    UncertaintyWeightedMultiTaskLoss
)

class ScheduledGumbelTeacherForcing:
    def __init__(self, warmup_epochs: int = 5, anneal_epochs: int = 15):
        self.warmup = warmup_epochs
        self.anneal = anneal_epochs

    def get_temperature(self, epoch: int) -> float:
        if epoch < self.warmup:
            return 10.0
        progress = min(1.0, (epoch - self.warmup) / self.anneal)
        return 10.0 * (0.01 ** progress)

    def get_mix_ratio(self, epoch: int) -> float:
        if epoch < self.warmup:
            return 1.0
        progress = min(1.0, (epoch - self.warmup) / self.anneal)
        return max(0.0, 1.0 - progress)

    @staticmethod
    def gumbel_soft_heads(arc_scores, temperature, gold_heads=None, mix_ratio=0.0):
        B, N, _ = arc_scores.shape
        logits = arc_scores.transpose(1, 2)
        if temperature > 0:
            gumbel_noise = -(-torch.empty_like(logits).uniform_().clamp(1e-8).log()).log()
            noisy_logits = (logits + gumbel_noise) / temperature
        else:
            noisy_logits = logits
        
        soft = torch.softmax(noisy_logits, dim=-1)
        
        if gold_heads is not None and mix_ratio > 0:
            gold_heads_clamped = gold_heads.clamp(min=0, max=N-1)
            gold_onehot = F.one_hot(gold_heads_clamped, N).float()
            soft = mix_ratio * gold_onehot + (1 - mix_ratio) * soft
        return soft

    @staticmethod
    def soft_head_representation(soft_heads, word_embeddings):
        return torch.bmm(soft_heads, word_embeddings)

class BiaffineScorer(nn.Module):
    def __init__(self, in_features, out_features, num_labels=1, bias=True):
        super(BiaffineScorer, self).__init__()
        self.num_labels = num_labels
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.Tensor(num_labels, in_features, out_features))
        if bias:
            self.bias_u = nn.Parameter(torch.Tensor(num_labels, in_features))
            self.bias_v = nn.Parameter(torch.Tensor(num_labels, out_features))
            self.bias = nn.Parameter(torch.Tensor(num_labels))
        else:
            self.register_parameter('bias_u', None)
            self.register_parameter('bias_v', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        if self.bias is not None:
            nn.init.zeros_(self.bias_u)
            nn.init.zeros_(self.bias_v)
            nn.init.zeros_(self.bias)

    def forward(self, u, v):
        # u: (B, N, in_features), v: (B, M, out_features)
        # Returns: (B, N, M, num_labels)
        u_expanded = u.unsqueeze(1)
        v_expanded = v.unsqueeze(1)

        b = torch.einsum('bni,lio,bmo->bnml', u, self.W, v)
        if self.bias is not None:
            bu = torch.einsum('bni,li->bnl', u, self.bias_u).unsqueeze(2)
            bv = torch.einsum('bmo,lo->bml', v, self.bias_v).unsqueeze(1)
            b = b + bu + bv + self.bias
        return b.squeeze(-1) if self.num_labels == 1 else b


class ParserConfig:
    def __init__(self, **kwargs):
        self.word_vocab = kwargs.get('word_vocab', 10000)
        self.char_vocab = kwargs.get('char_vocab', 300)
        self.bpe_vocab = kwargs.get('bpe_vocab', 8000)
        self.n_roots = kwargs.get('n_roots', 5000)
        self.n_patterns = kwargs.get('n_patterns', 200)
        self.n_pos_tags = kwargs.get('n_pos_tags', 64)
        
        self.word_dim = kwargs.get('word_dim', 384)
        self.n_heads = kwargs.get('n_heads', 6)
        self.n_relations = kwargs.get('n_relations', 50)
        self.n_cases = kwargs.get('n_cases', 5)
        
        self.n_transformer_layers = kwargs.get('n_transformer_layers', 3)
        self.n_gnn_rounds = kwargs.get('n_gnn_rounds', 3)
        self.latent_dim = kwargs.get('latent_dim', 64)
        self.n_archetypes = kwargs.get('n_archetypes', 12)
        self.dropout = kwargs.get('dropout', 0.15)


class ArabicHRMGridParserV2(nn.Module):
    def __init__(self, config: ParserConfig):
        super().__init__()
        
        D = config.word_dim
        n_heads = config.n_heads
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
        
        # Learned gate: decides per-word how much to trust word_embed vs morph_embed
        # This lets CharCNN (inside morph_encoder) dominate for rare/OOV words
        self.lexical_gate = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.Sigmoid()
        )
        
        self.input_projection = nn.Sequential(
            nn.Linear(D + D + 64, D),  # lexical_blend + morph + pos
            nn.GELU(),
            nn.LayerNorm(D),
            nn.Dropout(config.dropout)
        )
        
        # Case class weights (inverse frequency from PADT train data)
        # None/Mabni=48.1%, Nom=8.3%, Acc=10.2%, Gen=33.4%
        case_weights = torch.tensor([1.0, 5.8, 4.7, 1.4] + [1.0] * max(0, n_cases - 4))
        self.register_buffer('case_class_weights', case_weights[:n_cases])
        
        # ═══ Stage 2: Manager ═══
        self.manager = VariationalTreeManager(
            input_dim=D, latent_dim=config.latent_dim, n_archetypes=config.n_archetypes, goal_dim=D
        )
        
        self.goal_gate_weight = nn.Parameter(torch.empty(D, D))
        self.goal_gate_bias = nn.Parameter(torch.empty(D))
        nn.init.xavier_uniform_(self.goal_gate_weight)
        nn.init.zeros_(self.goal_gate_bias)
        
        # ═══ Stage 3: Self-Attention ═══
        self.transformer = StackedTransformerEncoder(
            d_model=D, n_heads=n_heads, d_ff=D*4, n_layers=config.n_transformer_layers, dropout=config.dropout
        )
        
        # ═══ Stage 4: HRM Grid Processing ═══
        self.grid_processor = HRMGridProcessor(hidden_dim=D, worker_dim=D, dropout=config.dropout)
        
        # ═══ Stage 5: Biaffine Parsing ═══
        self.arc_head_mlp = nn.Sequential(nn.Linear(D, D), nn.LeakyReLU(0.1), nn.Dropout(config.dropout))
        self.arc_dep_mlp = nn.Sequential(nn.Linear(D, D), nn.LeakyReLU(0.1), nn.Dropout(config.dropout))
        self.biaffine_arc = BiaffineScorer(D, D, 1)
        
        # ═══ Stage 6: Losses ═══
        self.tree_crf = DifferentiableTreeCRF()
        
        # ═══ Stage 7: GNN Refinement ═══
        self.gnn_refine = TreeGNNRefinement(d_model=D, n_rounds=config.n_gnn_rounds, n_heads=n_heads, dropout=config.dropout)
        
        # ═══ Stage 8: Second-Order ═══
        self.second_order = SecondOrderScorer(input_dim=D, scorer_dim=128)
        
        # ═══ Stage 9: Relation Classifier ═══
        self.rel_head_mlp = nn.Sequential(nn.Linear(D, D), nn.LeakyReLU(0.1), nn.Dropout(config.dropout))
        self.rel_dep_mlp = nn.Sequential(nn.Linear(D, D), nn.LeakyReLU(0.1), nn.Dropout(config.dropout))
        self.biaffine_rel = BiaffineScorer(D, D, num_labels=n_rels)
        
        # ═══ Stage 10: Case Classifier ═══
        self.case_classifier = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(D // 2, n_cases)
        )
        
        # ═══ More Losses ═══
        self.uncertainty_loss = UncertaintyWeightedMultiTaskLoss(n_tasks=5)
        self.contrastive_loss = ContrastiveTreeLoss(n_negatives=4, margin=2.0)
        self.struct_label_smooth = StructuralLabelSmoothing(n_classes=n_rels, smoothing=0.1)
        
        # ═══ Teacher Forcing Schedule ═══
        self.gumbel_scheduler = ScheduledGumbelTeacherForcing()
    
    def forward(self, batch, epoch=0, training=True):
        word_ids = batch['word_ids']
        char_ids = batch['char_ids']
        pos_tags = batch['pos_tags']
        bpe_ids = batch['bpe_ids']
        root_ids = batch['root_ids']
        pattern_ids = batch['pattern_ids']
        proclitic_ids = batch['proclitic_ids']
        enclitic_ids = batch['enclitic_ids']
        diac_ids = batch['diac_ids']
        mask = batch['mask']
        gold_heads = batch.get('heads')
        gold_rels = batch.get('relations')
        gold_cases = batch.get('cases')
        
        B, W = word_ids.shape
        device = word_ids.device
        
        # == STAGE 1: Encoding ==
        word_emb = self.word_embed(word_ids)
        morph_emb = self.morph_encoder(
            char_ids, bpe_ids, root_ids, pattern_ids,
            proclitic_ids, enclitic_ids, diac_ids
        )
        pos_emb = self.pos_embed(pos_tags)
        struct_pos = self.struct_pos_encoder(word_ids, pos_tags, mask.sum(dim=-1), mask)
        
        # Learned lexical gate: morph_emb (CharCNN) dominates for rare words,
        # word_emb contributes for frequent well-represented words
        gate = self.lexical_gate(torch.cat([word_emb, morph_emb], dim=-1))
        lexical_blend = gate * word_emb + (1 - gate) * morph_emb
        
        combined = torch.cat([lexical_blend + struct_pos, morph_emb, pos_emb], dim=-1)
        H = self.input_projection(combined)
        
        # == STAGE 2: Manager ==
        worker_goal, dist_bias, rel_prior, kl_loss, arch_logits = self.manager(H, mask, training=training)
        
        goal_gate = torch.sigmoid(F.linear(H, self.goal_gate_weight, self.goal_gate_bias))
        H = H + goal_gate * worker_goal.unsqueeze(1)
        
        # == STAGE 3: Self-Attention ==
        H = self.transformer(H, mask=mask)
        
        # == STAGE 4: Grid Processing ==
        H_grid = self.grid_processor(H, worker_goal, mask)
        H = H + H_grid
        
        # == STAGE 5: Biaffine Arc Scoring (SuPar-style) ==
        h_head = self.arc_head_mlp(H)
        h_dep = self.arc_dep_mlp(H)
        arc_scores = self.biaffine_arc(h_dep, h_head)  # (B, dep, head)
        
        # Distance bias
        positions = torch.arange(W, device=device)
        rel_pos = (positions.unsqueeze(1) - positions.unsqueeze(0)).clamp(-16, 16) + 16
        arc_scores = arc_scores + dist_bias[:, rel_pos]
        
        # SuPar masking: only mask HEAD dimension (invalid head candidates)
        # Do NOT mask dep dimension or diagonal — let the model learn
        cand_mask = mask.bool()
        arc_scores = arc_scores.masked_fill_(~cand_mask.unsqueeze(1), -1e4)
        
        # == STAGE 6: Relation Scoring ==
        r_head = self.rel_head_mlp(H)
        r_dep = self.rel_dep_mlp(H)
        rel_scores = self.biaffine_rel(r_dep, r_head)  # (B, dep, head, n_rels)
        
        # == STAGE 7: Case Classification ==
        case_logits = self.case_classifier(H)
        
        if training and gold_heads is not None and gold_rels is not None and gold_cases is not None:
            # --- Arc Loss (SuPar-style: flat masked cross-entropy) ---
            dep_mask = batch.get('dep_mask', mask).bool()  # excludes root words + padding
            
            # Flatten and apply mask
            arc_masked = arc_scores[dep_mask]  # (N_valid, W)
            heads_masked = gold_heads.clamp(0, W-1)[dep_mask]  # (N_valid,)
            arc_loss = F.cross_entropy(arc_masked, heads_masked)
            
            # --- Rel Loss (SuPar-style: extract rel at gold head position) ---
            # rel_scores: (B, dep, head, n_rels) → index with gold heads
            gold_h = gold_heads.clamp(0, W-1)  # (B, W)
            # Gather: for each (b, d), get rel_scores[b, d, gold_h[b,d], :]
            gold_h_expanded = gold_h.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, rel_scores.size(-1))
            rel_at_gold = rel_scores.gather(2, gold_h_expanded).squeeze(2)  # (B, W, n_rels)
            
            rel_masked = rel_at_gold[dep_mask]  # (N_valid, n_rels)
            rels_masked = gold_rels.clamp(0, rel_scores.size(-1)-1)[dep_mask]  # (N_valid,)
            rel_loss = F.cross_entropy(rel_masked, rels_masked)
            
            # --- Case Loss ---
            case_m = mask.bool()
            case_loss = F.cross_entropy(
                case_logits[case_m],
                gold_cases[case_m],
                weight=self.case_class_weights,
                label_smoothing=0.05
            )
            
            # SuPar-style total: arc + rel (primary) + case (auxiliary)
            total_loss = arc_loss + rel_loss + 0.3 * case_loss
            
            return {
                'loss': total_loss,
                'arc_loss': arc_loss.item(),
                'rel_loss': rel_loss.item(),
                'case_loss': case_loss.item(),
                'structural_loss': 0.0,
                'kl_loss': 0.0,
                'arc_scores': arc_scores,
                'rel_logits': rel_at_gold,
                'case_logits': case_logits,
            }
        else:
            # Inference: greedy decode (SuPar-style)
            pred_heads = arc_scores.argmax(dim=2)  # (B, W) — best head for each dep
            
            # Extract rel scores at predicted heads
            pred_h = pred_heads.clamp(0, W-1)
            pred_h_expanded = pred_h.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, rel_scores.size(-1))
            rel_at_pred = rel_scores.gather(2, pred_h_expanded).squeeze(2)  # (B, W, n_rels)
            pred_rels = rel_at_pred.argmax(dim=-1)
            
            pred_cases = case_logits.argmax(dim=-1)
            
            return {
                'pred_heads': pred_heads,
                'pred_rels': pred_rels,
                'pred_cases': pred_cases,
                'arc_scores': arc_scores,
            }
