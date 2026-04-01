import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalTreeManager(nn.Module):
    """
    The Manager uses a Variational Autoencoder (VAE) formulation to 
    infer a DISTRIBUTION over tree topology types and send a structured goal.
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
        self.pool_attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Posterior q(z|x): diagonal Gaussian
        self.posterior_mu = nn.Linear(input_dim, latent_dim)
        self.posterior_logvar = nn.Linear(input_dim, latent_dim)
        
        # ── Learned Mixture-of-Gaussians Prior ──
        self.prior_logits = nn.Parameter(torch.zeros(n_archetypes))
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
        self.distance_bias_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 33)  # 33 distance buckets (-16..+16)
        )
        
        self.relation_prior_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 50)  # n_relations
        )
        
        # ── Archetype Classifier (auxiliary loss) ──
        self.archetype_head = nn.Linear(latent_dim, n_archetypes)
    
    def _attention_pool(self, H, mask):
        scores = self.pool_attention(H).squeeze(-1)  # (B, N)
        scores = scores.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(scores, dim=-1)      # (B, N)
        pooled = torch.bmm(weights.unsqueeze(1), H).squeeze(1)  # (B, D)
        return pooled
    
    def encode(self, word_embeddings, mask):
        ctx_out, _ = self.context_encoder(word_embeddings)
        pooled = self._attention_pool(ctx_out, mask)
        
        mu = self.posterior_mu(pooled)
        logvar = self.posterior_logvar(pooled)
        logvar = logvar.clamp(-6, 2)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar, training=True):
        if training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def kl_divergence_mixture(self, q_mu, q_logvar):
        B = q_mu.shape[0]
        z = self.reparameterize(q_mu, q_logvar)  # (B, D)
        
        # log q(z|x)
        log_q = -0.5 * (
            self.latent_dim * math.log(2 * math.pi) + 
            q_logvar.sum(dim=-1) + 
            ((z - q_mu).pow(2) / q_logvar.exp()).sum(dim=-1)
        )
        
        # log p(z)
        prior_weights = F.log_softmax(self.prior_logits, dim=0)  # (K,)
        z_exp = z.unsqueeze(1)                    # (B, 1, D)
        p_mu = self.prior_mu.unsqueeze(0)         # (1, K, D)
        p_logvar = self.prior_logvar.unsqueeze(0)  # (1, K, D)
        
        log_p_k = -0.5 * (
            self.latent_dim * math.log(2 * math.pi) +
            p_logvar.sum(dim=-1) +
            ((z_exp - p_mu).pow(2) / p_logvar.exp()).sum(dim=-1)
        )
        
        log_p = torch.logsumexp(prior_weights.unsqueeze(0) + log_p_k, dim=-1)  # (B,)
        kl = (log_q - log_p).mean()
        return kl.clamp(min=0)
    
    def forward(self, word_embeddings, mask, training=True):
        mu, logvar = self.encode(word_embeddings, mask)
        z = self.reparameterize(mu, logvar, training)
        
        worker_goal = self.goal_decoder(z)
        distance_bias = self.distance_bias_head(z)
        relation_prior = F.log_softmax(self.relation_prior_head(z), dim=-1)
        
        kl_loss = self.kl_divergence_mixture(mu, logvar)
        archetype_logits = self.archetype_head(z.detach())
        
        return worker_goal, distance_bias, relation_prior, kl_loss, archetype_logits
