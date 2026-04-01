import torch
import torch.nn as nn
import torch.nn.functional as F

class HRMGridProcessor(nn.Module):
    """
    Worker RNN that runs a FAST clock over the sentence conditioned on the 
    Manager's global structured goal and local grid inputs. 
    This iterative step helps the model resolve ambiguities over multiple 
    passes (e.g., resolving nested dependencies).
    """
    def __init__(self, hidden_dim: int = 256, worker_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        
        # Worker RNN taking [current_word_state | manager_goal]
        self.worker_gru = nn.GRUCell(hidden_dim + worker_dim, worker_dim)
        
        self.context_fuse = nn.Sequential(
            nn.Linear(hidden_dim + worker_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru_drop = nn.Dropout(dropout)
        self.worker_dim = worker_dim
        
    def forward(self, H, manager_goal, mask, num_worker_steps: int = 3):
        """
        H: (B, N, D) - current word representations
        manager_goal: (B, D) - structured goal from Manager (VAE)
        mask: (B, N)
        """
        B, N, D = H.shape
        device = H.device
        
        # Worker states are updated iteratively word-by-word or in parallel?
        # Parallel GRU over steps (each word gets updated)
        worker_h = torch.zeros(B * N, self.worker_dim, device=device)
        
        H_flat = H.reshape(B * N, D)
        # Expand manager goal for each word
        goal_exp = manager_goal.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        
        w_in = torch.cat([H_flat, goal_exp], dim=-1)
        
        for w_step in range(num_worker_steps):
            worker_h = self.worker_gru(w_in, worker_h)
            worker_h = self.gru_drop(worker_h)
            
        worker_h = worker_h.view(B, N, self.worker_dim)
        
        fused = torch.cat([H, worker_h], dim=-1)
        H_enriched = self.context_fuse(fused)
        
        return H_enriched
