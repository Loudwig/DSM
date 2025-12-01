import torch
import torch.nn as nn

P = 10
Z = 10
class ScoreNetworkConditionned(nn.Module): 
    
    def __init__(self, x_dim,hidden_dim = 128*2,sigma_emb_dim=20):
        super().__init__()
        
        self.sigma_embedding = nn.Sequential(
            nn.Linear(1,sigma_emb_dim),
            nn.ReLU(),
            nn.Linear(sigma_emb_dim,sigma_emb_dim),
            nn.ReLU(),
        )
        
        self.score_mlp = nn.Sequential(
            nn.Linear(x_dim + sigma_emb_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,x_dim)
        )
    def forward(self,x_noisy,sigma):
        """
        x_noisy: (B, x_dim)
        sigma: (B, 1)
        """
        
        assert sigma.dim() == 2 and sigma.shape[0] == x_noisy.shape[0] and sigma.shape[1] == 1, f"sigma doit être (B,1), reçu {sigma.shape}"
        log_sigma = torch.log(sigma)
        s_e = self.sigma_embedding(log_sigma)
        x_stack = torch.cat([x_noisy, s_e], dim=-1) 
        score = self.score_mlp(x_stack)
        return score


import torch
import torch.nn as nn
import torch.nn.functional as F

P = 10
Z = 10

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, dim)
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        h = self.norm(h)
        return x + h


class ScoreNetworkConditionned1(nn.Module): 
    
    def __init__(self, x_dim, hidden_dim=128*2, sigma_emb_dim=20):
        super().__init__()
        
        # Embedding de log(sigma) comme avant
        self.sigma_embedding = nn.Sequential(
            nn.Linear(1, sigma_emb_dim),
            nn.SiLU(),
            nn.Linear(sigma_emb_dim, sigma_emb_dim),
            nn.SiLU(),
        )
        
        in_dim = x_dim + sigma_emb_dim
        
        # Petit "stem" pour projeter dans l'espace latent
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        
        # 2 blocs résiduels simples
        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)
        
        # Norm + projection finale vers x_dim
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_noisy, sigma):
        """
        x_noisy: (B, x_dim)
        sigma:   (B, 1)
        """
        assert sigma.dim() == 2 \
            and sigma.shape[0] == x_noisy.shape[0] \
            and sigma.shape[1] == 1, \
            f"sigma doit être (B,1), reçu {sigma.shape}"
        
        log_sigma = torch.log(sigma)
        s_e = self.sigma_embedding(log_sigma)          # (B, sigma_emb_dim)
        
        x_stack = torch.cat([x_noisy, s_e], dim=-1)    # (B, x_dim + sigma_emb_dim)
        
        h = self.in_proj(x_stack)                      # (B, hidden_dim)
        h = F.silu(h)
        
        h = self.block1(h)
        h = self.block2(h)
        
        h = self.out_norm(h)
        score = self.out_proj(h)                       # (B, x_dim)
        return score
 