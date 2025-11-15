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
   