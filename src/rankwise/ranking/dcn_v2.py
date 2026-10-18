"""
Multi-Task Deep & Cross Network v2 (DCN-v2) for Click and Long-View Ranking.
Features explicit cross layers with low-rank / matrix formulation, deep ReLU MLP,
and dual prediction heads.
"""

from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNetworkV2(nn.Module):
    """
    Cross Network v2 layer:
        x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
    where \odot is element-wise product.
    Captures explicit bounded-degree feature interactions without manual engineering.
    """
    def __init__(self, in_features: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(in_features, in_features))
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(in_features))
            for _ in range(num_layers)
        ])
        
        # Xavier initialization
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for l in range(self.num_layers):
            # linear projection: xl @ W_l + b_l
            proj = F.linear(xl, self.weights[l], self.biases[l])
            # Hadamard product with original input x0 and residual addition
            xl = x0 * proj + xl
        return xl


class DCNv2MultiTask(nn.Module):
    """
    Multi-Task DCN-v2 architecture:
    - User and item embedding layers
    - Dense continuous feature normalization
    - Cross Network v2 for feature crossing
    - Deep MLP for non-linear interactions
    - Dual heads: Click Probability (pCTR) and Long-View Probability (pLVR)
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 32,
        num_dense: int = 4,
        cross_layers: int = 3,
        deep_hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.num_dense = num_dense
        
        total_in_dim = embed_dim * 2 + num_dense
        
        # Explicit feature cross network
        self.cross_net = CrossNetworkV2(in_features=total_in_dim, num_layers=cross_layers)
        
        # Deep MLP
        deep_layers = []
        curr_dim = total_in_dim
        for h_dim in deep_hidden_dims:
            deep_layers.append(nn.Linear(curr_dim, h_dim))
            deep_layers.append(nn.BatchNorm1d(h_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        self.deep_net = nn.Sequential(*deep_layers)
        
        combined_dim = total_in_dim + curr_dim
        
        # Multi-task prediction heads
        self.click_head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.long_view_head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        dense_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        safe_u = torch.clamp(user_ids, 0, self.user_embedding.num_embeddings - 1)
        safe_i = torch.clamp(item_ids, 0, self.item_embedding.num_embeddings - 1)
        u_emb = self.user_embedding(safe_u)
        i_emb = self.item_embedding(safe_i)
        
        feats = [u_emb, i_emb]
        if dense_feats is not None and self.num_dense > 0:
            feats.append(dense_feats)
        elif self.num_dense > 0:
            zeros = torch.zeros(user_ids.size(0), self.num_dense, device=user_ids.device)
            feats.append(zeros)
            
        x0 = torch.cat(feats, dim=-1)
        
        # Compute parallel cross & deep representations
        x_cross = self.cross_net(x0)
        x_deep = self.deep_net(x0)
        
        x_combined = torch.cat([x_cross, x_deep], dim=-1)
        
        p_click = self.click_head(x_combined).squeeze(-1)
        p_long_view = self.long_view_head(x_combined).squeeze(-1)
        
        # Blended ranking score: 0.7 * click + 0.3 * long_view
        score = 0.7 * p_click + 0.3 * p_long_view
        
        return {
            "p_click": p_click,
            "p_long_view": p_long_view,
            "score": score,
        }
