"""
PyTorch Two-Tower candidate retrieval model with sampled softmax loss
and logQ popularity debiasing, exportable to ONNX for low-latency serving.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """
    User Tower embedding user ID and sequential interaction context into normalized vector.
    """
    def __init__(self, num_users: int, embed_dim: int = 64, hidden_dims: Tuple[int, ...] = (128, 64)):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
        
        layers = []
        in_dim = embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, embed_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        safe_ids = torch.clamp(user_ids, 0, self.user_embedding.num_embeddings - 1)
        x = self.user_embedding(safe_ids)
        out = self.mlp(x)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    """
    Item Tower embedding item ID and static metadata into normalized vector.
    """
    def __init__(self, num_items: int, embed_dim: int = 64, hidden_dims: Tuple[int, ...] = (128, 64)):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        
        layers = []
        in_dim = embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, embed_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        safe_ids = torch.clamp(item_ids, 0, self.item_embedding.num_embeddings - 1)
        x = self.item_embedding(safe_ids)
        out = self.mlp(x)
        return F.normalize(out, p=2, dim=-1)


class SampledSoftmaxWithLogQ(nn.Module):
    """
    In-batch softmax loss with logQ popularity correction:
        logits_{ij} = (u_i * v_j / tau) - ln Q(j)
    Prevents popular items from dominating negative gradients and debiases retrieval.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        user_embeds: torch.Tensor,
        item_embeds: torch.Tensor,
        item_log_q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            user_embeds: [B, D] normalized user representations
            item_embeds: [B, D] normalized item representations
            item_log_q: [B] log probability ln Q(j) for in-batch items
        """
        # Dot product similarity: [B, B]
        scores = torch.matmul(user_embeds, item_embeds.t()) / self.temperature

        # Apply logQ debiasing across columns: logits_{ij} - ln Q(j)
        # item_log_q is broadcasted along rows
        debiased_scores = scores - item_log_q.unsqueeze(0)

        # Ground truth is diagonal (i == j)
        targets = torch.arange(user_embeds.size(0), device=user_embeds.device)
        loss = F.cross_entropy(debiased_scores, targets)
        return loss


def export_user_tower_onnx(
    model: UserTower,
    output_path: Path,
    device: torch.device = torch.device("cpu"),
) -> Path:
    """
    Exports UserTower to ONNX for ultra-low latency inference in FastAPI serving.
    """
    model.eval()
    model.to(device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.tensor([1, 2], dtype=torch.long, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["user_ids"],
        output_names=["user_embeddings"],
        dynamic_axes={"user_ids": {0: "batch_size"}, "user_embeddings": {0: "batch_size"}},
        opset_version=18,
    )
    return output_path
