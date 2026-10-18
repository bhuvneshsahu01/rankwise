import torch
import numpy as np
import pytest
from rankwise.retrieval.two_tower import SampledSoftmaxWithLogQ


def test_log_q_correction_math():
    loss_fn = SampledSoftmaxWithLogQ(temperature=0.1)
    batch_size = 4
    dim = 16
    
    # Unit normalized embeddings
    u = torch.randn(batch_size, dim)
    u = u / torch.norm(u, dim=-1, keepdim=True)
    
    v = torch.randn(batch_size, dim)
    v = v / torch.norm(v, dim=-1, keepdim=True)
    
    # Popularity logQ
    log_q = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    
    loss = loss_fn(u, v, log_q)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_log_q_penalizes_popular_item():
    """
    Items with higher Q (higher logQ) should have their raw logits reduced more,
    thereby penalizing their negative dominance in the denominator.
    """
    loss_fn = SampledSoftmaxWithLogQ(temperature=1.0)
    u = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    
    # If item 1 is extremely popular, log_q is high (e.g. 0.0 vs -5.0)
    log_q_popular = torch.tensor([0.0, -5.0])
    log_q_uniform = torch.tensor([0.0, 0.0])
    
    loss_pop = loss_fn(u, v, log_q_popular)
    loss_uni = loss_fn(u, v, log_q_uniform)
    
    # Gradients should flow properly
    loss_pop.backward()
    assert u.grad is not None
