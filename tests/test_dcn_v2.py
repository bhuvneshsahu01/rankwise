import torch
import pytest
from rankwise.ranking.dcn_v2 import CrossNetworkV2, DCNv2MultiTask


def test_cross_network_v2_properties():
    batch_size = 8
    in_features = 24
    num_layers = 3
    
    cross_net = CrossNetworkV2(in_features=in_features, num_layers=num_layers)
    x0 = torch.randn(batch_size, in_features)
    out = cross_net(x0)
    
    assert out.shape == (batch_size, in_features)
    # Residual property: out is not identical to x0
    assert not torch.allclose(out, x0)


def test_dcn_v2_multitask_forward_and_backward():
    num_users = 50
    num_items = 100
    model = DCNv2MultiTask(
        num_users=num_users,
        num_items=num_items,
        embed_dim=16,
        num_dense=2,
        cross_layers=2,
        deep_hidden_dims=(32, 16),
    )
    
    batch_size = 4
    u_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    i_ids = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    dense_feats = torch.randn(batch_size, 2)
    
    outputs = model(u_ids, i_ids, dense_feats)
    
    assert "p_click" in outputs
    assert "p_long_view" in outputs
    assert "score" in outputs
    
    p_click = outputs["p_click"]
    p_long = outputs["p_long_view"]
    score = outputs["score"]
    
    assert p_click.shape == (batch_size,)
    assert (p_click >= 0.0).all() and (p_click <= 1.0).all()
    assert (p_long >= 0.0).all() and (p_long <= 1.0).all()
    assert (score >= 0.0).all() and (score <= 1.0).all()
    
    # Multi-task loss
    target_click = torch.tensor([1.0, 0.0, 1.0, 0.0])
    target_long = torch.tensor([0.0, 0.0, 1.0, 0.0])
    loss = torch.nn.functional.binary_cross_entropy(p_click, target_click) +            torch.nn.functional.binary_cross_entropy(p_long, target_long)
    loss.backward()
    
    assert model.user_embedding.weight.grad is not None
