import tempfile
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
import pytest

from rankwise.retrieval.two_tower import UserTower, ItemTower, export_user_tower_onnx
from rankwise.retrieval.indexer import HNSWIndexer


def test_two_tower_forward_and_normalization():
    num_users = 100
    num_items = 200
    embed_dim = 32
    
    user_tower = UserTower(num_users=num_users, embed_dim=embed_dim)
    item_tower = ItemTower(num_items=num_items, embed_dim=embed_dim)
    
    u_ids = torch.tensor([1, 5, 20], dtype=torch.long)
    i_ids = torch.tensor([10, 50, 150], dtype=torch.long)
    
    u_emb = user_tower(u_ids)
    i_emb = item_tower(i_ids)
    
    assert u_emb.shape == (3, embed_dim)
    assert i_emb.shape == (3, embed_dim)
    
    # Verify L2 normalization
    u_norms = torch.norm(u_emb, dim=-1).detach().numpy()
    i_norms = torch.norm(i_emb, dim=-1).detach().numpy()
    np.testing.assert_allclose(u_norms, 1.0, atol=1e-5)
    np.testing.assert_allclose(i_norms, 1.0, atol=1e-5)


def test_onnx_user_tower_parity():
    user_tower = UserTower(num_users=50, embed_dim=16)
    user_tower.eval()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = Path(tmp_dir) / "user_tower.onnx"
        export_user_tower_onnx(user_tower, onnx_file)
        
        assert onnx_file.exists()
        
        # Test onnx runtime inference
        session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        test_ids = np.array([3, 14, 25], dtype=np.int64)
        ort_out = session.run(None, {"user_ids": test_ids})[0]
        
        with torch.no_grad():
            torch_out = user_tower(torch.from_numpy(test_ids)).numpy()
            
        np.testing.assert_allclose(ort_out, torch_out, atol=1e-4)


def test_hnsw_indexer_accuracy():
    dim = 32
    n_items = 200
    np.random.seed(42)
    
    item_embeddings = np.random.randn(n_items, dim).astype(np.float32)
    item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    item_ids = np.arange(1, n_items + 1)
    
    indexer = HNSWIndexer(dim=dim, M=16, ef_construction=64, ef_search=32)
    indexer.build(item_embeddings, item_ids)
    
    query = item_embeddings[10:11]
    scores, retrieved_ids = indexer.search(query, top_k=5)
    
    # Exact match for identical query vector
    assert retrieved_ids[0, 0] == item_ids[10]
    assert np.isclose(scores[0, 0], 1.0, atol=1e-3)
