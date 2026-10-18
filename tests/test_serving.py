import numpy as np
import pytest
from fastapi.testclient import TestClient
from rankwise.serving.app import app, hnsw_indexer


def test_fastapi_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy", "service": "rankwise"}


def test_fastapi_recommend():
    # Build indexer with sample data
    dim = 64
    n_items = 100
    item_embs = np.random.randn(n_items, dim).astype(np.float32)
    item_embs /= np.linalg.norm(item_embs, axis=1, keepdims=True)
    item_ids = np.arange(1, n_items + 1)
    hnsw_indexer.build(item_embs, item_ids)
    
    client = TestClient(app)
    resp = client.post("/recommend", json={"user_id": 42, "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["user_id"] == 42
    assert len(data["recommendations"]) == 5
    assert "retrieval_latency_ms" in data
    assert "ranking_latency_ms" in data
    assert "total_latency_ms" in data
    
    first_item = data["recommendations"][0]
    assert "item_id" in first_item
    assert "score" in first_item
    assert "calibrated_prob" in first_item
    assert first_item["final_rank"] == 1
