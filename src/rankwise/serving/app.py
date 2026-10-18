"""
FastAPI online serving endpoint:
- In-memory / Redis feature store lookup
- ONNX Runtime user-tower embedding extraction
- FAISS HNSW candidate retrieval (top-K)
- PyTorch DCN-v2 re-ranking
- Isotonic probability calibration
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rankwise.serving.feature_store import FeatureStore
from rankwise.retrieval.indexer import HNSWIndexer


app = FastAPI(
    title="Rankwise Recommender API",
    description="Two-Stage Recommender Service with FAISS HNSW Retrieval, DCN-v2 Re-Ranking, and Isotonic Calibration",
    version="0.1.0",
)

feature_store = FeatureStore()
hnsw_indexer = HNSWIndexer(dim=64)
onnx_session = None
dcn_model = None
calibrator = None


class RecommendRequest(BaseModel):
    user_id: int = Field(..., json_schema_extra={"example": 42}, description="Target User ID")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations to return")


class RecommendItem(BaseModel):
    item_id: int
    score: float
    calibrated_prob: float
    retrieval_rank: int
    final_rank: int


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendItem]
    retrieval_latency_ms: float
    ranking_latency_ms: float
    total_latency_ms: float


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "rankwise"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    t_start = time.perf_counter()
    
    # 1. User Feature Extraction & Tower Inference
    t0 = time.perf_counter()
    user_id = req.user_id
    
    if onnx_session is not None:
        user_input = np.array([user_id], dtype=np.int64)
        user_emb = onnx_session.run(None, {"user_ids": user_input})[0]
    else:
        # Fallback dummy embedding if ONNX not loaded yet
        np.random.seed(user_id)
        user_emb = np.random.randn(1, 64).astype(np.float32)
        user_emb /= (np.linalg.norm(user_emb) + 1e-8)
        
    # 2. FAISS HNSW Candidate Retrieval
    retrieval_top_k = max(req.top_k * 5, 50)
    scores, candidate_ids = hnsw_indexer.search(user_emb, top_k=retrieval_top_k)
    candidate_ids = candidate_ids[0]
    t_retrieval = (time.perf_counter() - t0) * 1000.0
    
    # 3. Stage 2: DCN-v2 Multi-Task Re-ranking
    t1 = time.perf_counter()
    n_cands = len(candidate_ids)
    u_tensor = torch.full((n_cands,), user_id, dtype=torch.long)
    i_tensor = torch.from_numpy(candidate_ids).long()
    
    if dcn_model is not None:
        with torch.no_grad():
            outputs = dcn_model(u_tensor, i_tensor)
            rank_scores = outputs["score"].cpu().numpy()
    else:
        # Linear fallback
        rank_scores = scores[0]
        
    # 4. Isotonic Calibration
    if calibrator is not None and calibrator.is_fitted:
        calib_probs = calibrator.calibrate(rank_scores)
    else:
        calib_probs = 1.0 / (1.0 + np.exp(-rank_scores))
        
    # Top-K Selection
    top_indices = np.argsort(-rank_scores)[:req.top_k]
    
    recommendations = []
    for final_rank, idx in enumerate(top_indices, start=1):
        recommendations.append(
            RecommendItem(
                item_id=int(candidate_ids[idx]),
                score=float(rank_scores[idx]),
                calibrated_prob=float(calib_probs[idx]),
                retrieval_rank=int(idx + 1),
                final_rank=final_rank,
            )
        )
        
    t_ranking = (time.perf_counter() - t1) * 1000.0
    t_total = (time.perf_counter() - t_start) * 1000.0
    
    return RecommendResponse(
        user_id=user_id,
        recommendations=recommendations,
        retrieval_latency_ms=round(t_retrieval, 3),
        ranking_latency_ms=round(t_ranking, 3),
        total_latency_ms=round(t_total, 3),
    )

def configure_app(session=None, indexer=None, model=None, calib=None):
    global onnx_session, hnsw_indexer, dcn_model, calibrator
    if session is not None:
        onnx_session = session
    if indexer is not None:
        hnsw_indexer = indexer
    if model is not None:
        dcn_model = model
    if calib is not None:
        calibrator = calib
