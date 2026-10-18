"""
FAISS HNSW (Hierarchical Navigable Small World) candidate indexer
for sub-millisecond approximate nearest neighbor retrieval.
"""

from pathlib import Path
from typing import Tuple, List
import faiss
import numpy as np


class HNSWIndexer:
    """
    Wrapper around FAISS IndexHNSWFlat with cosine similarity (inner product on normalized vectors).
    """
    def __init__(self, dim: int = 64, M: int = 32, ef_construction: int = 64, ef_search: int = 32):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Inner product on L2-normalized vectors == Cosine similarity
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self.item_ids = np.array([], dtype=np.int64)

    def build(self, embeddings: np.ndarray, item_ids: np.ndarray):
        """
        Adds item embeddings to HNSW graph.
        """
        assert embeddings.shape[1] == self.dim, f"Embedding dim mismatch: {embeddings.shape[1]} vs {self.dim}"
        # Ensure L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (embeddings / norms).astype(np.float32)

        self.index.reset()
        self.index.add(normalized)
        self.item_ids = np.asarray(item_ids, dtype=np.int64)

    def search(self, query_embeddings: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (scores, retrieved_item_ids).
        """
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_queries = (query_embeddings / norms).astype(np.float32)

        scores, indices = self.index.search(normalized_queries, top_k)
        retrieved_ids = np.take(self.item_ids, indices, mode="clip")
        return scores, retrieved_ids

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(file_path))
        np.save(str(file_path.with_suffix(".ids.npy")), self.item_ids)

    def load(self, file_path: Path):
        self.index = faiss.read_index(str(file_path))
        self.item_ids = np.load(str(file_path.with_suffix(".ids.npy")))
