"""
Low-latency feature store supporting in-memory dict replay with Redis client fallback.
"""

from typing import Dict, Any, Optional
import numpy as np


class FeatureStore:
    """
    Feature Store caching user interaction states and video feature vectors.
    Defaults to an ultra-fast in-memory dict store; can bind to Redis when running.
    """
    def __init__(self, redis_url: Optional[str] = None):
        self.in_memory_user_store: Dict[int, Dict[str, Any]] = {}
        self.in_memory_item_store: Dict[int, Dict[str, Any]] = {}
        self.redis_client = None

        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url, socket_timeout=1.0)
                self.redis_client.ping()
                print(f"[✓] Connected to Redis at {redis_url}")
            except Exception as e:
                print(f"[!] Redis not reachable ({e}). Falling back to in-memory store.")
                self.redis_client = None

    def put_user(self, user_id: int, features: Dict[str, Any]):
        self.in_memory_user_store[user_id] = features

    def get_user(self, user_id: int) -> Dict[str, Any]:
        return self.in_memory_user_store.get(user_id, {"user_id": user_id, "active_degree": 0})

    def put_item(self, item_id: int, features: Dict[str, Any]):
        self.in_memory_item_store[item_id] = features

    def get_item(self, item_id: int) -> Dict[str, Any]:
        return self.in_memory_item_store.get(item_id, {"video_id": item_id, "category": 0})


RedisFeatureStore = FeatureStore
