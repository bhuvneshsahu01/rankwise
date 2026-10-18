"""
LightGBM LambdaMART baseline ranker trained on KuaiRand standard traffic.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb


class LambdaMARTBaseline:
    """
    LightGBM LambdaMART ranking model for candidate re-ranking.
    """
    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.model: Optional[lgb.LGBMRanker] = None
        self.feature_cols: List[str] = []

    def fit(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        user_col: str = "user_id",
        label_col: str = "is_click",
    ):
        self.feature_cols = feature_cols
        # LambdaMART requires data sorted by group
        df_sorted = df_train.sort_values(by=user_col).reset_index(drop=True)
        groups = df_sorted.groupby(user_col, sort=False).size().to_numpy()
        
        X = df_sorted[feature_cols].to_numpy()
        y = df_sorted[label_col].to_numpy()

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[10],
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(X, y, group=groups)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        X = df[self.feature_cols].to_numpy()
        return self.model.predict(X)
