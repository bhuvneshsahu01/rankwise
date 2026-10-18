"""
Ranking quality, retrieval, and calibration metrics for Rankwise.
"""

from typing import List, Dict, Union
import numpy as np


def compute_dcg_at_k(relevances: np.ndarray, k: int = 10) -> float:
    """
    Computes Discounted Cumulative Gain at rank K.
    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    relevances = np.asfarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(len(relevances)) + 2.0)
    return float(np.sum((np.power(2.0, relevances) - 1.0) / discounts))


def compute_ndcg_at_k(actual_relevances: np.ndarray, predicted_scores: np.ndarray, k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank K.
    """
    if len(actual_relevances) == 0:
        return 0.0
    
    # Sort items by predicted score descending
    rank_order = np.argsort(-predicted_scores)
    sorted_relevances = actual_relevances[rank_order]
    
    actual_dcg = compute_dcg_at_k(sorted_relevances, k=k)
    
    # Ideal DCG (sort ground truth relevances descending)
    ideal_order = np.argsort(-actual_relevances)
    ideal_relevances = actual_relevances[ideal_order]
    ideal_dcg = compute_dcg_at_k(ideal_relevances, k=k)
    
    if ideal_dcg <= 0.0:
        return 0.0
    return float(actual_dcg / ideal_dcg)


def compute_group_ndcg(
    df,
    user_col: str = "user_id",
    rel_col: str = "is_click",
    score_col: str = "predicted_score",
    k: int = 10,
    min_items_per_user: int = 5,
) -> float:
    """
    Computes mean NDCG@K across all users with at least min_items_per_user interactions.
    """
    user_ndcgs = []
    for user_id, group in df.groupby(user_col):
        if len(group) < min_items_per_user:
            continue
        rels = group[rel_col].to_numpy(dtype=float)
        if np.sum(rels) == 0:
            continue
        scores = group[score_col].to_numpy(dtype=float)
        ndcg = compute_ndcg_at_k(rels, scores, k=k)
        user_ndcgs.append(ndcg)
        
    return float(np.mean(user_ndcgs)) if user_ndcgs else 0.0


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Computes 95% bootstrap confidence interval for the sample mean.
    """
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    n = len(values)
    for b in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    mean = float(np.mean(values))
    return {"mean": mean, "ci_lower": low, "ci_upper": high}


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) with equal-width probability bins.
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    
    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
            
    return float(ece)
