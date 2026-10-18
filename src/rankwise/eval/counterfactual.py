"""
Counterfactual Off-Policy Evaluation (OPE):
- Item exposure propensity estimation e(x, a)
- Clipped Inverse Propensity Scoring (IPS)
- Self-Normalized Inverse Propensity Scoring (SNIPS)
- Relative error benchmarking against held-out unbiased random-exposure traffic.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from rankwise.eval.metrics import bootstrap_ci


class ExposurePropensityModel:
    """
    Estimates the logging policy's exposure propensity e(x, a) = P(Item a displayed | User Context x).
    On KuaiRand, standard feed exposures are heavily confounded by item popularity and author features.
    """
    def __init__(self, clip_min: float = 1e-4, clip_max: float = 0.5):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.item_propensities: Dict[int, float] = {}
        self.default_propensity: float = 1e-3

    def fit_from_logs(
        self,
        df_standard: pd.DataFrame,
        item_col: str = "video_id",
        user_col: str = "user_id",
    ):
        """
        Estimates empirical item exposure probability under logging policy.
        """
        total_impressions = len(df_standard)
        item_counts = df_standard[item_col].value_counts()
        
        for item_id, count in item_counts.items():
            raw_p = count / total_impressions
            # Clip propensity to avoid high variance in IPS
            clipped_p = float(np.clip(raw_p, self.clip_min, self.clip_max))
            self.item_propensities[int(item_id)] = clipped_p
            
        self.default_propensity = self.clip_min
        return self

    def predict_propensity(self, item_ids: np.ndarray) -> np.ndarray:
        return np.array([
            self.item_propensities.get(int(i), self.default_propensity)
            for i in item_ids
        ], dtype=float)


def compute_ips(
    rewards: np.ndarray,
    target_policy_probs: np.ndarray,
    logging_propensities: np.ndarray,
    clip_ratio: float = 50.0,
) -> float:
    """
    Clipped Inverse Propensity Scoring (IPS):
        V_hat(pi) = (1 / N) * sum_{i=1}^N min(pi(a_i | x_i) / e(x_i, a_i), M) * r_i
    """
    weights = target_policy_probs / np.clip(logging_propensities, 1e-6, 1.0)
    clipped_weights = np.clip(weights, 0.0, clip_ratio)
    return float(np.mean(clipped_weights * rewards))


def compute_snips(
    rewards: np.ndarray,
    target_policy_probs: np.ndarray,
    logging_propensities: np.ndarray,
    clip_ratio: float = 50.0,
) -> float:
    """
    Self-Normalized Inverse Propensity Scoring (SNIPS):
        V_hat_SNIPS(pi) = sum(w_i * r_i) / sum(w_i)
    Reduces variance of IPS while remaining asymptotically unbiased.
    """
    weights = target_policy_probs / np.clip(logging_propensities, 1e-6, 1.0)
    clipped_weights = np.clip(weights, 0.0, clip_ratio)
    total_w = np.sum(clipped_weights)
    if total_w <= 0.0:
        return 0.0
    return float(np.sum(clipped_weights * rewards) / total_w)


def evaluate_counterfactual_policy(
    df_logged: pd.DataFrame,
    df_random_ground_truth: pd.DataFrame,
    propensity_model: ExposurePropensityModel,
    target_scores_logged: np.ndarray,
    target_scores_random: np.ndarray,
    reward_col: str = "is_click",
    item_col: str = "video_id",
    clip_ratio: float = 50.0,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluates policy value using SNIPS on logged traffic and benchmarks against
    the true empirical reward on the held-out random-exposure traffic.
    """
    rewards_logged = df_logged[reward_col].to_numpy(dtype=float)
    items_logged = df_logged[item_col].to_numpy(dtype=int)
    
    # Target policy probability: softmax temperature scaling on scores
    tau = 1.0
    exp_scores = np.exp(np.clip(target_scores_logged / tau, -10.0, 10.0))
    # Approximate action selection probability of recommended item
    pi_logged = exp_scores / (1.0 + exp_scores)
    # Logging propensity e(x, a)
    e_logged = propensity_model.predict_propensity(items_logged)
    
    # Compute IPS & SNIPS estimates
    v_ips = compute_ips(rewards_logged, pi_logged, e_logged, clip_ratio=clip_ratio)
    v_snips = compute_snips(rewards_logged, pi_logged, e_logged, clip_ratio=clip_ratio)
    
    # Ground truth policy value on unbiased random traffic:
    # Under random exposure, evaluating policy corresponds to expected reward on top-ranked items
    # or empirical reward weighted by target policy decision
    rewards_rand = df_random_ground_truth[reward_col].to_numpy(dtype=float)
    exp_rand = np.exp(np.clip(target_scores_random / tau, -10.0, 10.0))
    pi_rand = exp_rand / (1.0 + exp_rand)
    
    # Ground truth unbiased policy value
    v_ground_truth = float(np.average(rewards_rand, weights=pi_rand))
    
    # Relative estimation error [E]% = |V_snips - V_ground_truth| / V_ground_truth * 100%
    rel_error_pct = abs(v_snips - v_ground_truth) / (v_ground_truth + 1e-8) * 100.0
    
    # 95% Bootstrap CI on SNIPS
    rng = np.random.default_rng(seed)
    n = len(rewards_logged)
    boot_snips = np.empty(500)
    weights = np.clip(pi_logged / np.clip(e_logged, 1e-6, 1.0), 0.0, clip_ratio)
    
    for b in range(500):
        idx = rng.choice(n, size=n, replace=True)
        w_sample = weights[idx]
        r_sample = rewards_logged[idx]
        sum_w = np.sum(w_sample)
        boot_snips[b] = np.sum(w_sample * r_sample) / (sum_w + 1e-8)
        
    ci_lower = float(np.percentile(boot_snips, 2.5))
    ci_upper = float(np.percentile(boot_snips, 97.5))
    
    return {
        "v_ips": v_ips,
        "v_snips": v_snips,
        "v_ground_truth": v_ground_truth,
        "relative_error_pct": float(round(rel_error_pct, 2)),
        "snips_ci_lower": ci_lower,
        "snips_ci_upper": ci_upper,
    }
