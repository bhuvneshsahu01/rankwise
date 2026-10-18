import numpy as np
import pytest
from rankwise.eval.counterfactual import compute_ips, compute_snips, ExposurePropensityModel


def test_ips_and_snips_unbiasedness_synthetic_bandit():
    """
    Simulates a contextual bandit where logging policy is biased:
    Action A: P(a=0) = 0.8, True Reward Mean = 0.2
    Action B: P(a=1) = 0.2, True Reward Mean = 0.8
    Target policy: pi(a=0) = 0.3, pi(a=1) = 0.7
    
    Ground truth target policy expected reward:
    V*(pi) = 0.3 * 0.2 + 0.7 * 0.8 = 0.06 + 0.56 = 0.620
    """
    rng = np.random.default_rng(42)
    n = 60000
    
    # Logging policy propensity: e(a)
    propensities = np.array([0.8, 0.2])
    actions = rng.choice([0, 1], size=n, p=propensities)
    
    # True rewards conditioned on action
    true_means = np.array([0.2, 0.8])
    rewards = rng.binomial(1, p=true_means[actions]).astype(float)
    
    logging_p = propensities[actions]
    target_policy_dist = np.array([0.3, 0.7])
    target_p = target_policy_dist[actions]
    
    v_true = float(np.sum(target_policy_dist * true_means))
    assert np.isclose(v_true, 0.62)
    
    v_ips = compute_ips(rewards, target_p, logging_p, clip_ratio=100.0)
    v_snips = compute_snips(rewards, target_p, logging_p, clip_ratio=100.0)
    
    # Verify estimates are within 5% of true policy value
    assert abs(v_ips - v_true) / v_true < 0.05
    assert abs(v_snips - v_true) / v_true < 0.05


def test_exposure_propensity_model():
    import pandas as pd
    df = pd.DataFrame({
        "video_id": [1, 1, 1, 1, 2, 2, 3],
        "user_id": [10, 11, 12, 13, 14, 15, 16]
    })
    model = ExposurePropensityModel(clip_min=0.01, clip_max=0.8)
    model.fit_from_logs(df)
    
    preds = model.predict_propensity(np.array([1, 2, 3, 999]))
    assert len(preds) == 4
    # Item 1 has 4/7 > Item 2 has 2/7 > Item 3 has 1/7
    assert preds[0] > preds[1] > preds[2]
    assert preds[3] == 0.01  # Default clipped propensity
