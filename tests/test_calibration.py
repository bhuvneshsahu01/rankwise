import numpy as np
import pytest
from rankwise.eval.metrics import compute_ece
from rankwise.ranking.calibration import RankerCalibrator


def test_ece_computation():
    # Perfectly calibrated
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert np.isclose(ece, 0.0)
    
    # Completely miscalibrated (predict 0.99 for all negatives)
    y_true_bad = np.zeros(100)
    y_prob_bad = np.full(100, 0.99)
    ece_bad = compute_ece(y_true_bad, y_prob_bad, n_bins=10)
    assert ece_bad > 0.9


def test_ranker_calibrator_monotonicity_and_reduction():
    rng = np.random.default_rng(42)
    scores_val = rng.uniform(-3, 3, size=500)
    # Sigmoidal ground truth with noise
    probs = 1.0 / (1.0 + np.exp(-scores_val))
    y_val = rng.binomial(1, probs)
    
    calibrator = RankerCalibrator()
    calibrator.fit(scores_val, y_val)
    
    # Test on test set
    scores_test = np.linspace(-3, 3, 50)
    calibrated = calibrator.calibrate(scores_test)
    
    # Monotonicity check
    assert np.all(np.diff(calibrated) >= -1e-6)
    assert (calibrated >= 0.0).all() and (calibrated <= 1.0).all()
