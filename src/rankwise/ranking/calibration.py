"""
Post-ranking probability calibration using Isotonic Regression and ECE validation.
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from rankwise.eval.metrics import compute_ece


class RankerCalibrator:
    """
    Fits non-parametric isotonic regression on validation predictions
    to map raw ranking logits/scores into well-calibrated posterior probabilities.
    Supports propensity-weighted calibration for debiasing under distribution shift.
    """
    def __init__(self):
        self.iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "RankerCalibrator":
        scores = np.asarray(scores, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
        self.iso_reg.fit(scores, y_true, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "Calibrator is not fitted yet."
        return self.iso_reg.predict(np.asarray(scores, dtype=float))

    def evaluate(
        self,
        scores_test: np.ndarray,
        y_test: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[float, float]:
        """
        Returns (uncalibrated_ece, calibrated_ece).
        """
        raw_prob = (scores_test - scores_test.min()) / (scores_test.max() - scores_test.min() + 1e-8)
        uncalib_ece = compute_ece(y_test, raw_prob, n_bins=n_bins)
        
        calib_prob = self.calibrate(scores_test)
        calib_ece = compute_ece(y_test, calib_prob, n_bins=n_bins)
        
        return uncalib_ece, calib_ece
