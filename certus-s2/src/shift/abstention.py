import numpy as np
from enum import Enum

class AbstentionTier(Enum):
    CERTIFIED = "CERTIFIED"
    DEGRADED = "DEGRADED"
    ABSTAIN = "ABSTAIN"

class ShiftEvaluator:
    """
    Evaluates geographic/distribution shift and enforces the abstention policy.
    
    Delta_hat (\hat{\Delta}) is an empirical estimate of the distribution shift between
    the calibration null scores and the test null scores. 
    
    This is explicitly NOT a certified bound on the coverage gap, but an estimate
    used to trigger abstention.
    """
    def __init__(self, degraded_threshold: float = 0.05, abstain_threshold: float = 0.20):
        self.degraded_threshold = degraded_threshold
        self.abstain_threshold = abstain_threshold
        self.calibration_scores = None
        self.is_calibrated = False

    def calibrate(self, calibration_scores: np.ndarray):
        """
        Stores the baseline risk score distribution.
        """
        assert len(calibration_scores) > 0, "Requires calibration scores."
        self.calibration_scores = np.sort(calibration_scores)
        self.is_calibrated = True

    def estimate_shift(self, test_scores: np.ndarray) -> float:
        """
        Estimates the shift \hat{\Delta} between the test scores and calibration baseline.
        We use a simple Kolmogorov-Smirnov (KS) distance as the shift metric.
        """
        if not self.is_calibrated:
            raise ValueError("ShiftEvaluator is not calibrated.")
            
        if len(test_scores) == 0:
            return 0.0
            
        # Compute KS distance
        test_sorted = np.sort(test_scores)
        
        # We evaluate the CDFs at the combined unique points
        all_points = np.unique(np.concatenate([self.calibration_scores, test_sorted]))
        
        cdf_cal = np.searchsorted(self.calibration_scores, all_points, side='right') / len(self.calibration_scores)
        cdf_test = np.searchsorted(test_sorted, all_points, side='right') / len(test_sorted)
        
        ks_distance = np.max(np.abs(cdf_cal - cdf_test))
        return float(ks_distance)

    def evaluate(self, test_scores: np.ndarray) -> tuple[AbstentionTier, float]:
        """
        Returns the abstention tier and the estimated shift score.
        """
        shift_score = self.estimate_shift(test_scores)
        
        if shift_score > self.abstain_threshold:
            tier = AbstentionTier.ABSTAIN
        elif shift_score > self.degraded_threshold:
            tier = AbstentionTier.DEGRADED
        else:
            tier = AbstentionTier.CERTIFIED
            
        return tier, shift_score
