import numpy as np

class TrackBCertifier:
    """
    Track B: Conformal p-values with Benjamini-Hochberg (BH) or Benjamini-Yekutieli (BY)
    for False Discovery Rate (FDR) control.
    
    Null hypothesis: The candidate detection is a false discovery (error).
    Therefore, small risk scores are evidence *against* the null.
    """
    def __init__(self, alpha: float = 0.1, method: str = 'BY'):
        assert method in ['BH', 'BY'], "Method must be 'BH' or 'BY'."
        self.alpha = alpha
        self.method = method
        self.null_scores = None
        self.is_calibrated = False

    def calibrate(self, calibration_scenes: list[dict]):
        """
        Pools null scores (risk scores on ground-truth false discoveries) from the calibration set.
        """
        nulls = []
        for scene in calibration_scenes:
            # We only keep risk scores for candidates that were ACTUALLY false discoveries (nulls)
            scores = scene['risk_scores']
            is_fd = scene['is_false_discovery']
            nulls.extend(scores[is_fd].tolist())
            
        assert len(nulls) > 0, "No null calibration samples available."
        self.null_scores = np.array(nulls)
        self.is_calibrated = True

    def compute_p_values(self, test_scores: np.ndarray) -> np.ndarray:
        """
        Computes conformal p-values for a set of test candidates.
        p_j = (1 + sum(s_k <= s_test)) / (m + 1)
        """
        m = len(self.null_scores)
        p_values = np.zeros_like(test_scores, dtype=float)
        
        # Vectorized computation of p-values
        # For each test score, count how many null scores are <= test score.
        # This can be optimized, but standard broadcasting works for moderate sizes.
        # test_scores: [N], null_scores: [M]
        # p_values: [N]
        # Using searchsorted for O(N log M) instead of O(N * M)
        sorted_nulls = np.sort(self.null_scores)
        # number of nulls <= s_test
        counts = np.searchsorted(sorted_nulls, test_scores, side='right')
        
        p_values = (1.0 + counts) / (m + 1.0)
        return p_values

    def certify(self, test_scores: np.ndarray) -> np.ndarray:
        """
        Applies BH or BY procedure to certify detections.
        Returns boolean mask corresponding to test_scores (True = certified/reject null).
        """
        if not self.is_calibrated:
            raise ValueError("Track B is not calibrated.")
            
        N = len(test_scores)
        if N == 0:
            return np.array([], dtype=bool)
            
        p_values = self.compute_p_values(test_scores)
        
        # Sort p-values
        sorted_indices = np.argsort(p_values)
        p_sorted = p_values[sorted_indices]
        
        # Compute thresholds
        k_array = np.arange(1, N + 1)
        
        if self.method == 'BH':
            thresholds = (k_array / N) * self.alpha
        else: # BY
            cm = np.sum(1.0 / np.arange(1, N + 1))
            thresholds = (k_array / (N * cm)) * self.alpha
            
        # Find largest k where p_(k) <= threshold
        valid_k = np.where(p_sorted <= thresholds)[0]
        
        certified_mask = np.zeros(N, dtype=bool)
        if len(valid_k) > 0:
            k_max = valid_k[-1]
            # All indices up to k_max are certified
            certified_indices = sorted_indices[:k_max + 1]
            certified_mask[certified_indices] = True
            
        return certified_mask
