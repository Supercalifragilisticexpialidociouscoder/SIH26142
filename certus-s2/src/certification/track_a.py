import numpy as np

class TrackACertifier:
    """
    Track A: Conformal Risk Control on a monotone normalized false-discovery loss.
    
    The loss for scene i is:
    L_i(\lambda) = FD_i(\lambda) / max(1, N_i)
    
    where N_i is fixed by the D1 proposal stage, and FD_i(\lambda) is the number of 
    proposed candidates with risk score <= \lambda that are actually false discoveries.
    
    Because N_i is fixed, L_i(\lambda) is monotonically non-decreasing in \lambda.
    We seek the largest \lambda such that:
    (1/n) \sum L_i(\lambda) + B/(n+1) <= \alpha
    """
    def __init__(self, alpha: float = 0.1, B: float = 1.0):
        self.alpha = alpha
        self.B = B
        self.lambda_hat = None
        self.is_calibrated = False

    def compute_loss(self, risk_scores: np.ndarray, is_false_discovery: np.ndarray, 
                     N: int, lam: float) -> float:
        """
        Computes L_i(\lambda) for a single scene.
        """
        # Selected candidates: risk score <= lambda
        selected = risk_scores <= lam
        # False discoveries among selected
        FD = np.sum(selected & is_false_discovery)
        
        return FD / max(1, N)

    def calibrate(self, calibration_scenes: list[dict]):
        """
        Calibrates \hat{\lambda} over a set of site-disjoint calibration scenes.
        
        calibration_scenes: list of dicts, each containing:
          - 'risk_scores': array of risk scores for proposed candidates in C(Y)
          - 'is_false_discovery': boolean array indicating ground-truth errors
          - 'N': total number of proposed candidates in C(Y) (fixed)
        """
        n = len(calibration_scenes)
        assert n > 0, "Need at least one calibration scene."
        
        # We need to find the supremum of \lambda \in [0, 1] satisfying the bound.
        # Since \lambda only changes the loss when it crosses one of the risk scores,
        # we can check all unique risk scores as candidate \lambdas.
        
        all_lambdas = [0.0, 1.0]
        for scene in calibration_scenes:
            all_lambdas.extend(scene['risk_scores'].tolist())
            
        # Sort lambdas descending so we find the largest valid one first
        all_lambdas = np.sort(np.unique(all_lambdas))[::-1]
        
        best_lambda = -1.0
        for lam in all_lambdas:
            losses = []
            for scene in calibration_scenes:
                loss_i = self.compute_loss(
                    scene['risk_scores'], 
                    scene['is_false_discovery'], 
                    scene['N'], 
                    lam
                )
                losses.append(loss_i)
                
            empirical_risk = np.mean(losses)
            bound = empirical_risk + self.B / (n + 1)
            
            if bound <= self.alpha:
                best_lambda = lam
                break
                
        # If no lambda satisfies the bound (even lambda=0), it's infeasible.
        # In a real scenario, this means n is too small or B is too large for the requested alpha.
        self.lambda_hat = best_lambda
        self.calibration_scenes = calibration_scenes
        self.is_calibrated = True

    def lambda_for_alpha(self, alpha: float) -> float:
        """
        Computes the calibrated lambda threshold for a specific alpha target.
        """
        if not self.is_calibrated or not hasattr(self, 'calibration_scenes') or not self.calibration_scenes:
            raise ValueError("Track A is not calibrated.")
            
        n = len(self.calibration_scenes)
        all_lambdas = [0.0, 1.0]
        for scene in self.calibration_scenes:
            all_lambdas.extend(scene['risk_scores'].tolist())
            
        all_lambdas = np.sort(np.unique(all_lambdas))[::-1]
        
        for lam in all_lambdas:
            losses = [
                self.compute_loss(
                    s['risk_scores'], 
                    s['is_false_discovery'], 
                    s['N'], 
                    lam
                )
                for s in self.calibration_scenes
            ]
            bound = float(np.mean(losses) + self.B / (n + 1))
            if bound <= alpha:
                return float(lam)
                
        return -1.0

    def certify(self, risk_scores: np.ndarray) -> np.ndarray:
        """
        Applies the certified threshold to a new scene's candidate risk scores.
        Returns a boolean mask of CERTIFIED detections (True if risk <= lambda_hat).
        """
        if not self.is_calibrated:
            raise ValueError("Track A is not calibrated.")
            
        if self.lambda_hat < 0:
            # Infeasible bound
            return np.zeros_like(risk_scores, dtype=bool)
            
        return risk_scores <= self.lambda_hat
