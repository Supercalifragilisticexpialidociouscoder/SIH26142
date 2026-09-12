import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

class RiskBackend(ABC):
    """
    Interface for the Risk Engine backends.
    Predicts a decision risk score r in [0, 1].
    """
    @abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray):
        pass

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        pass

class NumpyRiskBackend(RiskBackend):
    """
    Native NumPy fallback for dependency-constrained environments.
    Computes decision risk r in [0, 1] via logistic sigmoid.
    Not the intended final production model.
    """
    def __init__(self):
        self.is_fitted = False
        # Features: [support (s), epistemic uncertainty (sigma_ep), aleatoric uncertainty (sigma_al)]
        # Negative weight on support: stronger observational support decreases decision risk.
        # Positive weight on uncertainties: higher uncertainty increases decision risk.
        self.weights = np.array([[-2.0, 3.0, 1.5]]) 
        self.bias = np.array([0.5])
        
    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.is_fitted = True
        
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        # Sanitize features to avoid arithmetic invalidity
        clean_features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        logits = (clean_features @ self.weights.T).squeeze() + self.bias
        # Clip logits to prevent numerical overflow in exp
        logits = np.clip(logits, -50.0, 50.0)
        r_flat = 1.0 / (1.0 + np.exp(-logits))
        r_flat = np.nan_to_num(r_flat, nan=1.0)
        return np.clip(r_flat, 0.0, 1.0)

class SklearnRiskBackend(RiskBackend):
    """
    Scikit-learn logistic regression backend (the primary design).
    """
    def __init__(self):
        try:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(class_weight='balanced')
        except ImportError:
            raise ImportError("scikit-learn is not installed. Use NumpyRiskBackend fallback.")
            
    def fit(self, features: np.ndarray, labels: np.ndarray):
        clean_features = np.nan_to_num(features, nan=0.0)
        self.model.fit(clean_features, labels)
        
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        clean_features = np.nan_to_num(features, nan=0.0)
        # LogisticRegression.predict_proba returns [N, 2].
        # Column 1 is the score r in [0, 1] for class 1.
        # NOTE: r is a decision risk score, NOT a calibrated probability of hallucination.
        r = self.model.predict_proba(clean_features)[:, 1]
        r = np.nan_to_num(r, nan=1.0)
        return np.clip(r, 0.0, 1.0)

class RiskEngine:
    """
    Computes a downstream decision risk score r(p) in [0, 1] for candidate detections.

    NOTE: r(p) is a decision risk ranking metric for candidate retention (r_k <= lambda).
    Per SIH26142_v4_hardened.md and CLAIMS.md, r(p) is NOT a calibrated probability
    of hallucination or error unless an empirical calibration procedure establishes that interpretation.
    """
    def __init__(self, backend_type: str = "auto"):
        self.backend_type = backend_type
        if backend_type == "auto" or backend_type == "sklearn":
            try:
                self.backend = SklearnRiskBackend()
                logger.info("RiskEngine: Loaded SklearnRiskBackend")
                self.backend_type = "sklearn"
            except ImportError:
                if backend_type == "sklearn":
                    raise
                self.backend = NumpyRiskBackend()
                logger.warning("RiskEngine: scikit-learn missing. Falling back to NumpyRiskBackend (NOT intended for production).")
                self.backend_type = "numpy"
        elif backend_type == "numpy":
            self.backend = NumpyRiskBackend()
            logger.info("RiskEngine: Loaded NumpyRiskBackend explicitly.")
        else:
            raise ValueError(f"Unknown backend_type: {backend_type}")

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.backend.fit(features, labels)

    def compute_risk(self, support: np.ndarray, 
                     sigma_ep: np.ndarray, 
                     sigma_al: np.ndarray,
                     residual: Optional[np.ndarray] = None) -> np.ndarray:
        r"""
        Computes the spatial decision risk map r(p) in [0, 1].

        Args:
            support: Measurement support map s(p) in [0, 1], quantifying observational constraint.
            sigma_ep: Epistemic uncertainty map (model/reconstruction uncertainty).
            sigma_al: Aleatoric uncertainty map (sensor noise uncertainty).
            residual: Optional spatial residual e = |y - \hat{D}(\hat{x})|.

        Returns:
            Risk map r(p) in [0, 1] matching the spatial dimensions of support.
        """
        orig_shape = support.shape
        
        # Robust handling of invalid/NaN/Inf support:
        # A missing or invalid measurement provides ZERO observational constraint (support = 0.0).
        s_clean = np.nan_to_num(support, nan=0.0, posinf=1.0, neginf=0.0)
        s_clean = np.clip(s_clean, 0.0, 1.0)
        
        # Sanitize uncertainties (NaN uncertainty treated conservatively as high uncertainty = 1.0)
        ep_clean = np.nan_to_num(sigma_ep, nan=1.0, posinf=1.0, neginf=0.0)
        al_clean = np.nan_to_num(sigma_al, nan=1.0, posinf=1.0, neginf=0.0)
        
        s_flat = s_clean.flatten()
        ep_flat = ep_clean.flatten()
        al_flat = al_clean.flatten()
        
        X = np.stack([s_flat, ep_flat, al_flat], axis=1)
        r_flat = self.backend.predict_proba(X)
        r_flat = np.nan_to_num(r_flat, nan=1.0)
        r_flat = np.clip(r_flat, 0.0, 1.0)
        
        r_map = r_flat.reshape(orig_shape)
        return r_map
