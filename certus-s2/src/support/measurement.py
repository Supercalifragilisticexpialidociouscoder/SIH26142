import numpy as np
from src.observation.operator import EffectiveObservationOperator

def compute_residual(observed_lr: np.ndarray, reconstructed_hr: np.ndarray, operator: EffectiveObservationOperator) -> np.ndarray:
    """
    Computes the measurement residual e = |Y - \hat{D}(\hat{X})|.
    
    Args:
        observed_lr: The real L2A Sentinel-2 observation [C, H, W] or [H, W]
        reconstructed_hr: The SEN2SR output [C, 4H, 4W] or [4H, 4W]
        operator: The fitted EffectiveObservationOperator
        
    Returns:
        The spatial residual map matching the LR dimensions.
    """
    if not operator.is_fitted:
        raise ValueError("Operator must be fitted before computing residuals.")
        
    # Apply operator to HR (this includes blur, subpixel shift, and downsampling)
    expected_lr = operator.apply(reconstructed_hr)
    
    # operator.compute_residual can be used, or just manually compute the magnitude
    residual = np.abs(operator.compute_residual(observed_lr, expected_lr))
    return residual

def compute_support_map(residual: np.ndarray, sigma_b: float, scale_factor: int = 4) -> np.ndarray:
    """
    Estimates measurement support s(p) based on the residual and observation noise variance.
    
    This is an explicit calculation: s = exp(-residual^2 / (2 * sigma_b^2)).
    A value s ≈ 1 means the measurement perfectly constrains this region under the assumed model.
    A value s ≈ 0 means the measurement offers little constraint (potential hallucination).
    
    NOTE: As per CLAIMS.md, s=0 does NOT mean hallucinated, and s=1 does NOT mean correctness.
    It purely means how strongly the observation constrains the local structure.
    
    Args:
        residual: The computed residual map on the LR grid.
        sigma_b: The fitted operator error standard deviation.
        scale_factor: The spatial relationship mapping ratio (e.g. 4 for 10m -> 2.5m).
        
    Returns:
        Support map s(p) in [0, 1] on the HR grid.
    """
    # Create mask for valid pixels (not NaN)
    valid_mask = ~np.isnan(residual)
    
    # Initialize support map to 0 (NoData -> no observational constraint)
    s_lr = np.zeros_like(residual)
    
    # Calculate support on valid pixels, clipping residual to avoid overflow/underflow
    s_lr[valid_mask] = np.exp(-(residual[valid_mask]**2) / (2 * (sigma_b**2)))
    s_lr = np.clip(s_lr, 0.0, 1.0)
    
    # Explicitly map the 10m observational constraint back up to the 2.5m reconstruction grid
    # A single 10m observation uniformly constrains the 4x4 block of 2.5m pixels underneath it
    if scale_factor > 1:
        # Nearest-neighbor upsampling (Kronecker product-like)
        s_hr = np.repeat(s_lr, scale_factor, axis=-1)
        s_hr = np.repeat(s_hr, scale_factor, axis=-2)
    else:
        s_hr = s_lr
        
    return s_hr
