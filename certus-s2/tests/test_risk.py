import numpy as np
import pytest
from src.risk.engine import RiskEngine
from src.support.measurement import compute_support_map

def test_risk_engine():
    """Existing baseline test verifying basic forward pass and fitting."""
    engine = RiskEngine(backend_type="numpy")
    
    # 2x2 synthetic feature maps
    support = np.array([[1.0, 0.0], [0.5, 0.5]])
    sigma_ep = np.array([[0.01, 0.5], [0.1, 0.1]])
    sigma_al = np.array([[0.01, 0.2], [0.05, 0.05]])
    
    # Unfitted forward pass
    risk_map = engine.compute_risk(support, sigma_ep, sigma_al)
    
    assert risk_map.shape == (2, 2)
    assert np.all((risk_map >= 0.0) & (risk_map <= 1.0))
    
    # With mock weights:
    # High support (1.0), low uncertainty (0.01) -> Low risk
    # Low support (0.0), high uncertainty (0.5) -> High risk
    assert risk_map[0, 0] < risk_map[0, 1]
    
    # Fit the engine
    features = np.random.rand(100, 3)
    labels = np.random.randint(0, 2, 100)
    engine.fit(features, labels)
    
    assert engine.backend.is_fitted
    risk_map_fitted = engine.compute_risk(support, sigma_ep, sigma_al)
    assert risk_map_fitted.shape == (2, 2)
    assert np.all((risk_map_fitted >= 0.0) & (risk_map_fitted <= 1.0))

def test_risk_bounds_and_extremes():
    """Verify that risk output r(p) is strictly bounded in [0, 1] across extreme inputs."""
    engine = RiskEngine(backend_type="numpy")
    
    # Extreme inputs: highly positive, highly negative, zero
    support = np.array([[-100.0, 0.0], [1.0, 100.0]])
    sigma_ep = np.array([[0.0, 1e5], [1e-5, -10.0]])
    sigma_al = np.array([[-50.0, 1e5], [0.0, 50.0]])
    
    risk_map = engine.compute_risk(support, sigma_ep, sigma_al)
    assert risk_map.shape == (2, 2)
    assert np.all(risk_map >= 0.0), f"Risk map has negative values: {risk_map.min()}"
    assert np.all(risk_map <= 1.0), f"Risk map exceeds 1.0: {risk_map.max()}"

def test_risk_nan_resilience():
    """Verify that invalid/NaN/Inf support and uncertainties do NOT produce NaNs."""
    engine = RiskEngine(backend_type="numpy")
    
    # Inputs containing NaN, +inf, -inf
    support = np.array([[np.nan, 0.5], [np.inf, -np.inf]])
    sigma_ep = np.array([[0.1, np.nan], [0.2, 0.3]])
    sigma_al = np.array([[np.nan, 0.05], [0.05, np.inf]])
    
    risk_map = engine.compute_risk(support, sigma_ep, sigma_al)
    assert risk_map.shape == (2, 2)
    assert not np.any(np.isnan(risk_map)), "Risk map contains NaN values!"
    assert not np.any(np.isinf(risk_map)), "Risk map contains Inf values!"
    assert np.all((risk_map >= 0.0) & (risk_map <= 1.0))

def test_risk_monotonicity_support():
    """
    Verify that increasing measurement support s(p) strictly decreases or maintains
    risk r(p) when uncertainties are held constant (dr/ds <= 0).
    """
    engine = RiskEngine(backend_type="numpy")
    
    # Range of support values from 0.0 (no constraint) to 1.0 (full constraint)
    supports = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fixed_ep = np.full_like(supports, 0.1)
    fixed_al = np.full_like(supports, 0.05)
    
    risks = engine.compute_risk(supports, fixed_ep, fixed_al)
    
    # Verify strictly decreasing risk as support increases
    for i in range(len(risks) - 1):
        assert risks[i] >= risks[i + 1], (
            f"Monotonicity violation: support {supports[i]} -> risk {risks[i]}, "
            f"support {supports[i+1]} -> risk {risks[i+1]}"
        )

def test_risk_monotonicity_uncertainty():
    """
    Verify that increasing model or sensor uncertainty strictly increases or maintains
    risk r(p) when support is held constant (dr/d_sigma >= 0).
    """
    engine = RiskEngine(backend_type="numpy")
    
    fixed_support = np.full((5,), 0.5)
    fixed_al = np.full((5,), 0.05)
    ep_uncertainties = np.array([0.01, 0.1, 0.5, 1.0, 2.0])
    
    risks_ep = engine.compute_risk(fixed_support, ep_uncertainties, fixed_al)
    for i in range(len(risks_ep) - 1):
        assert risks_ep[i] <= risks_ep[i + 1], (
            f"Epistemic uncertainty violation: sigma {ep_uncertainties[i]} -> risk {risks_ep[i]}, "
            f"sigma {ep_uncertainties[i+1]} -> risk {risks_ep[i+1]}"
        )
        
    fixed_ep = np.full((5,), 0.1)
    al_uncertainties = np.array([0.01, 0.05, 0.2, 0.5, 1.0])
    risks_al = engine.compute_risk(fixed_support, fixed_ep, al_uncertainties)
    for i in range(len(risks_al) - 1):
        assert risks_al[i] <= risks_al[i + 1], (
            f"Aleatoric uncertainty violation: sigma {al_uncertainties[i]} -> risk {risks_al[i]}, "
            f"sigma {al_uncertainties[i+1]} -> risk {risks_al[i+1]}"
        )

def test_risk_residual_to_support_chain():
    """
    Verify the complete chain:
    residual e = |y - D(x)| -> support s = exp(-e^2/2sigma_b^2) -> risk r(s, sigma_ep, sigma_al).
    A higher observation discrepancy e must yield lower support s and higher risk r.
    """
    engine = RiskEngine(backend_type="numpy")
    sigma_b = 0.04
    
    # 10m grid residuals: pixel 0 has small discrepancy, pixel 1 has large discrepancy
    residuals = np.array([[0.001, 0.08]])
    
    # Compute support map (4x upsampled to 2.5m)
    support_hr = compute_support_map(residuals, sigma_b=sigma_b, scale_factor=4)
    assert support_hr.shape == (4, 8)
    
    # Support for pixel 0 block should be high, for pixel 1 block should be low
    assert support_hr[0, 0] > support_hr[0, 4]
    
    # Hold uncertainties constant across the entire grid
    sigma_ep = np.full_like(support_hr, 0.1)
    sigma_al = np.full_like(support_hr, 0.05)
    
    risk_hr = engine.compute_risk(support_hr, sigma_ep, sigma_al, residual=residuals)
    assert risk_hr.shape == (4, 8)
    assert np.all((risk_hr >= 0.0) & (risk_hr <= 1.0))
    
    # Higher residual -> lower support -> higher downstream decision risk
    assert risk_hr[0, 0] < risk_hr[0, 4], (
        f"Expected low-residual risk {risk_hr[0, 0]} < high-residual risk {risk_hr[0, 4]}"
    )

