import numpy as np
from src.observation.operator import EffectiveObservationOperator
from src.support.measurement import compute_residual, compute_support_map

def test_residual_and_support():
    op = EffectiveObservationOperator(kernel_sigma=0.5, subpixel_shift=(0.0, 0.0), sigma_b=0.05, scale_factor=4)
    op.is_fitted = True # mock fit
    
    # 40x40 HR image
    hr = np.ones((40, 40)) * 100.0
    # 10x10 LR image
    lr = np.ones((10, 10)) * 100.0
    
    residual = compute_residual(lr, hr, op)
    
    assert residual.shape == lr.shape
    # Since HR is uniform 100 and LR is uniform 100, blur and downsample should also yield 100
    # Residual should be close to 0
    assert np.allclose(residual, 0.0, atol=1e-2)
    
    support = compute_support_map(residual, sigma_b=0.05, scale_factor=4)
    # The output support map must match the HR reconstruction dimensions (40x40)
    assert support.shape == hr.shape
    # Residual ≈ 0 means Support ≈ 1
    assert np.allclose(support, 1.0, atol=1e-2)
    
    # Now introduce a massive discrepancy
    lr_bad = np.ones((10, 10)) * 50.0
    residual_bad = compute_residual(lr_bad, hr, op)
    support_bad = compute_support_map(residual_bad, sigma_b=0.05, scale_factor=4)
    
    # Residual should be around 50
    assert np.allclose(residual_bad, 50.0, atol=1e-2)
    # Support should be ~0 because exp(-50^2 / 2*0.05^2) is tiny
    assert np.allclose(support_bad, 0.0, atol=1e-5)
    
    # Bounds check
    assert np.all(support_bad >= 0.0)
    assert np.all(support_bad <= 1.0)
    
def test_support_nan_handling():
    # If LR observation has NoData/NaN, support must be explicitly 0.0
    residual = np.array([[0.0, np.nan], [np.nan, 50.0]])
    support = compute_support_map(residual, sigma_b=1.0, scale_factor=2)
    
    # Scale factor 2 means (2, 2) input -> (4, 4) output
    assert support.shape == (4, 4)
    
    # Top-left (0,0) mapped to [0:2, 0:2], residual 0 -> support 1
    assert np.allclose(support[0:2, 0:2], 1.0)
    
    # Top-right (0,1) mapped to [0:2, 2:4], residual NaN -> support 0
    assert np.allclose(support[0:2, 2:4], 0.0)
    
    # Bottom-left (1,0) mapped to [2:4, 0:2], residual NaN -> support 0
    assert np.allclose(support[2:4, 0:2], 0.0)
    
    # Bottom-right (1,1) mapped to [2:4, 2:4], residual 50 -> support ≈ 0
    assert np.all(support[2:4, 2:4] < 1e-10)
