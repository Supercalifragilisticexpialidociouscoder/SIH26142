import numpy as np
from src.observation.operator import EffectiveObservationOperator

def test_operator_apply():
    op = EffectiveObservationOperator(kernel_sigma=1.0, subpixel_shift=(0.5, 0.5), scale_factor=4)
    
    # 2D case
    hr_img = np.zeros((40, 40))
    hr_img[20:24, 20:24] = 100.0
    
    observed = op.apply(hr_img)
    
    # 40x40 downsampled by 4 -> 10x10
    assert observed.shape == (10, 10)
    
    # 3D case [C, H, W]
    hr_img_3d = np.zeros((4, 40, 40))
    hr_img_3d[:, 20:24, 20:24] = 100.0
    observed_3d = op.apply(hr_img_3d)
    
    assert observed_3d.shape == (4, 10, 10)
    
def test_compute_residual():
    op = EffectiveObservationOperator()
    y = np.ones((4, 10, 10))
    y_hat = np.full((4, 10, 10), 0.8)
    
    e = op.compute_residual(y, y_hat)
    assert e.shape == (4, 10, 10)
    assert np.allclose(e, 0.2)
    
    # Verify shape mismatch raises error
    y_wrong = np.ones((4, 40, 40))
    try:
        op.compute_residual(y_wrong, y_hat)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_operator_fit():
    op = EffectiveObservationOperator()
    assert not op.is_fitted
    
    lr_obs = [np.random.rand(3, 10, 10)]
    hr_ref = [np.random.rand(3, 40, 40)] # Usually HR is larger, but interface takes arrays
    
    op.fit(lr_obs, hr_ref, n_bootstrap=10)
    
    assert op.is_fitted
    summary = op.summary()
    assert "kernel_sigma" in summary
    assert "sigma_b_ci" in summary
