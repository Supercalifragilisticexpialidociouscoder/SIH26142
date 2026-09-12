import numpy as np
from src.certification.track_b import TrackBCertifier

def test_track_b_calibration_and_certification():
    # BH method
    certifier = TrackBCertifier(alpha=0.1, method='BH')
    
    # Create mock calibration scenes
    # We only care about the risk scores of the false discoveries (nulls).
    # Let's say nulls generally have high risk scores (e.g., 0.7, 0.8, 0.9)
    calibration_scenes = [
        {
            'risk_scores': np.array([0.1, 0.8, 0.9]),
            'is_false_discovery': np.array([False, True, True])
        },
        {
            'risk_scores': np.array([0.2, 0.75, 0.85]),
            'is_false_discovery': np.array([False, True, True])
        }
    ]
    
    certifier.calibrate(calibration_scenes)
    assert certifier.is_calibrated
    # The null scores should be [0.8, 0.9, 0.75, 0.85]
    assert len(certifier.null_scores) == 4
    
    # Test compute p-values
    # nulls sorted: [0.75, 0.8, 0.85, 0.9] (m=4)
    # p_j = (1 + count(nulls <= s)) / 5
    
    # test score 0.1: count = 0 -> p = 1/5 = 0.2
    # test score 0.8: count = 2 -> p = 3/5 = 0.6
    # test score 0.95: count = 4 -> p = 5/5 = 1.0
    p_vals = certifier.compute_p_values(np.array([0.1, 0.8, 0.95]))
    assert np.allclose(p_vals, [0.2, 0.6, 1.0])
    
    # Since alpha=0.1, none of these p-values are <= 0.1, so none are certified.
    cert_mask = certifier.certify(np.array([0.1, 0.8, 0.95]))
    assert not np.any(cert_mask)
    
    # Let's add more nulls so that a small score can get p <= 0.1
    # We need m >= 9 for p = 1/10 = 0.1
    nulls_large = np.linspace(0.5, 1.0, 99) # 99 nulls
    certifier.null_scores = nulls_large
    
    # test score 0.1: count = 0 -> p = 1/100 = 0.01
    # threshold for rank 1 (BH): (1/1) * 0.1 = 0.1
    # 0.01 <= 0.1, so it should be certified.
    cert_mask = certifier.certify(np.array([0.1]))
    assert cert_mask[0] == True
    
    # BY method should be more conservative
    certifier_by = TrackBCertifier(alpha=0.1, method='BY')
    certifier_by.null_scores = nulls_large
    certifier_by.is_calibrated = True
    
    # Test array with multiple scores
    test_scores = np.array([0.1, 0.4, 0.6]) # 0.1 will have p=0.01, 0.4 will have p=0.01, 0.6 will have p > 0.1
    mask_bh = certifier.certify(test_scores)
    mask_by = certifier_by.certify(test_scores)
    
    # Expect both to certify the first two (p=0.01 < thresholds)
    assert mask_bh[0] == True and mask_bh[1] == True
    assert mask_by[0] == True and mask_by[1] == True
