import numpy as np
from src.certification.track_a import TrackACertifier

def test_track_a_calibration():
    # Setup alpha = 0.2, B = 1.0
    certifier = TrackACertifier(alpha=0.2, B=1.0)
    
    # We need n large enough that B/(n+1) < alpha, so 1/(n+1) < 0.2 => n+1 > 5 => n > 4
    # Let's use 9 scenes so B/(n+1) = 0.1
    # We need average loss <= 0.1 to satisfy alpha=0.2
    
    calibration_scenes = []
    for _ in range(9):
        # N=10 proposals per scene
        # 5 true detections (risk 0.1 to 0.5), 5 false discoveries (risk 0.6 to 1.0)
        risk_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        is_fd = np.array([False, False, False, False, False, True, True, True, True, True])
        
        calibration_scenes.append({
            'risk_scores': risk_scores,
            'is_false_discovery': is_fd,
            'N': 10
        })
        
    certifier.calibrate(calibration_scenes)
    
    assert certifier.is_calibrated
    # At lambda = 0.5, selected FDs = 0. Loss = 0. Bound = 0 + 0.1 = 0.1 <= 0.2. Valid.
    # At lambda = 0.6, selected FDs = 1. Loss = 1/10 = 0.1. Bound = 0.1 + 0.1 = 0.2 <= 0.2. Valid.
    # At lambda = 0.7, selected FDs = 2. Loss = 0.2. Bound = 0.2 + 0.1 = 0.3 > 0.2. Invalid.
    # So lambda_hat should be 0.6.
    assert np.isclose(certifier.lambda_hat, 0.6)
    
    # Test apply
    test_scores = np.array([0.5, 0.6, 0.7])
    certified = certifier.certify(test_scores)
    assert certified[0] == True
    assert certified[1] == True
    assert certified[2] == False
