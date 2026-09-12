import numpy as np
from src.decision.proposal import D1ProposalStage

def test_d1_proposal():
    proposal = D1ProposalStage(c_0=0.8)
    
    sr_image = np.array([
        [0.1, 0.9],
        [0.85, 0.4]
    ])
    
    mask = proposal.propose(sr_image)
    assert mask.shape == (2, 2)
    assert mask[0, 1] == True
    assert mask[1, 0] == True
    assert mask[0, 0] == False
    assert mask[1, 1] == False
    
    # Test calibration
    proposal.calibrate([sr_image], target_yield=0.5)
    # The array has 4 pixels: 0.1, 0.4, 0.85, 0.9
    # top 50% means top 2 pixels (0.85, 0.9). 
    # Quantile at 0.5 is (0.4 + 0.85)/2 = 0.625
    assert proposal.is_calibrated
    assert np.isclose(proposal.c_0, 0.625)
    
    mask2 = proposal.propose(sr_image)
    assert mask2[0, 1] == True
    assert mask2[1, 0] == True
    assert mask2[0, 0] == False

def test_d1_proposal_multiband_and_candidates():
    # Setup 4-band image (4, 30, 30)
    # Background: low brightness, vegetation NDVI > 0.4
    # Structure 1: 5x5 block (25 px) at (5, 5), high brightness, low NDVI
    # Structure 2: 2x2 block (4 px) at (20, 20) -> too small for min_area=16
    img = np.zeros((4, 30, 30), dtype=np.float32)
    # Band order: B02, B03, B04, B08
    # Background: Red=0.1, NIR=0.4 (NDVI = 0.3/0.5 = 0.6)
    img[0] = 0.1 # Blue
    img[1] = 0.1 # Green
    img[2] = 0.1 # Red
    img[3] = 0.4 # NIR
    
    # Structure 1: 5x5 built structure at [5:10, 5:10] (25 pixels)
    # High visible, low NDVI: Red=0.4, NIR=0.4 (NDVI = 0.0)
    img[0, 5:10, 5:10] = 0.4
    img[1, 5:10, 5:10] = 0.4
    img[2, 5:10, 5:10] = 0.4
    img[3, 5:10, 5:10] = 0.4
    
    # Structure 2: 2x2 structure at [20:22, 20:22] (4 pixels)
    img[0, 20:22, 20:22] = 0.5
    img[1, 20:22, 20:22] = 0.5
    img[2, 20:22, 20:22] = 0.5
    img[3, 20:22, 20:22] = 0.5
    
    proposal = D1ProposalStage(c_0=0.01, min_area=16, max_area=400)
    cand_mask, candidates = proposal.extract_candidates(img)
    
    # Only Structure 1 (25 px) should pass min_area=16; Structure 2 (4 px) should be filtered out
    assert len(candidates) == 1
    c = candidates[0]
    assert c["id"] == 1
    assert c["area"] == 25
    assert np.isclose(c["centroid"][0], 7.0)
    assert np.isclose(c["centroid"][1], 7.0)
    assert cand_mask[7, 7] == True
    assert cand_mask[21, 21] == False # Area 4 filtered out

def test_d1_evaluate_against_reference():
    proposal = D1ProposalStage(c_0=0.01, min_area=16, max_area=400)
    
    # Candidate at (10.0, 10.0)
    candidates = [{
        "id": 1,
        "area": 25,
        "centroid": (10.0, 10.0),
        "bbox": (8, 8, 12, 12),
        "mask": np.zeros((30, 30), dtype=bool)
    }]
    
    # Reference image containing matching structure at (11.0, 10.0) (dist = 1.0 <= 2.0)
    ref_img = np.zeros((4, 30, 30), dtype=np.float32)
    ref_img[:, 9:14, 8:13] = 0.4 # 5x5 structure
    
    eval_res = proposal.evaluate_against_reference(candidates, ref_img, d_max=2.0)
    assert eval_res["total_candidates"] == 1
    assert eval_res["true_detections"] == 1
    assert eval_res["false_discoveries"] == 0
    assert eval_res["precision"] == 1.0
    assert candidates[0]["is_true_detection"] == True
    assert candidates[0]["dist_to_reference"] <= 2.0

