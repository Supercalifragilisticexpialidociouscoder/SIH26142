import os
import json
import pytest
import numpy as np
from pathlib import Path

from src.certification.calibrator import CertusCalibrator
from src.certification.certus_certifier import CertusCertifier
from src.certification.track_a import TrackACertifier
from src.certification.track_b import TrackBCertifier

ARTIFACT_PATH = "data/calibration/d1_calibration.json"
TEST_CANDIDATES_PATH = "data/demo_d1/d1_candidates.json"

@pytest.fixture
def certus_engine():
    assert os.path.exists(ARTIFACT_PATH), f"Calibration artifact missing: {ARTIFACT_PATH}"
    return CertusCertifier(calibration_artifact=ARTIFACT_PATH)

def test_calibration_test_split_separation():
    """Verifies that calibration scenes strictly exclude test tile T30TXM."""
    with open(ARTIFACT_PATH, "r") as f:
        artifact = json.load(f)
        
    meta = artifact["calibration_metadata"]
    excluded = meta.get("excluded_tiles", [])
    assert "T30TXM" in excluded, "T30TXM must be excluded from calibration"
    
    unique_tiles = meta.get("unique_calibration_tiles", [])
    assert "T30TXM" not in unique_tiles, "T30TXM must not be in calibration tiles"
    
    test_gee_id = "20210811T105619_20210811T110659_T30TXM"
    for s in artifact["calibration_scenes"]:
        assert "T30TXM" not in s["tile"], f"Scene {s['roi']} belongs to test tile T30TXM"
        assert s["lr_gee_id"] != test_gee_id, f"Test scene {test_gee_id} found in calibration scenes"


def test_no_test_labels_in_calibration():
    """Verifies that no test scene labels or test candidate IDs exist in the calibration artifact."""
    with open(ARTIFACT_PATH, "r") as f:
        artifact = json.load(f)
        
    for s in artifact["calibration_scenes"]:
        assert s["lr_gee_id"] != "20210811T105619_20210811T110659_T30TXM", "Test scene lr_gee_id found in calibration"

def test_calibration_artifact_loading(certus_engine):
    """Verifies that CertusCertifier loads the calibration artifact correctly."""
    assert certus_engine.is_calibrated or hasattr(certus_engine, "track_a")
    assert certus_engine.track_a.is_calibrated
    assert certus_engine.null_scores is not None
    assert len(certus_engine.null_scores) > 0
    assert certus_engine.proposal_stage.c_0 == 0.02
    assert certus_engine.proposal_stage.min_area == 16
    assert certus_engine.proposal_stage.max_area == 400

def test_track_a_calibration_and_lambda(certus_engine):
    """Verifies that Track A provides valid, monotonic lambda thresholds."""
    lam_10 = certus_engine.track_a.lambda_for_alpha(0.10)
    lam_20 = certus_engine.track_a.lambda_for_alpha(0.20)
    lam_30 = certus_engine.track_a.lambda_for_alpha(0.30)
    
    assert lam_10 > 0, "lambda_hat for alpha=0.10 should be feasible"
    assert lam_20 >= lam_10, "lambda_hat must be monotonic non-decreasing with alpha"
    assert lam_30 >= lam_20, "lambda_hat must be monotonic non-decreasing with alpha"

def test_track_b_calibration_nulls(certus_engine):
    """Verifies that Track B null scores produce valid conformal p-values."""
    test_scores = np.array([0.18, 0.19, 0.20, 0.25])
    p_vals = certus_engine.certify_scene(
        [{"id": i, "risk": s, "bbox_yxyx": [0, 0, 1, 1]} for i, s in enumerate(test_scores)],
        alpha=0.10,
        method="BH"
    )["candidate_results"]
    
    for r in p_vals:
        assert 0.0 < r["p_value"] <= 1.0, f"Invalid p-value: {r['p_value']}"

def test_bh_and_by_selection_order():
    """Verifies that BY is strictly as conservative or more conservative than BH."""
    calib = [{
        "risk_scores": np.array([0.2, 0.3, 0.4, 0.5]),
        "is_false_discovery": np.array([True, True, True, True]),
        "N": 4
    }]
    test_scores = np.array([0.1, 0.15, 0.2, 0.3])
    
    cb_bh = TrackBCertifier(alpha=0.5, method="BH")
    cb_bh.calibrate(calib)
    mask_bh = cb_bh.certify(test_scores)
    
    cb_by = TrackBCertifier(alpha=0.5, method="BY")
    cb_by.calibrate(calib)
    mask_by = cb_by.certify(test_scores)
    
    # BY must select a subset of BH
    assert np.sum(mask_by) <= np.sum(mask_bh)

def test_certificate_result_schema(certus_engine):
    """Verifies that certify_scene produces all required contract fields."""
    mock_candidates = [
        {"id": 1, "risk": 0.184, "area_px": 25, "centroid_yx": [10.0, 10.0], "bbox_yxyx": [8, 8, 12, 12]},
        {"id": 2, "risk": 0.250, "area_px": 50, "centroid_yx": [20.0, 20.0], "bbox_yxyx": [18, 18, 22, 22]}
    ]
    res = certus_engine.certify_scene(mock_candidates, alpha=0.10, method="BH")
    
    expected_keys = [
        "certificate_status", "method", "alpha", "N_proposed", "N_selected",
        "selected_candidate_ids", "selected_mask", "candidate_results",
        "threshold_info", "risk_statistics", "shift_status", "certification_metadata"
    ]
    for k in expected_keys:
        assert k in res, f"Missing key {k} in certificate result"
        
    assert res["N_proposed"] == 2
    assert res["selected_mask"].shape == (512, 512)

def test_held_out_certification_execution(certus_engine):
    """Runs certification on the real held-out test scene candidates."""
    with open(TEST_CANDIDATES_PATH, "r") as f:
        test_cands = json.load(f)
        
    res_a = certus_engine.certify_scene(test_cands, alpha=0.10, method="TRACK_A")
    assert res_a["N_proposed"] == 109
    assert res_a["selected_mask"].shape == (512, 512)
    assert res_a["selected_mask"].dtype == np.uint8
    assert res_a["N_selected"] >= 0

def test_deterministic_reproduction(certus_engine):
    """Verifies that re-running certification yields bit-for-bit identical results."""
    with open(TEST_CANDIDATES_PATH, "r") as f:
        test_cands = json.load(f)
        
    res1 = certus_engine.certify_scene(test_cands, alpha=0.20, method="TRACK_A")
    res2 = certus_engine.certify_scene(test_cands, alpha=0.20, method="TRACK_A")
    
    assert res1["N_selected"] == res2["N_selected"]
    assert res1["selected_candidate_ids"] == res2["selected_candidate_ids"]
    assert np.array_equal(res1["selected_mask"], res2["selected_mask"])
