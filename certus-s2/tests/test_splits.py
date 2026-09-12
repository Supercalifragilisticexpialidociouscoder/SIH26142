import json
import os
from pathlib import Path

def test_split_disjointness():
    """
    Enforces the CERTUS-S2 invariant:
    TRAIN ∩ CALIBRATE = ∅
    TRAIN ∩ TEST = ∅
    CALIBRATE ∩ TEST = ∅
    """
    manifest_path = Path(__file__).parent.parent / "data" / "split_manifest.json"
    assert manifest_path.exists(), f"Split manifest not found at {manifest_path}"
    
    with open(manifest_path, "r") as f:
        splits = json.load(f)
        
    train_sites = set(splits.get("train", []))
    calibrate_sites = set(splits.get("calibrate", []))
    test_sites = set(splits.get("test", []))
    
    assert len(train_sites) > 0, "TRAIN set cannot be empty"
    assert len(calibrate_sites) > 0, "CALIBRATE set cannot be empty"
    assert len(test_sites) > 0, "TEST set cannot be empty"
    
    train_cal_overlap = train_sites.intersection(calibrate_sites)
    train_test_overlap = train_sites.intersection(test_sites)
    cal_test_overlap = calibrate_sites.intersection(test_sites)
    
    assert not train_cal_overlap, f"Invariant violated: TRAIN and CALIBRATE share sites {train_cal_overlap}"
    assert not train_test_overlap, f"Invariant violated: TRAIN and TEST share sites {train_test_overlap}"
    assert not cal_test_overlap, f"Invariant violated: CALIBRATE and TEST share sites {cal_test_overlap}"
    
if __name__ == "__main__":
    test_split_disjointness()
    print("Split disjointness invariant holds.")
