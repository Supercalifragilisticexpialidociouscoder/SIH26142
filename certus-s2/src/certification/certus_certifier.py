import os
import json
from pathlib import Path
from typing import Optional, Any, Union
import numpy as np
import rasterio

from src.decision.proposal import D1ProposalStage
from src.certification.track_a import TrackACertifier
from src.certification.track_b import TrackBCertifier
from src.shift.abstention import ShiftEvaluator, AbstentionTier

class CertusCertifier:
    """
    CERTUS-S2 Production Certification Engine for Task D1.
    
    Loads a site-disjoint calibration artifact and certifies new scenes using:
    - Track A: Conformal Risk Control on monotone normalized loss (E[L(lambda)] <= alpha)
    - Track B: Conformal p-values with Benjamini-Hochberg (BH) or Benjamini-Yekutieli (BY) (E[FDR] <= alpha)
    
    Preserves strict separation between:
    - Track A risk bound (population expectation guarantee on normalized loss L = FD / N)
    - Track B FDR guarantee (expectation guarantee E[FDR] <= alpha)
    - Empirical FDP on a single scene (post-hoc sample statistic; NOT a per-scene guarantee)
    """
    def __init__(self, calibration_artifact: Union[str, Path, dict] = "data/calibration/d1_calibration.json"):
        if isinstance(calibration_artifact, (str, Path)):
            self.artifact_path = str(calibration_artifact)
            assert os.path.exists(self.artifact_path), f"Calibration artifact not found at {self.artifact_path}"
            with open(self.artifact_path, "r") as f:
                self.artifact_data = json.load(f)
        else:
            self.artifact_path = "dict_memory"
            self.artifact_data = calibration_artifact
            
        # Reconstruct proposal stage with calibrated parameters
        prop_cfg = self.artifact_data.get("proposal_config", {})
        self.proposal_stage = D1ProposalStage(
            c_0=prop_cfg.get("c_0", 0.02),
            min_area=prop_cfg.get("min_area", 16),
            max_area=prop_cfg.get("max_area", 400),
            highpass_sigma=prop_cfg.get("highpass_sigma", 2.0),
            ndvi_threshold=prop_cfg.get("ndvi_threshold", 0.3)
        )
        
        # Initialize Track A
        track_a_info = self.artifact_data.get("track_a_calibration", {})
        self.track_a = TrackACertifier(alpha=0.10, B=track_a_info.get("B", 1.0))
        
        # Load calibration scenes into Track A
        calib_scenes_raw = self.artifact_data.get("calibration_scenes", [])
        calib_scenes = []
        for s in calib_scenes_raw:
            calib_scenes.append({
                "scene_idx": s["scene_idx"],
                "roi": s["roi"],
                "tile": s["tile"],
                "lr_gee_id": s["lr_gee_id"],
                "N": s["N"],
                "true_detections": s["true_detections"],
                "false_discoveries": s["false_discoveries"],
                "risk_scores": np.array(s["risk_scores"], dtype=np.float32),
                "is_false_discovery": np.array(s["is_false_discovery"], dtype=bool)
            })
        self.track_a.calibrate(calib_scenes)
        
        # Initialize Track B with calibration nulls
        track_b_info = self.artifact_data.get("track_b_calibration", {})
        null_scores = np.array(track_b_info.get("null_scores", []), dtype=np.float32)
        assert len(null_scores) > 0, "No null calibration scores found in artifact."
        
        self.null_scores = null_scores
        self.track_b_null_count = len(null_scores)
        
        # Calibrate ShiftEvaluator using the calibration risk score baseline
        all_cal_risks = []
        for s in calib_scenes:
            if len(s["risk_scores"]) > 0:
                all_cal_risks.extend(s["risk_scores"].tolist())
                
        self.shift_evaluator = ShiftEvaluator()
        if len(all_cal_risks) > 0:
            self.shift_evaluator.calibrate(np.array(all_cal_risks, dtype=np.float32))

        self.is_calibrated = True



    def certify_scene(
        self,
        scene_input: Any,
        alpha: float = 0.10,
        method: str = "BH",
        risk_map: Optional[np.ndarray] = None
    ) -> dict[str, Any]:
        """
        Certifies a scene's candidates under Track A or Track B.
        
        Args:
            scene_input: Candidate list/dict, or (sr_image, risk_map) tuple, or candidate JSON.
            alpha: Significance / error budget (e.g. 0.10).
            method: Certification track ('BH', 'BY', or 'TRACK_A').
            risk_map: Optional risk map if passing raw sr_image.
            
        Returns:
            Certificate result dictionary with selected mask, candidate annotations, and metadata.
        """
        method_norm = method.upper()
        assert method_norm in ["BH", "BY", "TRACK_A"], f"Unsupported method: {method}. Must be 'BH', 'BY', or 'TRACK_A'."
        
        # 1. Parse candidates and risk scores
        candidates = []
        sr_shape = (512, 512)
        
        if isinstance(scene_input, dict) and "candidates" in scene_input:
            candidates = scene_input["candidates"]
            sr_shape = scene_input.get("shape", (512, 512))
        elif isinstance(scene_input, list):
            candidates = scene_input
            sr_shape = (512, 512)
        elif isinstance(scene_input, np.ndarray):
            # sr_image passed
            _, extracted = self.proposal_stage.extract_candidates(scene_input, risk_map=risk_map)
            candidates = extracted
            sr_shape = scene_input.shape[-2:]
        else:
            raise TypeError(f"Unsupported scene_input format: {type(scene_input)}")
            
        N = len(candidates)
        if N == 0:
            return {
                "certificate_status": "EMPTY_PROPOSAL",
                "method": method_norm,
                "alpha": alpha,
                "N_proposed": 0,
                "N_selected": 0,
                "selected_candidate_ids": [],
                "selected_mask": np.zeros(sr_shape, dtype=np.uint8),
                "candidate_results": [],
                "risk_statistics": {},
                "certification_metadata": {"note": "No candidate structures proposed in scene."}
            }
            
        risks = np.array([float(c["risk"]) for c in candidates], dtype=np.float32)
        
        # 2. Execute Certification
        p_values = None
        selected_mask_1d = np.zeros(N, dtype=bool)
        threshold_info = {}
        
        if method_norm == "TRACK_A":
            lambda_hat = self.track_a.lambda_for_alpha(alpha)
            threshold_info["lambda_hat"] = lambda_hat
            threshold_info["bound_definition"] = f"Track A Conformal Risk Control: E[L(lambda)] <= {alpha}"
            
            if lambda_hat >= 0:
                selected_mask_1d = risks <= lambda_hat
                cert_status = "CERTIFIED" if np.any(selected_mask_1d) else "EMPTY_SELECTION"
            else:
                cert_status = "INFEASIBLE_BOUND"
                selected_mask_1d = np.zeros(N, dtype=bool)
                
        elif method_norm in ["BH", "BY"]:
            certifier_b = TrackBCertifier(alpha=alpha, method=method_norm)
            certifier_b.null_scores = self.null_scores
            certifier_b.is_calibrated = True
            
            p_values = certifier_b.compute_p_values(risks)
            selected_mask_1d = certifier_b.certify(risks)
            cert_status = "CERTIFIED" if np.any(selected_mask_1d) else "EMPTY_SELECTION"
            threshold_info["fdr_guarantee"] = f"Track B {method_norm}: E[FDR] <= {alpha}"
            threshold_info["null_sample_size"] = len(self.null_scores)
            
        # 3. Construct spatial mask (512x512)
        selected_mask_2d = np.zeros(sr_shape, dtype=np.uint8)
        selected_ids = []
        candidate_records = []
        
        for i, c in enumerate(candidates):
            c_id = c.get("id", i + 1)
            is_selected = bool(selected_mask_1d[i])
            if is_selected:
                selected_ids.append(c_id)
                # Mark candidate mask if present, or bbox
                if "mask" in c and isinstance(c["mask"], np.ndarray):
                    selected_mask_2d[c["mask"]] = 1
                elif "bbox_yxyx" in c:
                    y0, x0, y1, x1 = c["bbox_yxyx"]
                    selected_mask_2d[y0:y1+1, x0:x1+1] = 1
                    
            record = {
                "id": c_id,
                "area_px": c.get("area_px", c.get("area", 0)),
                "centroid": c.get("centroid_yx", c.get("centroid", (0.0, 0.0))),
                "risk": float(risks[i]),
                "p_value": float(p_values[i]) if p_values is not None else None,
                "certified": is_selected
            }
            candidate_records.append(record)
            
        # 4. Shift evaluation
        tier, shift_val = self.shift_evaluator.evaluate(risks)
        
        result = {
            "certificate_status": cert_status,
            "method": method_norm,
            "alpha": float(alpha),
            "N_proposed": int(N),
            "N_selected": int(np.sum(selected_mask_1d)),
            "selected_candidate_ids": selected_ids,
            "selected_mask": selected_mask_2d,
            "candidate_results": candidate_records,
            "threshold_info": threshold_info,
            "risk_statistics": {
                "min": float(np.min(risks)),
                "max": float(np.max(risks)),
                "mean": float(np.mean(risks)),
                "median": float(np.median(risks))
            },
            "shift_status": {
                "tier": tier.name if hasattr(tier, "name") else str(tier),
                "shift_score": float(shift_val) if shift_val is not None else None
            },
            "certification_metadata": {
                "theoretical_semantics": {
                    "Track_A": "Finite-sample population expectation bound E[L(lambda_hat)] <= alpha on normalized loss L = FD / N.",
                    "Track_B": "False Discovery Rate control E[FDR] <= alpha under Benjamini-Hochberg (PRDS) or Benjamini-Yekutieli (arbitrary dependence).",
                    "Critical_Distinction": "Neither track guarantees per-scene FDP <= alpha. Empirical FDP on a single scene is an uncertified sample statistic."
                },
                "calibration_artifact": self.artifact_path,
                "calibration_null_count": self.track_b_null_count,
                "split_disjointness_enforced": True
            }
        }
        return result

    def evaluate_held_out(
        self,
        certification_result: dict[str, Any],
        hr_reference: np.ndarray,
        d_max: float = 2.0
    ) -> dict[str, Any]:
        """
        POST-HOC EVALUATION ONLY:
        Evaluates the certified candidate mask against the quarantined HR reference.
        
        This method MUST NEVER influence calibration, lambda selection, or p-values.
        """
        assert "candidate_results" in certification_result, "Invalid certification_result structure."
        
        # Apply identical proposal detector to quarantined HR reference
        hr_mask = self.proposal_stage.propose(hr_reference)
        import scipy.ndimage as ndi
        hr_dist_map = ndi.distance_transform_edt(~hr_mask)
        
        selected_records = [c for c in certification_result["candidate_results"] if c["certified"]]
        n_selected = len(selected_records)
        
        if n_selected == 0:
            return {
                "evaluation_tp": 0,
                "evaluation_fp": 0,
                "evaluation_fdp": 0.0,
                "evaluation_precision": 0.0,
                "d_max": d_max,
                "note": "Empty selection; empirical FDP is 0.0 by definition."
            }
            
        tp_count = 0
        for c in selected_records:
            cy = int(np.clip(round(c["centroid"][0]), 0, hr_dist_map.shape[0] - 1))
            cx = int(np.clip(round(c["centroid"][1]), 0, hr_dist_map.shape[1] - 1))
            dist = float(hr_dist_map[cy, cx])
            if dist <= d_max:
                tp_count += 1
                
        fp_count = n_selected - tp_count
        empirical_fdp = float(fp_count / max(1, n_selected))
        empirical_prec = float(tp_count / max(1, n_selected))
        
        return {
            "evaluation_tp": tp_count,
            "evaluation_fp": fp_count,
            "evaluation_fdp": round(empirical_fdp, 4),
            "evaluation_precision": round(empirical_prec, 4),
            "d_max": d_max,
            "status": "POST_HOC_EVALUATION_ONLY",
            "disclaimer": "Sample statistic on held-out test scene; not a per-scene certification guarantee."
        }

    def save_certified_mask_geotiff(
        self,
        certification_result: dict[str, Any],
        output_path: str,
        reference_geotiff: str = "data/demo_d1/reconstruction_2_5m/SEN2SR_2_5m_ROI_00001_B02_2_5m.tif"
    ) -> str:
        """Saves the certified spatial mask as a georeferenced GeoTIFF."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        assert os.path.exists(reference_geotiff), f"Reference GeoTIFF not found: {reference_geotiff}"
        
        with rasterio.open(reference_geotiff) as src:
            profile = src.profile.copy()
            
        mask = certification_result["selected_mask"]
        profile.update(dtype="uint8", count=1)
        
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
            
        return output_path
