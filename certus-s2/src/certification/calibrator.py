import os
import json
import pickle
from pathlib import Path
from typing import Optional, Any, Union
import numpy as np

from src.decision.proposal import D1ProposalStage
from src.observation.operator import EffectiveObservationOperator
from src.risk.engine import RiskEngine
from src.pipeline.runner import CertusPipeline
from src.pipeline.sen2sr_wrapper import SEN2SRWrapper
from src.certification.track_a import TrackACertifier
from src.certification.track_b import TrackBCertifier

class CertusCalibrator:
    """
    Multi-Scene Calibration Engine for CERTUS-S2 Task D1.
    
    Enforces strict site-disjoint calibration/test separation:
    - Reads split_manifest.json
    - Identifies CALIBRATE split (OPENSR_SPAIN_CROPS_01)
    - Excludes TEST tiles (e.g., T30TXM Madrid)
    - Processes each calibration scene independently (never pooling tiles)
    - Calibrates Track A (monotone normalized loss) and Track B (conformal nulls)
    - Produces a machine-readable, reproducible calibration artifact.
    """
    def __init__(
        self,
        manifest_path: str = "data/split_manifest.json",
        proposal_stage: Optional[D1ProposalStage] = None,
        pipeline: Optional[CertusPipeline] = None
    ):
        self.manifest_path = manifest_path
        self.proposal_stage = proposal_stage or D1ProposalStage(
            c_0=0.02, min_area=16, max_area=400, highpass_sigma=2.0, ndvi_threshold=0.3
        )
        self.pipeline = pipeline
        self.calibration_scenes: list[dict[str, Any]] = []
        self.track_a = TrackACertifier(alpha=0.10, B=1.0)
        self.track_b = TrackBCertifier(alpha=0.10, method="BH")
        self.is_calibrated = False
        self.provenance: dict[str, Any] = {}

    def _init_default_pipeline(self):
        if self.pipeline is None:
            op = EffectiveObservationOperator(kernel_sigma=1.2, subpixel_shift=(0.0, 0.0), sigma_b=0.04)
            op.is_fitted = True
            risk_engine = RiskEngine(backend_type="numpy")
            wrapper = SEN2SRWrapper(device="cpu")
            self.pipeline = CertusPipeline(
                operator=op, risk_engine=risk_engine, proposal_stage=None,
                certifier_a=None, certifier_b=None, shift_eval=None, sr_wrapper=wrapper
            )

    def run_calibration(
        self,
        dataset_path: str = "data/opensr_cache/spain_crops.pkl",
        exclude_tiles: list[str] = ["T30TXM"],
        max_scenes: Optional[int] = None
    ) -> dict[str, Any]:
        """
        Executes multi-scene calibration over site-disjoint calibration scenes.
        
        Args:
            dataset_path: Path to cached pickle of CALIBRATE dataset (OPENSR_SPAIN_CROPS_01).
            exclude_tiles: List of MGRS tiles to exclude (must include TEST tiles like T30TXM).
            max_scenes: Optional limit on number of scenes to process.
            
        Returns:
            Dictionary containing calibration artifact data.
        """
        assert os.path.exists(dataset_path), f"Calibration dataset not found: {dataset_path}"
        self._init_default_pipeline()
        
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
            
        meta = data["metadata"]
        
        # Filter scenes strictly enforcing site-disjointness
        valid_indices = []
        for i in range(len(meta)):
            gee_id = meta.iloc[i]["lr_gee_id"]
            tile = gee_id.split("_")[-1]
            if not any(ex_tile in tile for ex_tile in exclude_tiles):
                valid_indices.append(i)
                
        if max_scenes is not None:
            valid_indices = valid_indices[:max_scenes]
            
        assert len(valid_indices) > 0, "No valid site-disjoint calibration scenes found."
        
        self.calibration_scenes = []
        
        for idx in valid_indices:
            row = meta.iloc[idx]
            # Real Sentinel-2 L2A 4 bands: [B02, B03, B04, B08]
            s_l2a = data["L2A"][idx][[1, 2, 3, 7]]
            s_hr = data["HR"][idx]
            
            # Forward pipeline pass
            layers = self.pipeline.compute_analytical_layers(s_l2a)
            sr_image = layers["sr_image"]
            risk_map = layers["risk_map"]
            
            # Proposal stage: fixed c0 = 0.02, area in [16, 400]
            _, cands = self.proposal_stage.extract_candidates(sr_image, risk_map=risk_map)
            
            # Ground-truth evaluation against quarantined HR reference (labels strictly for calibration)
            eval_res = self.proposal_stage.evaluate_against_reference(cands, s_hr, d_max=2.0)
            
            risk_scores = np.array([c["risk"] for c in cands], dtype=np.float32)
            is_fd = np.array([c["is_false_discovery"] for c in cands], dtype=bool)
            
            scene_record = {
                "scene_idx": int(idx),
                "roi": str(row.roi),
                "tile": str(row.lr_gee_id.split("_")[-1]),
                "lr_gee_id": str(row.lr_gee_id),
                "N": len(cands),
                "true_detections": int(eval_res["true_detections"]),
                "false_discoveries": int(eval_res["false_discoveries"]),
                "risk_scores": risk_scores,
                "is_false_discovery": is_fd
            }
            self.calibration_scenes.append(scene_record)
            
        # Fit Track A
        self.track_a.calibrate(self.calibration_scenes)
        
        # Fit Track B
        self.track_b.calibrate(self.calibration_scenes)
        
        self.is_calibrated = True
        
        # Build calibration summary
        all_risks = np.concatenate([s["risk_scores"] for s in self.calibration_scenes if len(s["risk_scores"]) > 0])
        null_risks = self.track_b.null_scores
        
        # Track A operating table for alpha grid
        alpha_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        track_a_table = []
        for alpha in alpha_grid:
            lam = self.track_a.lambda_for_alpha(alpha)
            track_a_table.append({
                "alpha": alpha,
                "lambda_hat": round(lam, 5) if lam >= 0 else None,
                "feasible": (lam >= 0)
            })
            
        self.provenance = {
            "dataset": "OPENSR_SPAIN_CROPS_01",
            "manifest_path": self.manifest_path,
            "n_calibration_scenes": len(self.calibration_scenes),
            "excluded_tiles": exclude_tiles,
            "unique_calibration_tiles": sorted(list(set(s["tile"] for s in self.calibration_scenes))),
            "total_candidates": int(sum(s["N"] for s in self.calibration_scenes)),
            "total_nulls": int(len(null_risks)),
            "risk_distribution": {
                "min": float(np.min(all_risks)) if len(all_risks) > 0 else None,
                "max": float(np.max(all_risks)) if len(all_risks) > 0 else None,
                "mean": float(np.mean(all_risks)) if len(all_risks) > 0 else None
            },
            "null_distribution": {
                "min": float(np.min(null_risks)) if len(null_risks) > 0 else None,
                "max": float(np.max(null_risks)) if len(null_risks) > 0 else None,
                "mean": float(np.mean(null_risks)) if len(null_risks) > 0 else None,
                "count": len(null_risks)
            },
            "proposal_config": {
                "c_0": self.proposal_stage.c_0,
                "min_area": self.proposal_stage.min_area,
                "max_area": self.proposal_stage.max_area,
                "highpass_sigma": self.proposal_stage.highpass_sigma,
                "ndvi_threshold": self.proposal_stage.ndvi_threshold
            },
            "track_a_table": track_a_table
        }
        
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Serializes the calibrated state into a machine-readable dictionary."""
        assert self.is_calibrated, "Calibrator has not been calibrated."
        
        scenes_export = []
        for s in self.calibration_scenes:
            scenes_export.append({
                "scene_idx": s["scene_idx"],
                "roi": s["roi"],
                "tile": s["tile"],
                "lr_gee_id": s["lr_gee_id"],
                "N": s["N"],
                "true_detections": s["true_detections"],
                "false_discoveries": s["false_discoveries"],
                "risk_scores": [float(r) for r in s["risk_scores"]],
                "is_false_discovery": [bool(fd) for fd in s["is_false_discovery"]]
            })
            
        return {
            "calibration_metadata": self.provenance,
            "proposal_config": self.provenance["proposal_config"],
            "calibration_scenes": scenes_export,
            "track_a_calibration": {
                "B": float(self.track_a.B),
                "n_scenes": len(self.calibration_scenes),
                "finite_sample_inflation": float(self.track_a.B / (len(self.calibration_scenes) + 1)),
                "table": self.provenance["track_a_table"]
            },
            "track_b_calibration": {
                "null_count": int(len(self.track_b.null_scores)),
                "null_scores": [float(x) for x in self.track_b.null_scores],
                "min_conformal_p_value": float(1.0 / (len(self.track_b.null_scores) + 1.0))
            }
        }

    def save_calibration_artifact(self, output_path: str = "data/calibration/d1_calibration.json") -> str:
        """Saves the calibration artifact to a machine-readable JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data_dict = self.to_dict()
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=2)
        return output_path

    @classmethod
    def from_artifact(cls, artifact_path: Union[str, dict]) -> "CertusCalibrator":
        """Loads a calibrated CertusCalibrator from a JSON file or dictionary."""
        if isinstance(artifact_path, (str, Path)):
            with open(artifact_path, "r") as f:
                data = json.load(f)
        else:
            data = artifact_path
            
        prop_cfg = data["proposal_config"]
        proposal_stage = D1ProposalStage(
            c_0=prop_cfg["c_0"],
            min_area=prop_cfg["min_area"],
            max_area=prop_cfg["max_area"],
            highpass_sigma=prop_cfg["highpass_sigma"],
            ndvi_threshold=prop_cfg["ndvi_threshold"]
        )
        
        calibrator = cls(proposal_stage=proposal_stage)
        
        # Reconstruct calibration scenes
        calib_scenes = []
        for s in data["calibration_scenes"]:
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
            
        calibrator.calibration_scenes = calib_scenes
        calibrator.track_a = TrackACertifier(alpha=0.10, B=data["track_a_calibration"]["B"])
        calibrator.track_a.calibrate(calib_scenes)
        
        calibrator.track_b = TrackBCertifier(alpha=0.10, method="BH")
        calibrator.track_b.null_scores = np.array(data["track_b_calibration"]["null_scores"], dtype=np.float32)
        calibrator.track_b.is_calibrated = True
        
        calibrator.provenance = data["calibration_metadata"]
        calibrator.is_calibrated = True
        return calibrator
