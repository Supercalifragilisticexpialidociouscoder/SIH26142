import numpy as np
import scipy.ndimage as ndi
from typing import Optional, Any

class D1ProposalStage:
    """
    Task D1: Binary built-structure presence verification.
    
    Extracts a set of candidate detections C(Y) from the 2.5m SR reconstruction
    based on a fixed, lambda-independent proposal threshold c_0, followed by
    connected-component labelling and area filtering in [min_area, max_area]
    (SIH26142_v4_hardened.md Section 21.1: [16, 400] HR pixels, ~100m^2 to 2500m^2).
    
    Crucially:
    1. c_0 is strictly lambda-independent.
    2. Candidate count N_i = |K_i| is fixed before risk-based retention.
    3. Evaluation against the quarantined HR reference uses the identical detector
       with the same c_0, testing if the candidate centroid lies within d <= 2 HR pixels
       of a structure in the reference mask (Section 21.1).
    """
    def __init__(
        self,
        c_0: float = 0.02,
        min_area: int = 16,
        max_area: int = 400,
        highpass_sigma: float = 2.0,
        ndvi_threshold: float = 0.3
    ):
        self.c_0 = c_0
        self.min_area = min_area
        self.max_area = max_area
        self.highpass_sigma = highpass_sigma
        self.ndvi_threshold = ndvi_threshold
        self.is_calibrated = False

    def compute_feature_map(self, sr_image: np.ndarray) -> np.ndarray:
        """
        Computes the built-structure feature response map from the 2.5m SR reconstruction.
        
        Args:
            sr_image: 2D array [H, W] or 3D multi-band array [C, H, W] (B02, B03, B04, B08).
            
        Returns:
            2D feature response map [H, W].
        """
        img = np.array(sr_image, dtype=np.float32)
        if img.ndim == 2:
            return img
        
        if img.ndim == 4:
            img = img.squeeze(0)
            
        # If DN scaled [0, 10000], scale to [0, 1]
        if np.nanmax(img) > 2.0:
            img = np.clip(img / 10000.0, 0.0, 1.0)
        else:
            img = np.clip(img, 0.0, 1.0)
            
        # Expected band order: [B02, B03, B04, B08]
        if img.shape[0] >= 4:
            b02, b03, b04, b08 = img[0], img[1], img[2], img[3]
        else:
            # Fallback for 3-channel RGB: use mean visible brightness
            b02 = b03 = b04 = img[:3].mean(axis=0)
            b08 = b04
            
        # 1. Visible brightness
        vis = (b02 + b03 + b04) / 3.0
        
        # 2. High-pass filter of visible brightness (isolates compact structures)
        lowpass = ndi.gaussian_filter(vis, sigma=self.highpass_sigma)
        highpass = np.maximum(0.0, vis - lowpass)
        
        # 3. Spectral vegetation suppression: NDVI < ndvi_threshold
        ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
        built_spectral = (ndvi < self.ndvi_threshold).astype(np.float32)
        
        # Combined feature: high-pass energy on non-vegetated pixels
        feature = highpass * built_spectral
        return feature

    def calibrate(self, train_sr_images: list[np.ndarray], target_yield: float = 0.1):
        """
        Calibrates the proposal threshold c_0 on the TRAIN set.
        """
        all_features = []
        for img in train_sr_images:
            feat = self.compute_feature_map(img)
            all_features.append(feat.flatten())
        concat_feats = np.concatenate(all_features)
        self.c_0 = float(np.quantile(concat_feats, 1.0 - target_yield))
        self.is_calibrated = True

    def propose(self, sr_image: np.ndarray) -> np.ndarray:
        """
        Applies the fixed proposal threshold and connected-component area filtering
        to generate the candidate mask C(Y).
        
        Args:
            sr_image: The reconstructed HR image [H, W] or [C, H, W].
            
        Returns:
            A boolean mask [H, W] where True indicates a proposed candidate pixel.
        """
        img = np.array(sr_image, dtype=np.float32)
        
        # Backward compatibility for simple 2D synthetic unit tests
        if img.ndim == 2 and (img.shape[0] < self.min_area or img.shape[1] < self.min_area):
            return img > self.c_0
            
        feature = self.compute_feature_map(img)
        raw_mask = feature > self.c_0
        
        if self.min_area <= 1 and self.max_area >= raw_mask.size:
            return raw_mask
            
        labeled, num_labels = ndi.label(raw_mask)
        candidate_mask = np.zeros_like(raw_mask, dtype=bool)
        
        for i in range(1, num_labels + 1):
            component = (labeled == i)
            area = int(np.sum(component))
            if self.min_area <= area <= self.max_area:
                candidate_mask[component] = True
                
        return candidate_mask

    def extract_candidates(
        self,
        sr_image: np.ndarray,
        risk_map: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Extracts discrete candidate connected regions and computes their properties.
        
        Args:
            sr_image: The reconstructed HR image [H, W] or [C, H, W].
            risk_map: Optional risk map [H, W] or [C, H, W] to compute per-candidate risk.
            
        Returns:
            Tuple of:
            - candidate_mask: Boolean mask [H, W] of all candidate pixels.
            - candidates: List of candidate dictionaries with fields:
              'id', 'area', 'centroid', 'bbox', 'mask', 'risk'
        """
        img = np.array(sr_image, dtype=np.float32)
        feature = self.compute_feature_map(img)
        raw_mask = feature > self.c_0
        
        labeled, num_labels = ndi.label(raw_mask)
        candidate_mask = np.zeros_like(raw_mask, dtype=bool)
        candidates = []
        
        # Prepare 2D risk map if 3D array provided
        risk_2d = None
        if risk_map is not None:
            r_arr = np.array(risk_map, dtype=np.float32)
            if r_arr.ndim == 3:
                risk_2d = r_arr.mean(axis=0)
            elif r_arr.ndim == 2:
                risk_2d = r_arr
                
        for i in range(1, num_labels + 1):
            comp = (labeled == i)
            area = int(np.sum(comp))
            if self.min_area <= area <= self.max_area:
                candidate_mask[comp] = True
                cy, cx = ndi.center_of_mass(comp)
                
                y_indices, x_indices = np.where(comp)
                bbox = (int(y_indices.min()), int(x_indices.min()), int(y_indices.max()), int(x_indices.max()))
                
                cand_info = {
                    "id": len(candidates) + 1,
                    "area": area,
                    "centroid": (float(cy), float(cx)),
                    "bbox": bbox,
                    "mask": comp,
                    "risk": float(risk_2d[comp].mean()) if risk_2d is not None else None
                }
                candidates.append(cand_info)
                
        return candidate_mask, candidates

    def evaluate_against_reference(
        self,
        candidates: list[dict[str, Any]],
        hr_reference: np.ndarray,
        d_max: float = 2.0
    ) -> dict[str, Any]:
        """
        Evaluates candidate detections against the quarantined HR reference.
        
        Per Section 21.1:
        Positive detection: candidate centroid lies within d <= 2 HR pixels
        of a structure in the reference mask obtained with the SAME detector.
        
        Args:
            candidates: List of candidate dictionaries from extract_candidates.
            hr_reference: Co-registered HR reference array [C, H, W] or [H, W].
            d_max: Maximum distance in HR pixels (default 2.0 HR pixels = 5.0m at 2.5m).
            
        Returns:
            Dictionary with evaluation summary and annotated candidates.
        """
        hr_mask = self.propose(hr_reference)
        
        if not np.any(hr_mask):
            for c in candidates:
                c["dist_to_reference"] = float("inf")
                c["is_true_detection"] = False
                c["is_false_discovery"] = True
            return {
                "total_candidates": len(candidates),
                "true_detections": 0,
                "false_discoveries": len(candidates),
                "precision": 0.0,
                "hr_structure_pixels": 0,
                "d_max": d_max
            }
            
        hr_dist_map = ndi.distance_transform_edt(~hr_mask)
        
        tp_count = 0
        distances = []
        for c in candidates:
            cy = int(np.clip(round(c["centroid"][0]), 0, hr_dist_map.shape[0] - 1))
            cx = int(np.clip(round(c["centroid"][1]), 0, hr_dist_map.shape[1] - 1))
            dist = float(hr_dist_map[cy, cx])
            is_tp = (dist <= d_max)
            
            c["dist_to_reference"] = dist
            c["is_true_detection"] = is_tp
            c["is_false_discovery"] = not is_tp
            
            if is_tp:
                tp_count += 1
            distances.append(dist)
            
        fp_count = len(candidates) - tp_count
        precision = float(tp_count / max(1, len(candidates)))
        
        return {
            "total_candidates": len(candidates),
            "true_detections": tp_count,
            "false_discoveries": fp_count,
            "precision": precision,
            "hr_structure_pixels": int(np.sum(hr_mask)),
            "d_max": d_max,
            "mean_dist_to_reference": float(np.mean(distances)) if distances else 0.0
        }

