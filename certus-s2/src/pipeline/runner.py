import numpy as np
import logging
from typing import Optional, Any
from src.shift.abstention import AbstentionTier

logger = logging.getLogger(__name__)

class BlockedInfrastructureError(Exception):
    pass

class CertusPipeline:
    """
    CERTUS-S2 Pipeline Coordinator.
    
    Orchestrates the data flow:
    L2A -> SEN2SR Backbone -> Operator -> Support -> Risk -> Proposal -> Certification -> Output
    """
    def __init__(self, operator, risk_engine, proposal_stage, certifier_a, certifier_b, shift_eval, sr_wrapper: Optional[Any] = None):
        self.operator = operator
        self.risk_engine = risk_engine
        self.proposal = proposal_stage
        self.certifier_a = certifier_a
        self.certifier_b = certifier_b
        self.shift_eval = shift_eval
        self.sr_wrapper = sr_wrapper
        
        # Loaded state is True if an sr_wrapper is supplied, False otherwise
        self._sr_backbone_loaded = (sr_wrapper is not None)

    def reconstruct_sr(self, l2a_observation: np.ndarray) -> np.ndarray:
        """
        Executes real SEN2SR super-resolution forward pass.
        
        Args:
            l2a_observation: Sentinel-2 L2A observation array [C, H, W] or [1, C, H, W].
            
        Returns:
            Super-resolved 2.5m reconstruction [C, 4*H, 4*W] in reflectance [0, 1].
        """
        if not self._sr_backbone_loaded:
            raise BlockedInfrastructureError(
                "Real SEN2SR inference is blocked because no SR backbone was provided. "
                "Per SIH26142_v4_hardened.md rules, we DO NOT fabricate or mock the forward pass."
            )
        return self._invoke_backbone(l2a_observation)

    def run_inference(self, l2a_observation: np.ndarray, 
                      sigma_ep_map: np.ndarray = None, 
                      sigma_al_map: np.ndarray = None) -> dict:
        """
        Runs the full CERTUS-S2 forward pass on a new observation.
        """
        if not self._sr_backbone_loaded:
            raise BlockedInfrastructureError(
                "Real SEN2SR inference is blocked due to infrastructure limitations (PyTorch unavailable). "
                "Per SIH26142_v4_hardened.md rules, we DO NOT fabricate or mock the forward pass."
            )
            
        # 1. SR Inference
        sr_reconstruction = self.reconstruct_sr(l2a_observation)
        
        # Ensure observation is on reflectance scale [0, 1] for residual calculation
        l2a_eval = np.array(l2a_observation, dtype=np.float32)
        if l2a_eval.ndim == 4:
            l2a_eval = l2a_eval.squeeze(0)
        if np.nanmax(l2a_eval) > 1.0:
            l2a_eval = np.clip(l2a_eval / 10000.0, 0.0, 1.0)
            
        # 2. Residual & Support
        from src.support.measurement import compute_residual, compute_support_map
        residual = compute_residual(l2a_eval, sr_reconstruction, self.operator)
        support = compute_support_map(residual, self.operator.sigma_b)
        
        # 3. Uncertainty
        # In a real run, these are extracted from the LDSR-S2 backbone or an ensemble.
        if sigma_ep_map is None:
            sigma_ep_map = np.zeros_like(support)
        if sigma_al_map is None:
            sigma_al_map = np.zeros_like(support)
            
        # 4. Risk Engine
        risk_map = self.risk_engine.compute_risk(support, sigma_ep_map, sigma_al_map, residual=residual)
        
        # 5. D1 Proposal
        candidate_mask = self.proposal.propose(sr_reconstruction)
        
        # Filter risk scores to only candidates
        if risk_map.ndim == 3 and candidate_mask.ndim == 2:
            candidate_scores = risk_map.mean(axis=0)[candidate_mask]
        else:
            candidate_scores = risk_map[candidate_mask]
        
        # 6. Shift & Abstention
        tier, shift_val = self.shift_eval.evaluate(candidate_scores)
        
        # 7. Certification
        track_a_mask = np.zeros_like(candidate_mask, dtype=bool)
        track_b_mask = np.zeros_like(candidate_mask, dtype=bool)
        
        if tier != AbstentionTier.ABSTAIN:
            track_a_res = self.certifier_a.certify(candidate_scores)
            track_b_res = self.certifier_b.certify(candidate_scores)
            
            # Map back to full spatial dimensions
            track_a_mask[candidate_mask] = track_a_res
            track_b_mask[candidate_mask] = track_b_res

        return {
            "tier": tier,
            "shift_score": shift_val,
            "reconstruction": sr_reconstruction,
            "support_map": support,
            "risk_map": risk_map,
            "track_a_certified": track_a_mask,
            "track_b_certified": track_b_mask,
            "provenance": {
                "operator_fitted": self.operator.is_fitted,
                "n_calibration_sites_track_a": len(self.certifier_a.lambda_hat) if hasattr(self.certifier_a.lambda_hat, '__len__') else 1,
                "note": "CERTUS-S2 prototype"
            }
        }
        
    def compute_analytical_layers(self, l2a_observation: np.ndarray,
                                  sigma_ep_map: Optional[np.ndarray] = None,
                                  sigma_al_map: Optional[np.ndarray] = None) -> dict[str, Any]:
        """
        Executes the forward analytical chain:
        L2A -> SEN2SR -> Observation Residual -> Measurement Support -> Uncertainty -> Risk.
        
        Does not perform candidate proposal or conformal certification.
        
        Returns a dictionary of visualization-ready NumPy arrays with consistent spatial dimensions:
        - 'sr_image': Reconstructed 2.5m super-resolution image [C, 4*H, 4*W]
        - 'observation_residual': Measurement residual magnitude [C, 4*H, 4*W] (broadcasted from 10m)
        - 'observation_residual_10m': Measurement residual magnitude on native 10m grid [C, H, W]
        - 'measurement_support': Support map s(p) [C, 4*H, 4*W] in [0, 1]
        - 'epistemic_uncertainty': Model uncertainty map [C, 4*H, 4*W]
        - 'aleatoric_uncertainty': Data noise uncertainty map [C, 4*H, 4*W]
        - 'risk_map': Decision risk map r(p) [C, 4*H, 4*W] in [0, 1]
        - 'provenance': Metadata documenting terms, sources, and scaffold status.
        """
        if not self._sr_backbone_loaded:
            raise BlockedInfrastructureError(
                "Real SEN2SR inference is blocked because no SR backbone was provided. "
                "Per SIH26142_v4_hardened.md rules, we DO NOT fabricate or mock the forward pass."
            )
            
        # 1. Real SEN2SR Super-Resolution forward pass
        sr_reconstruction = self.reconstruct_sr(l2a_observation)
        
        # Ensure observation is on surface reflectance scale [0, 1]
        l2a_eval = np.array(l2a_observation, dtype=np.float32)
        if l2a_eval.ndim == 4:
            l2a_eval = l2a_eval.squeeze(0)
        if np.nanmax(l2a_eval) > 1.0:
            l2a_eval = np.clip(l2a_eval / 10000.0, 0.0, 1.0)
            
        # 2. Observation Operator Residual e = |y - \hat{D}(\hat{x})| on 10m grid
        from src.support.measurement import compute_residual, compute_support_map
        residual_10m = compute_residual(l2a_eval, sr_reconstruction, self.operator)
        
        # Broadcast residual to 2.5m grid for consistent spatial visualization dimensions
        residual_2_5m = np.repeat(residual_10m, 4, axis=-1)
        residual_2_5m = np.repeat(residual_2_5m, 4, axis=-2)
        
        # 3. Measurement Support Map s(p) on 2.5m grid
        support = compute_support_map(residual_10m, self.operator.sigma_b, scale_factor=4)
        
        # 4. Uncertainty maps
        # NOTE: If unprovided, these default to the current repository scaffold/placeholder (zeros).
        # We explicitly document whether uncertainties were supplied or are placeholder scaffolds.
        uncertainty_is_scaffold = False
        if sigma_ep_map is None:
            sigma_ep_map = np.zeros_like(support, dtype=np.float32)
            uncertainty_is_scaffold = True
        else:
            sigma_ep_map = np.array(sigma_ep_map, dtype=np.float32)
            
        if sigma_al_map is None:
            sigma_al_map = np.zeros_like(support, dtype=np.float32)
            uncertainty_is_scaffold = True
        else:
            sigma_al_map = np.array(sigma_al_map, dtype=np.float32)
            
        # 5. Risk Map r(p) on 2.5m grid
        risk_map = self.risk_engine.compute_risk(support, sigma_ep_map, sigma_al_map, residual=residual_10m)
        
        return {
            "sr_image": sr_reconstruction,
            "observation_residual": residual_2_5m,
            "observation_residual_10m": residual_10m,
            "measurement_support": support,
            "epistemic_uncertainty": sigma_ep_map,
            "aleatoric_uncertainty": sigma_al_map,
            "risk_map": risk_map,
            "provenance": {
                "sr_backbone": "SEN2SR (CNNSR + HardConstraint + predict_large)",
                "operator_fitted": self.operator.is_fitted,
                "sigma_b": self.operator.sigma_b,
                "uncertainty_status": "placeholder_scaffold (zeros)" if uncertainty_is_scaffold else "provided",
                "spatial_scale": "4x (10m -> 2.5m)",
                "dimensions_2_5m": list(sr_reconstruction.shape),
                "dimensions_10m": list(residual_10m.shape)
            }
        }

    def _invoke_backbone(self, x: np.ndarray) -> np.ndarray:
        """
        Invokes the plugged-in SEN2SR super-resolution wrapper.
        """
        if self.sr_wrapper is None:
            raise BlockedInfrastructureError("SEN2SR backbone is not loaded.")
            
        if hasattr(self.sr_wrapper, "predict_numpy"):
            return self.sr_wrapper.predict_numpy(x)
        elif hasattr(self.sr_wrapper, "predict"):
            import torch
            if isinstance(x, np.ndarray):
                tensor = torch.from_numpy(x).float()
                if tensor.ndim == 3:
                    tensor = tensor.unsqueeze(0)
                with torch.no_grad():
                    out = self.sr_wrapper.predict(tensor)
                return out.squeeze(0).cpu().numpy().astype(np.float32)
            else:
                with torch.no_grad():
                    out = self.sr_wrapper.predict(x)
                return out.squeeze(0).cpu().numpy().astype(np.float32)
        elif callable(self.sr_wrapper):
            return self.sr_wrapper(x)
        else:
            raise TypeError(f"Unsupported sr_wrapper type: {type(self.sr_wrapper)}")


