import os
import ssl
import numpy as np
import torch
import safetensors.torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from sen2sr.models.opensr_baseline.cnn import CNNSR
from sen2sr.models.tricks import HardConstraint
from sen2sr.nonreference import srmodel
from sen2sr.utils import predict_large
from typing import Optional

# Ensure SSL certificates on macOS do not block HuggingFace hub downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class SEN2SRWrapper:
    """
    Production wrapper for SEN2SR 4x super-resolution model.
    Uses real CNNSR backbone with HardConstraint and predict_large chunking.
    """
    def __init__(self, device: str = "cpu", model_id: str = "WEO-SAS/sen2sr"):
        self.device = device
        self.model_id = model_id
        self.model = self._load_model()
        
    def _load_model(self):
        print(f"Downloading/loading SEN2SR weights from {self.model_id}...")
        model_path = hf_hub_download(repo_id=self.model_id, filename="model.safetensor")
        hc_path = hf_hub_download(repo_id=self.model_id, filename="hard_constraint.safetensor")
        
        # Instantiate base CNN
        sr_model = CNNSR(4, 4, 24, 4, True, False, 6)
        sr_model.load_state_dict(safetensors.torch.load_file(model_path))
        sr_model = sr_model.eval().to(self.device)
        for param in sr_model.parameters():
            param.requires_grad = False
            
        # Instantiate HardConstraint
        hc_weights = safetensors.torch.load_file(hc_path)
        hard_constraint = HardConstraint(low_pass_mask=hc_weights["weights"].to(self.device), device=self.device)
        
        # Combine using nonreference srmodel wrapper
        compiled = srmodel(sr_model, hard_constraint, self.device)
        return compiled

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on an input tensor using chunking (predict_large).
        
        Args:
            x: Tensor of shape (1, C, H, W) or (C, H, W). 
               Must be normalized to [0, 1].
               
        Returns:
            Tensor of shape (1, C, 4*H, 4*W)
        """
        if x.ndim == 3:
            x_squeeze = x.to(self.device)
        else:
            x_squeeze = x.squeeze(0).to(self.device)
        
        # predict_large returns the assembled (C, 4*H, 4*W) on CPU
        out = predict_large(x_squeeze, self.model, overlap=32)
        
        return out.unsqueeze(0).to(self.device)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference directly on a NumPy array (convenience API for pipeline).
        
        Args:
            x: NumPy array of shape (C, H, W) or (1, C, H, W).
               Can be raw Sentinel-2 DNs (0..10000) or surface reflectance [0, 1].
               
        Returns:
            NumPy array of shape (C, 4*H, 4*W) with values in surface reflectance [0, 1].
        """
        x_clean = np.array(x, dtype=np.float32)
        if x_clean.ndim == 4:
            x_clean = x_clean.squeeze(0)
            
        # If values exceed 2.0, scale from Sentinel-2 DN (0..10000) to reflectance [0, 1]
        if np.nanmax(x_clean) > 2.0:
            x_clean = np.clip(x_clean / 10000.0, 0.0, 1.0)
        else:
            x_clean = np.clip(x_clean, 0.0, 1.0)
            
        x_tensor = torch.from_numpy(x_clean).unsqueeze(0).float()
        
        with torch.no_grad():
            out_tensor = self.predict(x_tensor)
            
        out_numpy = out_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        out_numpy = np.clip(out_numpy, 0.0, 1.0)
        return out_numpy

# Module-level singleton cache for efficient reuse
_GLOBAL_WRAPPER: Optional[SEN2SRWrapper] = None

def run_sen2sr_inference(l2a_observation: np.ndarray, 
                         device: str = "cpu", 
                         wrapper: Optional[SEN2SRWrapper] = None) -> np.ndarray:
    """
    Reusable functional entry point for SEN2SR inference.
    
    Args:
        l2a_observation: Sentinel-2 L2A array of shape (4, H, W) or (1, 4, H, W).
        device: Torch execution device ('cpu' or 'cuda').
        wrapper: Optional pre-instantiated SEN2SRWrapper.
        
    Returns:
        Super-resolved 2.5m reconstruction of shape (4, 4*H, 4*W).
    """
    global _GLOBAL_WRAPPER
    if wrapper is not None:
        active_wrapper = wrapper
    elif _GLOBAL_WRAPPER is not None and _GLOBAL_WRAPPER.device == device:
        active_wrapper = _GLOBAL_WRAPPER
    else:
        active_wrapper = SEN2SRWrapper(device=device)
        _GLOBAL_WRAPPER = active_wrapper
        
    return active_wrapper.predict_numpy(l2a_observation)


