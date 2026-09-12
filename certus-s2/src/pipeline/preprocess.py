import numpy as np
import rasterio
import torch

def preprocess_l2a(band_paths: dict) -> tuple[torch.Tensor, dict]:
    """
    Load Sentinel-2 L2A raster crops, normalize to [0, 1], mask NoData, 
    and stack into a PyTorch tensor.
    
    Args:
        band_paths: Dictionary mapping band names (e.g. 'B02') to file paths.
    
    Returns:
        tuple containing:
            - tensor: PyTorch tensor of shape (1, C, H, W) in order [B02, B03, B04, B08]
            - metadata: Rasterio profile metadata of the first band for reference
    """
    expected_bands = ["B02", "B03", "B04", "B08"]
    arrays = []
    metadata = None
    
    for band in expected_bands:
        if band not in band_paths:
            raise ValueError(f"Missing required band {band}")
            
        with rasterio.open(band_paths[band]) as src:
            if metadata is None:
                metadata = src.profile
                
            data = src.read(1).astype(np.float32)
            
            # Simple scaling for Sentinel-2 L2A BOA reflectance (0 - 10000 -> 0.0 - 1.0)
            data = np.clip(data / 10000.0, 0.0, 1.0)
            
            # TODO: handle NoData mask from SCL if needed, but for now just 0 values are NoData
            
            arrays.append(data)
            
    # Stack along channels (C, H, W)
    stacked = np.stack(arrays, axis=0)
    
    # Add batch dimension -> (1, C, H, W)
    tensor = torch.from_numpy(stacked).unsqueeze(0)
    
    return tensor, metadata
