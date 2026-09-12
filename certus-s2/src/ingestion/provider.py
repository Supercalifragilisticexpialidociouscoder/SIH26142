import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import rasterio
import numpy as np

class L2AProvider(ABC):
    """
    Provider-independent contract for acquiring Sentinel-2 L2A scenes.
    """
    @abstractmethod
    def acquire_scene(self, bbox: Tuple[float, float, float, float], datetime: str) -> Dict[str, Any]:
        """
        Acquires a single scene within the given bounding box and time range.
        
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            datetime: e.g. '2023-01-01/2023-01-31'
            
        Returns:
            A dictionary containing the validated scene data and metadata:
            {
                'bands': {
                    'B02': np.ndarray, # 10m Blue
                    'B03': np.ndarray, # 10m Green
                    'B04': np.ndarray, # 10m Red
                    'B08': np.ndarray, # 10m NIR
                },
                'scl': np.ndarray,     # Scene Classification Layer (20m -> 10m upsampled)
                'metadata': {
                    'crs': str,
                    'resolution': float,
                    'dtype': str,
                    'nodata': float,
                    'spatial_extent': Tuple,
                    'provenance': str
                }
            }
        """
        pass

class CDSEProvider(L2AProvider):
    """
    Copernicus Data Space Ecosystem (CDSE) provider.
    Uses CDSEClient with Sentinel Hub Process API.
    """
    def __init__(self, client_id: str = None, client_secret: str = None):
        from src.data.cdse_client import CDSEClient
        self.client = CDSEClient(client_id=client_id, client_secret=client_secret)

    def acquire_scene(self, bbox: Tuple[float, float, float, float], datetime: str) -> Dict[str, Any]:
        if not self.client.is_configured:
            raise RuntimeError("CDSE credentials not found. Please set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in .env.")
        
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        
        # Parse datetime interval if provided
        time_from = "2024-05-01T00:00:00Z"
        time_to = "2024-09-01T00:00:00Z"
        if "/" in datetime:
            parts = datetime.split("/")
            time_from = parts[0] if parts[0].endswith("Z") else f"{parts[0]}T00:00:00Z"
            time_to = parts[1] if parts[1].endswith("Z") else f"{parts[1]}T00:00:00Z"

        patch = self.client.fetch_sentinel2_l2a_patch(
            lat=center_lat,
            lon=center_lon,
            time_from=time_from,
            time_to=time_to
        )
        
        l2a = patch["l2a"]
        return {
            "bands": {
                "B02": l2a[0],
                "B03": l2a[1],
                "B04": l2a[2],
                "B08": l2a[3]
            },
            "metadata": patch["metadata"]
        }

