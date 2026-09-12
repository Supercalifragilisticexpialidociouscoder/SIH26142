import warnings
# Ignore warnings from rasterio/STAC regarding notgeoreferenced or tags
warnings.filterwarnings("ignore")

import numpy as np
import rasterio
from rasterio.windows import Window
from pystac_client import Client
import planetary_computer
from src.ingestion.provider import L2AProvider
from typing import Dict, Any, Tuple

class PlanetaryComputerProvider(L2AProvider):
    """
    Microsoft Planetary Computer STAC provider for Sentinel-2 L2A.
    """
    def __init__(self):
        self.catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.collection = "sentinel-2-l2a"
        
    def acquire_scene(self, bbox: Tuple[float, float, float, float], datetime: str) -> Dict[str, Any]:
        """
        Queries Planetary Computer for Sentinel-2 L2A data and downloads a small crop.
        """
        # 1. Query STAC
        catalog = Client.open(self.catalog_url, modifier=planetary_computer.sign_inplace)
        search = catalog.search(
            collections=[self.collection],
            bbox=bbox,
            datetime=datetime,
            query={"eo:cloud_cover": {"lt": 20}},
            limit=1
        )
        items = list(search.items())
        
        if not items:
            raise ValueError(f"No Sentinel-2 L2A scenes found for bbox={bbox} and datetime={datetime}")
            
        item = items[0]
        
        # 2. Extract Data (B02, B03, B04, B08) at 10m
        bands_to_fetch = ["B02", "B03", "B04", "B08"]
        band_data = {}
        
        # For simplicity in this crop, we will fetch a 256x256 window 
        # offset from the top-left to avoid reading the massive full scene into memory.
        # This honors: "Acquire ONE small real Sentinel-2 L2A scene/crop first, not a large dataset."
        window = Window(col_off=1000, row_off=1000, width=256, height=256)
        
        # We also need to capture metadata from the first band
        metadata = {}
        
        for b in bands_to_fetch:
            asset = item.assets[b]
            with rasterio.open(asset.href) as src:
                data = src.read(1, window=window)
                band_data[b] = data
                if not metadata:
                    metadata = {
                        'crs': src.crs.to_string(),
                        'resolution': src.res[0],
                        'dtype': src.dtypes[0],
                        'nodata': src.nodata if src.nodata is not None else 0.0,
                        'spatial_extent': src.window_bounds(window),
                        'provenance': f"Microsoft Planetary Computer, Item ID: {item.id}"
                    }
                    
        # 3. Extract SCL (Scene Classification Layer, naturally 20m)
        scl_asset = item.assets["SCL"]
        # Since SCL is 20m, the window for 10m is halved
        scl_window = Window(col_off=window.col_off // 2, 
                            row_off=window.row_off // 2, 
                            width=window.width // 2, 
                            height=window.height // 2)
                            
        with rasterio.open(scl_asset.href) as src:
            scl_data_20m = src.read(1, window=scl_window)
            
            # Upsample to 10m using nearest neighbor (Kronecker product)
            scl_data_10m = np.kron(scl_data_20m, np.ones((2, 2), dtype=scl_data_20m.dtype))
            
        return {
            'bands': band_data,
            'scl': scl_data_10m,
            'metadata': metadata
        }
