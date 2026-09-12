import os
import rasterio
from rasterio.windows import Window
import planetary_computer
from pystac_client import Client
import numpy as np

def download_crop():
    item_id = "S2B_MSIL2A_20231130T185739_R113_T10SEG_20241031T042024"
    catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    
    print(f"Connecting to {catalog_url}...")
    catalog = Client.open(catalog_url, modifier=planetary_computer.sign_inplace)
    
    # We can fetch the specific item by searching for its ID
    search = catalog.search(collections=["sentinel-2-l2a"], ids=[item_id])
    items = list(search.items())
    if not items:
        raise ValueError(f"Item {item_id} not found!")
        
    item = items[0]
    print(f"Found item: {item.id}")
    
    bands_to_fetch = ["B02", "B03", "B04", "B08"]
    
    # We want a 256x256 crop from the 10m bands
    window = Window(col_off=2000, row_off=2000, width=256, height=256)
    
    out_dir = "data/input_crop"
    os.makedirs(out_dir, exist_ok=True)
    
    metadata = None
    
    for b in bands_to_fetch:
        print(f"Downloading {b} crop...")
        asset = item.assets[b]
        
        with rasterio.open(asset.href) as src:
            if metadata is None:
                metadata = src.profile.copy()
                metadata.update({
                    "height": window.height,
                    "width": window.width,
                    "transform": rasterio.windows.transform(window, src.transform)
                })
                
            data = src.read(1, window=window)
            
            out_path = os.path.join(out_dir, f"{item_id}_{b}_10m.tif")
            with rasterio.open(out_path, "w", **metadata) as dest:
                dest.write(data, 1)
                
    # Also fetch SCL (20m) and upsample to 10m
    print(f"Downloading SCL crop...")
    scl_asset = item.assets["SCL"]
    scl_window = Window(col_off=window.col_off // 2, 
                        row_off=window.row_off // 2, 
                        width=window.width // 2, 
                        height=window.height // 2)
                        
    with rasterio.open(scl_asset.href) as src:
        scl_data_20m = src.read(1, window=scl_window)
        # Upsample to 10m
        scl_data_10m = np.kron(scl_data_20m, np.ones((2, 2), dtype=scl_data_20m.dtype))
        
        scl_out_path = os.path.join(out_dir, f"{item_id}_SCL_10m.tif")
        with rasterio.open(scl_out_path, "w", **metadata) as dest:
            dest.write(scl_data_10m, 1)
            
    print(f"Successfully downloaded 10m crops to {out_dir}/")

if __name__ == "__main__":
    download_crop()
