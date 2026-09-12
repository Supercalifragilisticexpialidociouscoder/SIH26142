import os
import sys
import torch
import rasterio

# Add src to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.preprocess import preprocess_l2a
from src.pipeline.sen2sr_wrapper import SEN2SRWrapper

def main():
    item_id = "S2B_MSIL2A_20231130T185739_R113_T10SEG_20241031T042024"
    data_dir = "data/input_crop"
    
    band_paths = {
        "B02": os.path.join(data_dir, f"{item_id}_B02_10m.tif"),
        "B03": os.path.join(data_dir, f"{item_id}_B03_10m.tif"),
        "B04": os.path.join(data_dir, f"{item_id}_B04_10m.tif"),
        "B08": os.path.join(data_dir, f"{item_id}_B08_10m.tif"),
    }
    
    # 1. Preprocess the real raster crops
    print("Preprocessing real raster data...")
    tensor, meta = preprocess_l2a(band_paths)
    print(f"Input tensor shape: {tensor.shape}") # Expected: (1, 4, 256, 256)
    
    # 2. Instantiate SEN2SR Model
    # Important: macOS SSL fix for huggingface hub
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Set torch hub dir to cache properly
    torch.hub.set_dir('/Users/sripranavireddypalle/.cache/torch/hub')
    
    print("Initializing SEN2SR Wrapper...")
    wrapper = SEN2SRWrapper(device="cpu")
    
    # 3. Inference
    print("Running SEN2SR forward pass (this may take a moment on CPU)...")
    with torch.no_grad():
        out_tensor = wrapper.predict(tensor)
        
    print(f"Output tensor shape: {out_tensor.shape}") # Expected: (1, 4, 1024, 1024)
    
    # 4. Save results to 2.5m Analysis Grid
    print("Saving 2.5m analysis grid outputs...")
    out_dir = "data/output_2_5m"
    os.makedirs(out_dir, exist_ok=True)
    
    out_meta = meta.copy()
    
    # Calculate new transform for 4x scaling (resolution / 4)
    from rasterio.transform import Affine
    new_transform = out_meta['transform'] * Affine.scale(0.25, 0.25)
    
    out_meta.update({
        "height": out_tensor.shape[2],
        "width": out_tensor.shape[3],
        "transform": new_transform,
        "dtype": "float32"
    })
    
    out_numpy = out_tensor.squeeze(0).cpu().numpy()
    
    # Save a multi-band TIF or separate TIFs. We will save separate ones to mimic input.
    bands = ["B02", "B03", "B04", "B08"]
    for i, band in enumerate(bands):
        out_path = os.path.join(out_dir, f"{item_id}_{band}_2_5m.tif")
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_numpy[i], 1)
            
    print(f"Success! Output artifacts saved to {out_dir}")

if __name__ == "__main__":
    main()
