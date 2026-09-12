import os
import sys
import rasterio
import numpy as np

# Add src to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.observation.operator import EffectiveObservationOperator
from src.pipeline.preprocess import preprocess_l2a

def main():
    item_id = "S2B_MSIL2A_20231130T185739_R113_T10SEG_20241031T042024"
    in_dir = "data/input_crop"
    out_dir = "data/output_2_5m"
    
    # 1. Load real 10m LR observation y
    band_paths = {b: os.path.join(in_dir, f"{item_id}_{b}_10m.tif") for b in ["B02", "B03", "B04", "B08"]}
    y, _ = preprocess_l2a(band_paths)
    y = y.squeeze(0).numpy() # (4, 256, 256)
    
    # 2. Load reconstructed 2.5m HR image \hat{x}
    x_hat = []
    for b in ["B02", "B03", "B04", "B08"]:
        with rasterio.open(os.path.join(out_dir, f"{item_id}_{b}_2_5m.tif")) as src:
            x_hat.append(src.read(1))
    x_hat = np.stack(x_hat, axis=0) # (4, 1024, 1024)
    
    # 3. Apply Effective Observation Operator
    print("Applying Effective Observation Operator...")
    op = EffectiveObservationOperator(kernel_sigma=1.2, subpixel_shift=(0.1, -0.2), scale_factor=4)
    op.is_fitted = True
    
    y_hat = op.apply(x_hat)
    print(f"Predicted LR shape: {y_hat.shape}")
    
    # 4. Compute Residual
    e = op.compute_residual(y, y_hat)
    
    # 5. Print statistics
    print("\n--- Numerical Sanity Checks ---")
    for i, b in enumerate(["B02", "B03", "B04", "B08"]):
        print(f"Band {b}:")
        print(f"  y     Mean: {np.mean(y[i]):.4f}, Std: {np.std(y[i]):.4f}")
        print(f"  y_hat Mean: {np.mean(y_hat[i]):.4f}, Std: {np.std(y_hat[i]):.4f}")
        print(f"  e     Mean: {np.mean(e[i]):.4f}, Std: {np.std(e[i]):.4f}, Max Abs: {np.max(np.abs(e[i])):.4f}")
        
if __name__ == "__main__":
    main()
