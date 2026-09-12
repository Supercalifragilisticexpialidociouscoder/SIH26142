import os
import rasterio
import numpy as np

def verify():
    print("=== SEN2SR BASELINE VERIFICATION ===")
    
    input_dir = "data/input_crop"
    output_dir = "data/output_2_5m"
    item_id = "S2B_MSIL2A_20231130T185739_R113_T10SEG_20241031T042024"
    bands = ["B02", "B03", "B04", "B08"]
    
    print("\n[INPUT RASTERS]")
    for b in bands:
        path = os.path.join(input_dir, f"{item_id}_{b}_10m.tif")
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        with rasterio.open(path) as src:
            data = src.read(1)
            print(f"Band: {b}")
            print(f"  Shape: {data.shape}")
            print(f"  CRS: {src.crs}")
            print(f"  Transform: {src.transform}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Min: {np.nanmin(data):.4f}, Max: {np.nanmax(data):.4f}, Mean: {np.nanmean(data):.4f}, Std: {np.nanstd(data):.4f}")
            print(f"  NaNs: {np.isnan(data).sum()}, Infs: {np.isinf(data).sum()}")
            print(f"  NoData value: {src.nodata}")

    print("\n[OUTPUT RASTERS]")
    for b in bands:
        path = os.path.join(output_dir, f"{item_id}_{b}_2_5m.tif")
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        with rasterio.open(path) as src:
            data = src.read(1)
            print(f"Band: {b}")
            print(f"  Shape: {data.shape}")
            print(f"  CRS: {src.crs}")
            print(f"  Transform: {src.transform}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Min: {np.nanmin(data):.4f}, Max: {np.nanmax(data):.4f}, Mean: {np.nanmean(data):.4f}, Std: {np.nanstd(data):.4f}")
            print(f"  NaNs: {np.isnan(data).sum()}, Infs: {np.isinf(data).sum()}")
            print(f"  NoData value: {src.nodata}")

if __name__ == "__main__":
    verify()
