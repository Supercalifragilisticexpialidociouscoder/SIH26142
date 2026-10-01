"""
CERTUS-S2: Certified Super-Resolution Reconstruction & Statistical Decision Trust for Sentinel-2
Root entry point for Streamlit Community Cloud and cloud container environments.
"""
import sys
import runpy
from pathlib import Path

# Add the certus-s2 directory to sys.path so modules resolve cleanly
ROOT_DIR = Path(__file__).resolve().parent
CERTUS_DIR = ROOT_DIR / "certus-s2"

if str(CERTUS_DIR) not in sys.path:
    sys.path.insert(0, str(CERTUS_DIR))

# Target script path
TARGET_APP = CERTUS_DIR / "ui" / "app.py"

if __name__ == "__main__" or True:
    runpy.run_path(str(TARGET_APP), run_name="__main__")
