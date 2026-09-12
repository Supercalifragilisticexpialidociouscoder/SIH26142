"""
Data loader and analytical layer computation for CERTUS-S2 Streamlit UI.
Supports:
1. Guaranteed held-out benchmark scene (Madrid ROI_00001 with 25cm PNOA reference).
2. Domain-shifted agricultural benchmark (Castile Cropland from OpenSR-S2).
3. Live satellite acquisitions via Copernicus Data Space Ecosystem (CDSE)
   for curated global locations (Barcelona, Seville, Valencia, Paris, Rome, Athens)
   or arbitrary custom coordinates.
"""

import os
import io
import json
import math
import pickle
from pathlib import Path
from typing import Any, Optional, Dict, Tuple, List
import numpy as np
import rasterio
from rasterio.warp import transform_bounds, transform
import scipy.ndimage as ndi
import streamlit as st

from src.observation.operator import EffectiveObservationOperator
from src.support.measurement import compute_residual, compute_support_map
from src.risk.engine import RiskEngine
from src.certification.certus_certifier import CertusCertifier
from src.pipeline.sen2sr_wrapper import SEN2SRWrapper
from src.data.cdse_client import CDSEClient

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "demo_d1"
CALIB_ARTIFACT_PATH = BASE_DIR / "data" / "calibration" / "d1_calibration.json"
CACHE_DIR = BASE_DIR / "data" / "opensr_cache"

# Curated global catalog
SCENE_CATALOG = {
    "demo_madrid": {
        "id": "demo_madrid",
        "name": "Madrid Urban (Held-out Benchmark)",
        "city": "Madrid / Guadalajara, Spain",
        "lat": 40.6406,
        "lon": -3.1678,
        "type": "held_out_benchmark",
        "has_hr": True,
        "description": "Guaranteed held-out test scene (ROI_00001, T30TXM) with 25cm PNOA airborne orthophoto ground truth."
    },
    "castile_crops": {
        "id": "castile_crops",
        "name": "Castile Cropland (Domain-Shift Benchmark)",
        "city": "Castile-La Mancha / León, Spain",
        "lat": 38.9995,
        "lon": -2.0177,
        "type": "domain_shift_benchmark",
        "has_hr": True,
        "description": "Agricultural crop parcels from OpenSR-S2 demonstrating honest Tier 3 (Abstain) shift detection."
    },
    "rome": {
        "id": "rome",
        "name": "Rome EUR & Historic (Live CDSE)",
        "city": "Rome, Lazio, Italy",
        "lat": 41.9028,
        "lon": 12.4964,
        "type": "live_cdse",
        "has_hr": False,
        "description": "Historic monuments, complex topography, and modern suburban built structures."
    },
    "athens": {
        "id": "athens",
        "name": "Athens Metropolitan / Greece (Live CDSE)",
        "city": "Athens, Attica, Greece",
        "lat": 37.9838,
        "lon": 23.7275,
        "type": "live_cdse",
        "has_hr": False,
        "description": "High-density concrete canopy surrounded by Mediterranean mountainous topography."
    },
    "paris": {
        "id": "paris",
        "name": "Paris Urban Core (Live CDSE)",
        "city": "Paris, Île-de-France, France",
        "lat": 48.8566,
        "lon": 2.3522,
        "type": "live_cdse",
        "has_hr": False,
        "description": "Dense European metropolis with Haussmannian geometric rooflines and high building density."
    },
    "barcelona": {
        "id": "barcelona",
        "name": "Barcelona Urban & Port (Live CDSE)",
        "city": "Barcelona, Catalonia, Spain",
        "lat": 41.3879,
        "lon": 2.1699,
        "type": "live_cdse",
        "has_hr": False,
        "description": "Dense coastal urban grid with maritime port infrastructure and Mediterranean atmosphere."
    },
    "seville": {
        "id": "seville",
        "name": "Seville Historic & River (Live CDSE)",
        "city": "Seville, Andalusia, Spain",
        "lat": 37.3891,
        "lon": -5.9845,
        "type": "live_cdse",
        "has_hr": False,
        "description": "Historic urban core along the Guadalquivir river channel with dense clay roof structures."
    },
    "valencia": {
        "id": "valencia",
        "name": "Valencia Coastal City (Live CDSE)",
        "city": "Valencia, Spain",
        "lat": 39.4699,
        "lon": -0.3763,
        "type": "live_cdse",
        "has_hr": False,
        "description": "Mediterranean coastal metropolis with complex urban fabric and linear parks."
    },
    "hyderabad": {
        "id": "hyderabad",
        "name": "Hyderabad Urban & Tech Corridor (Live CDSE)",
        "city": "Hyderabad, Telangana, India",
        "lat": 17.3850,
        "lon": 78.4867,
        "type": "live_cdse",
        "has_hr": False,
        "description": "High-density tech corridors, urban lakes, and historic core in Telangana, India."
    },
    "bengaluru": {
        "id": "bengaluru",
        "name": "Bengaluru Innovation Hub (Live CDSE)",
        "city": "Bengaluru, Karnataka, India",
        "lat": 12.9716,
        "lon": 77.5946,
        "type": "live_cdse",
        "has_hr": False,
        "description": "High-density urban innovation hub and garden metropolis in Karnataka, India."
    },
    "delhi": {
        "id": "delhi",
        "name": "Delhi National Capital Region (Live CDSE)",
        "city": "Delhi NCR, India",
        "lat": 28.6139,
        "lon": 77.2090,
        "type": "live_cdse",
        "has_hr": False,
        "description": "Dense capital metropolis with monumental architecture and Yamuna floodplain."
    },
    "custom": {
        "id": "custom",
        "name": "Custom Geographic Coordinates (Live CDSE)",
        "city": "Custom Coordinates",
        "lat": 0.0,
        "lon": 0.0,
        "type": "live_cdse_custom",
        "has_hr": False,
        "description": "User-specified latitude and longitude anywhere on Earth via Copernicus Data Space Ecosystem."
    }
}


@st.cache_resource(show_spinner=False)
def get_certifier() -> CertusCertifier:
    """Initializes and caches the CertusCertifier backend."""
    return CertusCertifier(calibration_artifact=str(CALIB_ARTIFACT_PATH))


@st.cache_resource(show_spinner=False)
def get_sr_model() -> SEN2SRWrapper:
    """Initializes and caches the SEN2SR super-resolution wrapper."""
    return SEN2SRWrapper(device="cpu")


@st.cache_resource(show_spinner=False)
def get_cdse_client() -> CDSEClient:
    """Initializes and caches the CDSE acquisition client."""
    return CDSEClient()


# ----------------------------------------------------------------------
# Backward-compatible loaders for Madrid Demo Scene (tests and baseline)
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_scene_metadata() -> dict[str, Any]:
    """Loads metadata for held-out Madrid ROI_00001 demo scene."""
    meta_path = DATA_DIR / "scene_metadata.json"
    with open(meta_path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_demo_rasters() -> dict[str, Any]:
    """
    Loads Sentinel-2 L2A (10m), SEN2SR reconstruction (2.5m), and quarantined HR reference.
    Returns RGB composites and multichannel reflectance arrays for Madrid demo scene.
    """
    l2a_dir = DATA_DIR / "input_10m"
    b2_10m = rasterio.open(l2a_dir / "S2_L2A_20210811T105619_20210811T110659_T30TXM_ROI_00001_B02_10m.tif").read(1)
    b3_10m = rasterio.open(l2a_dir / "S2_L2A_20210811T105619_20210811T110659_T30TXM_ROI_00001_B03_10m.tif").read(1)
    b4_10m = rasterio.open(l2a_dir / "S2_L2A_20210811T105619_20210811T110659_T30TXM_ROI_00001_B04_10m.tif").read(1)
    b8_10m = rasterio.open(l2a_dir / "S2_L2A_20210811T105619_20210811T110659_T30TXM_ROI_00001_B08_10m.tif").read(1)

    l2a_raw = np.stack([b2_10m, b3_10m, b4_10m, b8_10m], axis=0).astype(np.float32)
    l2a = np.clip(l2a_raw / 10000.0, 0.0, 1.0) if np.nanmax(l2a_raw) > 2.0 else np.clip(l2a_raw, 0.0, 1.0)

    # 10m RGB composite (B04, B03, B02)
    l2a_rgb = np.stack([l2a[2], l2a[1], l2a[0]], axis=-1)
    p2, p98 = np.percentile(l2a_rgb, (2, 98))
    l2a_rgb_stretched = np.clip((l2a_rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

    # 2.5m SEN2SR reconstruction
    sr_dir = DATA_DIR / "reconstruction_2_5m"
    with rasterio.open(sr_dir / "SEN2SR_2_5m_ROI_00001_B02_2_5m.tif") as src:
        b2_sr = src.read(1)
        sr_profile = src.profile.copy()
        bounds = src.bounds
        crs_str = str(src.crs)

    b3_sr = rasterio.open(sr_dir / "SEN2SR_2_5m_ROI_00001_B03_2_5m.tif").read(1)
    b4_sr = rasterio.open(sr_dir / "SEN2SR_2_5m_ROI_00001_B04_2_5m.tif").read(1)
    b8_sr = rasterio.open(sr_dir / "SEN2SR_2_5m_ROI_00001_B08_2_5m.tif").read(1)

    sr = np.stack([b2_sr, b3_sr, b4_sr, b8_sr], axis=0).astype(np.float32)
    sr = np.clip(sr, 0.0, 1.0)

    # 2.5m RGB composite (B04, B03, B02)
    sr_rgb = np.stack([sr[2], sr[1], sr[0]], axis=-1)
    p2_sr, p98_sr = np.percentile(sr_rgb, (2, 98))
    sr_rgb_stretched = np.clip((sr_rgb - p2_sr) / (p98_sr - p2_sr + 1e-6), 0.0, 1.0)

    # Quarantined HR reference
    hr_dir = DATA_DIR / "ground_truth_hr"
    b2_hr = rasterio.open(hr_dir / "HR__ROI_00001__PNOA_ANUAL_2021_OF_ETRS89_HU30_h25_0383-2_B02_2_5m.tif").read(1)
    b3_hr = rasterio.open(hr_dir / "HR__ROI_00001__PNOA_ANUAL_2021_OF_ETRS89_HU30_h25_0383-2_B03_2_5m.tif").read(1)
    b4_hr = rasterio.open(hr_dir / "HR__ROI_00001__PNOA_ANUAL_2021_OF_ETRS89_HU30_h25_0383-2_B04_2_5m.tif").read(1)
    b8_hr = rasterio.open(hr_dir / "HR__ROI_00001__PNOA_ANUAL_2021_OF_ETRS89_HU30_h25_0383-2_B08_2_5m.tif").read(1)
    hr = np.stack([b2_hr, b3_hr, b4_hr, b8_hr], axis=0).astype(np.float32)
    hr = np.clip(hr / 10000.0, 0.0, 1.0) if np.nanmax(hr) > 2.0 else np.clip(hr, 0.0, 1.0)

    return {
        "l2a": l2a,
        "l2a_rgb": l2a_rgb_stretched,
        "sr": sr,
        "sr_rgb": sr_rgb_stretched,
        "hr_reference": hr,
        "sr_profile": sr_profile,
        "bounds": bounds,
        "crs": crs_str
    }


@st.cache_data(show_spinner=False)
def compute_analytical_layers_data() -> dict[str, Any]:
    """
    Computes measurement support map s(p) and continuous risk map r(p) for Madrid demo.
    """
    rasters = load_demo_rasters()
    l2a = rasters["l2a"]
    sr = rasters["sr"]

    op = EffectiveObservationOperator(kernel_sigma=1.2, subpixel_shift=(0.0, 0.0), sigma_b=0.04)
    op.is_fitted = True

    residual_10m = compute_residual(l2a, sr, op)
    support_4d = compute_support_map(residual_10m, op.sigma_b, scale_factor=4)

    risk_engine = RiskEngine(backend_type="numpy")
    ep = np.zeros_like(support_4d)
    al = np.zeros_like(support_4d)
    risk_4d = risk_engine.compute_risk(support_4d, ep, al, residual=residual_10m)

    support_2d = np.mean(support_4d, axis=0)
    risk_2d = np.mean(risk_4d, axis=0)

    return {
        "residual_10m": residual_10m,
        "support_2d": support_2d,
        "risk_2d": risk_2d,
        "support_stats": {
            "mean": float(np.mean(support_2d)),
            "min": float(np.min(support_2d)),
            "max": float(np.max(support_2d)),
            "median": float(np.median(support_2d))
        },
        "risk_stats": {
            "mean": float(np.mean(risk_2d)),
            "min": float(np.min(risk_2d)),
            "max": float(np.max(risk_2d)),
            "median": float(np.median(risk_2d))
        }
    }


@st.cache_data(show_spinner=False)
def load_aoi_spatial_info() -> dict[str, Any]:
    """
    Computes exact WGS84 geographic footprint, bounds, and centroid for Madrid ROI_00001.
    """
    sr_tif = DATA_DIR / "reconstruction_2_5m" / "SEN2SR_2_5m_ROI_00001_B02_2_5m.tif"
    with rasterio.open(sr_tif) as src:
        crs = src.crs
        bounds = src.bounds

    w, s, e, n = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
    center_lon = float((w + e) / 2.0)
    center_lat = float((s + n) / 2.0)

    polygon = [
        [float(w), float(s)],
        [float(e), float(s)],
        [float(e), float(n)],
        [float(w), float(n)],
        [float(w), float(s)]
    ]

    width_km = float((bounds.right - bounds.left) / 1000.0)
    height_km = float((bounds.top - bounds.bottom) / 1000.0)
    area_km2 = float(width_km * height_km)

    return {
        "polygon_coords": polygon,
        "bounds_wgs84": {
            "west": float(w),
            "south": float(s),
            "east": float(e),
            "north": float(n)
        },
        "center_wgs84": {
            "lat": center_lat,
            "lon": center_lon
        },
        "dimensions_km": (width_km, height_km),
        "area_km2": area_km2,
        "area_ha": area_km2 * 100.0,
        "utm_crs": str(crs)
    }


@st.cache_data(show_spinner=False)
def load_proposal_candidates() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Loads fixed D1 proposal candidates (N=109), binary proposal mask, and labeled component map.
    Attaches exact pixel masks and geographic WGS84 coordinates to each candidate.
    """
    mask_path = DATA_DIR / "d1_candidate_mask.tif"
    with rasterio.open(mask_path) as src:
        prop_mask = src.read(1)
        transform_affine = src.transform
        crs = src.crs

    labeled, n_components = ndi.label(prop_mask)

    cand_path = DATA_DIR / "d1_candidates.json"
    with open(cand_path, "r") as f:
        candidates = json.load(f)

    xs, ys = [], []
    for c in candidates:
        cy, cx = c["centroid_yx"]
        x_utm, y_utm = rasterio.transform.xy(transform_affine, cy, cx)
        xs.append(x_utm)
        ys.append(y_utm)

    lons, lats = transform(crs, "EPSG:4326", xs, ys)

    for i, c in enumerate(candidates):
        c_id = c["id"]
        c["mask"] = (labeled == c_id)
        c["lon"] = float(lons[i])
        c["lat"] = float(lats[i])

    return candidates, prop_mask, labeled


# ----------------------------------------------------------------------
# Dynamic Multi-Scene Pipeline Execution
# ----------------------------------------------------------------------

def _compute_scene_layers(l2a: np.ndarray, sr: np.ndarray) -> dict[str, Any]:
    """
    Runs observation operator residual, support map, and risk engine on any scene.
    """
    op = EffectiveObservationOperator(kernel_sigma=1.2, subpixel_shift=(0.0, 0.0), sigma_b=0.04)
    op.is_fitted = True

    residual_10m = compute_residual(l2a, sr, op)
    support_4d = compute_support_map(residual_10m, op.sigma_b, scale_factor=4)

    risk_engine = RiskEngine(backend_type="numpy")
    ep = np.zeros_like(support_4d)
    al = np.zeros_like(support_4d)
    risk_4d = risk_engine.compute_risk(support_4d, ep, al, residual=residual_10m)

    support_2d = np.mean(support_4d, axis=0)
    risk_2d = np.mean(risk_4d, axis=0)

    return {
        "residual_10m": residual_10m,
        "support_2d": support_2d,
        "risk_2d": risk_2d,
        "support_stats": {
            "mean": float(np.mean(support_2d)),
            "min": float(np.min(support_2d)),
            "max": float(np.max(support_2d)),
            "median": float(np.median(support_2d))
        },
        "risk_stats": {
            "mean": float(np.mean(risk_2d)),
            "min": float(np.min(risk_2d)),
            "max": float(np.max(risk_2d)),
            "median": float(np.median(risk_2d))
        }
    }


def _extract_scene_candidates(
    sr: np.ndarray,
    risk_2d: np.ndarray,
    bounds_wgs84: dict[str, float]
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Extracts candidate built-structure proposals from SEN2SR reconstruction,
    labels connected components, and projects centroids to WGS84 coordinates.
    """
    certifier = get_certifier()
    prop_mask, raw_candidates = certifier.proposal_stage.extract_candidates(sr, risk_map=risk_2d)

    labeled, num_labels = ndi.label(prop_mask)

    west = bounds_wgs84["west"]
    south = bounds_wgs84["south"]
    east = bounds_wgs84["east"]
    north = bounds_wgs84["north"]

    candidates = []
    for c in raw_candidates:
        cy, cx = c["centroid"]
        c_id = c["id"]
        # Pixel coordinates 0..512 mapped linearly to WGS84
        c_lon = float(west + (cx / 512.0) * (east - west))
        c_lat = float(north - (cy / 512.0) * (north - south))

        cand_record = {
            "id": c_id,
            "area_px": c["area"],
            "area_m2": c["area"] * 6.25,  # 2.5m x 2.5m pixel = 6.25 m2
            "centroid_yx": [float(cy), float(cx)],
            "centroid": [float(cy), float(cx)],
            "bbox_yxyx": c["bbox"],
            "risk": float(c["risk"]) if c["risk"] is not None else float(risk_2d.mean()),
            "mask": (labeled == c_id),
            "lon": c_lon,
            "lat": c_lat,
            "is_true_detection": None,
            "dist_to_reference_px": None
        }
        candidates.append(cand_record)

    return candidates, prop_mask, labeled


# Geocoding dictionary for instantaneous, reliable response
POPULAR_LOCATIONS = {
    "hyderabad": (17.3850, 78.4867, "Hyderabad, Telangana, India"),
    "bengaluru": (12.9716, 77.5946, "Bengaluru, Karnataka, India"),
    "bangalore": (12.9716, 77.5946, "Bengaluru, Karnataka, India"),
    "delhi": (28.6139, 77.2090, "New Delhi, NCR, India"),
    "new delhi": (28.6139, 77.2090, "New Delhi, NCR, India"),
    "mumbai": (19.0760, 72.8777, "Mumbai, Maharashtra, India"),
    "chennai": (13.0827, 80.2707, "Chennai, Tamil Nadu, India"),
    "kolkata": (22.5726, 88.3639, "Kolkata, West Bengal, India"),
    "madrid": (40.4168, -3.7038, "Madrid, Spain"),
    "barcelona": (41.3879, 2.1699, "Barcelona, Catalonia, Spain"),
    "seville": (37.3891, -5.9845, "Seville, Andalusia, Spain"),
    "valencia": (39.4699, -0.3763, "Valencia, Spain"),
    "paris": (48.8566, 2.3522, "Paris, Île-de-France, France"),
    "london": (51.5074, -0.1278, "London, United Kingdom"),
    "rome": (41.9028, 12.4964, "Rome, Lazio, Italy"),
    "athens": (37.9838, 23.7275, "Athens, Attica, Greece"),
    "tokyo": (35.6762, 139.6503, "Tokyo, Japan")
}


def geocode_location(query: str) -> Optional[dict]:
    """Geocodes a search string to exact lat/lon coordinates."""
    if not query:
        return None
    q = query.strip().lower()
    if q in POPULAR_LOCATIONS:
        lat, lon, full_name = POPULAR_LOCATIONS[q]
        return {"name": full_name, "lat": lat, "lon": lon}

    for k, (lat, lon, full_name) in POPULAR_LOCATIONS.items():
        if k in q or q in k:
            return {"name": full_name, "lat": lat, "lon": lon}

    try:
        import requests
        url = f"https://nominatim.openstreetmap.org/search?format=json&limit=1&q={requests.utils.quote(query)}"
        resp = requests.get(url, headers={"User-Agent": "CERTUS-S2-Research/1.0"}, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                return {
                    "name": data[0].get("display_name", query),
                    "lat": float(data[0]["lat"]),
                    "lon": float(data[0]["lon"])
                }
    except Exception:
        pass
    return None


def search_available_observations(lat: float, lon: float) -> list[dict]:
    """Queries CDSE for real Sentinel-2 L2A observations for the AOI."""
    client = get_cdse_client()
    return client.search_sentinel2_l2a_scenes(lat=lat, lon=lon)


@st.cache_data(show_spinner=False)
def load_scene_data(
    scene_key: str,
    custom_lat: Optional[float] = None,
    custom_lon: Optional[float] = None,
    selected_date: Optional[str] = None,
    location_name: Optional[str] = None
) -> dict[str, Any]:
    """
    Unified multi-scene loader for CERTUS-S2.
    Loads or acquires L2A, runs real SEN2SR super-resolution, computes analytical layers,
    and extracts candidate built-structures with geographic coordinates.
    """
    info = dict(SCENE_CATALOG.get(scene_key, SCENE_CATALOG["demo_madrid"]))

    if scene_key == "demo_madrid":
        # Guaranteed baseline Madrid demo
        rasters = load_demo_rasters()
        layers = compute_analytical_layers_data()
        spatial_info = load_aoi_spatial_info()
        candidates, prop_mask, labeled = load_proposal_candidates()
        scene_meta = load_scene_metadata()

        return {
            "scene_key": "demo_madrid",
            "info": info,
            "l2a": rasters["l2a"],
            "l2a_rgb": rasters["l2a_rgb"],
            "sr": rasters["sr"],
            "sr_rgb": rasters["sr_rgb"],
            "hr_reference": rasters["hr_reference"],
            "bounds_wgs84": spatial_info["bounds_wgs84"],
            "center_wgs84": spatial_info["center_wgs84"],
            "polygon_coords": spatial_info["polygon_coords"],
            "dimensions_km": spatial_info["dimensions_km"],
            "area_km2": spatial_info["area_km2"],
            "analytical_layers": layers,
            "candidates": candidates,
            "prop_mask": prop_mask,
            "labeled_mask": labeled,
            "metadata": scene_meta,
            "has_hr": True
        }

    elif scene_key == "castile_crops":
        # Castile agricultural crops domain-shift benchmark
        crops_path = CACHE_DIR / "spain_crops.pkl"
        with open(crops_path, "rb") as f:
            data = pickle.load(f)

        l2a_full = data["L2A"][0]  # Shape (12, 128, 128)
        # OpenSR band ordering: B02=idx 1, B03=idx 2, B04=idx 3, B08=idx 7
        l2a = l2a_full[[1, 2, 3, 7]].astype(np.float32)
        l2a = np.clip(l2a / 10000.0, 0.0, 1.0) if np.nanmax(l2a) > 2.0 else np.clip(l2a, 0.0, 1.0)

        # 10m RGB
        rgb = np.stack([l2a[2], l2a[1], l2a[0]], axis=-1)
        p2, p98 = np.percentile(rgb, (2, 98))
        l2a_rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

        # Real SEN2SR inference
        sr_wrapper = get_sr_model()
        sr = sr_wrapper.predict_numpy(l2a)
        sr = np.clip(sr, 0.0, 1.0)

        sr_rgb = np.stack([sr[2], sr[1], sr[0]], axis=-1)
        p2_sr, p98_sr = np.percentile(sr_rgb, (2, 98))
        sr_rgb = np.clip((sr_rgb - p2_sr) / (p98_sr - p2_sr + 1e-6), 0.0, 1.0)

        # Quarantined HR reference
        hr_full = data["HR"][0].astype(np.float32)
        hr = np.clip(hr_full / 10000.0, 0.0, 1.0) if np.nanmax(hr_full) > 2.0 else np.clip(hr_full, 0.0, 1.0)

        lat, lon = info["lat"], info["lon"]
        bbox = CDSEClient.calculate_aoi_bbox(lat, lon, size_meters=1280.0)
        bounds_wgs84 = {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]}
        center_wgs84 = {"lat": lat, "lon": lon}
        polygon_coords = [
            [bbox[0], bbox[1]], [bbox[2], bbox[1]],
            [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]
        ]

        layers = _compute_scene_layers(l2a, sr)
        candidates, prop_mask, labeled = _extract_scene_candidates(sr, layers["risk_2d"], bounds_wgs84)

        meta = {
            "source": "OpenSR-S2 Agricultural Cropland Archive",
            "tile": "T30SXJ",
            "roi": "ROI_CROPS_00001",
            "acquisition_date": "2021-07-17",
            "center": center_wgs84,
            "dimensions_km": [1.28, 1.28],
            "note": "Domain-shift test scene with dense vegetation cover (expected Tier 3 Abstain)"
        }

        return {
            "scene_key": "castile_crops",
            "info": info,
            "l2a": l2a,
            "l2a_rgb": l2a_rgb,
            "sr": sr,
            "sr_rgb": sr_rgb,
            "hr_reference": hr,
            "bounds_wgs84": bounds_wgs84,
            "center_wgs84": center_wgs84,
            "polygon_coords": polygon_coords,
            "dimensions_km": (1.28, 1.28),
            "area_km2": 1.6384,
            "analytical_layers": layers,
            "candidates": candidates,
            "prop_mask": prop_mask,
            "labeled_mask": labeled,
            "metadata": meta,
            "has_hr": True
        }

    else:
        # Live CDSE acquisition (curated city or custom coordinates)
        if scene_key == "custom":
            lat = float(custom_lat if custom_lat is not None else 17.3850)
            lon = float(custom_lon if custom_lon is not None else 78.4867)
            city_str = location_name or f"Coordinates ({lat:.4f}°N, {lon:.4f}°E)"
            info["city"] = city_str
            info["name"] = city_str
            info["lat"] = lat
            info["lon"] = lon
        else:
            lat = info["lat"]
            lon = info["lon"]

        cdse = get_cdse_client()
        patch = cdse.fetch_sentinel2_l2a_patch(lat=lat, lon=lon, date=selected_date)

        l2a = patch["l2a"]
        l2a_rgb = patch["l2a_rgb"]

        # Run real SEN2SR Super-Resolution forward pass
        sr_wrapper = get_sr_model()
        sr = sr_wrapper.predict_numpy(l2a)
        sr = np.clip(sr, 0.0, 1.0)

        sr_rgb = np.stack([sr[2], sr[1], sr[0]], axis=-1)
        p2_sr, p98_sr = np.percentile(sr_rgb, (2, 98))
        sr_rgb = np.clip((sr_rgb - p2_sr) / (p98_sr - p2_sr + 1e-6), 0.0, 1.0)

        bounds_wgs84 = patch["bounds_wgs84"]
        center_wgs84 = patch["center"]
        polygon_coords = [
            [bounds_wgs84["west"], bounds_wgs84["south"]],
            [bounds_wgs84["east"], bounds_wgs84["south"]],
            [bounds_wgs84["east"], bounds_wgs84["north"]],
            [bounds_wgs84["west"], bounds_wgs84["north"]],
            [bounds_wgs84["west"], bounds_wgs84["south"]]
        ]

        layers = _compute_scene_layers(l2a, sr)
        candidates, prop_mask, labeled = _extract_scene_candidates(sr, layers["risk_2d"], bounds_wgs84)

        meta = patch.get("metadata", {
            "source": "Copernicus Data Space Ecosystem (CDSE) Process API",
            "center": center_wgs84,
            "dimensions_km": [1.28, 1.28],
            "date": selected_date or "recent"
        })

        return {
            "scene_key": scene_key,
            "info": info,
            "l2a": l2a,
            "l2a_rgb": l2a_rgb,
            "sr": sr,
            "sr_rgb": sr_rgb,
            "hr_reference": None,
            "bounds_wgs84": bounds_wgs84,
            "center_wgs84": center_wgs84,
            "polygon_coords": polygon_coords,
            "dimensions_km": (1.28, 1.28),
            "area_km2": 1.6384,
            "analytical_layers": layers,
            "candidates": candidates,
            "prop_mask": prop_mask,
            "labeled_mask": labeled,
            "metadata": meta,
            "has_hr": False
        }

