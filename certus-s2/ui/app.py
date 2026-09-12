"""
CERTUS-S2: Certified Super-Resolution Reconstruction & Statistical Decision Trust for Sentinel-2
Professional Geospatial Analysis Console & 3D Earth Interface

Run with:
    streamlit run ui/app.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Any, Optional, Dict, List

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ui.data_loader import (
    SCENE_CATALOG,
    get_certifier,
    load_scene_data,
    load_scene_metadata,
    load_demo_rasters,
    compute_analytical_layers_data,
    load_proposal_candidates,
    load_aoi_spatial_info,
    geocode_location,
    search_available_observations,
    POPULAR_LOCATIONS
)
from ui.visualizer import (
    plot_reconstruction,
    plot_support_map,
    plot_risk_map,
    plot_proposed_detections,
    plot_certified_detections
)
from ui.cesium_component import generate_cesium_html


# Page configuration
st.set_page_config(
    page_title="CERTUS-S2 | Geospatial Trust Console",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Clean CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 2.5rem;
        max-width: 96%;
    }
    
    /* Top Professional Header */
    .app-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        padding: 1.0rem 1.4rem;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        margin-bottom: 1.0rem;
    }
    .app-header h1 {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .app-header .subtitle {
        font-size: 0.95rem;
        font-weight: 500;
        color: #38bdf8;
        margin: 0.2rem 0 0.4rem 0;
    }
    .app-header .description {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.45;
        margin: 0;
    }
    .precaution-box {
        margin-top: 0.5rem;
        padding: 0.4rem 0.8rem;
        background: rgba(15, 23, 42, 0.9);
        border-left: 3px solid #f59e0b;
        border-radius: 4px;
        font-size: 0.80rem;
        color: #fbbf24;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* AOI Summary Card */
    .aoi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    .aoi-title {
        font-size: 0.90rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .aoi-detail {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 0.25rem;
        line-height: 1.35;
    }
    .aoi-detail strong {
        color: #cbd5e1;
    }
    .aoi-badge {
        display: inline-block;
        background: #0369a1;
        color: #e0f2fe;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* Dual Status Cards */
    .status-card-cert {
        background: #064e3b;
        border-left: 4px solid #10b981;
        border-radius: 6px;
        padding: 0.8rem 1.0rem;
        margin-bottom: 0.6rem;
    }
    .status-card-empty {
        background: #451a03;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 0.8rem 1.0rem;
        margin-bottom: 0.6rem;
    }
    .status-card-shift {
        background: #451a03;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 0.8rem 1.0rem;
        margin-bottom: 0.6rem;
    }
    .status-heading {
        font-size: 0.90rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: 0.02em;
    }

    /* Metric Cards */
    .metric-container {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 0.7rem 0.8rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0.15rem 0;
    }
    .metric-sub {
        font-size: 0.70rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)


def render_header(mode: str = "LIVE ANALYSIS"):
    """Renders professional header without decorative icons."""
    mode_color = "#38bdf8" if mode == "LIVE ANALYSIS" else "#10b981"
    st.markdown(f"""
    <div class="app-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1>CERTUS-S2</h1>
            <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid {mode_color}; color: {mode_color}; padding: 3px 12px; border-radius: 4px; font-size: 0.80rem; font-weight: 700; letter-spacing: 0.05em;">
                MODE: {mode}
            </div>
        </div>
        <div class="subtitle">Certified Super-Resolution Reconstruction & Statistical Decision Trust for Sentinel-2</div>
        <p class="description">
            SEN2SR reconstructs a 2.5m representation from 10m Sentinel-2 observations.
            CERTUS-S2 evaluates measurement support, reconstruction consistency, and decision risk before certifying downstream detections.
        </p>
        <div class="precaution-box">
            <strong>Scientific Precaution:</strong> 2.5m is the reconstruction grid spacing; it is not a claim of uniformly resolved 2.5m physical detail.
            Decision risk r(p) measures downstream empirical loss potential, not a pixel hallucination probability.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renders clean, functional sidebar controls."""
    with st.sidebar:
        st.markdown("### ANALYSIS CONTROLS")

        # 1. Mode Selection
        mode_choice = st.radio(
            "Operating Mode",
            options=["Live Analysis (Global CDSE)", "Benchmark Demo (Madrid / Castile)"],
            index=0 if st.session_state.get("is_live_mode", True) else 1,
            help="Live Analysis queries Copernicus Data Space Ecosystem for any chosen location. Benchmark Demo loads the held-out Madrid scene with 25cm airborne reference or Castile crops."
        )
        is_live = (mode_choice == "Live Analysis (Global CDSE)")
        if is_live != st.session_state.get("is_live_mode", True):
            st.session_state.is_live_mode = is_live
            if not is_live and st.session_state.get("active_scene_key") not in ("demo_madrid", "castile_crops"):
                st.session_state.active_scene_key = "demo_madrid"
                st.session_state.active_lat = 40.6406
                st.session_state.active_lon = -3.1678
                st.session_state.active_location_name = "Madrid / Guadalajara, Spain"
            elif is_live and st.session_state.get("active_scene_key") in ("demo_madrid", "castile_crops"):
                st.session_state.active_scene_key = "hyderabad"
                st.session_state.active_lat = 17.3850
                st.session_state.active_lon = 78.4867
                st.session_state.active_location_name = "Hyderabad, Telangana, India"
            st.rerun()

        # 2. Location & Scene Preset Selector in Sidebar
        all_sidebar_presets = [
            ("Madrid Urban (Held-out Benchmark)", "demo_madrid", 40.6406, -3.1678, "Madrid / Guadalajara, Spain", False),
            ("Castile Cropland (Domain Shift)", "castile_crops", 38.9995, -2.0177, "Castile-La Mancha / León, Spain", False),
            ("Rome, Italy (Live CDSE)", "rome", 41.9028, 12.4964, "Rome, Lazio, Italy", True),
            ("Athens / Greece (Live CDSE)", "athens", 37.9838, 23.7275, "Athens, Attica, Greece", True),
            ("Paris, France (Live CDSE)", "paris", 48.8566, 2.3522, "Paris, Île-de-France, France", True),
            ("Barcelona, Spain (Live CDSE)", "barcelona", 41.3879, 2.1699, "Barcelona, Catalonia, Spain", True),
            ("Seville, Spain (Live CDSE)", "seville", 37.3891, -5.9845, "Seville, Andalusia, Spain", True),
            ("Valencia, Spain (Live CDSE)", "valencia", 39.4699, -0.3763, "Valencia, Spain", True),
            ("Hyderabad, India (Live CDSE)", "hyderabad", 17.3850, 78.4867, "Hyderabad, Telangana, India", True),
            ("Bengaluru, India (Live CDSE)", "bengaluru", 12.9716, 77.5946, "Bengaluru, Karnataka, India", True),
            ("Delhi NCR, India (Live CDSE)", "delhi", 28.6139, 77.2090, "Delhi NCR, India", True),
        ]
        sidebar_keys = [p[1] for p in all_sidebar_presets]
        active_key = st.session_state.get("active_scene_key", "hyderabad" if is_live else "demo_madrid")
        side_idx = sidebar_keys.index(active_key) if active_key in sidebar_keys else 0
        side_sel = st.selectbox(
            "Target Location & Scene",
            options=range(len(all_sidebar_presets)),
            format_func=lambda i: all_sidebar_presets[i][0],
            index=side_idx,
            help="Select any curated benchmark or live CDSE global location."
        )
        if side_sel is not None and all_sidebar_presets[side_sel][1] != active_key:
            chosen = all_sidebar_presets[side_sel]
            st.session_state.active_scene_key = chosen[1]
            st.session_state.active_lat = chosen[2]
            st.session_state.active_lon = chosen[3]
            st.session_state.active_location_name = chosen[4]
            st.session_state.is_live_mode = chosen[5]
            st.rerun()

        st.markdown("---")

        # 2. Downstream Task
        st.selectbox(
            "Downstream Task",
            options=["Built-structure presence verification (Task D1)"],
            index=0,
            help="Task D1: Verify presence of built structures in 2.5m reconstruction."
        )
        st.caption("Proposal detector: Fixed threshold c₀ = 0.02, Area ∈ [16, 400] px (100–2,500 m²).")

        st.markdown("---")

        # 3. Certification Method
        method_choice = st.selectbox(
            "Certification Method",
            options=[
                "Track A — Conformal Risk Control",
                "Track B — BH (Benjamini-Hochberg)",
                "Track B — BY (Benjamini-Yekutieli)"
            ],
            index=0,
            help="Track A bounds normalized loss E[L(λ)] ≤ α. Track B controls False Discovery Rate E[FDR] ≤ α."
        )

        if "Track A" in method_choice:
            method_code = "TRACK_A"
            method_name = "Track A (Conformal Risk Control)"
        elif "BH" in method_choice:
            method_code = "BH"
            method_name = "Track B (Benjamini-Hochberg)"
        else:
            method_code = "BY"
            method_name = "Track B (Benjamini-Yekutieli)"

        # 4. Alpha slider (default explicitly 0.10)
        alpha_val = st.slider(
            "Risk Tolerance (α)",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
            help="Significance level / error budget α. For Track A: E[L] ≤ α. For Track B: E[FDR] ≤ α."
        )

        if method_code == "TRACK_A":
            st.caption(f"Track A guarantees expected loss E[L(λ̂)] ≤ {alpha_val:.2f} under exchangeability.")
        else:
            st.caption(f"Track B controls False Discovery Rate E[FDR] ≤ {alpha_val:.2f} under PRDS.")

        st.markdown("---")
        st.caption("**Calibration Provenance:**")
        st.caption("• 24 site-disjoint calibration scenes (OPENSR_SPAIN_CROPS_01)")
        st.caption("• 191 null calibration risk scores")
        st.caption("• Test tile T30TXM strictly excluded from calibration")

    return is_live, method_code, method_name, alpha_val


def execute_certification_workflow(
    certifier: Any,
    candidates: list[dict[str, Any]],
    hr_reference: Optional[np.ndarray],
    method_code: str,
    alpha: float,
    scene_id: str = "demo_madrid",
    bounds_wgs84: Optional[dict[str, float]] = None
) -> dict[str, Any]:
    """Executes certification and handles live scenes without HR."""
    has_runtime = False
    try:
        from streamlit.runtime import exists
        has_runtime = exists()
    except Exception:
        has_runtime = False

    out_dir = Path(__file__).resolve().parent.parent / "data" / "demo_d1" / "certified"
    os.makedirs(out_dir, exist_ok=True)
    geotiff_path = str(out_dir / f"certified_{method_code.lower()}_alpha_{int(alpha*100):02d}_{scene_id}.tif")
    ref_tif = str(Path(__file__).resolve().parent.parent / "data" / "demo_d1" / "reconstruction_2_5m" / "SEN2SR_2_5m_ROI_00001_B02_2_5m.tif")

    cert_res = certifier.certify_scene(candidates, alpha=alpha, method=method_code)

    if hr_reference is not None:
        eval_res = certifier.evaluate_held_out(cert_res, hr_reference, d_max=2.0)
    else:
        eval_res = {
            "evaluation_mode": "operational_live_satellite",
            "has_hr_reference": False,
            "evaluation_tp": 0,
            "evaluation_fp": 0,
            "evaluation_fdp": 0.0,
            "note": "Operational Live Acquisition: Certified via Conformal Bounds (No 25cm Airborne Reference)."
        }

    # Generate GeoTIFF
    if scene_id == "demo_madrid" and os.path.exists(ref_tif):
        certifier.save_certified_mask_geotiff(cert_res, geotiff_path, reference_geotiff=ref_tif)
    elif bounds_wgs84:
        try:
            import rasterio
            from rasterio.transform import from_bounds
            w, s, e, n = bounds_wgs84["west"], bounds_wgs84["south"], bounds_wgs84["east"], bounds_wgs84["north"]
            with rasterio.open(
                geotiff_path, "w", driver="GTiff", height=512, width=512, count=1,
                dtype=rasterio.uint8, crs="EPSG:4326",
                transform=from_bounds(w, s, e, n, 512, 512)
            ) as dst:
                dst.write(cert_res["selected_mask"], 1)
        except Exception:
            pass

    return {
        "cert_res": cert_res,
        "eval_res": eval_res,
        "geotiff_path": geotiff_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }


# Backwards compatibility alias
execute_certification = execute_certification_workflow


def render_globe_section(scene_data: dict[str, Any], cert_res: dict[str, Any]):
    """Renders the real 3D Earth Globe with AOI footprint and candidate overlays."""
    aoi_info = {
        "center_wgs84": scene_data["center_wgs84"],
        "bounds_wgs84": scene_data["bounds_wgs84"],
        "polygon_coords": scene_data["polygon_coords"],
        "dimensions_km": scene_data["dimensions_km"],
        "area_km2": scene_data["area_km2"]
    }
    candidates = scene_data["candidates"]
    certified_ids = cert_res.get("selected_candidate_ids", [])
    info = scene_data["info"]

    html = generate_cesium_html(
        aoi_info=aoi_info,
        candidates=candidates,
        certified_ids=certified_ids,
        show_footprint=True,
        show_certified=True,
        show_proposed=False,
        auto_fly_to_aoi=False,
        scene_name=info["name"]
    )
    components.html(html, height=560, scrolling=False)
    st.caption(
        "**Interactive 3D Earth:** Mouse drag to rotate • Scroll to zoom • Right-click/Ctrl+drag to tilt. "
        "Use quick chips or search bar to fly to any city. AOI footprint is rendered in cyan."
    )


def render_dual_status_panel(cert_res: dict[str, Any], method_name: str, alpha: float):
    """Renders statistical certification status and domain transfer status."""
    st.markdown('<div class="section-title">2. CERTIFICATION & TRUST STATUS</div>', unsafe_allow_html=True)
    col_cert, col_shift = st.columns(2)

    cert_status = cert_res.get("certificate_status", "UNKNOWN")
    n_selected = cert_res.get("N_selected", 0)

    with col_cert:
        if cert_status == "CERTIFIED":
            lam = cert_res.get("threshold_info", {}).get("lambda_hat")
            lam_text = f"r_k ≤ λ̂ = {lam:.5f}" if lam is not None else "FDR criterion met"
            st.markdown(f"""
            <div class="status-card-cert">
                <div class="status-heading" style="color: #10b981;">CERTIFICATION: CERTIFIED ({n_selected} SELECTED)</div>
                <p style="margin: 0; color: #e2e8f0; font-size: 0.85rem;">
                    <strong>Statistical Criterion:</strong> {n_selected} candidate(s) met conformal risk threshold ({lam_text}) under {method_name} at α = {alpha:.2f}.
                    Guarantees calibrated risk control across exchangeable test scenes.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif cert_status == "EMPTY_SELECTION":
            st.markdown(f"""
            <div class="status-card-empty">
                <div class="status-heading" style="color: #f59e0b;">CERTIFICATION: EMPTY SELECTION (0 SELECTED)</div>
                <p style="margin: 0; color: #e2e8f0; font-size: 0.85rem;">
                    <strong>Statistical Criterion:</strong> No candidates met the selection criterion at α = {alpha:.2f}.
                    All candidate risk scores exceeded the rejection threshold. The system safely abstains rather than manufacturing false discoveries.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-shift">
                <div class="status-heading" style="color: #ef4444;">CERTIFICATION: {cert_status}</div>
                <p style="margin: 0; color: #e2e8f0; font-size: 0.85rem;">Preconditions not met for statistical certification.</p>
            </div>
            """, unsafe_allow_html=True)

    shift_info = cert_res.get("shift_status", {})
    shift_tier = shift_info.get("tier", "ABSTAIN")
    shift_score = shift_info.get("shift_score", 0.147)

    with col_shift:
        if shift_tier == "ABSTAIN":
            st.markdown(f"""
            <div class="status-card-shift">
                <div class="status-heading" style="color: #f59e0b;">TRANSFER STATUS: ABSTAIN (SHIFT DETECTED)</div>
                <p style="margin: 0; color: #e2e8f0; font-size: 0.85rem;">
                    <strong>Estimated Shift Score:</strong> Δ̂ = {shift_score:.3f} (> threshold 0.20).
                    Distribution shift relative to the calibration baseline triggers transfer abstention. Requires recalibration before field deployment.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif shift_tier == "DEGRADED":
            st.markdown(f"""
            <div class="status-card-shift">
                <div class="status-heading" style="color: #f59e0b;">TRANSFER STATUS: DEGRADED (MODERATE SHIFT)</div>
                <p style="margin: 0; color: #e2e8f0; font-size: 0.85rem;">
                    <strong>Estimated Shift Score:</strong> Δ̂ = {shift_score:.3f} (moderate shift in [0.10, 0.20]).
                    Target distribution differs from supported calibration regime. Decision bounds widen for safety.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-cert">
                <div class="status-heading" style="color: #10b981;">TRANSFER STATUS: CERTIFIED (IN-DOMAIN)</div>
                <p style="margin: 0; color: #e2e8f0; font-size: 0.85rem;">
                    <strong>Estimated Shift Score:</strong> Δ̂ = {shift_score:.3f} (negligible shift ≤ 0.10).
                    Scene matches the calibration distribution profile.
                </p>
            </div>
            """, unsafe_allow_html=True)


def render_metrics_cards(cert_res: dict[str, Any], eval_res: dict[str, Any], method_name: str, alpha: float):
    """Renders the 7 key metrics cards."""
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Tolerance α</div>
            <div class="metric-val">{alpha:.2f}</div>
            <div class="metric-sub">Error budget</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Proposed N</div>
            <div class="metric-val">{cert_res.get('N_proposed', 0)}</div>
            <div class="metric-sub">Fixed c₀ = 0.02</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        n_sel = cert_res.get('N_selected', 0)
        sel_color = "#10b981" if n_sel > 0 else "#94a3b8"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Certified</div>
            <div class="metric-val" style="color: {sel_color};">{n_sel}</div>
            <div class="metric-sub">{n_sel}/{cert_res.get('N_proposed', 0)} selected</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        method_short = "Track A (CRC)" if cert_res.get("method") == "TRACK_A" else cert_res.get("method")
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Method</div>
            <div class="metric-val" style="font-size: 1.00rem;">{method_short}</div>
            <div class="metric-sub">Conformal bounds</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        if cert_res.get("method") == "TRACK_A":
            lam = cert_res.get("threshold_info", {}).get("lambda_hat", 0.0)
            lam_str = f"{lam:.4f}"
            lam_sub = "Risk threshold"
        else:
            lam_str = "N/A"
            lam_sub = "λ: not applicable"

        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Threshold λ̂</div>
            <div class="metric-val" style="font-size: 1.10rem;">{lam_str}</div>
            <div class="metric-sub">{lam_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        shift_val = cert_res.get("shift_status", {}).get("shift_score", 0.0)
        shift_tier = cert_res.get("shift_status", {}).get("tier", "NOMINAL")
        shift_val_str = f"{shift_val:.3f}" if shift_val is not None else "0.000"
        tier_color = "#10b981" if shift_tier == "CERTIFIED" else "#f59e0b"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Estimated Shift</div>
            <div class="metric-val" style="color: {tier_color}; font-size: 1.15rem;">{shift_val_str}</div>
            <div class="metric-sub">Tier: {shift_tier}</div>
        </div>
        """, unsafe_allow_html=True)

    with col7:
        has_hr = eval_res.get("has_hr_reference", True)
        if has_hr:
            fdp = eval_res.get("evaluation_fdp", 0.0)
            tp = eval_res.get("evaluation_tp", 0)
            fp = eval_res.get("evaluation_fp", 0)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Empirical FDP</div>
                <div class="metric-val" style="font-size: 1.20rem;">{fdp:.2f}</div>
                <div class="metric-sub">{tp} TP / {fp} FP (post-hoc)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Empirical FDP</div>
                <div class="metric-val" style="font-size: 1.00rem; color: #38bdf8;">Live S2</div>
                <div class="metric-sub">Conformal active</div>
            </div>
            """, unsafe_allow_html=True)


def render_analytical_tabs(
    rasters: dict[str, Any],
    analytical: dict[str, Any],
    proposal_mask: np.ndarray,
    candidates: list[dict[str, Any]],
    cert_res: dict[str, Any],
    method_name: str,
    alpha: float,
    has_hr: bool = False
):
    """Renders the 5 scientific analytical tabs."""
    st.markdown('<div class="section-title">3. SCIENTIFIC EVIDENCE LAYERS</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Observation vs Reconstruction",
        "2. Measurement Support",
        "3. Decision Risk",
        "4. Proposed Detections",
        "5. Certified Detections"
    ])

    # TAB 1: Observation vs Reconstruction
    with tab1:
        st.markdown("#### Sentinel-2 L2A (10m) vs SEN2SR (2.5m) Representation")
        st.caption("Direct comparison: Observed Sentinel-2 vs Super-resolution reconstruction. Extent: 1.28 km × 1.28 km.")

        view_mode_choice = st.radio(
            "Display Mode",
            options=["Split Comparison (Side-by-Side)", "Original Sentinel-2 L2A (10m)", "SEN2SR Reconstruction (2.5m)"],
            horizontal=True
        )
        mode_key = "SPLIT" if "Split" in view_mode_choice else ("ORIGINAL" if "Original" in view_mode_choice else "RECONSTRUCTION")

        fig1 = plot_reconstruction(
            rasters["sr_rgb"],
            l2a_rgb=rasters["l2a_rgb"],
            view_mode=mode_key
        )
        st.pyplot(fig1, use_container_width=True)

        st.info(
            "**Core Scientific Principle:** 2.5m is the reconstruction output grid spacing; "
            "it is not a claim of uniformly resolved 2.5m physical detail. "
            "Downstream detection decisions require empirical verification via CERTUS-S2."
        )

    # TAB 2: Measurement Support
    with tab2:
        st.markdown("#### Measurement Support Map s(p)")
        st.caption("Higher support indicates reconstructed pixels are strongly supported by the Sentinel-2 measurement. Residual e = |y - D̂(x̂)|.")
        fig2 = plot_support_map(analytical["support_2d"])
        st.pyplot(fig2, use_container_width=True)

        stats = analytical["support_stats"]
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Mean Support", f"{stats['mean']:.3f}")
        col_s2.metric("Median Support", f"{stats['median']:.3f}")
        col_s3.metric("Min Support", f"{stats['min']:.3e}")
        col_s4.metric("Max Support", f"{stats['max']:.3f}")

    # TAB 3: Decision Risk
    with tab3:
        st.markdown("#### Continuous Decision Risk Map r(p)")
        st.caption("Continuous decision-risk score r(p) ∈ [0, 1] evaluated across reconstructed pixels.")
        fig3 = plot_risk_map(analytical["risk_2d"])
        st.pyplot(fig3, use_container_width=True)

        rstats = analytical["risk_stats"]
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Mean Decision Risk", f"{rstats['mean']:.3f}")
        col_r2.metric("Median Risk", f"{rstats['median']:.3f}")
        col_r3.metric("Min Risk", f"{rstats['min']:.3f}")
        col_r4.metric("Max Risk", f"{rstats['max']:.3f}")

    # TAB 4: Proposed Detections
    with tab4:
        st.markdown("#### Fixed D1 Proposed Detections")
        st.caption(f"Total candidates proposed: N = {len(candidates)} structures (threshold c₀ = 0.02, area [16, 400] px).")
        fig4 = plot_proposed_detections(rasters["sr_rgb"], proposal_mask, candidates)
        st.pyplot(fig4, use_container_width=True)

        total_area_m2 = sum(c.get("area_m2", c.get("area_px", 0) * 6.25) for c in candidates)
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Total Candidates (N)", f"{len(candidates)}")
        col_p2.metric("Proposed Built Footprint", f"{total_area_m2:,.1f} m²")
        col_p3.metric("Proposal Threshold c₀", "0.02")

    # TAB 5: Certified Detections
    with tab5:
        st.markdown("#### Certified Candidate Selection")
        st.caption("Candidate structures meeting statistical guarantees under the specified track.")

        cert_mask = cert_res.get("selected_mask", np.zeros((512, 512), dtype=np.uint8))
        cert_records = [c for c in cert_res.get("candidate_results", []) if c.get("certified", False)]

        lam_label = None
        if cert_res.get("method") == "TRACK_A":
            lam = cert_res.get("threshold_info", {}).get("lambda_hat", 0.0)
            lam_label = f"λ̂={lam:.4f}"

        fig5 = plot_certified_detections(
            rasters["sr_rgb"],
            cert_mask,
            cert_records,
            method_name=method_name,
            alpha=alpha,
            threshold_label=lam_label
        )
        st.pyplot(fig5, use_container_width=True)

        if len(cert_records) == 0:
            st.info(
                "**Statistical Abstention:** No candidates met the selection criterion at this α. "
                "Rather than manufacturing false positive detections, CERTUS-S2 safely abstains."
            )


def render_candidate_table(cert_res: dict[str, Any], candidates: list[dict[str, Any]]):
    """Renders candidate evaluation table with coordinates."""
    st.markdown('<div class="section-title">4. CANDIDATE EVALUATION TABLE</div>', unsafe_allow_html=True)

    cand_count = len(candidates)
    filter_choice = st.radio(
        "Filter",
        options=[f"All Candidates (N={cand_count})", "Certified Only"],
        horizontal=True
    )

    cand_dict = {c["id"]: c for c in candidates}
    rows = []
    for r in cert_res.get("candidate_results", []):
        c_id = r["id"]
        c_meta = cand_dict.get(c_id, {})
        is_cert = r.get("certified", False)

        if filter_choice == "Certified Only" and not is_cert:
            continue

        area_px = r.get("area_px", c_meta.get("area_px", 0))
        area_m2 = area_px * 6.25
        risk_score = r.get("risk", c_meta.get("risk", 0.0))
        p_val = r.get("p_value")

        is_td = c_meta.get("is_true_detection", None)
        dist_hr = c_meta.get("dist_to_reference_px", None)
        if is_td is True:
            hr_label = "True Detection"
        elif is_td is False:
            hr_label = "False Discovery"
        else:
            hr_label = "Operational (Conformal)"

        rows.append({
            "ID": f"#{c_id}",
            "Risk (r_k)": f"{risk_score:.4f}",
            "Area (m²)": f"{area_m2:.1f}",
            "Area (px)": area_px,
            "Latitude": f"{c_meta.get('lat', 0.0):.5f}°N",
            "Longitude": f"{c_meta.get('lon', 0.0):.5f}°E",
            "Certified": "Certified" if is_cert else "Rejected",
            "p-value": f"{p_val:.4f}" if p_val is not None else "N/A",
            "Validation": hr_label,
            "Distance to HR (px)": f"{dist_hr:.2f}" if dist_hr is not None else "N/A"
        })

    if len(rows) > 0:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No candidates match the current filter (0 candidates certified).")


def render_provenance_and_downloads(
    cert_res: dict[str, Any],
    meta: dict[str, Any],
    eval_res: dict[str, Any],
    geotiff_path: str,
    timestamp: str,
    scene_id: str = "demo_madrid"
):
    """Renders provenance audit section and download buttons."""
    st.markdown('<div class="section-title">5. PROVENANCE & ARTIFACT EXPORT</div>', unsafe_allow_html=True)

    with st.expander("Certificate / Provenance Audit Trail", expanded=False):
        cert_export = {
            "scene_id": scene_id,
            "scene_identifier": meta.get("scene_identifier", scene_id),
            "roi": meta.get("roi", "CUSTOM_ROI"),
            "tile": meta.get("mgrs_tile", meta.get("tile", "GLOBAL")),
            "timestamp": timestamp,
            "task": "D1 Built-Structure Presence Verification",
            "method": cert_res.get("method"),
            "alpha": cert_res.get("alpha"),
            "certificate_status": cert_res.get("certificate_status"),
            "N_proposed": cert_res.get("N_proposed"),
            "N_selected": cert_res.get("N_selected"),
            "selected_candidate_ids": cert_res.get("selected_candidate_ids", []),
            "threshold_info": cert_res.get("threshold_info", {}),
            "risk_statistics": cert_res.get("risk_statistics", {}),
            "shift_status": cert_res.get("shift_status", {}),
            "calibration_artifact": cert_res.get("certification_metadata", {}).get("calibration_artifact"),
            "post_hoc_evaluation": eval_res,
            "theoretical_semantics": cert_res.get("certification_metadata", {}).get("theoretical_semantics", {})
        }
        st.json(cert_export)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if os.path.exists(geotiff_path):
            with open(geotiff_path, "rb") as f:
                geo_bytes = f.read()
            st.download_button(
                label="Download certified GeoTIFF",
                data=geo_bytes,
                file_name=f"certus_s2_certified_{cert_res.get('method', 'track_a').lower()}_{scene_id}.tif",
                mime="image/tiff",
                use_container_width=True
            )
        else:
            st.button("Download certified GeoTIFF (Not available)", disabled=True, use_container_width=True)

    with col_d2:
        cert_export = {
            "scene_id": scene_id,
            "timestamp": timestamp,
            "method": cert_res.get("method"),
            "alpha": cert_res.get("alpha"),
            "certificate_status": cert_res.get("certificate_status"),
            "N_proposed": cert_res.get("N_proposed"),
            "N_selected": cert_res.get("N_selected"),
            "selected_candidate_ids": cert_res.get("selected_candidate_ids", []),
            "threshold_info": cert_res.get("threshold_info", {}),
            "risk_statistics": cert_res.get("risk_statistics", {}),
            "shift_status": cert_res.get("shift_status", {}),
            "candidate_results": cert_res.get("candidate_results", []),
            "post_hoc_evaluation": eval_res,
            "certification_metadata": cert_res.get("certification_metadata", {})
        }
        json_str = json.dumps(cert_export, indent=2)
        st.download_button(
            label="Download certificate JSON",
            data=json_str,
            file_name=f"certus_s2_certificate_{cert_res.get('method', 'track_a').lower()}_{scene_id}.json",
            mime="application/json",
            use_container_width=True
        )


def main():
    """Main application orchestrator."""
    # 1. State initialization
    if "is_live_mode" not in st.session_state:
        st.session_state.is_live_mode = True
    if "active_lat" not in st.session_state:
        st.session_state.active_lat = 17.3850  # Default Hyderabad
    if "active_lon" not in st.session_state:
        st.session_state.active_lon = 78.4867
    if "active_location_name" not in st.session_state:
        st.session_state.active_location_name = "Hyderabad, Telangana, India"
    if "active_date" not in st.session_state:
        st.session_state.active_date = "2024-04-27"
    if "active_scene_key" not in st.session_state:
        st.session_state.active_scene_key = "custom"
    if "available_scenes" not in st.session_state:
        st.session_state.available_scenes = []

    # 2. Sidebar parameters
    is_live, method_code, method_name, alpha = render_sidebar()

    # Header
    current_mode_str = "LIVE ANALYSIS" if is_live else "BENCHMARK DEMO (Madrid / Guadalajara)"
    render_header(current_mode_str)

    # 3. Location & Navigation Action Bar
    st.markdown('<div class="section-title">1. 3D EARTH EXPLORER & TARGET AOI</div>', unsafe_allow_html=True)

    col_act1, col_act2, col_act3 = st.columns([5, 3, 2])

    with col_act1:
        search_query = st.text_input(
            "Search Location",
            value="",
            placeholder="Search city (e.g. Hyderabad, Bengaluru, Delhi, Rome, Athens, Paris, Madrid)...",
            label_visibility="collapsed"
        )
        if search_query:
            geo_res = geocode_location(search_query)
            if geo_res:
                st.session_state.active_lat = geo_res["lat"]
                st.session_state.active_lon = geo_res["lon"]
                st.session_state.active_location_name = geo_res["name"]
                st.session_state.active_scene_key = "custom"
                st.session_state.is_live_mode = True
                st.success(f"Located: {geo_res['name']} ({geo_res['lat']:.4f}°N, {geo_res['lon']:.4f}°E)")
            else:
                st.warning(f"Could not find coordinates for '{search_query}'.")

    with col_act2:
        # Quick location selector with all benchmark and live locations
        chip_options = [
            ("Madrid (Benchmark)", 40.6406, -3.1678, "Madrid / Guadalajara, Spain", "demo_madrid"),
            ("Castile Cropland", 38.9995, -2.0177, "Castile-La Mancha / León, Spain", "castile_crops"),
            ("Rome", 41.9028, 12.4964, "Rome, Lazio, Italy", "rome"),
            ("Athens / Greece", 37.9838, 23.7275, "Athens, Attica, Greece", "athens"),
            ("Paris", 48.8566, 2.3522, "Paris, Île-de-France, France", "paris"),
            ("Barcelona", 41.3879, 2.1699, "Barcelona, Catalonia, Spain", "barcelona"),
            ("Seville", 37.3891, -5.9845, "Seville, Andalusia, Spain", "seville"),
            ("Valencia", 39.4699, -0.3763, "Valencia, Spain", "valencia"),
            ("Hyderabad", 17.3850, 78.4867, "Hyderabad, Telangana, India", "hyderabad"),
            ("Bengaluru", 12.9716, 77.5946, "Bengaluru, Karnataka, India", "bengaluru"),
            ("Delhi", 28.6139, 77.2090, "Delhi NCR, India", "delhi")
        ]
        chip_names = [c[0] for c in chip_options]
        chip_keys = [c[4] for c in chip_options]
        active_chip_key = st.session_state.get("active_scene_key", "hyderabad" if is_live else "demo_madrid")
        selected_chip_idx = (chip_keys.index(active_chip_key) + 1) if active_chip_key in chip_keys else 0
        sel_chip = st.selectbox(
            "Quick Location Preset",
            options=["-- Quick Locations --"] + chip_names,
            index=selected_chip_idx,
            label_visibility="collapsed"
        )
        if sel_chip != "-- Quick Locations --":
            for c in chip_options:
                if c[0] == sel_chip and c[4] != active_chip_key:
                    st.session_state.active_lat = c[1]
                    st.session_state.active_lon = c[2]
                    st.session_state.active_location_name = c[3]
                    target_key = c[4]
                    st.session_state.active_scene_key = target_key
                    if target_key in ("demo_madrid", "castile_crops"):
                        st.session_state.is_live_mode = False
                    else:
                        st.session_state.is_live_mode = True
                    st.rerun()
                    break

    with col_act3:
        if is_live:
            if st.button("GO TO BENCHMARK", use_container_width=True):
                st.session_state.is_live_mode = False
                st.session_state.active_scene_key = "demo_madrid"
                st.session_state.active_lat = 40.6406
                st.session_state.active_lon = -3.1678
                st.session_state.active_location_name = "Madrid / Guadalajara, Spain"
                st.rerun()
        else:
            if st.button("SWITCH TO LIVE", use_container_width=True):
                st.session_state.is_live_mode = True
                st.session_state.active_scene_key = "hyderabad"
                st.session_state.active_lat = 17.3850
                st.session_state.active_lon = 78.4867
                st.session_state.active_location_name = "Hyderabad, Telangana, India"
                st.rerun()

    # Determine active scene key
    scene_key_to_load = st.session_state.get("active_scene_key", "demo_madrid")
    if not is_live and scene_key_to_load not in ("demo_madrid", "castile_crops"):
        scene_key_to_load = "demo_madrid"

    # 4. Load Scene Data
    with st.spinner("Loading geospatial data & executing SEN2SR super-resolution..."):
        scene_data = load_scene_data(
            scene_key_to_load,
            custom_lat=st.session_state.active_lat,
            custom_lon=st.session_state.active_lon,
            selected_date=st.session_state.active_date,
            location_name=st.session_state.active_location_name
        )

    candidates = scene_data["candidates"]
    hr_reference = scene_data.get("hr_reference")

    # 5. Execute Certification
    certifier = get_certifier()
    cert_execution = execute_certification_workflow(
        certifier=certifier,
        candidates=candidates,
        hr_reference=hr_reference,
        method_code=method_code,
        alpha=alpha,
        scene_id=scene_key_to_load,
        bounds_wgs84=scene_data["bounds_wgs84"]
    )
    active_cert = cert_execution["cert_res"]
    active_eval = cert_execution["eval_res"]
    active_tif = cert_execution["geotiff_path"]
    active_time = cert_execution["timestamp"]

    # 6. Render 3D Earth Globe
    render_globe_section(scene_data, active_cert)

    # 7. Selected Location & CDSE Observation Workflow (Matching Section 13)
    col_loc1, col_loc2 = st.columns([1, 1])

    with col_loc1:
        bounds = scene_data["bounds_wgs84"]
        center = scene_data["center_wgs84"]
        if scene_key_to_load == "demo_madrid":
            source_badge = "HELD-OUT BENCHMARK (25cm PNOA)"
        elif scene_key_to_load == "castile_crops":
            source_badge = "DOMAIN-SHIFT BENCHMARK (CROPLAND)"
        else:
            source_badge = "COPERNICUS DATA SPACE (CDSE L2A)"

        st.markdown(f"""
        <div class="aoi-card">
            <div class="aoi-title">SELECTED LOCATION: {st.session_state.active_location_name}</div>
            <div class="aoi-detail"><strong>Coordinates:</strong> Lat: {center['lat']:.4f}°N, Lon: {center['lon']:.4f}°E</div>
            <div class="aoi-detail"><strong>AOI Footprint:</strong> 1.28 km × 1.28 km (1.64 km² / 163.8 ha)</div>
            <div class="aoi-detail"><strong>Bounding Box (WGS84):</strong> [{bounds['west']:.4f}°, {bounds['south']:.4f}°] to [{bounds['east']:.4f}°, {bounds['north']:.4f}°]</div>
            <div class="aoi-detail"><strong>Resolution:</strong> Sentinel-2 L2A (10m) → SEN2SR (2.5m grid, 512×512 px)</div>
            <div class="aoi-badge">{source_badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_loc2:
        if is_live:
            st.markdown('<div class="aoi-title" style="margin-top: 5px;">COPERNICUS DATA SPACE ECOSYSTEM (CDSE)</div>', unsafe_allow_html=True)
            col_btn, col_txt = st.columns([1, 2])
            with col_btn:
                if st.button("SEARCH CDSE", type="primary", use_container_width=True):
                    with st.spinner("Querying CDSE STAC Catalog for Sentinel-2 L2A scenes..."):
                        scenes = search_available_observations(st.session_state.active_lat, st.session_state.active_lon)
                        st.session_state.available_scenes = scenes
                        if len(scenes) > 0:
                            st.session_state.active_date = scenes[0]["date"]
            with col_txt:
                st.caption(f"Queries real Sentinel-2 L2A acquisitions intersecting ({st.session_state.active_lat:.3f}°N, {st.session_state.active_lon:.3f}°E).")

            # Display real available Sentinel-2 scenes
            if len(st.session_state.available_scenes) > 0:
                st.markdown("**Available Sentinel-2 L2A Observations:**")
                scene_labels = [
                    f"{s['date']} | Cloud: {s['cloud_cover']}% | Tile: {s['tile']} | {s['platform']}"
                    for s in st.session_state.available_scenes
                ]
                sel_scene_idx = st.selectbox(
                    "Select Acquisition",
                    options=range(len(scene_labels)),
                    format_func=lambda i: scene_labels[i],
                    index=0
                )
                selected_scene = st.session_state.available_scenes[sel_scene_idx]
                if selected_scene["date"] != st.session_state.active_date:
                    st.session_state.active_date = selected_scene["date"]
                    st.rerun()

                run_btn = st.button("RUN CERTUS-S2 ON SELECTED SCENE", type="primary", use_container_width=True)
                if run_btn:
                    st.rerun()
            else:
                # Default quick observation notice
                st.info(f"Active Sentinel-2 Observation Date: **{st.session_state.active_date}** (Tile intersecting AOI). Click **SEARCH CDSE** to query recent catalog.")
                if st.button("RUN CERTUS-S2", type="primary", use_container_width=True):
                    st.rerun()
        else:
            st.markdown('<div class="aoi-title" style="margin-top: 5px;">REPRODUCIBLE BENCHMARK DATASET</div>', unsafe_allow_html=True)
            if scene_key_to_load == "castile_crops":
                st.write("Domain-shift test scene: **Castile Cropland (OpenSR-S2 Agricultural Benchmark)**.")
                st.caption("Demonstrates honest Tier 3 Abstain under severe distribution shift (dense agricultural crop parcels).")
                if st.button("RUN CERTUS-S2 (CASTILE CROPS)", type="primary", use_container_width=True):
                    st.rerun()
            else:
                st.write("Held-out test scene: **Madrid / Guadalajara (T30TXM, ROI_00001)**.")
                st.caption("Includes quarantined 25cm airborne orthophoto (PNOA) for post-hoc validation.")
                if st.button("RUN CERTUS-S2 (BENCHMARK)", type="primary", use_container_width=True):
                    st.rerun()

    st.markdown("---")

    # 8. Dual Status Panel
    render_dual_status_panel(active_cert, method_name, alpha)

    # 9. Key Metrics Cards
    render_metrics_cards(active_cert, active_eval, method_name, alpha)

    st.markdown("---")

    # 10. Analytical Layers (5 Scientific Tabs)
    render_analytical_tabs(
        rasters={
            "l2a": scene_data["l2a"],
            "l2a_rgb": scene_data["l2a_rgb"],
            "sr": scene_data["sr"],
            "sr_rgb": scene_data["sr_rgb"],
            "hr_reference": scene_data.get("hr_reference")
        },
        analytical=scene_data["analytical_layers"],
        proposal_mask=scene_data["prop_mask"],
        candidates=candidates,
        cert_res=active_cert,
        method_name=method_name,
        alpha=alpha,
        has_hr=scene_data.get("has_hr", False)
    )

    st.markdown("---")

    # 11. Candidate Evaluation Table
    render_candidate_table(active_cert, candidates)

    st.markdown("---")

    # 12. Provenance & Artifact Export
    render_provenance_and_downloads(
        cert_res=active_cert,
        meta=scene_data["metadata"],
        eval_res=active_eval,
        geotiff_path=active_tif,
        timestamp=active_time,
        scene_id=scene_key_to_load
    )


if __name__ == "__main__":
    main()
