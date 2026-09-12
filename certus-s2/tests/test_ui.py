"""
Tests for CERTUS-S2 Streamlit UI components and data pipelines.
"""

import os
import json
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pydeck as pdk

from ui.data_loader import (
    get_certifier,
    load_scene_metadata,
    load_demo_rasters,
    compute_analytical_layers_data,
    load_proposal_candidates,
    load_aoi_spatial_info
)
from ui.visualizer import (
    plot_reconstruction,
    plot_support_map,
    plot_risk_map,
    plot_proposed_detections,
    plot_certified_detections
)
from ui.map_component import build_geospatial_deck
from ui.cesium_component import generate_cesium_html
from ui.app import execute_certification


def test_ui_data_loader_metadata():
    """Verifies scene metadata loading for Madrid demo scene."""
    meta = load_scene_metadata()
    assert meta["roi"] == "ROI_00001"
    assert "T30TXM" in meta["mgrs_tile"]
    assert meta["crs"] == "EPSG:32630"


def test_ui_aoi_spatial_info():
    """Verifies exact WGS84 bounding box, center, and area computation."""
    aoi = load_aoi_spatial_info()
    assert "center_wgs84" in aoi
    assert 41.6 < aoi["center_wgs84"]["lat"] < 41.7
    assert -1.05 < aoi["center_wgs84"]["lon"] < -0.95
    assert len(aoi["polygon_coords"]) == 5
    assert 1.2 <= aoi["dimensions_km"][0] <= 1.35
    assert 1.5 <= aoi["area_km2"] <= 1.8


def test_ui_geospatial_deck():
    """Verifies pydeck Deck.gl map generation with AOI and candidate layers."""
    aoi = load_aoi_spatial_info()
    cands, _, _ = load_proposal_candidates()
    deck = build_geospatial_deck(
        aoi_info=aoi,
        candidates=cands,
        certified_ids=[15, 16],
        show_footprint=True,
        show_certified=True,
        show_proposed=True
    )
    assert isinstance(deck, pdk.Deck)
    assert len(deck.layers) == 4  # Footprint + Center Marker + Proposed + Certified


def test_ui_cesium_globe_html():
    """Verifies CesiumJS 3D Earth Globe HTML generation with realistic satellite basemap and overlays."""
    aoi = load_aoi_spatial_info()
    cands, _, _ = load_proposal_candidates()
    html = generate_cesium_html(
        aoi_info=aoi,
        candidates=cands,
        certified_ids=[15, 16],
        show_footprint=True,
        show_certified=True,
        show_proposed=False,
        auto_fly_to_aoi=False
    )
    assert isinstance(html, str)
    assert "cesiumContainer" in html
    assert "CERTUS-S2 DEMO AOI" in html
    assert "btnFlyAOI" in html
    assert "btnGlobal" in html
    assert "coordsDisplay" in html
    assert "ArcGIS/rest/services/World_Imagery" in html


def test_ui_data_loader_rasters():
    """Verifies raster loading and RGB stretching."""
    rasters = load_demo_rasters()
    assert rasters["l2a"].shape == (4, 128, 128)
    assert rasters["sr"].shape == (4, 512, 512)
    assert rasters["sr_rgb"].shape == (512, 512, 3)
    assert rasters["hr_reference"].shape == (4, 512, 512)
    assert 0.0 <= np.min(rasters["sr_rgb"]) <= 1.0
    assert 0.0 <= np.max(rasters["sr_rgb"]) <= 1.0


def test_ui_analytical_layers():
    """Verifies observation operator, support map, and risk map precomputation."""
    layers = compute_analytical_layers_data()
    support = layers["support_2d"]
    risk = layers["risk_2d"]
    
    assert support.shape == (512, 512)
    assert risk.shape == (512, 512)
    assert 0.0 <= layers["support_stats"]["min"] <= layers["support_stats"]["max"] <= 1.0
    assert 0.0 <= layers["risk_stats"]["min"] <= layers["risk_stats"]["max"] <= 1.0
    assert np.all(np.isfinite(support))
    assert np.all(np.isfinite(risk))


def test_ui_proposal_candidates_count():
    """Verifies fixed D1 proposal candidates count and masks."""
    cands, mask, labeled = load_proposal_candidates()
    assert len(cands) == 109
    assert mask.shape == (512, 512)
    assert np.sum(mask > 0) == 4740
    
    # Check that each candidate has an exact spatial mask attached
    for c in cands:
        assert "mask" in c
        assert isinstance(c["mask"], np.ndarray)
        assert c["mask"].shape == (512, 512)


def test_ui_certification_track_a():
    """Verifies Track A certification execution in the UI flow."""
    cands, _, _ = load_proposal_candidates()
    rasters = load_demo_rasters()
    certifier = get_certifier()
    
    # Alpha = 0.10 -> 1 certified candidate
    res10 = execute_certification(certifier, cands, rasters["hr_reference"], "TRACK_A", alpha=0.10)
    assert res10["cert_res"]["certificate_status"] == "CERTIFIED"
    assert res10["cert_res"]["N_selected"] == 1
    assert res10["eval_res"]["evaluation_tp"] == 1
    assert res10["eval_res"]["evaluation_fp"] == 0
    assert res10["eval_res"]["evaluation_fdp"] == 0.0
    assert os.path.exists(res10["geotiff_path"])
    
    # Alpha = 0.20 -> 7 certified candidates
    res20 = execute_certification(certifier, cands, rasters["hr_reference"], "TRACK_A", alpha=0.20)
    assert res20["cert_res"]["certificate_status"] == "CERTIFIED"
    assert res20["cert_res"]["N_selected"] == 7
    assert res20["eval_res"]["evaluation_tp"] == 4
    assert res20["eval_res"]["evaluation_fp"] == 3


def test_ui_certification_track_b_honest_empty():
    """Verifies Track B honest EMPTY_SELECTION abstention in UI flow."""
    cands, _, _ = load_proposal_candidates()
    rasters = load_demo_rasters()
    certifier = get_certifier()
    
    # Track B BH alpha = 0.10 -> EMPTY_SELECTION
    res_bh = execute_certification(certifier, cands, rasters["hr_reference"], "BH", alpha=0.10)
    assert res_bh["cert_res"]["certificate_status"] == "EMPTY_SELECTION"
    assert res_bh["cert_res"]["N_selected"] == 0
    assert res_bh["eval_res"]["evaluation_tp"] == 0
    assert res_bh["eval_res"]["evaluation_fp"] == 0
    assert res_bh["eval_res"]["evaluation_fdp"] == 0.0
    
    # Track B BY alpha = 0.10 -> EMPTY_SELECTION
    res_by = execute_certification(certifier, cands, rasters["hr_reference"], "BY", alpha=0.10)
    assert res_by["cert_res"]["certificate_status"] == "EMPTY_SELECTION"
    assert res_by["cert_res"]["N_selected"] == 0


def test_ui_visualizer_renders():
    """Verifies all 5 scientific visualizer plotting functions produce valid figures."""
    rasters = load_demo_rasters()
    layers = compute_analytical_layers_data()
    cands, mask, _ = load_proposal_candidates()
    
    fig1 = plot_reconstruction(rasters["sr_rgb"], rasters["l2a_rgb"], show_comparison=True)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)
    
    fig2 = plot_support_map(layers["support_2d"])
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)
    
    fig3 = plot_risk_map(layers["risk_2d"])
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)
    
    fig4 = plot_proposed_detections(rasters["sr_rgb"], mask, cands)
    assert isinstance(fig4, plt.Figure)
    plt.close(fig4)
    
    # Test certified visualization with empty selection
    fig5_empty = plot_certified_detections(rasters["sr_rgb"], np.zeros((512, 512)), [], "Track B (BH)", 0.10)
    assert isinstance(fig5_empty, plt.Figure)
    plt.close(fig5_empty)
    
    # Test certified visualization with selected candidates
    fig5_cert = plot_certified_detections(rasters["sr_rgb"], mask, [cands[0]], "Track A (CRC)", 0.10, "lambda=0.184")
    assert isinstance(fig5_cert, plt.Figure)
    plt.close(fig5_cert)
