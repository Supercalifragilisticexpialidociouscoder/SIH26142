"""
Unit tests for CDSE Client and Live Satellite Ingestion.
Verifies OAuth token lifecycle, bounding box calculation, array shape & normalization,
and multi-scene loader integration.
"""

import os
import pytest
import numpy as np
from src.data.cdse_client import CDSEClient
from src.ingestion.provider import CDSEProvider
from ui.data_loader import SCENE_CATALOG, load_scene_data


def test_cdse_calculate_aoi_bbox():
    """Verifies WGS84 bounding box geometry calculation."""
    lat, lon = 40.4168, -3.7038  # Madrid center
    bbox = CDSEClient.calculate_aoi_bbox(lat, lon, size_meters=1280.0)

    assert len(bbox) == 4
    west, south, east, north = bbox

    assert west < lon < east
    assert south < lat < north
    # Ensure reasonable geographic span (~0.01 degree)
    assert 0.005 < (east - west) < 0.03
    assert 0.005 < (north - south) < 0.03


def test_cdse_client_configured_status():
    """Verifies credential detection."""
    client = CDSEClient()
    # If .env is present with user credentials, it should be configured
    assert client.is_configured is True
    assert client.client_id.startswith("sh-")


def test_cdse_token_acquisition():
    """Verifies OAuth2 Bearer token acquisition from CDSE identity service."""
    client = CDSEClient()
    token = client.get_token()
    assert isinstance(token, str)
    assert len(token) > 50

    # Repeat should return cached token without re-requesting
    cached_token = client.get_token()
    assert cached_token == token


def test_cdse_fetch_live_patch():
    """Verifies live acquisition and normalization of 4-band Sentinel-2 L2A crop."""
    client = CDSEClient()
    # Test on Barcelona coordinates
    patch = client.fetch_sentinel2_l2a_patch(lat=41.3879, lon=2.1699, use_cache=True)

    assert "l2a" in patch
    assert "l2a_rgb" in patch
    assert "bounds_wgs84" in patch

    l2a = patch["l2a"]
    assert l2a.shape == (4, 128, 128)
    assert l2a.dtype == np.float32
    assert 0.0 <= np.min(l2a) <= np.max(l2a) <= 1.0

    l2a_rgb = patch["l2a_rgb"]
    assert l2a_rgb.shape == (128, 128, 3)
    assert 0.0 <= np.min(l2a_rgb) <= np.max(l2a_rgb) <= 1.0


def test_cdse_provider_contract():
    """Verifies CDSEProvider adheres to L2AProvider interface."""
    provider = CDSEProvider()
    bbox = (-3.72, 40.40, -3.68, 40.44)
    scene = provider.acquire_scene(bbox=bbox, datetime="2024-05-01/2024-08-31")

    assert "bands" in scene
    assert "B02" in scene["bands"]
    assert "B03" in scene["bands"]
    assert "B04" in scene["bands"]
    assert "B08" in scene["bands"]
    assert scene["bands"]["B02"].shape == (128, 128)


def test_multi_scene_catalog_and_loader():
    """Verifies scene catalog definitions and multi-scene loading."""
    assert "demo_madrid" in SCENE_CATALOG
    assert "barcelona" in SCENE_CATALOG
    assert "castile_crops" in SCENE_CATALOG
    assert len(SCENE_CATALOG) >= 8

    # Test loading Castile crops domain-shift scene
    crops = load_scene_data("castile_crops")
    assert crops["l2a"].shape == (4, 128, 128)
    assert crops["sr"].shape == (4, 512, 512)
    assert crops["analytical_layers"]["support_2d"].shape == (512, 512)
    assert crops["analytical_layers"]["risk_2d"].shape == (512, 512)
    assert len(crops["candidates"]) >= 0
