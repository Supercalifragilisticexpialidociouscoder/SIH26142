"""
Copernicus Data Space Ecosystem (CDSE) Live Sentinel-2 Acquisition Client.
Connects to CDSE OAuth2 token service and Sentinel Hub Process API to retrieve
calibrated Sentinel-2 L2A surface reflectance (B02, B03, B04, B08) for arbitrary
geographic coordinates.
"""

import os
import io
import time
import math
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import requests
import numpy as np
import tifffile
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / "data" / "cdse_cache"


class CDSEClient:
    """
    Client for acquiring live Sentinel-2 L2A observations from the
    Copernicus Data Space Ecosystem (CDSE).
    """
    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        env_file: Optional[str] = None
    ):
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv(BASE_DIR / ".env")

        self.client_id = client_id or os.getenv("CDSE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("CDSE_CLIENT_SECRET")

        self._access_token: Optional[str] = None
        self._token_expiry_timestamp: float = 0.0

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_configured(self) -> bool:
        """Checks if credentials are configured."""
        return bool(self.client_id and self.client_secret)

    def get_token(self, force_refresh: bool = False) -> str:
        """
        Retrieves or refreshes the OAuth2 Bearer token from CDSE identity service.
        """
        if not self.is_configured:
            raise ValueError(
                "CDSE credentials missing. Please set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in .env."
            )

        now = time.time()
        if (
            not force_refresh
            and self._access_token
            and now < (self._token_expiry_timestamp - 60)
        ):
            return self._access_token

        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials"
        }

        try:
            resp = requests.post(self.TOKEN_URL, data=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            self._access_token = data["access_token"]
            expires_in = data.get("expires_in", 1800)
            self._token_expiry_timestamp = now + expires_in
            return self._access_token
        except Exception as e:
            raise ConnectionError(f"Failed to authenticate with CDSE: {e}")

    @staticmethod
    def calculate_aoi_bbox(
        lat: float,
        lon: float,
        size_meters: float = 1280.0
    ) -> List[float]:
        """
        Calculates a WGS84 bounding box [west, south, east, north] centered at (lat, lon)
        corresponding to approximately `size_meters` x `size_meters`.
        """
        # 1 deg latitude is approx 111,320 m
        meters_per_deg_lat = 111320.0
        dlat = (size_meters / 2.0) / meters_per_deg_lat

        # Longitude distance depends on latitude
        lat_rad = math.radians(lat)
        cos_lat = max(math.cos(lat_rad), 0.01)
        meters_per_deg_lon = meters_per_deg_lat * cos_lat
        dlon = (size_meters / 2.0) / meters_per_deg_lon

        west = round(lon - dlon, 6)
        south = round(lat - dlat, 6)
        east = round(lon + dlon, 6)
        north = round(lat + dlat, 6)

        return [west, south, east, north]

    CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"

    def search_sentinel2_l2a_scenes(
        self,
        lat: float,
        lon: float,
        start_date: str = "2024-01-01",
        end_date: str = "2024-09-01",
        limit: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Queries CDSE STAC / Catalog for available real Sentinel-2 L2A acquisitions
        intersecting the given coordinates.
        """
        token = self.get_token()
        bbox = self.calculate_aoi_bbox(lat, lon, size_meters=1280.0)

        body = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "limit": limit
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        try:
            resp = requests.post(self.CATALOG_URL, json=body, headers=headers, timeout=15)
            resp.raise_for_status()
            features = resp.json().get("features", [])

            scenes = []
            for f in features:
                props = f.get("properties", {})
                raw_dt = props.get("datetime", "")
                date_str = raw_dt[:10] if len(raw_dt) >= 10 else "Unknown"
                cloud_pct = float(props.get("eo:cloud_cover", 0.0))
                p_id = f.get("id", "")
                parts = p_id.split("_")
                tile = parts[-2] if len(parts) > 2 else "L2A"
                platform = props.get("platform", "sentinel-2").upper()

                scenes.append({
                    "id": p_id,
                    "date": date_str,
                    "datetime": raw_dt,
                    "cloud_cover": round(cloud_pct, 1),
                    "tile": tile,
                    "platform": platform,
                    "epsg": props.get("proj:epsg")
                })
            scenes.sort(key=lambda s: (s["cloud_cover"], s["date"]))
            return scenes
        except Exception as e:
            print(f"CDSE Catalog search warning: {e}")
            return []

    def fetch_sentinel2_l2a_patch(
        self,
        lat: float,
        lon: float,
        width: int = 128,
        height: int = 128,
        date: Optional[str] = None,
        time_from: str = "2024-05-01T00:00:00Z",
        time_to: str = "2024-09-01T00:00:00Z",
        max_cloud_coverage: int = 25,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Acquires a 10m 4-band Sentinel-2 L2A crop [B02, B03, B04, B08]
        centered around (lat, lon).
        """
        if date:
            time_from = f"{date}T00:00:00Z"
            time_to = f"{date}T23:59:59Z"
            date_tag = date
        else:
            date_tag = "recent"

        bbox = self.calculate_aoi_bbox(lat, lon, size_meters=1280.0)
        cache_key = f"{lat:.4f}_{lon:.4f}_{date_tag}_{width}x{height}"
        cache_file = CACHE_DIR / f"{cache_key}.npz"
        cache_meta = CACHE_DIR / f"{cache_key}.json"

        if use_cache and cache_file.exists() and cache_meta.exists():
            try:
                npz_data = np.load(cache_file)
                l2a = npz_data["l2a"]
                l2a_rgb = npz_data["l2a_rgb"]
                with open(cache_meta, "r") as f:
                    meta = json.load(f)
                return {
                    "l2a": l2a,
                    "l2a_rgb": l2a_rgb,
                    "bbox": bbox,
                    "center": {"lat": lat, "lon": lon},
                    "bounds_wgs84": {
                        "west": bbox[0],
                        "south": bbox[1],
                        "east": bbox[2],
                        "north": bbox[3]
                    },
                    "dimensions_km": (1.28, 1.28),
                    "cached": True,
                    "metadata": meta
                }
            except Exception:
                pass  # Fall back to live fetch if cache corrupted

        token = self.get_token()

        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B02", "B03", "B04", "B08"],
            output: { bands: 4, sampleType: "FLOAT32" }
          };
        }
        function evaluatePixel(sample) {
          return [sample.B02, sample.B03, sample.B04, sample.B08];
        }
        """

        sh_body = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": time_from,
                            "to": time_to
                        },
                        "maxCloudCoverage": max_cloud_coverage
                    }
                }]
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "image/tiff"}
                }]
            },
            "evalscript": evalscript
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        resp = requests.post(self.PROCESS_URL, json=sh_body, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"CDSE Process API returned HTTP {resp.status_code}: {resp.text[:400]}"
            )

        with io.BytesIO(resp.content) as f:
            raw_arr = tifffile.imread(f)  # Shape (H, W, 4)

        if raw_arr.ndim != 3 or raw_arr.shape[-1] != 4:
            raise ValueError(f"Unexpected array shape from CDSE: {raw_arr.shape}")

        # Transpose from (H, W, 4) to (4, H, W) -> [B02, B03, B04, B08]
        l2a = np.transpose(raw_arr, (2, 0, 1)).astype(np.float32)
        l2a = np.nan_to_num(l2a, nan=0.0, posinf=1.0, neginf=0.0)
        l2a = np.clip(l2a, 0.0, 1.0)

        # Build RGB composite (B04, B03, B02) -> Red, Green, Blue
        rgb = np.stack([l2a[2], l2a[1], l2a[0]], axis=-1)
        p2, p98 = np.percentile(rgb, (2, 98))
        if p98 > p2:
            l2a_rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
        else:
            l2a_rgb = np.clip(rgb, 0.0, 1.0)

        meta = {
            "source": "Copernicus Data Space Ecosystem (CDSE) Sentinel Hub Process API",
            "center": {"lat": lat, "lon": lon},
            "bbox": bbox,
            "bands": ["B02", "B03", "B04", "B08"],
            "resolution_m": 10.0,
            "dimensions_km": [1.28, 1.28],
            "time_window": f"{time_from} to {time_to}",
            "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        # Save to disk cache
        try:
            np.savez_compressed(cache_file, l2a=l2a, l2a_rgb=l2a_rgb)
            with open(cache_meta, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

        return {
            "l2a": l2a,
            "l2a_rgb": l2a_rgb,
            "bbox": bbox,
            "center": {"lat": lat, "lon": lon},
            "bounds_wgs84": {
                "west": bbox[0],
                "south": bbox[1],
                "east": bbox[2],
                "north": bbox[3]
            },
            "dimensions_km": (1.28, 1.28),
            "cached": False,
            "metadata": meta
        }
