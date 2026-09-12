"""
Geospatial interactive map component for CERTUS-S2 Mission Console.
Renders Deck.gl maps via pydeck with real geographic footprints, WGS84 coordinates,
and interactive detection overlays.
"""

from typing import Any, Optional
import pydeck as pdk


def build_geospatial_deck(
    aoi_info: dict[str, Any],
    candidates: list[dict[str, Any]],
    certified_ids: list[int],
    show_footprint: bool = True,
    show_certified: bool = True,
    show_proposed: bool = False,
    map_style: str = "dark"
) -> pdk.Deck:
    """
    Constructs a pydeck Deck object centered on the target AOI with interactive layers.
    
    Args:
        aoi_info: Spatial metadata including WGS84 polygon, center, bounds, area.
        candidates: List of candidate dicts with lon, lat, id, risk, area_m2.
        certified_ids: List of candidate IDs that met the certification criteria.
        show_footprint: Toggle for AOI bounding polygon.
        show_certified: Toggle for certified candidate points.
        show_proposed: Toggle for all proposed candidate points.
        map_style: Mapbox / Deck.gl basemap style ("dark", "road", "satellite").
    """
    layers = []
    center_lat = aoi_info["center_wgs84"]["lat"]
    center_lon = aoi_info["center_wgs84"]["lon"]
    
    # 1. AOI Footprint Polygon Layer
    if show_footprint:
        polygon_data = [{
            "polygon": aoi_info["polygon_coords"],
            "name": "Target AOI: Spain Urban — Madrid / Guadalajara",
            "tile": "Sentinel-2 Tile T30TXM | ROI_00001",
            "footprint": f"{aoi_info['dimensions_km'][0]:.2f} km × {aoi_info['dimensions_km'][1]:.2f} km ({aoi_info['area_km2']:.2f} km²)",
            "center": f"{center_lat:.4f}°N, {center_lon:.4f}°E",
            "crs": aoi_info.get("utm_crs", "EPSG:32630")
        }]
        
        footprint_layer = pdk.Layer(
            "PolygonLayer",
            data=polygon_data,
            get_polygon="polygon",
            get_fill_color=[14, 165, 233, 30],       # Translucent cyan fill
            get_line_color=[56, 189, 248, 240],      # Bright cyan boundary
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True
        )
        layers.append(footprint_layer)

    # 2. AOI Center Target Marker Layer
    center_data = [{
        "position": [center_lon, center_lat],
        "name": "Scene Center Target",
        "coords": f"{center_lat:.5f}°N, {center_lon:.5f}°E"
    }]
    center_layer = pdk.Layer(
        "ScatterplotLayer",
        data=center_data,
        get_position="position",
        get_fill_color=[56, 189, 248, 200],
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
        get_radius=20,
        radius_min_pixels=5,
        radius_max_pixels=12,
        pickable=True
    )
    layers.append(center_layer)

    # 3. Proposed Candidates Layer (Amber)
    if show_proposed:
        proposed_points = []
        for c in candidates:
            if "lon" in c and "lat" in c:
                area_m2 = c.get("area_m2", c.get("area_px", 0) * 6.25)
                proposed_points.append({
                    "position": [c["lon"], c["lat"]],
                    "id": c["id"],
                    "risk": float(c.get("risk", 0.0)),
                    "area_m2": float(area_m2),
                    "status": "Proposed Structure (Fixed c₀=0.02)",
                    "coords": f"{c['lat']:.5f}°N, {c['lon']:.5f}°E"
                })
                
        if len(proposed_points) > 0:
            prop_layer = pdk.Layer(
                "ScatterplotLayer",
                data=proposed_points,
                get_position="position",
                get_fill_color=[245, 158, 11, 190],   # Amber
                get_line_color=[217, 119, 6, 255],
                line_width_min_pixels=1,
                get_radius=8,
                radius_min_pixels=3,
                radius_max_pixels=8,
                pickable=True,
                auto_highlight=True
            )
            layers.append(prop_layer)

    # 4. Certified Candidates Layer (Emerald Green)
    if show_certified and len(certified_ids) > 0:
        cert_id_set = set(certified_ids)
        certified_points = []
        for c in candidates:
            if c["id"] in cert_id_set and "lon" in c and "lat" in c:
                area_m2 = c.get("area_m2", c.get("area_px", 0) * 6.25)
                certified_points.append({
                    "position": [c["lon"], c["lat"]],
                    "id": c["id"],
                    "risk": float(c.get("risk", 0.0)),
                    "area_m2": float(area_m2),
                    "status": "CERTIFIED Built Structure",
                    "coords": f"{c['lat']:.5f}°N, {c['lon']:.5f}°E"
                })
                
        if len(certified_points) > 0:
            cert_layer = pdk.Layer(
                "ScatterplotLayer",
                data=certified_points,
                get_position="position",
                get_fill_color=[16, 185, 129, 240],   # Emerald green
                get_line_color=[255, 255, 255, 255],  # Crisp white outline
                line_width_min_pixels=2,
                get_radius=14,
                radius_min_pixels=6,
                radius_max_pixels=16,
                pickable=True,
                auto_highlight=True
            )
            layers.append(cert_layer)

    # View state centered on AOI with optimal tilt & zoom
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=14.1,
        pitch=0,
        bearing=0
    )

    # Tooltip configuration
    tooltip = {
        "html": """
            <div style="font-family: -apple-system, sans-serif; font-size: 12px; color: #f8fafc; padding: 4px;">
                <b>{name}</b>
                {status}
                <div style="color: #94a3b8; margin-top: 2px;">{tile}</div>
                <div style="color: #38bdf8;">Coordinates: {coords}</div>
                <div style="color: #cbd5e1;">Footprint: {footprint}</div>
                <div style="color: #10b981;">{area_m2}</div>
            </div>
        """,
        "style": {
            "backgroundColor": "#0f172a",
            "border": "1px solid #334155",
            "borderRadius": "6px",
            "padding": "6px 10px"
        }
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_style,
        tooltip=tooltip
    )
    return deck
