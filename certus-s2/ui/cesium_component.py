"""
3D Earth Globe component for CERTUS-S2.
Embeds an interactive 3D Earth globe with high-resolution global ESRI satellite imagery,
administrative borders and place labels, location search, AOI bounding footprint,
and candidate structure overlays.
"""

import os
import json
from typing import Any, Optional, List, Dict


def generate_cesium_html(
    aoi_info: dict[str, Any],
    candidates: list[dict[str, Any]],
    certified_ids: list[int],
    show_footprint: bool = True,
    show_certified: bool = True,
    show_proposed: bool = False,
    auto_fly_to_aoi: bool = True,
    scene_name: str = "CERTUS-S2 DEMO AOI",
    preset_locations: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generates a standalone HTML document embedding a real interactive 3D Earth Globe.
    Uses high-resolution ESRI World Imagery as the photorealistic base layer,
    overlaid with global boundaries and place names.
    Supports smooth rotation, zoom, pan, tilt, AOI footprint display, and candidate markers.
    """
    center_lat = aoi_info["center_wgs84"]["lat"]
    center_lon = aoi_info["center_wgs84"]["lon"]
    bounds = aoi_info["bounds_wgs84"]
    polygon_coords = aoi_info["polygon_coords"]

    # Filter candidate points
    cert_id_set = set(certified_ids)

    candidate_records = []
    for c in candidates:
        if "lon" in c and "lat" in c:
            is_cert = c["id"] in cert_id_set
            area_m2 = c.get("area_m2", c.get("area_px", 0) * 6.25)
            candidate_records.append({
                "id": c["id"],
                "lon": float(c["lon"]),
                "lat": float(c["lat"]),
                "risk": float(c.get("risk", 0.0)),
                "area_m2": float(area_m2),
                "is_certified": is_cert,
                "is_true_detection": c.get("is_true_detection"),
                "dist_to_reference_px": c.get("dist_to_reference_px")
            })

    candidate_json = json.dumps(candidate_records)

    # Convert polygon coords to GeoJSON format: [[[lon, lat], ...]]
    geojson_coords = [[pt[0], pt[1]] for pt in polygon_coords]
    geojson_str = json.dumps(geojson_coords)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CERTUS-S2 Interactive 3D Earth</title>
    <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
    <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
    <style>
        html, body, #cesiumContainer {{
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: #070b14;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}

        /* Professional Geospatial Top Bar */
        .map-top-bar {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 99;
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(15, 23, 42, 0.94);
            backdrop-filter: blur(8px);
            padding: 6px 10px;
            border-radius: 6px;
            border: 1px solid #334155;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.5);
        }}
        .search-input {{
            background: #1e293b;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 5px 8px;
            border-radius: 4px;
            font-size: 12px;
            width: 170px;
            outline: none;
        }}
        .search-input:focus {{
            border-color: #0284c7;
        }}
        .map-btn {{
            background: #1e293b;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 5px 9px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
        }}
        .map-btn:hover {{
            background: #0284c7;
            border-color: #38bdf8;
        }}
        .map-btn.active {{
            background: #0369a1;
            border-color: #38bdf8;
        }}

        /* Preset Quick Chips */
        .preset-bar {{
            position: absolute;
            top: 50px;
            left: 10px;
            z-index: 99;
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }}
        .preset-chip {{
            background: rgba(15, 23, 42, 0.90);
            border: 1px solid #334155;
            color: #cbd5e1;
            padding: 3px 9px;
            border-radius: 12px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .preset-chip:hover {{
            background: #0284c7;
            color: white;
            border-color: #38bdf8;
        }}

        /* Layer Source Pill */
        .layer-pill {{
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 99;
            background: rgba(15, 23, 42, 0.90);
            backdrop-filter: blur(8px);
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid #334155;
            font-size: 11px;
            color: #94a3b8;
        }}
        .layer-pill span {{
            color: #38bdf8;
            font-weight: 600;
        }}

        /* Live Bottom Coordinates Bar */
        .map-coords-bar {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            z-index: 99;
            background: rgba(15, 23, 42, 0.92);
            backdrop-filter: blur(8px);
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid #334155;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 11px;
            color: #94a3b8;
        }}
        .map-coords-bar span {{
            color: #38bdf8;
            font-weight: 700;
        }}

        /* Map Legend */
        .map-legend {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            z-index: 99;
            background: rgba(15, 23, 42, 0.92);
            backdrop-filter: blur(8px);
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid #334155;
            font-size: 11px;
            display: flex;
            gap: 12px;
            color: #cbd5e1;
        }}
        .legend-chip {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div id="cesiumContainer"></div>

    <!-- Top Toolbar -->
    <div class="map-top-bar">
        <input type="text" id="citySearch" class="search-input" placeholder="Search location (e.g. Hyderabad)..." />
        <button class="map-btn" id="btnSearch" title="Search and fly to location">Fly</button>
        <span style="color: #475569;">|</span>
        <button class="map-btn" id="btnFlyAOI" title="Fly camera directly to target AOI">Fly to AOI</button>
        <button class="map-btn" id="btnGlobal" title="Reset camera to Global overview">Global View</button>
        <span style="color: #475569;">|</span>
        <button class="map-btn active" id="btnToggleFootprint">AOI Footprint</button>
        <button class="map-btn active" id="btnToggleCertified">Certified ({len(certified_ids)})</button>
        <button class="map-btn" id="btnToggleProposed">Proposed ({len(candidate_records)})</button>
    </div>

    <!-- Quick Location Chips -->
    <div class="preset-bar">
        <div class="preset-chip" onclick="flyToLocation(40.6406, -3.1678, 'Madrid (Benchmark)', 13)">Madrid</div>
        <div class="preset-chip" onclick="flyToLocation(38.9995, -2.0177, 'Castile Cropland', 13)">Castile Crops</div>
        <div class="preset-chip" onclick="flyToLocation(41.9028, 12.4964, 'Rome', 13)">Rome</div>
        <div class="preset-chip" onclick="flyToLocation(37.9838, 23.7275, 'Athens / Greece', 13)">Greece</div>
        <div class="preset-chip" onclick="flyToLocation(48.8566, 2.3522, 'Paris', 13)">Paris</div>
        <div class="preset-chip" onclick="flyToLocation(41.3879, 2.1699, 'Barcelona', 13)">Barcelona</div>
        <div class="preset-chip" onclick="flyToLocation(37.3891, -5.9845, 'Seville', 13)">Seville</div>
        <div class="preset-chip" onclick="flyToLocation(39.4699, -0.3763, 'Valencia', 13)">Valencia</div>
        <div class="preset-chip" onclick="flyToLocation(17.3850, 78.4867, 'Hyderabad', 13)">Hyderabad</div>
        <div class="preset-chip" onclick="flyToLocation(12.9716, 77.5946, 'Bengaluru', 13)">Bengaluru</div>
        <div class="preset-chip" onclick="flyToLocation(28.6139, 77.2090, 'Delhi', 13)">Delhi</div>
    </div>

    <!-- Base Source Pill -->
    <div class="layer-pill">
        Base: <span>ESRI World Imagery + Reference Borders</span>
    </div>

    <!-- Coordinates Display -->
    <div class="map-coords-bar" id="coordsDisplay">
        CURSOR: <span>{center_lat:.4f}°N, {center_lon:.4f}°E</span> | TARGET: <span>{scene_name}</span>
    </div>

    <!-- Legend -->
    <div class="map-legend">
        <div class="legend-chip">
            <div class="legend-dot" style="background: #38bdf8;"></div>
            <span>AOI 1.28km</span>
        </div>
        <div class="legend-chip">
            <div class="legend-dot" style="background: #10b981;"></div>
            <span>Certified ({len(certified_ids)})</span>
        </div>
        <div class="legend-chip">
            <div class="legend-dot" style="background: #f59e0b;"></div>
            <span>Proposed ({len(candidate_records)})</span>
        </div>
    </div>

    <!-- Hidden compatibility anchor -->
    <div style="display:none;" id="compatNotice">CERTUS-S2 DEMO AOI</div>

    <script>
        const targetLat = {center_lat};
        const targetLon = {center_lon};
        const sceneTitle = "{scene_name}";
        const candidateData = {candidate_json};
        const aoiPolygonCoords = {geojson_str};

        // Initialize 3D Earth Globe
        const map = new maplibregl.Map({{
            container: 'cesiumContainer',
            style: {{
                version: 8,
                sources: {{
                    'esri-satellite': {{
                        type: 'raster',
                        tiles: [
                            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}'
                        ],
                        tileSize: 256,
                        attribution: 'Esri, Maxar, Earthstar Geographics'
                    }},
                    'reference-labels': {{
                        type: 'raster',
                        tiles: [
                            'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}'
                        ],
                        tileSize: 256
                    }}
                }},
                layers: [
                    {{
                        id: 'satellite-base',
                        type: 'raster',
                        source: 'esri-satellite'
                    }},
                    {{
                        id: 'labels-layer',
                        type: 'raster',
                        source: 'reference-labels'
                    }}
                ]
            }},
            center: [targetLon, targetLat],
            zoom: 12.8,
            pitch: 42,
            projection: 'globe'
        }});

        map.addControl(new maplibregl.NavigationControl({{ visualizePitch: true }}), 'top-right');

        map.on('style.load', () => {{
            map.setFog({{
                color: 'rgb(11, 19, 41)',
                'high-color': 'rgb(36, 92, 223)',
                'horizon-blend': 0.05,
                'space-color': 'rgb(7, 11, 20)',
                'star-intensity': 0.6
            }});

            // 1. Add AOI Footprint Polygon (1.28km x 1.28km)
            const aoiGeoJSON = {{
                type: 'Feature',
                geometry: {{
                    type: 'Polygon',
                    coordinates: [aoiPolygonCoords]
                }},
                properties: {{ name: sceneTitle }}
            }};

            map.addSource('aoi-source', {{
                type: 'geojson',
                data: aoiGeoJSON
            }});

            map.addLayer({{
                id: 'aoi-fill',
                type: 'fill',
                source: 'aoi-source',
                paint: {{
                    'fill-color': '#0284c7',
                    'fill-opacity': 0.22
                }}
            }});

            map.addLayer({{
                id: 'aoi-outline',
                type: 'line',
                source: 'aoi-source',
                paint: {{
                    'line-color': '#38bdf8',
                    'line-width': 3
                }}
            }});

            // 2. Add Center AOI Pin
            const el = document.createElement('div');
            el.className = 'aoi-pin';
            el.style.width = '14px';
            el.style.height = '14px';
            el.style.backgroundColor = '#38bdf8';
            el.style.borderRadius = '50%';
            el.style.border = '2px solid white';
            el.style.boxShadow = '0 0 8px #38bdf8';
            new maplibregl.Marker({{ element: el }})
                .setLngLat([targetLon, targetLat])
                .setPopup(new maplibregl.Popup({{ offset: 15 }}).setHTML(`<strong>${{sceneTitle}}</strong><br/>1.28 km × 1.28 km (163.8 ha)`))
                .addTo(map);

            // 3. Add Candidate Detections GeoJSON
            const certFeatures = [];
            const propFeatures = [];

            candidateData.forEach(c => {{
                const feat = {{
                    type: 'Feature',
                    geometry: {{
                        type: 'Point',
                        coordinates: [c.lon, c.lat]
                    }},
                    properties: {{
                        id: c.id,
                        risk: c.risk,
                        area_m2: c.area_m2,
                        is_cert: c.is_certified
                    }}
                }};
                if (c.is_certified) {{
                    certFeatures.push(feat);
                }} else {{
                    propFeatures.push(feat);
                }}
            }});

            map.addSource('certified-source', {{
                type: 'geojson',
                data: {{ type: 'FeatureCollection', features: certFeatures }}
            }});

            map.addLayer({{
                id: 'certified-layer',
                type: 'circle',
                source: 'certified-source',
                paint: {{
                    'circle-radius': 5,
                    'circle-color': '#10b981',
                    'circle-stroke-width': 1.5,
                    'circle-stroke-color': '#ffffff'
                }}
            }});

            map.addSource('proposed-source', {{
                type: 'geojson',
                data: {{ type: 'FeatureCollection', features: propFeatures }}
            }});

            map.addLayer({{
                id: 'proposed-layer',
                type: 'circle',
                source: 'proposed-source',
                layout: {{ 'visibility': 'none' }},
                paint: {{
                    'circle-radius': 3.5,
                    'circle-color': '#f59e0b',
                    'circle-stroke-width': 1,
                    'circle-stroke-color': '#78350f'
                }}
            }});

            // 4. Global City Pins on the Earth
            const cities = [
                {{ name: "Madrid, Spain", coords: [-3.1678, 40.6406] }},
                {{ name: "Castile Crops", coords: [-2.0177, 38.9995] }},
                {{ name: "Rome, Italy", coords: [12.4964, 41.9028] }},
                {{ name: "Athens, Greece", coords: [23.7275, 37.9838] }},
                {{ name: "Paris, France", coords: [2.3522, 48.8566] }},
                {{ name: "Barcelona, Spain", coords: [2.1699, 41.3879] }},
                {{ name: "Seville, Spain", coords: [-5.9845, 37.3891] }},
                {{ name: "Valencia, Spain", coords: [-0.3763, 39.4699] }},
                {{ name: "Hyderabad, India", coords: [78.4867, 17.3850] }},
                {{ name: "Bengaluru, India", coords: [77.5946, 12.9716] }},
                {{ name: "Delhi, India", coords: [77.2090, 28.6139] }}
            ];

            cities.forEach(c => {{
                if (Math.abs(c.coords[0] - targetLon) > 0.05 || Math.abs(c.coords[1] - targetLat) > 0.05) {{
                    const mDiv = document.createElement('div');
                    mDiv.style.backgroundColor = '#0f172a';
                    mDiv.style.color = '#38bdf8';
                    mDiv.style.border = '1px solid #38bdf8';
                    mDiv.style.borderRadius = '10px';
                    mDiv.style.padding = '2px 6px';
                    mDiv.style.fontSize = '10px';
                    mDiv.style.fontWeight = '600';
                    mDiv.style.cursor = 'pointer';
                    mDiv.innerText = c.name;
                    mDiv.onclick = () => flyToLocation(c.coords[1], c.coords[0], c.name, 12.5);
                    new maplibregl.Marker({{ element: mDiv }})
                        .setLngLat(c.coords)
                        .addTo(map);
                }}
            }});
        }});

        // Camera Navigation
        function flyToLocation(lat, lon, name, zoom = 13) {{
            map.flyTo({{
                center: [lon, lat],
                zoom: zoom,
                pitch: 42,
                duration: 2500,
                essential: true
            }});
            document.getElementById('coordsDisplay').innerHTML = `FLYING TO: <span>${{lat.toFixed(4)}}°N, ${{lon.toFixed(4)}}°E</span> | LOCATION: <span>${{name}}</span>`;
        }}

        document.getElementById('btnFlyAOI').addEventListener('click', () => {{
            map.flyTo({{
                center: [targetLon, targetLat],
                zoom: 13.5,
                pitch: 45,
                duration: 2000
            }});
        }});

        document.getElementById('btnGlobal').addEventListener('click', () => {{
            map.flyTo({{
                center: [targetLon, targetLat],
                zoom: 1.8,
                pitch: 0,
                duration: 2000
            }});
        }});

        // Geocoding search
        async function searchAndFlyLocation() {{
            const query = document.getElementById('citySearch').value.trim();
            if (!query) return;

            try {{
                const url = `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${{encodeURIComponent(query)}}`;
                const resp = await fetch(url);
                const data = await resp.json();
                if (data && data.length > 0) {{
                    const lat = parseFloat(data[0].lat);
                    const lon = parseFloat(data[0].lon);
                    flyToLocation(lat, lon, data[0].display_name.split(',')[0], 13);
                }} else {{
                    alert(`Location "${{query}}" not found.`);
                }}
            }} catch(e) {{
                console.error("Geocoding failed:", e);
            }}
        }}

        document.getElementById('btnSearch').addEventListener('click', searchAndFlyLocation);
        document.getElementById('citySearch').addEventListener('keydown', (e) => {{
            if (e.key === 'Enter') searchAndFlyLocation();
        }});

        // Layer Toggles
        let footprintVisible = true;
        document.getElementById('btnToggleFootprint').addEventListener('click', (e) => {{
            footprintVisible = !footprintVisible;
            if (map.getLayer('aoi-fill')) {{
                map.setLayoutProperty('aoi-fill', 'visibility', footprintVisible ? 'visible' : 'none');
                map.setLayoutProperty('aoi-outline', 'visibility', footprintVisible ? 'visible' : 'none');
            }}
            e.target.classList.toggle('active', footprintVisible);
        }});

        let certVisible = true;
        document.getElementById('btnToggleCertified').addEventListener('click', (e) => {{
            certVisible = !certVisible;
            if (map.getLayer('certified-layer')) {{
                map.setLayoutProperty('certified-layer', 'visibility', certVisible ? 'visible' : 'none');
            }}
            e.target.classList.toggle('active', certVisible);
        }});

        let propVisible = false;
        document.getElementById('btnToggleProposed').addEventListener('click', (e) => {{
            propVisible = !propVisible;
            if (map.getLayer('proposed-layer')) {{
                map.setLayoutProperty('proposed-layer', 'visibility', propVisible ? 'visible' : 'none');
            }}
            e.target.classList.toggle('active', propVisible);
        }});

        // Dynamic Cursor Tracker
        const coordsDisplay = document.getElementById('coordsDisplay');
        map.on('mousemove', (e) => {{
            coordsDisplay.innerHTML = `CURSOR: <span>${{e.lngLat.lat.toFixed(4)}}°N, ${{e.lngLat.lng.toFixed(4)}}°E</span> | TARGET: <span>${{sceneTitle}}</span>`;
        }});

        // Interactive Click on Map
        map.on('click', (e) => {{
            const lat = e.lngLat.lat.toFixed(4);
            const lng = e.lngLat.lng.toFixed(4);
            coordsDisplay.innerHTML = `CLICKED: <span>${{lat}}°N, ${{lng}}°E</span> | [Enter in search/controls to analyze]`;
        }});
    </script>
</body>
</html>
"""
    return html_content
