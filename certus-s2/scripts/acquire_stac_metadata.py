import urllib.request
import json
import ssl
from datetime import datetime

def fetch_stac_item():
    """
    Fetches the first Sentinel-2 L2A item from Microsoft Planetary Computer
    STAC API for a specific bounding box and time.
    """
    # A small bbox in a known Sentinel-2 tile area (e.g. over a city in Spain or similar)
    # Let's just use a generic bbox
    bbox_str = "-122.0,37.0,-121.9,37.1"
    time_str = "2023-01-01T00:00:00Z/2023-12-31T00:00:00Z"
    
    # Planetary Computer STAC Search endpoint
    search_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [-122.0, 37.0, -121.9, 37.1],
        "datetime": time_str,
        "query": {"eo:cloud_cover": {"lt": 20}},
        "limit": 1
    }
    
    req = urllib.request.Request(search_url, data=json.dumps(payload).encode('utf-8'), 
                                 headers={'Content-Type': 'application/json'})
                                 
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
                                 
    with urllib.request.urlopen(req, context=ctx) as response:
        data = json.loads(response.read().decode('utf-8'))
        
    if not data['features']:
        print("No items found.")
        return
        
    item = data['features'][0]
    
    print("=== ACQUIRED STAC ITEM ===")
    print(f"ID: {item['id']}")
    print(f"Collection: {item['collection']}")
    
    # Validate Metadata from STAC Properties
    props = item['properties']
    print("\n=== VALIDATION ===")
    
    # EPSG / CRS is usually in proj:epsg
    crs = props.get('proj:epsg', 'Unknown')
    print(f"CRS (proj:epsg): EPSG:{crs}")
    
    # Extract resolution and dtype from a 10m band asset (e.g., B04)
    b4_asset = item['assets'].get('B04', {})
    if 'raster:bands' in b4_asset:
        b4_raster = b4_asset['raster:bands'][0]
        print(f"Resolution (B04): {b4_raster.get('spatial_resolution', 10)}m")
        print(f"Data Type (B04): {b4_raster.get('data_type')}")
        print(f"NoData Value (B04): {b4_raster.get('nodata')}")
    else:
        # Fallback to standard knowledge for S2
        print("Resolution (B04): 10m (Standard)")
        print("Data Type (B04): uint16 (Standard)")
        print("NoData Value (B04): 0 (Standard)")
        
    # Check Bands Availability
    available_bands = list(item['assets'].keys())
    required_bands = ["B02", "B03", "B04", "B08", "SCL"]
    missing = [b for b in required_bands if b not in available_bands]
    if missing:
        print(f"Bands: MISSING {missing}")
    else:
        print(f"Bands: All required bands present (B02, B03, B04, B08)")
        
    print(f"SCL Availability: {'SCL' in available_bands}")
    
    print(f"Spatial Extent (bbox): {item['bbox']}")
    print(f"Metadata/Provenance: Microsoft Planetary Computer STAC v1, Platform: {props.get('platform')}")
    
    print("\n=== RAW B04 URL (For Download) ===")
    print(b4_asset.get('href'))

if __name__ == "__main__":
    fetch_stac_item()
