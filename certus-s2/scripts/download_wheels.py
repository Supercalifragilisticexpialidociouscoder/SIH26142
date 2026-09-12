import urllib.request
import json
import ssl
import sys
import os

def download_wheel(package_name):
    # Setup SSL context
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f"Fetching metadata for {package_name}...")
    url = f"https://pypi.org/pypi/{package_name}/json"
    
    try:
        with urllib.request.urlopen(url, context=ctx) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Failed to fetch metadata for {package_name}: {e}")
        return False
        
    version = data['info']['version']
    releases = data['releases'][version]
    
    # Try to find arm64 mac wheel, then mac universal, then any mac, then any any
    target_wheel = None
    fallback_wheel = None
    
    for r in releases:
        if r['packagetype'] == 'bdist_wheel':
            filename = r['filename']
            # Prioritize macosx arm64
            if 'macosx' in filename and 'arm64' in filename:
                target_wheel = r
                break
            elif 'macosx' in filename and 'universal2' in filename:
                target_wheel = r
            elif 'any' in filename and not target_wheel:
                fallback_wheel = r
                
    if not target_wheel:
        target_wheel = fallback_wheel
        
    if not target_wheel:
        print(f"No suitable wheel found for {package_name}")
        return False
        
    download_url = target_wheel['url']
    filename = target_wheel['filename']
    
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return True
        
    print(f"Downloading {filename} from {download_url}...")
    
    # Download with a chunked approach to see progress
    try:
        req = urllib.request.Request(download_url)
        with urllib.request.urlopen(req, context=ctx, timeout=600) as response, open(filename, 'wb') as out_file:
            # We don't really need a progress bar for this automated step, just read and write
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0 and downloaded % (1024*1024) < block_size:
                    print(f"Downloaded {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
                    
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

if __name__ == "__main__":
    packages = ["rasterio", "scikit-learn", "torchvision", "einops", "timm", "scikit-image"]
    for pkg in packages:
        download_wheel(pkg)
