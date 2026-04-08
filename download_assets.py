import os
import urllib.request

def download_file(url, local_path):
    print(f"Downloading {url} to {local_path}...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
        
assets = {
    "static/chart.umd.min.js": "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js",
    "static/all.min.css": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    "static/webfonts/fa-solid-900.woff2": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-solid-900.woff2",
    "static/webfonts/fa-regular-400.woff2": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-regular-400.woff2",
    "static/webfonts/fa-brands-400.woff2": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-brands-400.woff2",
    "static/webfonts/fa-solid-900.ttf": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-solid-900.ttf",
    "static/webfonts/fa-regular-400.ttf": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-regular-400.ttf",
    "static/webfonts/fa-brands-400.ttf": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-brands-400.ttf",
    "static/webfonts/fa-v4compat.woff2": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-v4compat.woff2",
    "static/webfonts/fa-v4compat.ttf": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-v4compat.ttf",
}

for path, url in assets.items():
    try:
        download_file(url, path)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

print("Done downloading assets.")
