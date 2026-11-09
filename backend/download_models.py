"""
Download model files from external storage during deployment.
Models can be hosted on: Google Drive, Dropbox, Hugging Face, or cloud storage.
"""

import os
import requests
from pathlib import Path

# Model URLs - Replace these with your actual URLs
MODEL_URLS = {
    "resnet_realistic.pth": "YOUR_RESNET_URL_HERE",
    "densenet_realistic.pth": "YOUR_DENSENET_URL_HERE", 
    "ghostnet_realistic.pth": "YOUR_GHOSTNET_URL_HERE",
    "agri_realistic.pth": "YOUR_AGRI_URL_HERE"
}

def download_file(url, destination):
    """Download a file from URL to destination."""
    print(f"Downloading {destination}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    progress = (downloaded / total_size) * 100
                    print(f"Progress: {progress:.1f}%", end='\r')
    
    print(f"\nDownloaded {destination}")

def download_all_models():
    """Download all model files."""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    for model_name, url in MODEL_URLS.items():
        model_path = models_dir / model_name
        
        # Skip if already downloaded
        if model_path.exists():
            print(f"✓ {model_name} already exists")
            continue
        
        if url == f"YOUR_{model_name.upper().replace('.', '_').replace('_PTH', '')}_URL_HERE":
            print(f"⚠ Skipping {model_name} - URL not configured")
            continue
        
        try:
            download_file(url, model_path)
            print(f"✓ Successfully downloaded {model_name}")
        except Exception as e:
            print(f"✗ Error downloading {model_name}: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("Downloading Model Files")
    print("=" * 50)
    download_all_models()
    print("\nDone!")
