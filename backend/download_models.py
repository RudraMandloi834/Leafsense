"""
LeafSense Pre-trained Model Weights Downloader
Downloads model weights (.pth) from Hugging Face Hub or custom external URLs.
"""

import os
import sys
from pathlib import Path
import requests

MODEL_FILES = [
    "resnet_realistic.pth",
    "densenet_realistic.pth",
    "ghostnet_realistic.pth",
    "agri_realistic.pth",
]

DEFAULT_HF_REPO = os.getenv("HF_MODEL_REPO_ID", "adb043/agriscan_models")


def download_file_http(url: str, destination: Path):
    """Download a file from an HTTP/HTTPS URL with streaming and progress."""
    print(f"Downloading {destination.name} from {url}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 16384
    downloaded = 0

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"Progress: {percent:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="\r")

    print(f"\n✓ Successfully downloaded {destination.name}")


def download_all_models(models_dir: Path = None, repo_id: str = None) -> bool:
    """Download all required model weights into models_dir."""
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    repo_id = repo_id or DEFAULT_HF_REPO

    missing_models = [m for m in MODEL_FILES if not (models_dir / m).exists()]

    if not missing_models:
        print(f"✓ All {len(MODEL_FILES)} model files already present in {models_dir}")
        return True

    print(f"Found {len(missing_models)} missing models: {missing_models}")
    print(f"Initiating download from Hugging Face Hub: {repo_id}")

    try:
        from huggingface_hub import hf_hub_download

        for model_file in missing_models:
            print(f"📥 Fetching {model_file}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=model_file,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False,
            )
            print(f"✓ Saved {model_file} to {models_dir}")

        print("\n✅ All LeafSense models downloaded successfully!")
        return True

    except ImportError:
        print("✗ huggingface_hub is not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"✗ Error downloading models from Hugging Face ({repo_id}): {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🌿 LeafSense Model Weights Downloader")
    print("=" * 60)
    success = download_all_models()
    sys.exit(0 if success else 1)
