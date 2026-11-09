"""
LeafSense Model Weights Uploader to Hugging Face Hub
Uploads trained PyTorch model checkpoints to Hugging Face repository.
"""

import os
import sys
from pathlib import Path

# Repository ID defaults to verified upstream hub unless overridden
REPO_ID = os.getenv("HF_MODEL_REPO_ID", "adb043/agriscan_models")

MODEL_FILES = [
    "models/resnet_realistic.pth",
    "models/densenet_realistic.pth",
    "models/ghostnet_realistic.pth",
    "models/agri_realistic.pth",
]


def upload_models(repo_id: str = REPO_ID):
    """Upload all model checkpoints to Hugging Face Hub."""
    print("=" * 60)
    print("🌿 LeafSense — Uploading Models to Hugging Face")
    print("=" * 60)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("✗ huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    try:
        user = api.whoami()
        print(f"✓ Authenticated as: {user['name']}")
    except Exception as e:
        print("✗ Not authenticated with Hugging Face!")
        print("\nPlease run: huggingface-cli login")
        print("Obtain your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    try:
        print(f"\n📦 Ensuring repository exists: {repo_id}")
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        print(f"✓ Repository online: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"ℹ Repository status: {e}")

    backend_dir = Path(__file__).resolve().parent
    uploaded_count = 0

    for rel_path in MODEL_FILES:
        model_path = backend_dir / rel_path

        if not model_path.exists():
            print(f"⚠ Skipping {rel_path} — file not found locally")
            continue

        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\n📤 Uploading {model_path.name} ({size_mb:.1f} MB)...")

        try:
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"✓ Uploaded {model_path.name}")
            uploaded_count += 1
        except Exception as e:
            print(f"✗ Error uploading {model_path.name}: {e}")

    print("\n" + "=" * 60)
    print(f"✅ Upload Complete ({uploaded_count} files processed)")
    print("=" * 60)
    print(f"🔗 Repository: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    target_repo = sys.argv[1] if len(sys.argv) > 1 else REPO_ID
    upload_models(target_repo)
