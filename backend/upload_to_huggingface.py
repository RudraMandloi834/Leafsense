"""
Upload model files to Hugging Face Hub.

Before running:
1. Create account at https://huggingface.co/join
2. Get your token from https://huggingface.co/settings/tokens
3. Run: hf auth login
4. Paste your token when prompted
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

# Configuration
REPO_ID = "adb043/agriscan_models"  # Your HuggingFace username
MODEL_FILES = [
    "models/resnet_realistic.pth",
    "models/densenet_realistic.pth",
    "models/ghostnet_realistic.pth",
    "models/agri_realistic.pth",
]


def upload_models():
    """Upload all model files to Hugging Face."""

    print("=" * 60)
    print("🤗 Uploading Models to Hugging Face")
    print("=" * 60)

    # Initialize API
    api = HfApi()

    # Check if logged in
    try:
        user = api.whoami()
        print(f"✓ Logged in as: {user['name']}")
    except Exception as e:
        print("✗ Not logged in!")
        print("\nPlease run: hf auth login")
        print("Get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Create repository (if doesn't exist)
    try:
        print(f"\n📦 Creating repository: {REPO_ID}")
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)
        print(f"✓ Repository ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"⚠ Repository creation: {e}")

    # Upload each model file
    print("\n📤 Uploading model files...")
    for model_file in MODEL_FILES:
        model_path = Path(model_file)

        if not model_path.exists():
            print(f"✗ File not found: {model_file}")
            continue

        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\n📁 Uploading {model_path.name} ({file_size_mb:.1f} MB)...")

        try:
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"✓ Uploaded {model_path.name}")
        except Exception as e:
            print(f"✗ Error uploading {model_path.name}: {e}")

    print("\n" + "=" * 60)
    print("✅ Upload Complete!")
    print("=" * 60)
    print(f"\n🔗 View your models at: https://huggingface.co/{REPO_ID}")
    print("\n📝 Next steps:")
    print("1. Update backend/app.py to download from Hugging Face")
    print("2. Test locally: python test_model_download.py")
    print("3. Deploy your app!")


if __name__ == "__main__":
    upload_models()
