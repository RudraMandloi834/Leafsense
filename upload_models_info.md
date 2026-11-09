# 📤 How to Upload Your Models for Deployment

## Quick Steps (Google Drive - Easiest)

### 1. Upload to Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Create a folder called "agriscan-models"
3. Upload these 4 files:
   - `backend/models/resnet_realistic.pth`
   - `backend/models/densenet_realistic.pth`
   - `backend/models/ghostnet_realistic.pth`
   - `backend/models/agri_realistic.pth`

### 2. Get Shareable Links

For each file:
1. Right-click → Share
2. Change to "Anyone with the link"
3. Copy link
4. Extract FILE_ID from the link

**Example:**
- Link: `https://drive.google.com/file/d/1ABC123xyz/view?usp=sharing`
- FILE_ID: `1ABC123xyz`
- Direct download: `https://drive.google.com/uc?export=download&id=1ABC123xyz`

### 3. Update download_models.py

Edit `backend/download_models.py` and replace URLs:

```python
MODEL_URLS = {
    "resnet_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_RESNET_FILE_ID",
    "densenet_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_DENSENET_FILE_ID",
    "ghostnet_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_GHOSTNET_FILE_ID",
    "agri_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_AGRI_FILE_ID"
}
```

### 4. Test Download Locally

```bash
cd backend
python download_models.py
```

### 5. Commit and Push

```bash
git add backend/download_models.py
git commit -m "feat: Add model download script with URLs"
git push
```

---

## Alternative: Hugging Face (Better for ML)

### Why Hugging Face?
- ✅ Free unlimited storage for models
- ✅ Version control for models
- ✅ Easy integration with PyTorch
- ✅ Better for ML community

### Steps:

1. **Create account:** [huggingface.co/join](https://huggingface.co/join)

2. **Install CLI:**
```bash
pip install huggingface_hub
```

3. **Login:**
```bash
huggingface-cli login
```

4. **Create repository:**
- Go to huggingface.co → New → Model
- Name: `agriscan-models`
- Make it public

5. **Upload models:**
```bash
cd backend/models
huggingface-cli upload arghaDEVIL/agriscan-models resnet_realistic.pth
huggingface-cli upload arghaDEVIL/agriscan-models densenet_realistic.pth
huggingface-cli upload arghaDEVIL/agriscan-models ghostnet_realistic.pth
huggingface-cli upload arghaDEVIL/agriscan-models agri_realistic.pth
```

6. **Update app.py to download from Hugging Face:**

Add this function to `backend/app.py`:

```python
from huggingface_hub import hf_hub_download
import os

def ensure_models():
    """Download models from Hugging Face if not present."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_files = [
        "resnet_realistic.pth",
        "densenet_realistic.pth", 
        "ghostnet_realistic.pth",
        "agri_realistic.pth"
    ]
    
    for model_file in model_files:
        model_path = models_dir / model_file
        if not model_path.exists():
            print(f"Downloading {model_file}...")
            hf_hub_download(
                repo_id="arghaDEVIL/agriscan-models",
                filename=model_file,
                local_dir="models",
                local_dir_use_symlinks=False
            )
            print(f"✓ Downloaded {model_file}")

# Call at startup
ensure_models()
```

---

## Dataset Handling

The dataset is NOT needed for deployment! It's only for training.

For deployment, you only need:
- ✅ Trained model files (.pth)
- ✅ Backend code
- ✅ Frontend code

The app will:
1. User uploads image
2. Model predicts on that image
3. Returns results

**No dataset needed at runtime!**

---

## File Sizes

Check your model sizes:

```bash
cd backend/models
ls -lh *.pth
```

Typical sizes:
- ResNet50: ~100MB
- DenseNet121: ~30MB
- GhostNet: ~20MB
- AgriFusionNet: ~50MB

Total: ~200MB (easily fits in free tiers)

---

## Which Option Should You Choose?

| Option | Best For | Pros | Cons |
|--------|----------|------|------|
| **Google Drive** | Quick deploy | Easy, familiar | Manual process |
| **Hugging Face** | ML projects | Professional, free | Need account |
| **Git LFS** | Small teams | Integrated | Storage limits |
| **Cloud Storage** | Production | Scalable | Costs money |

**Recommendation:** Start with **Google Drive** for quick testing, move to **Hugging Face** for production.
