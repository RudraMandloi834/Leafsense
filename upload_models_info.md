# 📤 LeafSense — Model Weights Management Guide

This guide explains how pre-trained PyTorch model weights are managed, verified, and hosted for LeafSense.

---

## 📦 Model Artifacts

LeafSense utilizes 4 model weights:
- `resnet_realistic.pth`: ResNet50 classifier (~90 MB)
- `densenet_realistic.pth`: DenseNet121 classifier & Grad-CAM backbone (~28 MB)
- `ghostnet_realistic.pth`: GhostNetV2 lightweight classifier (~25 MB)
- `agri_realistic.pth`: Custom AgriFusionNet (EfficientNet-B4) (~75 MB)

All weights are excluded from Git version control due to GitHub's file size limitations.

---

## 🌐 Default Hosted Repository

The verified pre-trained weights are hosted on Hugging Face Hub:
```
https://huggingface.co/adb043/agriscan_models
```

On server startup or via `download_models.py`, LeafSense queries this repository to download any missing weights automatically into `backend/models/`.

---

## 🛠️ Uploading Replacement or Custom Models

If you re-train the models on a custom dataset or wish to host them in your personal Hugging Face account:

1. **Install Hugging Face CLI**:
   ```bash
   pip install huggingface_hub
   ```

2. **Log In to Hugging Face**:
   ```bash
   huggingface-cli login
   ```
   Provide your Hugging Face write token from `https://huggingface.co/settings/tokens`.

3. **Configure Repository in `backend/upload_to_huggingface.py`**:
   ```python
   REPO_ID = "YOUR_USERNAME/leafsense_models"
   ```

4. **Run Upload Script**:
   ```bash
   cd backend
   python upload_to_huggingface.py
   ```

5. **Update Backend Configuration**:
   Set `HF_MODEL_REPO_ID` in `backend/.env`:
   ```bash
   HF_MODEL_REPO_ID="YOUR_USERNAME/leafsense_models"
   ```
