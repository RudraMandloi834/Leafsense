# 🚀 LeafSense — Production Deployment Guide

This guide covers deploying the **LeafSense** plant disease detection system, including configuring model weights, setting up background services, and hosting on cloud platforms.

---

## 1. Architecture Overview

LeafSense consists of two primary services:
1. **FastAPI Backend (`backend/`)**: Serves prediction APIs, quality checks, Grad-CAM generation, and multilingual translation.
2. **React Frontend (`frontend/`)**: Modern responsive web application built with Vite and Tailwind CSS.

---

## 2. Managing Pre-Trained Model Weights

The 4 deep learning models (`resnet_realistic.pth`, `densenet_realistic.pth`, `ghostnet_realistic.pth`, `agri_realistic.pth`) total ~250MB.

### Automatic Download from Hugging Face Hub (Default)
LeafSense includes native automated model downloading on startup. The default verified repository is:
```
adb043/agriscan_models
```
You can configure a custom repository ID via environment variable:
```bash
export HF_MODEL_REPO_ID="your_username/your_models_repo"
```

To pre-download all weights prior to deployment:
```bash
cd backend
python download_models.py
```

---

## 3. Deployment Profiles

LeafSense supports two runtime profiles configured via the `LEAFSENSE_MODE` environment variable:

### Profile A: Full Ensemble Mode (`LEAFSENSE_MODE=full`)
- Loads all 4 models: ResNet50 (22%), DenseNet121 (30%), GhostNetV2 (28%), AgriFusionNet (20%).
- Provides weighted ensemble confidence scores and Grad-CAM disease localization.
- **System Requirements**: 1.5GB+ available RAM. Recommended for Render Starter, AWS EC2, GCP Cloud Run, and dedicated VPS.

### Profile B: Memory-Optimized Lite Mode (`LEAFSENSE_MODE=lite`)
- Runs exclusively on **DenseNet121** (individual accuracy 99.52%).
- Generates identical color-coded Grad-CAM heatmaps while consuming under 350MB RAM.
- **System Requirements**: 512MB RAM. Ideal for Render Free Tier, Fly.io free tier, or edge deployments.

---

## 4. Deploying to Cloud Platforms

### Deploying on Render

#### Step 1: Deploy Backend Web Service
1. In the Render Dashboard, click **New +** → **Web Service**.
2. Connect your Git repository: `https://github.com/RudraMandloi834/Leafsense`.
3. Configure settings:
   - **Name**: `leafsense-backend`
   - **Region**: Choose region closest to your users (e.g., Singapore, Frankfurt, Oregon).
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Set Environment Variables:
   - `SARVAM_API_KEY`: *(Optional)* Your Sarvam AI key
   - `LEAFSENSE_MODE`: `full` (for Starter tier) or `lite` (for Free tier)
5. Click **Create Web Service**. Note the deployed backend URL (e.g. `https://leafsense-backend.onrender.com`).

#### Step 2: Deploy Frontend Static Site
1. Click **New +** → **Static Site**.
2. Connect the same repository.
3. Configure settings:
   - **Name**: `leafsense-frontend`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`
4. Set Environment Variable:
   - `VITE_API_URL`: `https://leafsense-backend.onrender.com`
5. Click **Create Static Site**.

---

## 5. Docker Deployment

### Run with Docker Compose
```bash
docker compose up -d
```

### Run Backend Standalone
```bash
docker build -t leafsense-backend -f backend/Dockerfile backend/
docker run -d -p 8000:8000 \
  -e SARVAM_API_KEY="your_api_key" \
  --name leafsense-api \
  leafsense-backend
```

### Run Frontend Standalone
```bash
docker build -t leafsense-frontend -f frontend/Dockerfile frontend/
docker run -d -p 80:80 \
  --name leafsense-ui \
  leafsense-frontend
```

---

## 6. Verification and Health Monitoring

Once deployed, verify system status via the healthcheck endpoint:
```bash
curl https://your-backend-domain.com/health
```

Expected JSON response:
```json
{
  "status": "healthy",
  "app": "LeafSense API",
  "version": "1.0.0",
  "device": "cpu",
  "models_loaded": {
    "resnet": true,
    "densenet": true,
    "ghostnet": true,
    "agri": true
  },
  "classes_count": 17
}
```
