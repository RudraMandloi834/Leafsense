# 🚀 LeafSense — Deployment Architecture & Options

This guide summarizes deployment options for LeafSense, covering both full ensemble deployment and memory-optimized deployment for resource-constrained environments.

---

## 📊 Deployment Profiles

| Deployment Profile | Target Environments | Memory Required | Models Active | Best For |
|---|---|---|---|---|
| **Full Ensemble (Standard)** | Cloud VPS (Render Starter+, AWS EC2, GCP, Railway, Docker) | ~1.2 GB - 2 GB RAM | ResNet50, DenseNet121, GhostNetV2, AgriFusionNet | Production, High-Accuracy Classification |
| **Lightweight (Single-Model)** | Free-Tier Cloud (Render Free, Fly.io, Hugging Face Spaces) or Edge | ~350 MB - 500 MB RAM | DenseNet121 (Primary Model + Grad-CAM) | Academic Demos, Low-Cost Hosting, Edge Devices |

---

## 🌟 Option 1: Full Ensemble Cloud Deployment (Recommended)

### Recommended Platforms
- **Render** (Starter Plan, 1 CPU / 1GB or 2GB RAM)
- **Railway** / **Fly.io** / **DigitalOcean App Platform**
- **AWS Elastic Beanstalk** or **GCP Cloud Run**

### Backend Deployment Steps:
1. Connect your repository: `https://github.com/RudraMandloi834/Leafsense`
2. **Root Directory**: `backend`
3. **Environment**: Python 3
4. **Build Command**: `pip install -r requirements.txt`
5. **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. **Environment Variables**:
   - `SARVAM_API_KEY`: *(Optional)* Your Sarvam AI key
   - `HF_MODEL_REPO_ID`: `adb043/agriscan_models`
   - `CORS_ORIGINS`: `https://your-frontend-domain.com`

---

## 🌿 Option 2: Memory-Constrained Deployment (Lite Mode)

For hosts with limited memory (such as Render's 512MB free tier), running 4 heavy deep learning models concurrently can exceed limits.

LeafSense provides an out-of-the-box lightweight mode that utilizes **DenseNet121** (the highest individual accuracy model at 99.52% which also drives Grad-CAM explainability):

### Using Environment Variable in `app.py`:
Set:
```bash
LEAFSENSE_MODE=lite
```
When `LEAFSENSE_MODE=lite` is set, the application initializes DenseNet121, using under 350 MB RAM, and executes inference and Grad-CAM smoothly without OOM risks.

Alternatively, you can run the standalone lightweight app directly:
```bash
uvicorn app_lite:app --host 0.0.0.0 --port $PORT
```

---

## 🐳 Option 3: Docker Containerized Deployment

LeafSense includes Docker definitions for rapid, portable deployment.

### 1. Build and Run Backend
```bash
docker build -t leafsense-backend -f backend/Dockerfile backend/
docker run -d -p 8000:8000 \
  -e SARVAM_API_KEY="your_key" \
  --name leafsense-backend \
  leafsense-backend
```

### 2. Build and Run Frontend
```bash
docker build -t leafsense-frontend -f frontend/Dockerfile frontend/
docker run -d -p 80:80 \
  --name leafsense-frontend \
  leafsense-frontend
```

---

## 💻 Option 4: Local Development

```bash
# Terminal 1: Backend
cd backend
source .venv/bin/activate
uvicorn app:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```
Accessible at `http://localhost:5173`.
