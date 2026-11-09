# 🚀 Deployment Guide

This guide covers deploying AgriScan AI with large model files and dataset.

## 📦 Large Files Challenge

The following files are too large for GitHub:
- Model files (`.pth`): ~100-500MB each
- Dataset: ~2GB+

## 🎯 Deployment Options

### Option 1: Google Drive + Direct Download (Easiest)

1. **Upload models to Google Drive**
2. **Get shareable links**
3. **Download during deployment**

#### Steps:

**1. Upload your model files to Google Drive:**
```
backend/models/
├── resnet_realistic.pth
├── densenet_realistic.pth
├── ghostnet_realistic.pth
└── agri_realistic.pth
```

**2. Get direct download links:**
- Right-click file → Share → Copy link
- Convert to direct download link:
  - From: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
  - To: `https://drive.google.com/uc?export=download&id=FILE_ID`

**3. Update `backend/download_models.py`:**
```python
MODEL_URLS = {
    "resnet_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_RESNET_ID",
    "densenet_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_DENSENET_ID",
    "ghostnet_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_GHOSTNET_ID",
    "agri_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_AGRI_ID"
}
```

**4. Run before deployment:**
```bash
cd backend
python download_models.py
```

---

### Option 2: Hugging Face Hub (Recommended for ML Projects)

**1. Create account at [huggingface.co](https://huggingface.co)**

**2. Install Hugging Face CLI:**
```bash
pip install huggingface_hub
huggingface-cli login
```

**3. Upload models:**
```bash
huggingface-cli upload arghaDEVIL/agriscan-models backend/models/resnet_realistic.pth
huggingface-cli upload arghaDEVIL/agriscan-models backend/models/densenet_realistic.pth
huggingface-cli upload arghaDEVIL/agriscan-models backend/models/ghostnet_realistic.pth
huggingface-cli upload arghaDEVIL/agriscan-models backend/models/agri_realistic.pth
```

**4. Download in deployment:**
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="arghaDEVIL/agriscan-models",
    filename="resnet_realistic.pth",
    cache_dir="backend/models"
)
```

---

### Option 3: Cloud Storage (AWS S3, Google Cloud Storage)

**AWS S3:**
```bash
# Upload
aws s3 cp backend/models/ s3://your-bucket/models/ --recursive

# Download during deployment
aws s3 sync s3://your-bucket/models/ backend/models/
```

**Google Cloud Storage:**
```bash
# Upload
gsutil -m cp -r backend/models/* gs://your-bucket/models/

# Download during deployment
gsutil -m cp -r gs://your-bucket/models/* backend/models/
```

---

### Option 4: Git LFS (Large File Storage)

**Install Git LFS:**
```bash
git lfs install
```

**Track large files:**
```bash
git lfs track "*.pth"
git lfs track "dataset/**"
```

**Push to GitHub:**
```bash
git add .gitattributes
git add backend/models/*.pth
git commit -m "Add model files with Git LFS"
git push
```

**Note:** GitHub LFS has storage limits (1GB free, then paid)

---

## 🌐 Deployment Platforms

### 1. **Render (Recommended - Free Tier Available)**

**Backend Deployment:**
```yaml
# render.yaml
services:
  - type: web
    name: agriscan-backend
    env: python
    buildCommand: |
      cd backend
      pip install -r requirements.txt
      python download_models.py
    startCommand: cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: SARVAM_API_KEY
        sync: false
```

**Frontend Deployment:**
```yaml
  - type: web
    name: agriscan-frontend
    env: static
    buildCommand: cd frontend && npm install && npm run build
    staticPublishPath: ./frontend/dist
```

### 2. **Railway**

**Deploy with one click:**
- Connect GitHub repo
- Add environment variables
- Railway auto-detects Python + Node.js
- Add build command: `python backend/download_models.py`

### 3. **Heroku**

**Create `Procfile`:**
```
web: cd backend && python download_models.py && uvicorn app:app --host 0.0.0.0 --port $PORT
```

**Deploy:**
```bash
heroku create agriscan-backend
heroku config:set SARVAM_API_KEY=your_key
git push heroku main
```

### 4. **Docker (Any Platform)**

**Create `Dockerfile`:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY backend/ .

# Download models
RUN python download_models.py

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and deploy:**
```bash
docker build -t agriscan-backend .
docker run -p 8000:8000 -e SARVAM_API_KEY=your_key agriscan-backend
```

### 5. **DigitalOcean App Platform**

- Connect GitHub repo
- Add build command: `python backend/download_models.py`
- Set environment variables
- Deploy with automatic HTTPS

---

## 📋 Pre-Deployment Checklist

- [ ] Models uploaded to external storage (Google Drive/Hugging Face/S3)
- [ ] Model download URLs configured in `download_models.py`
- [ ] Environment variables set (SARVAM_API_KEY)
- [ ] Frontend API URL updated to backend URL
- [ ] Requirements.txt includes all dependencies
- [ ] Test model download script locally
- [ ] CORS configured for frontend domain
- [ ] .env file NOT committed to git

---

## 🔧 Configuration Updates Needed

### Backend: Update CORS for production

**`backend/app.py`:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4173",
        "https://your-frontend-domain.com"  # Add your frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Frontend: Update API endpoint

**`frontend/src/App.jsx`:**
```javascript
const API_URL = import.meta.env.PROD 
  ? 'https://your-backend-url.com'  // Production
  : 'http://127.0.0.1:8000';         // Development
```

Or use environment variables:
```javascript
// Create frontend/.env.production
VITE_API_URL=https://your-backend-url.com
```

```javascript
// In App.jsx
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
```

---

## 🚀 Quick Deploy Steps

### Using Render (Recommended):

1. **Upload models to Google Drive and update URLs**
2. **Push code to GitHub**
3. **Create Render account**
4. **New Web Service → Connect GitHub repo**
5. **Configure:**
   - Build Command: `cd backend && pip install -r requirements.txt && python download_models.py`
   - Start Command: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Environment: Add `SARVAM_API_KEY`
6. **Deploy frontend:**
   - New Static Site → Connect GitHub repo
   - Build Command: `cd frontend && npm install && npm run build`
   - Publish Directory: `frontend/dist`

---

## 🐛 Troubleshooting

**Models not downloading:**
- Check if URLs are direct download links
- Verify network access in deployment environment
- Check disk space on server

**Out of memory:**
- Use smaller batch size
- Deploy on server with more RAM (2GB minimum)
- Consider model quantization

**Slow predictions:**
- Use CPU-optimized PyTorch build
- Consider using ONNX Runtime
- Add caching for frequently used models

---

## 💰 Cost Estimate

**Free Tier Options:**
- **Render**: 750 hours/month free (sleeps after inactivity)
- **Railway**: $5 credit/month
- **Heroku**: 1000 dyno hours/month (with credit card)

**Paid Options (if needed):**
- **Render**: $7/month (always on)
- **Railway**: Pay as you go (~$5-10/month)
- **DigitalOcean**: $12/month (2GB RAM)

**Storage:**
- **Google Drive**: 15GB free
- **Hugging Face**: Unlimited free for models
- **Git LFS**: 1GB free, $5/month for 50GB

---

## 📞 Support

For deployment issues, check:
1. Server logs for errors
2. Model download completion
3. Environment variables set correctly
4. CORS configuration
5. API endpoint URL in frontend

Need help? Open an issue on GitHub!
