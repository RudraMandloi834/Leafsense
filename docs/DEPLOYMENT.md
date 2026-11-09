# LeafSense Production Deployment Guide

This guide details the deployment architecture, containerization, environment configuration, and hosting considerations for the LeafSense full-stack application.

> **Status Notice:** LeafSense is currently configured for local and containerized deployment. Automated cloud deployment has not yet been executed. Follow the guidelines below when taking the platform to production.

---

## 1. Recommended Deployment Architecture

```
                  +-----------------------------------+
                  |     Vercel / Cloudflare Pages     |
                  |     (React 19 + Vite Static SPA)  |
                  +-----------------+-----------------+
                                    | HTTPS API Calls
                                    v
                  +-----------------------------------+
                  |    Persistent Python Container    |
                  |    (FastAPI + PyTorch + ChromaDB) |
                  |    e.g. Render / Fly.io / AWS ECS |
                  +--------+-----------------+--------+
                           |                 |
         +-----------------+--+           +--+--------------------+
         |   IBM watsonx.ai   |           |  Persistent Storage   |
         |   Foundation LLM   |           |  - ChromaDB Database  |
         |   (Granite 4 Small)|           |  - Model Checkpoints  |
         +--------------------+           +-----------------------+
```

### Critical Architectural Constraint: Serverless vs. Persistent Backend
- **Frontend on Vercel:** Ideal for the React SPA. Generates fast global CDN static assets (`dist/`).
- **Backend MUST Run on a Persistent Container Service:**
  - **PyTorch Dependencies:** PyTorch and Torchvision binaries exceed Vercel's 50 MB / 250 MB serverless function bundle limits.
  - **Local ChromaDB State:** ChromaDB stores vector embeddings on local disk (`backend/knowledge_base/chroma_db/`). Ephemeral serverless functions lose this state on cold termination.
  - **Inference Latency:** Loading neural network weights into GPU/CPU memory takes several seconds; persistent services keep models resident in RAM for instant inference.

---

## 2. Docker & Containerized Deployment

LeafSense provides production Docker configurations for both frontend and backend services.

### 2.1. Backend Dockerfile (`backend/Dockerfile`)
The backend image bundles Python 3.11/3.13, PyTorch, ChromaDB, and FastAPI:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.2. Deploying with Docker Compose
To launch both services together locally or on a VPS:
```bash
docker compose up --build -d
```

Check running container status:
```bash
docker compose ps
docker compose logs -f backend
```

---

## 3. Production Environment Variables

### Backend (`backend/.env`)
Ensure these variables are securely populated on your container host:

```bash
# IBM watsonx.ai Credentials
WATSONX_API_KEY=your_production_api_key_here
WATSONX_PROJECT_ID=your_production_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Model Identifiers
WATSONX_MODEL_ID=ibm/granite-4-h-small
WATSONX_EMBEDDING_MODEL_ID=ibm/granite-embedding-278m-multilingual

# Operational Mode
LEAFSENSE_MODE=full # Or "lite" for low-memory (1 GB) servers

# Production CORS (Comma-separated allowed domains, no trailing slashes)
CORS_ORIGINS=https://leafsense.yourdomain.com,https://leafsense.vercel.app

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### Frontend (`frontend/.env.production`)
Configure the frontend build with the production backend API URL:
```bash
VITE_API_URL=https://api.leafsense.yourdomain.com
```

---

## 4. Hardware Sizing & Memory Specifications

| Configuration Mode | Minimum RAM | Recommended RAM | CPU Cores | Supported Models |
|:---|:---:|:---:|:---:|:---|
| **Full Ensemble (`LEAFSENSE_MODE=full`)** | 2.5 GB | 4.0 GB | 2+ vCPU | ResNet50 + DenseNet121 + GhostNetV2 + AgriFusionNet + ChromaDB |
| **Lite Single-Model (`LEAFSENSE_MODE=lite`)** | 1.0 GB | 2.0 GB | 1 vCPU | DenseNet121 (Primary + Grad-CAM) + ChromaDB |

---

## 5. Persistent Storage & Data Volumes

When deploying the backend container to cloud providers (e.g., Render Disks, Fly.io Volumes, or AWS EBS), mount persistent volumes for:
1. `/app/knowledge_base/chroma_db`: Ensures indexed agricultural vectors persist across container restarts.
2. `/app/models`: Caches downloaded neural network checkpoints to avoid redownloading on restart.
