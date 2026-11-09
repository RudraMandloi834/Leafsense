# 🚀 LeafSense — Quick Start Deployment Guide

This guide gets your **LeafSense** instance deployed in minutes.

---

## 📋 Prerequisites

1. Pre-trained model weights are hosted on Hugging Face at `adb043/agriscan_models`.
2. A free [Render.com](https://render.com) or [Vercel](https://vercel.com) account.

---

## Option 1: Standard Cloud Deployment (Render)

### Step 1: Deploy Backend Web Service
1. Log in to [Render.com](https://render.com).
2. Click **New +** → **Web Service**.
3. Select your repository: `https://github.com/RudraMandloi834/Leafsense`.
4. Configure:
   - **Name**: `leafsense-backend`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Starter (or Free tier with `LEAFSENSE_MODE=lite`)
5. Click **Create Web Service**. Copy the generated URL (e.g., `https://leafsense-backend.onrender.com`).

### Step 2: Deploy Frontend
1. In Render, click **New +** → **Static Site**.
2. Connect your repository.
3. Configure:
   - **Name**: `leafsense-frontend`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`
4. Add Environment Variable:
   - `VITE_API_URL`: `https://leafsense-backend.onrender.com`
5. Click **Create Static Site**.

---

## Option 2: Local Development Setup

### Terminal 1: Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Terminal 2: Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to test the application.
