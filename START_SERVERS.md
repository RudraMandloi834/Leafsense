# 🚀 Starting the LeafSense Application

This guide explains how to start both the **LeafSense FastAPI Backend** and the **LeafSense React Frontend** in your development environment.

---

## Prerequisites

- **Python**: 3.10+ installed
- **Node.js**: v18+ and `npm` installed
- Open **two terminal tabs/windows**:
  - Terminal 1: Backend service (FastAPI)
  - Terminal 2: Frontend client (Vite + React)

---

## 1. Start Backend (Terminal 1)

### Linux / macOS
```bash
# Navigate to the backend directory
cd backend

# Create virtual environment if not already created
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server with hot-reload enabled
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Windows (PowerShell)
```powershell
# Navigate to the backend directory
cd backend

# Create virtual environment if not already created
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start the server with hot-reload enabled
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Expected Startup Output:**
```
INFO:     Will watch for changes in directories: ['.../backend']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

- **API Base URL**: `http://127.0.0.1:8000`
- **Health Check**: `http://127.0.0.1:8000/health`
- **Interactive API Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`

---

## 2. Start Frontend (Terminal 2)

### Linux / macOS / Windows
```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies (only needed once)
npm install

# Start development server
npm run dev
```

**Expected Startup Output:**
```
  VITE v8.x.x  ready in 250 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

- **Frontend Application URL**: `http://localhost:5173`

> [!TIP]
> If you wish to test a production build locally, run:
> ```bash
> npm run build
> npm run preview
> ```
> This serves the optimized build at `http://localhost:4173`.

---

## 3. Verify End-to-End Application Flow

1. Open your browser and navigate to `http://localhost:5173`.
2. Select your preferred user interface language (English or one of 10 Indian regional languages).
3. Upload a leaf image from your device or use the built-in camera capture.
4. LeafSense will first run an automated **Image Quality Check** (evaluating sharpness, lighting, and resolution).
5. Click **"Predict Diseases"**.
6. The dashboard displays:
   - **Disease Predictions**: Detection results sorted with confidence percentages and severity levels.
   - **Explainable AI (Grad-CAM)**: Switch seamlessly between **Combined View** (all affected regions highlighted) and **Individual View** (card-by-card breakdown with color borders).
   - **Multilingual Advisory**: Step-by-step Cause, Symptoms, Treatment, and Prevention advice with dynamic language switching.
7. Click **"Download Report"** to export an image of your diagnostic findings.

---

## 4. Troubleshooting

### Port Conflicts
If port 8000 or 5173 is already in use:
- **Backend on alternative port**:
  ```bash
  uvicorn app:app --reload --port 8080
  ```
- Make sure to update `frontend/.env` with `VITE_API_URL=http://127.0.0.1:8080`.

### Missing Pretrained Weights
If model weights (`.pth` files) are not yet downloaded:
- LeafSense attempts auto-downloading them from Hugging Face on startup.
- You can manually trigger the download script:
  ```bash
  cd backend
  python download_models.py
  ```
- If running in a test or offline environment without weights, the backend gracefully stays online in test mode so APIs and frontend integrations can be verified.
