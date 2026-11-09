# 🚀 Quick Start Guide - Deploy AgriScan AI

## 📋 Prerequisites

You need to upload your models first since they're not in GitHub.

## Option A: Google Drive (Fastest - 10 minutes)

### Step 1: Upload Models to Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Create folder: `agriscan-models`
3. Upload these 4 files from `backend/models/`:
   - `resnet_realistic.pth`
   - `densenet_realistic.pth`
   - `ghostnet_realistic.pth`
   - `agri_realistic.pth`

### Step 2: Get Download Links

For EACH file:
1. Right-click → Share → "Anyone with the link"
2. Copy the link (looks like: `https://drive.google.com/file/d/FILE_ID/view`)
3. Extract the `FILE_ID` from the middle
4. Create direct download link: `https://drive.google.com/uc?export=download&id=FILE_ID`

**Example:**
```
Original: https://drive.google.com/file/d/1a2B3c4D5e6F7g8H9/view?usp=sharing
FILE_ID: 1a2B3c4D5e6F7g8H9
Direct: https://drive.google.com/uc?export=download&id=1a2B3c4D5e6F7g8H9
```

### Step 3: Update Download Script

Edit `backend/download_models.py` (lines 11-16):

```python
MODEL_URLS = {
    "resnet_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_RESNET_ID",
    "densenet_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_DENSENET_ID",
    "ghostnet_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_GHOSTNET_ID",
    "agri_realistic.pth": "https://drive.google.com/uc?export=download&id=YOUR_AGRI_ID"
}
```

### Step 4: Test Download

```bash
cd backend
python download_models.py
```

✅ If successful, you should see all 4 models downloaded!

---

## Option B: Hugging Face (Professional - 15 minutes)

### Step 1: Create Hugging Face Account

1. Go to [huggingface.co/join](https://huggingface.co/join)
2. Create account
3. Get token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Name: "agriscan-upload"
   - Type: "Write"
   - Copy the token

### Step 2: Login

```bash
cd backend
hf auth login
```

Paste your token when prompted (it won't show, just paste and press Enter)

### Step 3: Edit Upload Script

Open `backend/upload_to_huggingface.py` and change line 16:

```python
REPO_ID = "YOUR_USERNAME/agriscan-models"  # Replace YOUR_USERNAME
```

### Step 4: Upload Models

```bash
python upload_to_huggingface.py
```

Wait for all 4 models to upload (~5-10 minutes depending on internet).

### Step 5: Enable Auto-Download

Open `backend/app.py`, find this comment (around line 30):

```python
# UNCOMMENT THIS BLOCK AFTER UPLOADING TO HUGGING FACE:
```

**Uncomment** the block below it (remove the `"""` quotes) and update `REPO_ID`:

```python
try:
    from huggingface_hub import hf_hub_download
    REPO_ID = "YOUR_USERNAME/agriscan-models"  # Change this!
    
    for model_file in missing_models:
        print(f"📥 Downloading {model_file}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=model_file,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded {model_file}")
except Exception as e:
    print(f"✗ Error: {e}")
```

---

## 🌐 Deploy to Render (Free)

### Step 1: Commit Changes

```bash
git add .
git commit -m "Add model download configuration"
git push
```

### Step 2: Sign Up for Render

1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Authorize Render to access your repositories

### Step 3: Deploy Backend

1. Click **"New +"** → **"Web Service"**
2. Connect your `slmdd` repository
3. Configure:
   - **Name**: `agriscan-backend`
   - **Region**: Choose nearest
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt && python download_models.py
     ```
   - **Start Command**:
     ```bash
     uvicorn app:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: `Free`
4. Add Environment Variable:
   - Key: `SARVAM_API_KEY`
   - Value: (your API key from `.env` file)
5. Click **"Create Web Service"**

⏳ Wait 5-10 minutes for deployment...

✅ When done, you'll get a URL like: `https://agriscan-backend.onrender.com`

### Step 4: Deploy Frontend

1. Click **"New +"** → **"Static Site"**
2. Connect `slmdd` repository again
3. Configure:
   - **Name**: `agriscan-frontend`
   - **Branch**: `main`
   - **Root Directory**: `frontend`
   - **Build Command**:
     ```bash
     npm install && npm run build
     ```
   - **Publish Directory**: `dist`
4. Add Environment Variable:
   - Key: `VITE_API_URL`
   - Value: `https://agriscan-backend.onrender.com` (your backend URL)
5. Click **"Create Static Site"**

⏳ Wait 3-5 minutes...

✅ Frontend URL: `https://agriscan-frontend.onrender.com`

### Step 5: Update CORS

Edit `backend/app.py` (around line 35):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # For testing
        "https://agriscan-frontend.onrender.com"  # Add your frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Commit and push:
```bash
git add backend/app.py
git commit -m "Update CORS for production"
git push
```

Render will auto-redeploy! ✨

---

## 🧪 Test Your Deployment

1. Go to your frontend URL
2. Upload a leaf image
3. Wait for prediction (~10 seconds first time, faster after)
4. Check disease localization with colored overlays

---

## ⚠️ Important Notes

**Free Tier Limitations:**
- Backend "sleeps" after 15 min of inactivity
- First request after sleep takes ~30 seconds
- 750 hours/month free (enough for testing)

**To Keep Always On:**
- Upgrade to paid plan ($7/month)
- Or use a cron job to ping every 10 minutes

**Models Download:**
- First deployment: Downloads all models (~200MB, takes 5-10 min)
- Subsequent deployments: Models cached, instant startup

---

## 🐛 Troubleshooting

### Models not downloading?
```bash
# Test locally first
cd backend
python download_models.py
```

### Backend crash?
Check Render logs:
- Go to your service
- Click "Logs" tab
- Look for errors

### Frontend can't connect?
1. Check backend URL in frontend environment variables
2. Check CORS settings in backend
3. Try `https://` (not `http://`)

### Need help?
- Check full `DEPLOYMENT_GUIDE.md`
- Open GitHub issue
- Check Render docs

---

## 🎉 You're Done!

Share your deployment:
- Frontend: `https://agriscan-frontend.onrender.com`
- API Docs: `https://agriscan-backend.onrender.com/docs`

Enjoy your AI-powered plant disease detector! 🌿✨
