# How to Start the AgriScan AI Application

## Issue Fixed ✅
- **Problem**: `app.py` was importing `generate_gradcam_for_image` which no longer exists
- **Solution**: Removed the unused import. The code already uses the correct `generate_multi_disease_visualization` function

---

## Start Backend (Terminal 1)

```powershell
cd C:\Users\Asus\Downloads\slmdd\backend
.venv\Scripts\Activate
uvicorn app:app --reload
```

**Expected Output:**
```
INFO:     Will watch for changes in these directories: ['C:\\Users\\Asus\\Downloads\\slmdd\\backend']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using WatchFiles
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Backend will be available at: **http://127.0.0.1:8000**

---

## Start Frontend (Terminal 2)

```powershell
cd C:\Users\Asus\Downloads\slmdd\frontend
npm run preview
```

**Expected Output:**
```
> frontend@0.0.0 preview
> vite preview

  ➜  Local:   http://localhost:4173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

Frontend will be available at: **http://localhost:4173**

---

## Test the Application

1. Open browser: `http://localhost:4173`
2. Upload a plant leaf image with disease
3. Click "Predict Diseases"
4. View the predictions with confidence scores
5. Scroll down to see **Disease Localization** section
6. Toggle between "Show Combined" and "Show Individual" views
7. Test multiple languages (11 supported)

---

## Multi-Disease Localization Features

### What You'll See:
- **Individual View**: Each detected disease shown separately with unique colors:
  - Disease #1 → Red overlay
  - Disease #2 → Yellow overlay
  - Disease #3 → Blue overlay
  - Disease #4 → Green overlay
  - And so on...

- **Combined View**: All diseases merged into one visualization with color legend

### Supported Languages:
1. English (en)
2. हिंदी - Hindi (hi)
3. বাংলা - Bengali (bn)
4. தமிழ் - Tamil (ta)
5. తెలుగు - Telugu (te)
6. मराठी - Marathi (mr)
7. ગુજરાતી - Gujarati (gu)
8. ਪੰਜਾਬੀ - Punjabi (pa)
9. മലയാളം - Malayalam (ml)
10. ಕನ್ನಡ - Kannada (kn)
11. ଓଡ଼ିଆ - Odia (or)

---

## Troubleshooting

### Backend Issues:

**ImportError: cannot import name 'xxx'**
- ✅ **FIXED**: Removed old import from app.py

**ModuleNotFoundError: No module named 'timm'**
- Solution: `pip install -r requirements.txt`

**CUDA out of memory**
- The code automatically falls back to CPU if CUDA is not available

### Frontend Issues:

**Port 4173 already in use**
- Kill the existing process or use: `npm run preview -- --port 4174`

**Blank page or missing styles**
- Clear browser cache (Ctrl+Shift+R)
- Rebuild: `npm run build`

**CORS errors**
- Ensure backend is running on http://127.0.0.1:8000
- Check browser console for exact error

---

## API Endpoints

### 1. Health Check
```
GET http://127.0.0.1:8000/
Response: {"message": "Plant Disease Ensemble API Running"}
```

### 2. Image Quality Check
```
POST http://127.0.0.1:8000/check-quality
Body: multipart/form-data with 'file'
Response: Quality metrics and suggestions
```

### 3. Disease Prediction
```
POST http://127.0.0.1:8000/predict
Body: 
  - file: multipart/form-data
  - language: form field (default: "en")
Response: 
  - predictions
  - detected_diseases
  - visualizations (array of colored overlays)
  - combined_visualization
  - advice
```

### 4. Translate Advice
```
POST http://127.0.0.1:8000/translate-advice
Body: {"diseases": [...], "target_language": "hi"}
Response: Translated advice
```

---

## Architecture

### Models Used:
1. **ResNet50** (22% weight)
2. **DenseNet121** (30% weight) ← Used for Grad-CAM
3. **GhostNetV2** (28% weight)
4. **AgriFusionNet** (20% weight)

### Grad-CAM Settings:
- **Model**: DenseNet121 (highest single model accuracy)
- **Target Layer**: `norm5` (best for disease localization)
- **Threshold**: 0.7 (shows top 30% activations)
- **Alpha**: 0.7 (overlay transparency)

---

## Project Structure

```
slmdd/
├── backend/
│   ├── app.py                  # FastAPI server ✅ FIXED
│   ├── gradcam.py             # Multi-disease Grad-CAM ✅ NEW
│   ├── image_quality.py       # Quality checker
│   ├── sarvam_helper.py       # Multilingual advice
│   ├── utils.py               # Ensemble prediction
│   └── models/                # Trained model weights
│       ├── resnet_realistic.pth
│       ├── densenet_realistic.pth
│       ├── ghostnet_realistic.pth
│       └── agri_realistic.pth
└── frontend/
    ├── src/
    │   ├── App.jsx            # Main UI ✅ UPDATED
    │   └── translations.js    # 11 languages ✅ COMPLETE
    └── dist/                  # Built files
```

---

## Success Indicators

✅ Backend starts without import errors
✅ Frontend shows styled interface
✅ Image upload works (camera/gallery/drag-drop)
✅ Quality checker runs before prediction
✅ Disease prediction returns results
✅ Disease Localization section appears
✅ Individual disease overlays show in different colors
✅ Combined view merges all diseases
✅ Toggle button switches views
✅ All 11 languages work correctly

---

## Notes

- First prediction may be slow (model loading)
- Subsequent predictions are faster
- GPU will be used if available (much faster)
- CPU fallback is automatic
- Image quality check is mandatory before prediction
- Multilingual advice uses Sarvam AI API

---

## Ready to Test! 🚀

Your AgriScan AI with Multi-Disease Localization is ready for demonstration!
