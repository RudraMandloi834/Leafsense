# 🌿 AgriScan AI - Multi-Disease Plant Detection System

Advanced plant disease detection system with **explainable AI** powered by ensemble deep learning models.

## ✨ Features

- **Multi-Disease Detection**: Identify multiple diseases in a single leaf image
- **Explainable AI**: Color-coded Grad-CAM visualizations showing affected areas
  - Disease #1 → RED overlay
  - Disease #2 → YELLOW overlay
  - Disease #3 → BLUE overlay
- **Ensemble Models**: Combines 4 deep learning architectures
  - ResNet50 (25% weight)
  - DenseNet121 (30% weight)
  - GhostNet (20% weight)
  - AgriFusionNet (25% weight)
- **Multi-Language Support**: 11 Indian languages + English
  - English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Punjabi, Malayalam, Kannada, Odia
- **Image Quality Check**: Validates image quality before prediction
- **Real-Time Analysis**: Instant AI-powered predictions
- **Camera Support**: Take photos directly or upload from gallery

## 🏗️ Architecture

### Backend
- **Framework**: FastAPI (Python)
- **Models**: PyTorch-based ensemble
- **Grad-CAM**: DenseNet121 with custom color overlays
- **Translation**: Sarvam AI API for multilingual advice

### Frontend
- **Framework**: React + Vite
- **Styling**: TailwindCSS
- **Features**: Dark mode, responsive design, PWA support

## 📋 Requirements

### Backend
```bash
cd backend
pip install -r requirements.txt
```

**Key Dependencies:**
- fastapi
- uvicorn
- torch
- torchvision
- opencv-python
- pillow
- timm
- requests

### Frontend
```bash
cd frontend
npm install
```

## 🚀 Getting Started

### 1. Setup Backend

```bash
cd backend

# Create .env file
echo "SARVAM_API_KEY=your_api_key_here" > .env

# Start the backend server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Backend will run on: `http://127.0.0.1:8000`

### 2. Setup Frontend

```bash
cd frontend

# Install dependencies
npm install

# Build for production
npm run build

# Preview production build
npm run preview
```

Frontend will run on: `http://localhost:4173`

## 📁 Project Structure

```
slmdd/
├── backend/
│   ├── app.py                    # FastAPI application
│   ├── gradcam.py                # Grad-CAM visualization
│   ├── image_quality.py          # Image quality checker
│   ├── utils.py                  # Utility functions
│   ├── sarvam_helper.py          # Translation API helper
│   ├── requirements.txt
│   └── models/                   # Model weights (.pth files)
│       ├── resnet_realistic.pth
│       ├── densenet_realistic.pth
│       ├── ghostnet_realistic.pth
│       └── agri_realistic.pth
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Main React component
│   │   ├── translations.js       # Multi-language translations
│   │   ├── CameraCapture.jsx     # Camera component
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
├── train_resnet.py               # Training scripts
├── train_densenet.py
├── train_ghostnet.py
├── train_agri.py
└── README.md
```

## 🎯 How It Works

1. **Upload Image**: User uploads or captures a leaf image
2. **Quality Check**: System validates image quality
3. **Ensemble Prediction**: 4 models predict diseases with confidence scores
4. **Grad-CAM Generation**: 
   - DenseNet121 generates attention maps for each detected disease
   - Each disease gets a unique color (RED, YELLOW, BLUE)
   - 50/50 blend of original image and colored heatmap
5. **Visualization**: Individual disease cards show affected areas
6. **Translation**: Treatment advice translated to user's language

## 🔬 Technical Details

### Grad-CAM Implementation
- **Model**: DenseNet121 (best performing in ensemble)
- **Target Layer**: `features.norm5`
- **Threshold**: 0.4 (40% activation threshold)
- **Blending**: 50% original image + 50% colored heatmap
- **Colors**: Pure RGB (Red: 255,0,0 | Yellow: 255,255,0 | Blue: 0,0,255)

### Ensemble Voting
- Weighted average of model predictions
- Threshold: 0.5 confidence for disease detection
- Supports multiple simultaneous disease detection

## 🛠️ Training Models

Model weights are excluded from the repository due to size. To train models:

```bash
# Train ResNet50
python train_resnet.py

# Train DenseNet121
python train_densenet.py

# Train GhostNet
python train_ghostnet.py

# Train AgriFusionNet
python train_agri.py
```

**Note**: Place your dataset in `dataset/` directory (excluded from git)

## 📝 API Endpoints

### POST /predict
Predict diseases from an uploaded image.

**Request:**
```json
{
  "file": "<image_file>"
}
```

**Response:**
```json
{
  "predictions": {...},
  "detected_diseases": ["Disease1", "Disease2"],
  "visualizations": [
    {
      "disease": "Disease1",
      "color": "red",
      "color_rgb": [255, 0, 0],
      "image": "data:image/png;base64,..."
    }
  ]
}
```

### POST /check-quality
Check image quality before prediction.

### POST /translate
Translate text using Sarvam AI API.

## 🌍 Supported Languages

1. English (en)
2. Hindi (hi)
3. Bengali (bn)
4. Tamil (ta)
5. Telugu (te)
6. Marathi (mr)
7. Gujarati (gu)
8. Punjabi (pa)
9. Malayalam (ml)
10. Kannada (kn)
11. Odia (or)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**arghaDEVIL**
- GitHub: [@arghaDEVIL](https://github.com/arghaDEVIL)

## 🙏 Acknowledgments

- PlantVillage Dataset
- PyTorch Team
- FastAPI Framework
- Sarvam AI for translation API

---

**Note**: Model files (.pth) and dataset are not included in the repository due to size constraints. Please train models or contact the author for pre-trained weights.
