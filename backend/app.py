from fastapi import Form
from sarvam_helper import get_advice
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import timm
import io
from pathlib import Path
from image_quality import quality_checker
from gradcam import pil_to_base64


# ============= AUTO-DOWNLOAD MODELS FROM HUGGING FACE =============
def ensure_models_from_huggingface():
    """Download models from Hugging Face if not present locally."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_files = [
        "resnet_realistic.pth",
        "densenet_realistic.pth",
        "ghostnet_realistic.pth",
        "agri_realistic.pth",
    ]

    missing_models = [f for f in model_files if not (models_dir / f).exists()]

    if missing_models:
        print(f"⚠ Missing models: {missing_models}")
        print("📥 Downloading from Hugging Face...")

        try:
            from huggingface_hub import hf_hub_download

            REPO_ID = "adb043/agriscan_models"

            for model_file in missing_models:
                print(f"📥 Downloading {model_file}...")
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=model_file,
                    local_dir=str(models_dir),
                    local_dir_use_symlinks=False,
                )
                print(f"✓ Downloaded {model_file}")

            print("✅ All models downloaded successfully!")
        except Exception as e:
            print(f"✗ Error downloading models: {e}")
            print(
                "📝 Make sure huggingface_hub is installed: pip install huggingface_hub"
            )
    else:
        print("✓ All model files found locally")


# Download models if needed (runs on startup)
ensure_models_from_huggingface()
# ==================================================================

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LABELS ----------------

label_cols = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Potato___Early_blight",
    "Potato___Late_blight",
]

num_classes = len(label_cols)

# ---------------- TRANSFORM ----------------

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ---------------- RESNET ----------------

resnet = models.resnet50(weights=None)

resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

resnet.load_state_dict(torch.load("models/resnet_realistic.pth", map_location=device))

resnet = resnet.to(device)
resnet.eval()

# ---------------- DENSENET ----------------

densenet = models.densenet121(weights=None)

densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)

densenet.load_state_dict(
    torch.load("models/densenet_realistic.pth", map_location=device)
)

densenet = densenet.to(device)
densenet.eval()

# ---------------- GHOSTNET ----------------

ghostnet = timm.create_model(
    "ghostnetv2_100", pretrained=False, num_classes=num_classes
)

ghostnet.load_state_dict(
    torch.load("models/ghostnet_realistic.pth", map_location=device)
)

ghostnet = ghostnet.to(device)
ghostnet.eval()

# ---------------- AGRIFUSIONNET ----------------


class AgriFusionNet(nn.Module):
    def __init__(self, num_classes):

        super().__init__()

        self.backbone = models.efficientnet_b4(weights=None)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Identity()

        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.out = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.backbone(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.drop2(x)

        return self.out(x)


agri = AgriFusionNet(num_classes)

agri.load_state_dict(torch.load("models/agri_realistic.pth", map_location=device))

agri = agri.to(device)
agri.eval()

# ---------------- WEIGHTS ----------------

W_RESNET = 0.22
W_DENSENET = 0.30
W_GHOSTNET = 0.28
W_AGRI = 0.20

# ---------------- HOME ----------------


@app.get("/")
def home():

    return {"message": "Plant Disease Ensemble API Running"}


# ---------------- PREDICT ----------------


@app.post("/check-quality")
async def check_image_quality(file: UploadFile = File(...)):
    """
    Check image quality before prediction
    Returns quality metrics and suggestions
    """
    try:
        # Read image bytes
        image_bytes = await file.read()

        # Check quality
        quality_results = quality_checker.check_quality(image_bytes)

        return {"success": True, "quality": quality_results}

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "quality": {
                "is_valid": False,
                "warnings": ["Failed to analyze image quality"],
                "suggestions": ["Please try uploading a different image"],
                "quality_score": 0,
            },
        }


@app.post("/predict")
async def predict(file: UploadFile = File(...), language: str = Form("en")):

    image_bytes = await file.read()

    # Store original image for Grad-CAM visualization
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Transform for model input
    image = transform(original_image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = torch.sigmoid(resnet(image))

        p2 = torch.sigmoid(densenet(image))

        p3 = torch.sigmoid(ghostnet(image))

        p4 = torch.sigmoid(agri(image))

        final_probs = W_RESNET * p1 + W_DENSENET * p2 + W_GHOSTNET * p3 + W_AGRI * p4

    probs = final_probs.cpu().numpy()[0]

    results = {}

    threshold = 0.50

    for i, prob in enumerate(probs):
        if prob >= threshold:
            results[label_cols[i]] = round(float(prob), 4)

    if len(results) == 0:
        best_idx = probs.argmax()

        results[label_cols[best_idx]] = round(float(probs[best_idx]), 4)
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    detected_diseases = list(results.keys())

    advice = get_advice(detected_diseases, language)

    # Generate Grad-CAM for all detected diseases with unique colors
    visualizations = []
    combined_visualization = None

    print(f"\n=== GRAD-CAM GENERATION ===")
    print(f"Detected diseases: {detected_diseases}")

    try:
        # Define color mapping for diseases
        disease_colors = [
            ("red", (255, 0, 0)),
            ("yellow", (255, 255, 0)),
            ("blue", (0, 0, 255)),
            ("green", (0, 255, 0)),
            ("purple", (255, 0, 255)),
            ("orange", (255, 165, 0)),
            ("cyan", (0, 255, 255)),
            ("pink", (255, 192, 203)),
        ]

        # Create a new tensor with gradients enabled for Grad-CAM
        gradcam_image = transform(original_image).unsqueeze(0).to(device)
        gradcam_image.requires_grad = True

        # Get disease indices and create color map
        disease_indices = [label_cols.index(disease) for disease in detected_diseases]
        color_map = {
            idx: disease_colors[i % len(disease_colors)][1]
            for i, idx in enumerate(disease_indices)
        }

        # Import the new multi-disease function
        from gradcam import generate_multi_disease_visualization

        # Generate visualizations
        result = generate_multi_disease_visualization(
            densenet,
            "densenet",
            gradcam_image,
            original_image,
            disease_indices,
            color_map,
        )

        print(f"Grad-CAM result: {result is not None}")
        if result:
            print(f"Individual visualizations: {len(result.get('individual', {}))}")
            print(f"Combined visualization: {result.get('combined') is not None}")

        if result:
            # Create individual disease visualizations
            for i, disease in enumerate(detected_diseases):
                disease_idx = label_cols.index(disease)
                color_name = disease_colors[i % len(disease_colors)][0]

                if disease_idx in result["individual"]:
                    visualizations.append(
                        {
                            "disease": disease,
                            "color": color_name,
                            "color_rgb": list(
                                disease_colors[i % len(disease_colors)][1]
                            ),
                            "image": pil_to_base64(result["individual"][disease_idx]),
                        }
                    )

            # Combined visualization
            combined_visualization = pil_to_base64(result["combined"])

    except Exception as e:
        print(f"Error generating Grad-CAM visualizations: {e}")
        import traceback

        traceback.print_exc()

    return {
        "predictions": results,
        "detected_diseases": detected_diseases,
        "language": language,
        "advice": advice,
        "visualizations": visualizations,
        "combined_visualization": combined_visualization,
    }


# ---------------- TRANSLATE ADVICE ----------------


class TranslateRequest(BaseModel):
    diseases: list[str]
    target_language: str


@app.post("/translate-advice")
async def translate_advice(request: TranslateRequest):
    """
    Translate advice for detected diseases to a different language
    """
    try:
        advice = get_advice(request.diseases, request.target_language)
        return {
            "success": True,
            "advice": advice,
            "language": request.target_language,
            "diseases": request.diseases,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
