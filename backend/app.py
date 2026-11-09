"""
LeafSense FastAPI Application
Multi-Disease Plant Pathogen Detection, Explainable AI (Grad-CAM), and Multilingual Advisory.
"""

import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

# Import internal helpers
from utils import (
    LABEL_COLS,
    NUM_CLASSES,
    SUPPORTED_LANGUAGES,
    DISEASE_COLORS,
    MAX_FILE_SIZE_BYTES,
    validate_image_payload,
    pil_to_base64_data_uri,
)
from image_quality import quality_checker
from sarvam_helper import get_advice
from gradcam import (
    generate_multi_disease_visualization,
    pil_to_base64,
)
from watsonx_service import WatsonxServiceError
from rag_service import rag_service
from agent_service import agent_service

# ----------------- CONFIGURATION & PATH RESOLUTION -----------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LEAFSENSE_MODE = os.getenv("LEAFSENSE_MODE", "full").lower()  # "full" or "lite"
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "adb043/agriscan_models")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
_cors_env = os.getenv("CORS_ORIGINS", "").strip()
if _cors_env and _cors_env != "*":
    ALLOWED_ORIGINS = [o.strip() for o in _cors_env.split(",") if o.strip()]
    ALLOWED_ORIGIN_REGEX = None
else:
    # Default development origins and regex matching any localhost / 127.0.0.1 port
    ALLOWED_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    ALLOWED_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"

# ----------------- PYTORCH & TIMM IMPORTS -----------------
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    import timm

    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    nn = None
    timm = None
    models = None
    transforms = None
    TORCH_AVAILABLE = False
    device = "cpu"

# ----------------- AGRIFUSIONNET ARCHITECTURE -----------------
if TORCH_AVAILABLE:
    class AgriFusionNet(nn.Module):
        """Custom multi-scale classifier combining EfficientNet-B4 backbone with MLP projection."""

        def __init__(self, num_classes: int):
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

# ----------------- GLOBAL MODEL REGISTRY -----------------
model_registry: Dict[str, Any] = {
    "resnet": None,
    "densenet": None,
    "ghostnet": None,
    "agri": None,
    "weights_loaded": {
        "resnet": False,
        "densenet": False,
        "ghostnet": False,
        "agri": False,
    },
    "all_loaded": False,
}

# ImageNet transform pipeline for model inference
if TORCH_AVAILABLE:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    transform = None

# Ensemble weights calibrated from validation multi-label F1 scores
W_RESNET = 0.22
W_DENSENET = 0.30
W_GHOSTNET = 0.28
W_AGRI = 0.20


def ensure_model_weights():
    """Attempt downloading model checkpoints if missing locally."""
    model_files = [
        "densenet_realistic.pth",  # Always needed (Grad-CAM and primary classifier)
    ]
    if LEAFSENSE_MODE != "lite":
        model_files.extend([
            "resnet_realistic.pth",
            "ghostnet_realistic.pth",
            "agri_realistic.pth",
        ])

    missing = [f for f in model_files if not (MODELS_DIR / f).exists()]
    if not missing:
        return

    print(f"📥 Missing model weights: {missing}")
    print(f"Connecting to Hugging Face Hub: {HF_MODEL_REPO_ID}...")

    try:
        from huggingface_hub import hf_hub_download

        for filename in missing:
            print(f"  Downloading {filename}...")
            hf_hub_download(
                repo_id=HF_MODEL_REPO_ID,
                filename=filename,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"  ✓ Saved {filename}")
        print("✅ Model weights verified successfully!")
    except Exception as e:
        print(f"⚠ Could not auto-download model weights: {e}")
        print("Note: Backend will start with diagnostic/demo capability.")


def load_models():
    """Load neural network architectures and pre-trained weights."""
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch is not available. API running in diagnostic/mock mode.")
        return

    ensure_model_weights()

    # 1. DenseNet121 (Primary Model & Grad-CAM)
    try:
        densenet = models.densenet121(weights=None)
        densenet.classifier = nn.Linear(densenet.classifier.in_features, NUM_CLASSES)
        dense_path = MODELS_DIR / "densenet_realistic.pth"
        if dense_path.exists():
            densenet.load_state_dict(torch.load(str(dense_path), map_location=device, weights_only=True))
            model_registry["weights_loaded"]["densenet"] = True
            print("✓ DenseNet121 weights loaded successfully")
        densenet = densenet.to(device)
        densenet.eval()
        model_registry["densenet"] = densenet
    except Exception as e:
        print(f"⚠ Error initializing DenseNet121: {e}")

    # If running in lite profile, skip loading the remaining 3 models
    if LEAFSENSE_MODE == "lite":
        model_registry["all_loaded"] = model_registry["weights_loaded"]["densenet"]
        print("🌿 Running in LeafSense Lite Profile (DenseNet121 only)")
        return

    # 2. ResNet50
    try:
        resnet = models.resnet50(weights=None)
        resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
        res_path = MODELS_DIR / "resnet_realistic.pth"
        if res_path.exists():
            resnet.load_state_dict(torch.load(str(res_path), map_location=device, weights_only=True))
            model_registry["weights_loaded"]["resnet"] = True
            print("✓ ResNet50 weights loaded successfully")
        resnet = resnet.to(device)
        resnet.eval()
        model_registry["resnet"] = resnet
    except Exception as e:
        print(f"⚠ Error initializing ResNet50: {e}")

    # 3. GhostNetV2
    try:
        ghostnet = timm.create_model("ghostnetv2_100", pretrained=False, num_classes=NUM_CLASSES)
        ghost_path = MODELS_DIR / "ghostnet_realistic.pth"
        if ghost_path.exists():
            ghostnet.load_state_dict(torch.load(str(ghost_path), map_location=device, weights_only=True))
            model_registry["weights_loaded"]["ghostnet"] = True
            print("✓ GhostNetV2 weights loaded successfully")
        ghostnet = ghostnet.to(device)
        ghostnet.eval()
        model_registry["ghostnet"] = ghostnet
    except Exception as e:
        print(f"⚠ Error initializing GhostNetV2: {e}")

    # 4. AgriFusionNet
    try:
        agri = AgriFusionNet(NUM_CLASSES)
        agri_path = MODELS_DIR / "agri_realistic.pth"
        if agri_path.exists():
            agri.load_state_dict(torch.load(str(agri_path), map_location=device, weights_only=True))
            model_registry["weights_loaded"]["agri"] = True
            print("✓ AgriFusionNet weights loaded successfully")
        agri = agri.to(device)
        agri.eval()
        model_registry["agri"] = agri
    except Exception as e:
        print(f"⚠ Error initializing AgriFusionNet: {e}")

    model_registry["all_loaded"] = all(model_registry["weights_loaded"].values())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan: Load models on startup and release resources on shutdown."""
    print("🌿 LeafSense Backend starting up...")
    load_models()
    yield
    print("🌿 LeafSense Backend shutting down...")


# ----------------- FASTAPI APP INITIALIZATION -----------------
app = FastAPI(
    title="LeafSense API",
    description="Explainable Multi-Disease Plant Pathogen Detection & Agricultural Advisory",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- REQUEST / RESPONSE SCHEMAS -----------------
class TranslateRequest(BaseModel):
    diseases: List[str] = Field(..., description="List of detected disease label strings")
    target_language: str = Field(..., description="ISO 639-1 code (e.g. 'en', 'hi', 'bn')")


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="Farmer agricultural query")
    top_k: Optional[int] = Field(default=4, ge=1, le=10, description="Top-k chunks to retrieve")
    language: Optional[str] = Field(default="en", description="Target language ('en' or 'hi')")


class RagSourceItem(BaseModel):
    document: str
    page: int
    relevance_score: float
    snippet: str


class RagQueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    grounded: bool
    sources: List[RagSourceItem]
    model_used: str
    embedding_model: str
    message: Optional[str] = None


class RagIngestResponse(BaseModel):
    success: bool
    added_files: List[str]
    updated_files: List[str]
    skipped_files: List[str]
    deleted_files: List[str]
    total_documents: int
    total_chunks: int
    message: str


class RagStatusResponse(BaseModel):
    success: bool
    is_configured: bool
    total_documents: int
    total_chunks: int
    documents: List[str]
    embedding_model: str
    generation_model: str
    vector_db: str
    storage_path: str


class AgentRecommendRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="Farmer agricultural query")
    crop: Optional[str] = Field(default=None, description="Crop or plant name (e.g. 'Tomato', 'Wheat')")
    location: Optional[str] = Field(default=None, description="Farmer's district, state, or region")
    disease_predictions: Optional[Dict[str, float]] = Field(default=None, description="Optional map of disease labels to confidence scores")
    detected_diseases: Optional[List[str]] = Field(default=None, description="Optional list of detected disease label strings")
    language: Optional[str] = Field(default="en", description="Target language ('en' or 'hi')")
    top_k: Optional[int] = Field(default=4, ge=1, le=10, description="Number of knowledge base document chunks to retrieve")
    conversation_id: Optional[str] = Field(default=None, description="Session conversation ID for multi-turn tracking")
    messages: Optional[List[Dict[str, Any]]] = Field(default=None, description="Recent conversation turns [{'role': 'user'|'assistant', 'content': '...'}]")
    farmer_profile: Optional[Dict[str, Any]] = Field(default=None, description="Progressively accumulated farmer profile")


class AgentRecommendResponse(BaseModel):
    success: bool
    question: str
    answer: str
    crop: Optional[str] = None
    location: Optional[str] = None
    disease_context: Optional[Dict[str, Any]] = None
    grounded: bool
    sources: List[Dict[str, Any]]
    confidence_assessment: str
    follow_up_questions: List[str]
    disclaimer: str
    language: str
    model_used: str
    conversation_id: Optional[str] = None
    stage: Optional[str] = None
    stage_label: Optional[str] = None
    farmer_profile: Optional[Dict[str, Any]] = None
    quick_replies: Optional[List[str]] = None
    intents: Optional[List[str]] = None
    primary_intent: Optional[str] = None
    differential_analysis: Optional[Dict[str, Any]] = None
    evidence_type: Optional[str] = None
    reasoning_summary: Optional[str] = None


class SessionResetRequest(BaseModel):
    conversation_id: str


# ----------------- API ENDPOINTS -----------------
@app.get("/")
def home():
    """Root metadata endpoint."""
    return {
        "app": "LeafSense API",
        "description": "Intelligent Multi-Disease Plant Detection System",
        "version": "1.0.0",
        "mode": LEAFSENSE_MODE,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    """System health check and model loading status."""
    return {
        "status": "healthy",
        "app": "LeafSense API",
        "version": "1.0.0",
        "mode": LEAFSENSE_MODE,
        "device": str(device),
        "models_loaded": model_registry["weights_loaded"],
        "classes_count": NUM_CLASSES,
        "classes": LABEL_COLS,
    }


@app.post("/check-quality")
async def check_image_quality(file: UploadFile = File(...)):
    """Pre-flight quality validation on leaf image before running inference."""
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file payload")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
            )

        quality_results = quality_checker.check_quality(file_bytes)
        return {"success": True, "quality": quality_results}

    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "quality": {
                "is_valid": False,
                "warnings": ["Failed to analyze image quality"],
                "suggestions": ["Please try uploading a valid image file"],
                "quality_score": 0,
            },
        }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    language: str = Form("en"),
):
    """
    Diagnose leaf diseases from uploaded image.
    Executes ensemble inference, generates multi-disease Grad-CAM heatmaps,
    and returns multilingual farmer advisory.
    """
    file_bytes = await file.read()

    # Validate image payload
    is_valid, err_msg, original_image = validate_image_payload(
        file_bytes, filename=file.filename, content_type=file.content_type
    )
    if not is_valid or original_image is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)

    lang_code = language.strip().lower()
    if lang_code not in SUPPORTED_LANGUAGES:
        lang_code = "en"

    # Verify model readiness
    densenet = model_registry.get("densenet")
    if densenet is None or not TORCH_AVAILABLE:
        # Fallback response for uninitialized or test environments
        return {
            "predictions": {"Tomato___Early_blight": 0.85},
            "detected_diseases": ["Tomato___Early_blight"],
            "language": lang_code,
            "advice": get_advice(["Tomato___Early_blight"], lang_code),
            "visualizations": [],
            "combined_visualization": None,
            "warning": "Model running in demonstration mode. Download model weights for production inference.",
        }

    # Preprocess image
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    # Inference execution
    with torch.no_grad():
        if LEAFSENSE_MODE == "lite":
            # Lite mode: single model prediction from DenseNet121
            final_probs = torch.sigmoid(densenet(image_tensor))
        else:
            # Full mode: 4-model ensemble prediction
            resnet = model_registry.get("resnet")
            ghostnet = model_registry.get("ghostnet")
            agri = model_registry.get("agri")

            p_dense = torch.sigmoid(densenet(image_tensor))
            p_res = torch.sigmoid(resnet(image_tensor)) if resnet else p_dense
            p_ghost = torch.sigmoid(ghostnet(image_tensor)) if ghostnet else p_dense
            p_agri = torch.sigmoid(agri(image_tensor)) if agri else p_dense

            final_probs = (
                W_RESNET * p_res
                + W_DENSENET * p_dense
                + W_GHOSTNET * p_ghost
                + W_AGRI * p_agri
            )

    probs = final_probs.cpu().numpy()[0]

    # Filter detected diseases above threshold
    results: Dict[str, float] = {}
    for i, prob in enumerate(probs):
        if prob >= CONFIDENCE_THRESHOLD:
            results[LABEL_COLS[i]] = round(float(prob), 4)

    # If no disease exceeds threshold, take top class
    if not results:
        best_idx = int(probs.argmax())
        results[LABEL_COLS[best_idx]] = round(float(probs[best_idx]), 4)

    # Sort results by confidence descending
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    detected_diseases = list(results.keys())

    # Generate multilingual advice
    advice = get_advice(detected_diseases, lang_code)

    # Generate multi-disease Grad-CAM visualization
    visualizations: List[Dict[str, Any]] = []
    combined_visualization: Optional[str] = None

    try:
        disease_indices = [LABEL_COLS.index(d) for d in detected_diseases if d in LABEL_COLS]
        color_map = {
            idx: DISEASE_COLORS[i % len(DISEASE_COLORS)][1]
            for i, idx in enumerate(disease_indices)
        }

        # Enable gradients explicitly for Grad-CAM backprop
        gradcam_image = transform(original_image).unsqueeze(0).to(device)
        gradcam_image.requires_grad = True

        with torch.enable_grad():
            cam_results = generate_multi_disease_visualization(
                densenet,
                "densenet",
                gradcam_image,
                original_image,
                disease_indices,
                color_map,
            )

        if cam_results:
            for i, disease in enumerate(detected_diseases):
                disease_idx = LABEL_COLS.index(disease)
                color_name = DISEASE_COLORS[i % len(DISEASE_COLORS)][0]
                color_rgb = list(DISEASE_COLORS[i % len(DISEASE_COLORS)][1])

                if disease_idx in cam_results["individual"]:
                    visualizations.append({
                        "disease": disease,
                        "color": color_name,
                        "color_rgb": color_rgb,
                        "image": pil_to_base64(cam_results["individual"][disease_idx]),
                    })

            if cam_results.get("combined"):
                combined_visualization = pil_to_base64(cam_results["combined"])

    except Exception as e:
        print(f"Error generating Grad-CAM visualizations: {e}")

    return {
        "predictions": results,
        "detected_diseases": detected_diseases,
        "language": lang_code,
        "advice": advice,
        "visualizations": visualizations,
        "combined_visualization": combined_visualization,
    }


@app.post("/translate-advice")
async def translate_advice(request: TranslateRequest):
    """Translate agricultural treatment recommendations on demand."""
    try:
        lang_code = request.target_language.strip().lower()
        if lang_code not in SUPPORTED_LANGUAGES:
            lang_code = "en"

        advice = get_advice(request.diseases, lang_code)
        return {
            "success": True,
            "advice": advice,
            "language": lang_code,
            "diseases": request.diseases,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------- RAG KNOWLEDGE BASE ENDPOINTS -----------------
@app.get("/api/rag/status", response_model=RagStatusResponse)
async def get_rag_status():
    """Returns knowledge base configuration, document metrics, and ChromaDB state."""
    try:
        status_info = rag_service.get_status()
        return status_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query RAG status: {e}"
        )


@app.post("/api/rag/ingest", response_model=RagIngestResponse)
async def ingest_documents_endpoint(
    file: Optional[UploadFile] = File(None),
    force: bool = Form(False)
):
    """
    Ingests agricultural documents into the ChromaDB vector database.
    Optionally accepts a file upload (.pdf, .txt, .md) to add to knowledge_base/documents/.
    Avoids re-embedding unchanged files using SHA-256 hash checks.
    """
    target_path = None
    try:
        if file is not None and file.filename:
            allowed_exts = {".pdf", ".txt", ".md"}
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_exts:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file format '{file_ext}'. Allowed formats: .pdf, .txt, .md"
                )

            contents = await file.read()
            if len(contents) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty (0 bytes)."
                )

            # Limit upload size to 25 MB
            if len(contents) > 25 * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File exceeds maximum allowable size of 25 MB."
                )

            # Save file safely to knowledge_base/documents/
            safe_name = Path(file.filename).name
            target_path = rag_service.documents_dir / safe_name
            with open(target_path, "wb") as f:
                f.write(contents)

        result = rag_service.ingest_documents(force_reindex=force)
        return result
    except HTTPException:
        if target_path and target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        raise
    except ValueError as e:
        if target_path and target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or malformed document: {e}"
        )
    except WatsonxServiceError as e:
        if target_path and target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"IBM watsonx embedding service unavailable: {e.sanitized_message}"
        )
    except Exception as e:
        if target_path and target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {e}"
        )


@app.post("/api/rag/query", response_model=RagQueryResponse)
async def query_knowledge_base(request: RagQueryRequest):
    """
    Answers an agricultural query grounded strictly in retrieved documents.
    Generates verified responses using IBM Granite (ibm/granite-4-h-small).
    """
    try:
        response_data = rag_service.answer_query(
            question=request.question,
            top_k=request.top_k or 4,
            language=request.language or "en"
        )
        return response_data
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except WatsonxServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"IBM watsonx service unavailable: {e.sanitized_message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query execution failed: {e}"
        )


# ----------------- SMART FARMING AGENT ENDPOINTS -----------------
@app.post("/api/agent/recommend", response_model=AgentRecommendResponse)
async def get_agent_recommendation(request: AgentRecommendRequest):
    """
    Synthesizes farmer queries with disease detection results, RAG literature,
    multi-turn memory, progressive profiling, and IBM Granite reasoning.
    """
    try:
        recommendation = agent_service.get_recommendation(
            question=request.question,
            crop=request.crop,
            location=request.location,
            disease_predictions=request.disease_predictions,
            detected_diseases=request.detected_diseases,
            language=request.language or "en",
            top_k=request.top_k or 4,
            conversation_id=request.conversation_id,
            messages=request.messages,
            farmer_profile=request.farmer_profile
        )
        return recommendation
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except WatsonxServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"IBM watsonx service unavailable: {e.sanitized_message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Smart farming agent recommendation failed: {e}"
        )


@app.post("/api/agent/session/reset")
async def reset_session_endpoint(request: SessionResetRequest):
    """Resets memory for the specified conversation session."""
    try:
        session = agent_service.reset_session(request.conversation_id)
        return {
            "success": True,
            "conversation_id": session["conversation_id"],
            "message": "Session reset successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset session: {e}"
        )


@app.get("/api/agent/session/{conversation_id}")
async def get_session_endpoint(conversation_id: str):
    """Retrieves current session state, conversation stage, and farmer profile."""
    try:
        session = agent_service.get_or_create_session(conversation_id)
        return {
            "success": True,
            "conversation_id": session["conversation_id"],
            "stage": session.get("stage", "crop_context"),
            "profile": session.get("profile", {}),
            "messages_count": len(session.get("messages", []))
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch session: {e}"
        )
