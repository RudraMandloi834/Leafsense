# LeafSense API Reference

LeafSense exposes a robust RESTful API built with **FastAPI**. All endpoints run locally by default at `http://localhost:8000`. Interactive OpenAPI documentation is accessible at `http://localhost:8000/docs` and Redoc at `http://localhost:8000/redoc`.

---

## Endpoint Summary

| Category | Method | Endpoint | Description |
|:---|:---:|:---|:---|
| **System** | `GET` | `/` | Service root status and active operation mode |
| **System** | `GET` | `/health` | Application health, PyTorch device, and model registry status |
| **Diagnostics** | `POST` | `/check-quality` | Pre-flight validation of leaf image resolution, blur, and contrast |
| **Diagnostics** | `POST` | `/predict` | Multi-disease ensemble inference, Grad-CAM overlays, and initial advice |
| **Advisory** | `POST` | `/translate-advice` | Ad-hoc disease treatment translation into 11 Indian languages |
| **Knowledge Base** | `GET` | `/api/rag/status` | Vector collection status, indexed document count, and active models |
| **Knowledge Base** | `POST` | `/api/rag/ingest` | Upload and chunk PDF, TXT, or MD documents into ChromaDB |
| **Knowledge Base** | `POST` | `/api/rag/query` | Semantic vector search over agricultural documents |
| **AI Assistant** | `POST` | `/api/agent/recommend` | Multi-turn Smart Farming consultation powered by IBM Granite 4 |
| **AI Assistant** | `POST` | `/api/agent/session/reset` | Reset conversational memory and farmer profile for a given session |
| **AI Assistant** | `GET` | `/api/agent/session/{conversation_id}` | Retrieve stored session state and dialogue history |

---

## Detailed Endpoint Specifications

### 1. Root & Health Check

#### `GET /`
Returns the status of the LeafSense backend, the operational mode (`full` or `lite`), and model loading counts.

**Response `200 OK`:**
```json
{
  "status": "online",
  "service": "LeafSense Multi-Disease Plant Diagnosis & Smart Farming Agent",
  "version": "2.0.0",
  "mode": "full",
  "models_loaded": 4,
  "supported_classes": 17,
  "supported_languages": 11,
  "rag_knowledge_base": "operational",
  "agent_service": "operational"
}
```

**Curl Example:**
```bash
curl -X GET http://localhost:8000/
```

---

#### `GET /health`
Provides infrastructure diagnostics, including GPU/CPU allocation, PyTorch availability, and checkpoint load states.

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "torch_available": true,
  "device": "cuda",
  "models_loaded": {
    "resnet": true,
    "densenet": true,
    "ghostnet": true,
    "agri": true
  },
  "all_models_loaded": true,
  "num_classes": 17,
  "rag_status": "ready"
}
```

**Curl Example:**
```bash
curl -X GET http://localhost:8000/health
```

---

### 2. Disease Diagnostics & Vision

#### `POST /check-quality`
Pre-flight screening to ensure foliage photos meet minimum diagnostic thresholds.

**Request:** `multipart/form-data` with form field `file` (image file up to 15 MB).

**Response `200 OK`:**
```json
{
  "valid": true,
  "sharpness": 284.15,
  "is_blurry": false,
  "contrast": 68.42,
  "is_low_contrast": false,
  "resolution": [800, 600],
  "is_low_resolution": false,
  "issues": []
}
```

**Curl Example:**
```bash
curl -X POST http://localhost:8000/check-quality \
  -F "file=@sample_leaf.jpg"
```

---

#### `POST /predict`
Executes multi-disease ensemble prediction across the 4 neural networks, generates color-coded Grad-CAM heatmaps, and produces initial actionable advisory.

**Request:** `multipart/form-data`
- `file` (required): Foliage image file.
- `language` (optional, default `"en"`): Target language code (`en`, `hi`, `bn`, `ta`, `te`, `mr`, `gu`, `pa`, `ml`, `kn`, `or`).

**Response `200 OK`:**
```json
{
  "success": true,
  "predictions": [
    {
      "label": "Tomato___Early_blight",
      "confidence": 0.884,
      "percentage": "88%",
      "color": "red"
    },
    {
      "label": "Tomato___Septoria_leaf_spot",
      "confidence": 0.542,
      "percentage": "54%",
      "color": "yellow"
    }
  ],
  "is_multi_disease": true,
  "visualizations": [
    {
      "disease": "Tomato___Early_blight",
      "color": "red",
      "image": "data:image/jpeg;base64,..."
    }
  ],
  "combined_visualization": "data:image/jpeg;base64,...",
  "advice": {
    "crop": "Tomato",
    "disease": "Tomato - Early blight",
    "cause": "Fungal pathogen Alternaria solani...",
    "symptoms": "Concentric rings forming target spots on lower leaves...",
    "treatment": "Apply copper-based fungicides or bio-fungicides...",
    "prevention": "Ensure drip irrigation, remove crop debris, rotate crops..."
  },
  "detected_diseases": [
    "Tomato___Early_blight",
    "Tomato___Septoria_leaf_spot"
  ]
}
```

**Curl Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@tomato_leaf.jpg" \
  -F "language=en"
```

---

#### `POST /translate-advice`
Translates agricultural advisory into any of the 11 supported Indian regional languages.

**Request `application/json`:**
```json
{
  "advice": {
    "crop": "Tomato",
    "disease": "Tomato Early Blight",
    "cause": "Alternaria solani fungus",
    "treatment": "Copper fungicide application"
  },
  "target_language": "hi"
}
```

**Response `200 OK`:**
```json
{
  "success": true,
  "language": "hi",
  "advice": {
    "crop": "टमाटर",
    "disease": "टमाटर अगेती झुलसा",
    "cause": "अल्टरनेरिया सोलानी कवक",
    "treatment": "कॉपर आधारित फफूंदनाशक का छिड़काव"
  }
}
```

---

### 3. Agriculture RAG Knowledge Base

#### `GET /api/rag/status`
Returns ChromaDB indexing statistics, document counts, and embedding model IDs.

**Response `200 OK`:**
```json
{
  "status": "ready",
  "collection_name": "agriculture_knowledge",
  "total_documents": 8,
  "total_chunks": 142,
  "persist_directory": "backend/knowledge_base/chroma_db",
  "embedding_model": "ibm/granite-embedding-278m-multilingual",
  "watsonx_configured": true
}
```

---

#### `POST /api/rag/ingest`
Uploads and indexes an agricultural document (`.pdf`, `.txt`, `.md`) into ChromaDB.

**Request:** `multipart/form-data` with field `file`.

**Response `200 OK`:**
```json
{
  "success": true,
  "filename": "icar_tomato_management.pdf",
  "file_type": "pdf",
  "chunks_created": 24,
  "total_collection_chunks": 166,
  "message": "Successfully ingested icar_tomato_management.pdf into knowledge base."
}
```

---

#### `POST /api/rag/query`
Executes semantic vector retrieval against the indexed agricultural documents.

**Request `application/json`:**
```json
{
  "query": "How to treat yellow rust in wheat organically",
  "top_k": 3,
  "crop": "Wheat"
}
```

**Response `200 OK`:**
```json
{
  "query": "How to treat yellow rust in wheat organically",
  "results_count": 3,
  "sources": [
    {
      "document": "icar_wheat_bulletin.pdf",
      "page": 4,
      "snippet": "Neem oil spray (5ml/L) and bio-fungicides like Trichoderma viride provide early suppression of yellow rust...",
      "relevance_score": 0.89,
      "authority_type": "official_guidance"
    }
  ]
}
```

---

### 4. Smart Farming Conversational Agent

#### `POST /api/agent/recommend`
Executes a multi-turn agricultural consultation using IBM Granite 4, ChromaDB RAG, and conversational memory.

**Request `application/json`:**
```json
{
  "question": "What is the difference between Early Blight and Septoria, and how should I treat it?",
  "crop": "Tomato",
  "location": "Nashik, Maharashtra",
  "language": "en",
  "top_k": 4,
  "conversation_id": "session-1234",
  "detected_diseases": ["Tomato___Early_blight", "Tomato___Septoria_leaf_spot"],
  "disease_predictions": {
    "Tomato___Early_blight": 0.88,
    "Tomato___Septoria_leaf_spot": 0.54
  }
}
```

**Response `200 OK`:**
```json
{
  "answer": "Early Blight and Septoria Leaf Spot both attack tomato foliage, but they have distinct markers:\n\n1. Early Blight: Produces large concentric target rings...\n2. Septoria: Produces numerous small circular spots with dark borders and black specks...\n\nManagement: Apply copper fungicide following label instructions and rotate with non-solanaceous crops.",
  "grounded": true,
  "sources": [
    {
      "document": "icar_tomato_guide.pdf",
      "page": 7,
      "snippet": "Septoria spots remain small with black fruiting pycnidia, unlike large concentric rings of early blight.",
      "relevance_score": 0.92,
      "authority_type": "official_guidance"
    }
  ],
  "confidence_assessment": "High Certainty (88% Confidence) - Supported by verified ICAR extension guidance.",
  "follow_up_questions": [
    "Are the spots appearing mostly on bottom leaves?",
    "Have you noticed white fungal mildew under the leaves?"
  ],
  "quick_replies": [
    "Lower leaves only",
    "Spreading to fruit",
    "Show organic treatments"
  ],
  "disclaimer": "Assistive AI guidance. Confirm with local KVK before chemical spraying.",
  "model_used": "ibm/granite-4-h-small",
  "conversation_id": "session-1234",
  "farmer_profile": {
    "crop": "Tomato",
    "location": "Nashik, Maharashtra"
  },
  "stage": "symptom_deepdive",
  "stage_label": "Symptom Deep-Dive & Verification",
  "intents": ["differential_diagnosis", "treatment_options"],
  "primary_intent": "differential_diagnosis",
  "differential_analysis": {
    "disease_a": "Tomato Early Blight",
    "disease_b": "Tomato Septoria Leaf Spot",
    "shared_symptoms": "Brown circular spots on tomato leaves with chlorotic yellow halos",
    "disease_a_markers": "Concentric rings resembling a target board; larger lesions (up to 1.5 cm); attacks stems and fruit",
    "disease_b_markers": "Numerous small circular spots (2-3 mm) with dark brown margins and tiny black specks (pycnidia)",
    "target_probe": "Inspect the center of the leaf spots with a magnifier: tiny black pepper-like dots indicate Septoria; concentric target rings indicate Early Blight."
  },
  "evidence_type": "official_guidance",
  "reasoning_summary": "Detected differential diagnosis intent comparing Early Blight and Septoria. Retrieved verified ICAR tomato guidelines. Synthesized pairwise distinguishing markers and safe treatment steps."
}
```

---

#### `POST /api/agent/session/reset`
Resets the conversational memory for a given session.

**Request `application/json`:**
```json
{
  "conversation_id": "session-1234"
}
```

**Response `200 OK`:**
```json
{
  "success": true,
  "conversation_id": "session-1234",
  "message": "Session reset successfully"
}
```
