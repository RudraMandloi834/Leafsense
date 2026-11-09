"""
LeafSense Phase 6 Test Suite - End-to-End Edge Cases & Failure Scenarios
========================================================================
Comprehensive edge case, responsible AI, and failure resilience verification:
- Empty and excessive length user messages
- Empty file uploads and malformed documents
- Watonx API outages and network timeouts (503 Service Unavailable)
- Credential sanitization (zero secret leakage)
- Low confidence uncertainty warnings and disclaimers
- Context detachment and general farming queries
- Multilingual toggle consistency (English and Hindi)
- CORS origin verification
"""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import app
from watsonx_service import (
    WatsonxGraniteService,
    WatsonxServiceError,
    _sanitize_error_message,
)
from rag_service import rag_service
from agent_service import SmartFarmingAgent, agent_service

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Validation & Input Edge Cases
# ---------------------------------------------------------------------------

def test_empty_agent_query_rejected():
    """Verify empty or 1-character chat queries are rejected with HTTP 422."""
    resp = client.post("/api/agent/recommend", json={"question": ""})
    assert resp.status_code == 422

    resp2 = client.post("/api/agent/recommend", json={"question": "a"})
    assert resp2.status_code == 422


def test_excessive_length_agent_query_rejected():
    """Verify queries exceeding 1000 characters are rejected with HTTP 422."""
    long_question = "What is this disease? " * 60  # >1200 characters
    assert len(long_question) > 1000

    resp = client.post("/api/agent/recommend", json={"question": long_question})
    assert resp.status_code == 422


def test_empty_rag_query_rejected():
    """Verify empty RAG query is rejected with HTTP 422."""
    resp = client.post("/api/rag/query", json={"question": ""})
    assert resp.status_code == 422


def test_excessive_length_rag_query_rejected():
    """Verify RAG query exceeding 1000 characters is rejected with HTTP 422."""
    long_question = "Explain powdery mildew symptoms. " * 40
    assert len(long_question) > 1000

    resp = client.post("/api/rag/query", json={"question": long_question})
    assert resp.status_code == 422


def test_empty_file_upload_rejected():
    """Verify 0-byte file upload to RAG ingestion is rejected with HTTP 400."""
    empty_pdf = io.BytesIO(b"")
    resp = client.post(
        "/api/rag/ingest",
        files={"file": ("empty.pdf", empty_pdf, "application/pdf")}
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_corrupt_file_upload_cleaned_up():
    """Verify malformed PDF is rejected with HTTP 400 and deleted from documents directory."""
    corrupt_bytes = io.BytesIO(b"%PDF-corrupted-random-binary-content-12345")
    corrupt_filename = "corrupt_test_leafsense.pdf"
    target_path = rag_service.documents_dir / corrupt_filename

    # Ensure file doesn't exist prior to test
    if target_path.exists():
        target_path.unlink()

    resp = client.post(
        "/api/rag/ingest",
        files={"file": (corrupt_filename, corrupt_bytes, "application/pdf")}
    )

    assert resp.status_code == 400
    assert "malformed" in resp.json()["detail"].lower() or "unable to parse" in resp.json()["detail"].lower()
    # Critical: Ensure corrupt file was removed and not left in knowledge_base/documents/
    assert not target_path.exists()


# ----------------- 2. Watsonx Error Handling & Credential Sanitization -----------------

def test_watsonx_service_error_returns_503():
    """Verify WatsonxServiceError returns HTTP 503 with safe sanitized message."""
    with patch(
        "agent_service.watsonx_service.generate_text",
        side_effect=WatsonxServiceError("Connection refused by watsonx endpoint: api_key=abc123secret456xyz7890123456789012")
    ):
        resp = client.post(
            "/api/agent/recommend",
            json={"question": "How to control aphids on mustard?"}
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data
        assert "503" in str(resp.status_code)
        # Verify secret was sanitized
        assert "abc123secret" not in data["detail"]
        assert "[REDACTED]" in data["detail"] or "unavailable" in data["detail"].lower()


def test_error_message_sanitization_removes_secrets():
    """Verify regex sanitization replaces API keys, tokens, and bearer credentials."""
    dirty_message = (
        "watsonx.ai error: apikey=my_super_secret_api_key_1234567890123456 "
        "Authorization: Bearer 9876543210abcdef9876543210abcdef failed with 401 Unauthorized"
    )
    clean = _sanitize_error_message(dirty_message)

    assert "my_super_secret" not in clean
    assert "9876543210abcdef" not in clean
    assert "[REDACTED]" in clean or "[REDACTED_TOKEN]" in clean


def test_unconfigured_watsonx_service_raises_cleanly():
    """Verify unconfigured WatsonxGraniteService without API key raises WatsonxServiceError."""
    service = WatsonxGraniteService(api_key="", project_id="")
    with pytest.raises(WatsonxServiceError) as exc_info:
        service.generate_text("Test prompt")
    assert "WATSONX_API_KEY" in str(exc_info.value) or "credentials" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 3. Responsible AI & Uncertainty Handling
# ---------------------------------------------------------------------------

def test_responsible_ai_low_confidence_uncertainty_explanation():
    """
    Verify low model confidence (<0.50) triggers an explicit ambiguity warning
    and directs the farmer to physical extension officers (KVK).
    """
    agent = SmartFarmingAgent()
    disease_info = {
        "has_prediction": True,
        "max_confidence": 0.38,
        "primary_disease": "Tomato - Late blight",
        "diseases": [
            {"disease": "Tomato - Late blight", "raw_name": "Tomato___Late_blight", "confidence": 0.38, "percentage": "38.0%"}
        ]
    }

    assessment = agent._assess_confidence(disease_info, grounded=False, language="en")
    assert "Low / Uncertain" in assessment
    assert "KVK" in assessment or "extension" in assessment
    assert "ambiguous" in assessment.lower() or "uncertain" in assessment.lower()


def test_responsible_ai_disclaimer_preservation_en_and_hi():
    """Verify mandatory medical/chemical disclaimer is always attached in English and Hindi."""
    from agent_service import STANDARD_DISCLAIMER_EN, STANDARD_DISCLAIMER_HI

    assert "Krishi Vigyan Kendra (KVK)" in STANDARD_DISCLAIMER_EN
    assert "CIB&RC" in STANDARD_DISCLAIMER_EN

    assert "कृषि विज्ञान केंद्र (KVK)" in STANDARD_DISCLAIMER_HI
    assert "अस्वीकरण" in STANDARD_DISCLAIMER_HI


def test_chat_without_disease_scan_works_gracefully():
    """
    Verify the agent handles general agronomic questions without any scan context
    (e.g., when opened from the header button or floating FAB).
    """
    resp = client.post(
        "/api/agent/recommend",
        json={
            "question": "What is the recommended soil preparation for planting wheat in north India?",
            "crop": "Wheat",
            "location": "Punjab",
            "language": "en"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["crop"] == "Wheat"
    assert data["location"] == "Punjab"
    assert data["disease_context"] is None
    assert len(data["answer"]) > 0
    # Must offer follow-up questions
    assert isinstance(data["follow_up_questions"], list)
    assert len(data["follow_up_questions"]) > 0


def test_chat_with_detached_scan_context():
    """
    Verify that if the farmer clicks 'Detach scan', the request proceeds
    with disease_predictions=None and detected_diseases=None cleanly.
    """
    resp = client.post(
        "/api/agent/recommend",
        json={
            "question": "Can I use compost tea as an organic foliar spray?",
            "disease_predictions": None,
            "detected_diseases": None,
            "language": "en"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["disease_context"] is None


# ---------------------------------------------------------------------------
# 4. Multilingual & Header / CORS Verification
# ---------------------------------------------------------------------------

def test_cors_development_headers():
    """Verify CORS headers allow local Vite dev server."""
    resp = client.options(
        "/api/agent/recommend",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert resp.headers.get("access-control-allow-credentials") == "true"
