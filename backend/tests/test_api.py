"""
LeafSense Backend API Test Suite
Tests health, quality assessment, prediction endpoints, advice translation, and input validation.
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from app import app
from utils import LABEL_COLS, NUM_CLASSES, format_disease_name, validate_image_payload
from image_quality import quality_checker

client = TestClient(app)


def create_test_image(width=300, height=300, color=(34, 139, 34)):
    """Generate a valid test leaf image in memory."""
    img = Image.new("RGB", (width, height), color=color)
    # Add some texture so it's not totally monochrome
    arr = np.array(img)
    arr[50:150, 50:150] = [139, 69, 19]  # Brown spot simulating disease lesion
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_root_endpoint():
    """Verify root metadata endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "LeafSense API"
    assert "version" in data


def test_health_endpoint():
    """Verify health endpoint reports status and 17 disease classes."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "LeafSense API"
    assert data["classes_count"] == NUM_CLASSES
    assert len(data["classes"]) == 17
    assert "Tomato___Early_blight" in data["classes"]


def test_check_quality_valid_image():
    """Verify check-quality endpoint with a normal resolution image."""
    img_buf = create_test_image(400, 400)
    response = client.post(
        "/check-quality",
        files={"file": ("leaf.jpg", img_buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "quality" in data
    assert data["quality"]["quality_score"] > 0


def test_check_quality_low_res_image():
    """Verify check-quality flags low resolution images."""
    img_buf = create_test_image(100, 100)
    response = client.post(
        "/check-quality",
        files={"file": ("leaf_tiny.jpg", img_buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["quality"]["is_valid"] is False
    assert any("too low" in w.lower() for w in data["quality"]["warnings"])


def test_check_quality_empty_payload():
    """Verify check-quality rejects empty payloads."""
    response = client.post(
        "/check-quality",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code in (400, 200)
    if response.status_code == 200:
        data = response.json()
        assert data["quality"]["is_valid"] is False


def test_translate_advice_english():
    """Verify translation service returns formatted advice."""
    response = client.post(
        "/translate-advice",
        json={
            "diseases": ["Tomato___Early_blight"],
            "target_language": "en",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Tomato" in data["advice"] or "Detected Diseases" in data["advice"]


def test_translate_advice_hindi():
    """Verify translation service handles Indian regional language codes."""
    response = client.post(
        "/translate-advice",
        json={
            "diseases": ["Apple___Black_rot"],
            "target_language": "hi",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["advice"]) > 0


def test_predict_endpoint_with_valid_image():
    """Verify predict endpoint receives valid leaf image and returns diagnosis."""
    img_buf = create_test_image(250, 250)
    response = client.post(
        "/predict",
        files={"file": ("test_leaf.jpg", img_buf, "image/jpeg")},
        data={"language": "en"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "detected_diseases" in data
    assert "advice" in data
    assert len(data["detected_diseases"]) > 0


def test_predict_endpoint_corrupted_file():
    """Verify predict endpoint rejects corrupt non-image bytes."""
    corrupt_bytes = b"NOT_AN_IMAGE_FILE_DATA_CORRUPT"
    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
        data={"language": "en"},
    )
    assert response.status_code == 400


def test_format_disease_name_utility():
    """Verify human-readable label formatting utility."""
    formatted = format_disease_name("Apple___Apple_scab")
    assert formatted == "Apple - Apple scab"
    formatted2 = format_disease_name("Tomato___Tomato_Yellow_Leaf_Curl_Virus")
    assert "Tomato - Tomato Yellow Leaf Curl Virus" in formatted2


def test_validate_image_payload_oversized():
    """Verify file size limit enforcement."""
    huge_bytes = b"X" * (11 * 1024 * 1024)  # 11 MB
    is_valid, msg, img = validate_image_payload(huge_bytes, "huge.jpg")
    assert is_valid is False
    assert "exceeds" in msg.lower()
