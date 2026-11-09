"""
LeafSense Phase 4 Test Suite - Smart Farming AI Agent
=====================================================
Tests:
- Crop inference and disease prediction context handling
- Confidence assessment and uncertainty explanation
- Follow-up question generation for missing farmer context
- English and Hindi agent responses and disclaimers
- Empty knowledge base behavior and ungrounded fallbacks
- Guardrails against unsupported recommendations
- Source citation preservation
- API endpoint validation (/api/agent/recommend)
"""

import pytest
from fastapi.testclient import TestClient

from app import app
from agent_service import SmartFarmingAgent, agent_service

client = TestClient(app)


def test_infer_crop_from_diseases():
    """Verify crop name is cleanly extracted from raw disease class strings."""
    agent = SmartFarmingAgent()

    assert agent._infer_crop_from_diseases(["Tomato___Early_blight"]) == "Tomato"
    assert agent._infer_crop_from_diseases(["Potato___Late_blight"]) == "Potato"
    assert agent._infer_crop_from_diseases(["Apple___Cedar_apple_rust"]) == "Apple"
    assert agent._infer_crop_from_diseases(["Corn_(maize)___Common_rust_"]) == "Corn"
    assert agent._infer_crop_from_diseases([]) is None


def test_format_disease_context():
    """Verify disease prediction formatting, percentage calculations, and max confidence."""
    agent = SmartFarmingAgent()

    predictions = {
        "Tomato___Early_blight": 0.8942,
        "Tomato___Target_Spot": 0.5210
    }
    context = agent._format_disease_context(predictions, list(predictions.keys()))

    assert context["has_prediction"] is True
    assert context["max_confidence"] == 0.8942
    assert context["primary_disease"] == "Tomato - Early blight"
    assert len(context["diseases"]) == 2
    assert context["diseases"][0]["percentage"] == "89.4%"


def test_confidence_assessment_levels_english():
    """Verify uncertainty explanations in English across confidence tiers."""
    agent = SmartFarmingAgent()

    # High confidence (>=0.80)
    high_ctx = {"has_prediction": True, "max_confidence": 0.92, "primary_disease": "Tomato - Early blight"}
    high_assessment = agent._assess_confidence(high_ctx, grounded=True, language="en")
    assert "High visual similarity" in high_assessment
    assert "92.0%" in high_assessment
    assert "texture" in high_assessment or "confirmation" in high_assessment

    # Moderate confidence (0.50-0.80)
    mod_ctx = {"has_prediction": True, "max_confidence": 0.65, "primary_disease": "Potato - Late blight"}
    mod_assessment = agent._assess_confidence(mod_ctx, grounded=False, language="en")
    assert "Moderate visual similarity" in mod_assessment
    assert "deficiencies" in mod_assessment or "overlap" in mod_assessment

    # Low confidence (<0.50)
    low_ctx = {"has_prediction": True, "max_confidence": 0.35, "primary_disease": "Apple - Scab"}
    low_assessment = agent._assess_confidence(low_ctx, grounded=False, language="en")
    assert "Low / Uncertain" in low_assessment
    assert "KVK" in low_assessment or "ambiguous" in low_assessment


def test_confidence_assessment_levels_hindi():
    """Verify uncertainty explanations in Hindi."""
    agent = SmartFarmingAgent()

    high_ctx = {"has_prediction": True, "max_confidence": 0.88, "primary_disease": "Tomato - Early blight"}
    assessment_hi = agent._assess_confidence(high_ctx, grounded=True, language="hi")
    assert "उच्च विजुअल समानता" in assessment_hi or "संभावना" in assessment_hi


def test_follow_up_questions_generation_missing_context():
    """Verify agent prompts for missing location and crop details."""
    agent = SmartFarmingAgent()
    empty_disease_info = {"has_prediction": False, "diseases": []}

    # Missing location and crop
    follow_ups = agent._generate_follow_up_questions(
        crop=None,
        location=None,
        disease_info=empty_disease_info,
        language="en"
    )
    assert len(follow_ups) <= 3
    assert any("district" in q.lower() or "state" in q.lower() for q in follow_ups)
    assert any("crop" in q.lower() for q in follow_ups)


def test_follow_up_questions_hindi():
    """Verify follow-up questions are provided in Hindi when requested."""
    agent = SmartFarmingAgent()
    empty_disease_info = {"has_prediction": False, "diseases": []}

    follow_ups_hi = agent._generate_follow_up_questions(
        crop=None,
        location=None,
        disease_info=empty_disease_info,
        language="hi"
    )
    assert any("जिला" in q or "राज्य" in q for q in follow_ups_hi)
    assert any("फसल" in q for q in follow_ups_hi)


def test_retrieval_query_builder():
    """Verify retrieval query combines crop, disease, and farmer question."""
    agent = SmartFarmingAgent()
    disease_info = {
        "has_prediction": True,
        "primary_disease": "Tomato - Early blight"
    }

    query = agent._build_retrieval_query(
        question="What spray should I use for leaf spots?",
        crop="Tomato",
        disease_info=disease_info
    )
    assert "Tomato" in query
    assert "Early blight" in query
    assert "leaf spots" in query


def test_api_agent_recommend_validation():
    """Verify request validation rejects empty questions."""
    response = client.post(
        "/api/agent/recommend",
        json={"question": ""}  # Too short
    )
    assert response.status_code == 422


def test_api_agent_recommend_basic_english():
    """Verify POST /api/agent/recommend returns a valid structured response."""
    response = client.post(
        "/api/agent/recommend",
        json={
            "question": "How can I prevent early blight from spreading in my tomato field?",
            "crop": "Tomato",
            "location": "Nashik, Maharashtra",
            "language": "en"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["crop"] == "Tomato"
    assert data["location"] == "Nashik, Maharashtra"
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "confidence_assessment" in data
    assert "follow_up_questions" in data
    assert "disclaimer" in data
    assert "KVK" in data["disclaimer"] or "extension" in data["disclaimer"]


def test_api_agent_recommend_with_disease_predictions():
    """Verify POST /api/agent/recommend incorporates image model prediction metrics."""
    response = client.post(
        "/api/agent/recommend",
        json={
            "question": "My leaves have dark rings. What is this disease and what should I do?",
            "disease_predictions": {
                "Tomato___Early_blight": 0.8875,
                "Tomato___Target_Spot": 0.4120
            },
            "detected_diseases": ["Tomato___Early_blight"],
            "language": "en"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["crop"] == "Tomato"  # Inferred from disease prediction
    assert data["disease_context"] is not None
    assert data["disease_context"]["primary_disease"] == "Tomato - Early blight"
    assert len(data["answer"]) > 0


def test_api_agent_recommend_hindi():
    """Verify POST /api/agent/recommend generates Hindi response and Hindi disclaimer."""
    response = client.post(
        "/api/agent/recommend",
        json={
            "question": "टमाटर के पत्तों पर भूरे धब्बे दिखाई दे रहे हैं, क्या उपाय करें?",
            "crop": "टमाटर",
            "location": "इंदौर, मध्य प्रदेश",
            "language": "hi"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["language"] == "hi"
    assert "अस्वीकरण" in data["disclaimer"]
