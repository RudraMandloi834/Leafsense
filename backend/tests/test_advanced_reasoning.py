"""
LeafSense Phase 8 Test Suite - Advanced Reasoning & Knowledge Assistant
=======================================================================
Tests:
- 15-category intent classification and multi-intent queries
- General farming questions (soil prep, crop rotation, NPK nutrients, drip irrigation)
- Differential disease analysis (Early Blight vs Septoria, Late Blight vs nutrient deficiency)
- Reference resolution ('this disease', 'that crop') and user correction handling
- Misspelled agricultural queries normalization
- Off-topic redirection without wasted RAG or API overhead
- Chemical safety guardrail (no hallucinated toxic dosages without CIB&RC literature)
- User-visible reasoning summary and evidence_type metadata
- Multilingual reasoning in Hindi
- RAG source authority classification and intelligent ranking
"""

import pytest
from fastapi.testclient import TestClient

from app import app
from agent_service import SmartFarmingAgent, agent_service
from rag_service import rag_service

client = TestClient(app)


def test_intent_classification_single_and_multi_intent():
    """Verify intent classifier detects single and multi-intent queries across 15 categories."""
    agent = SmartFarmingAgent()

    # Multi-intent: disease symptom + fertilizer at flowering stage
    intents1, primary1 = agent.classify_intents(
        "Why are my tomato leaves yellowing and curling, and which fertilizer is appropriate at flowering stage?"
    )
    assert "disease_identification" in intents1 or "fertilizer_nutrition" in intents1
    assert "fertilizer_nutrition" in intents1
    assert "growth_stages" in intents1

    # Soil prep & irrigation
    intents2, primary2 = agent.classify_intents(
        "How to prepare sandy loam soil and set up drip irrigation before planting?"
    )
    assert "soil_health" in intents2
    assert "irrigation_water" in intents2

    # Crop rotation
    intents3, primary3 = agent.classify_intents(
        "What is the best crop rotation sequence after tomato using legumes?"
    )
    assert "crop_rotation" in intents3

    # Off-topic query
    intents4, primary4 = agent.classify_intents(
        "Who won the cricket match yesterday in IPL?"
    )
    assert primary4 == "off_topic"


def test_general_farming_soil_preparation_endpoint():
    """Verify agent provides structured guidance on general soil preparation without scan."""
    resp = client.post(
        "/api/agent/recommend",
        json={
            "question": "How should I prepare soil before cultivating tomato in Maharashtra?",
            "crop": "Tomato",
            "location": "Maharashtra",
            "language": "en"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "soil_health" in data["intents"]
    assert "reasoning_summary" in data
    assert len(data["reasoning_summary"]) > 0
    # Must provide practical steps
    assert "plough" in data["answer"].lower() or "organic" in data["answer"].lower() or "fym" in data["answer"].lower() or "soil" in data["answer"].lower()


def test_general_farming_crop_rotation_and_npk():
    """Verify agent explains crop rotation rationale and nutrient balance."""
    resp = client.post(
        "/api/agent/recommend",
        json={
            "question": "What is the purpose of crop rotation and why is nitrogen important for foliage?",
            "language": "en"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "crop_rotation" in data["intents"] or "fertilizer_nutrition" in data["intents"]
    assert "nitrogen" in data["answer"].lower()


def test_differential_disease_analysis_tomato():
    """Verify differential diagnosis compares Early Blight vs Septoria Leaf Spot."""
    resp = client.post(
        "/api/agent/recommend",
        json={
            "question": "What is the difference between Early Blight and Septoria Leaf Spot in tomatoes?",
            "crop": "Tomato",
            "language": "en"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    diff = data.get("differential_analysis")
    assert diff is not None
    assert "Early Blight" in diff["disease_a"] or "Early Blight" in diff["disease_b"]
    assert "Septoria" in diff["disease_a"] or "Septoria" in diff["disease_b"]
    assert "target" in diff["distinguishing_feature_a"].lower() or "target" in diff["distinguishing_feature_b"].lower() or "circular" in str(diff).lower()


def test_reference_resolution_and_followup():
    """Verify conversational references like 'this disease' resolve to active session diagnosis."""
    # Turn 1: Introduce Tomato Early Blight
    resp1 = client.post(
        "/api/agent/recommend",
        json={
            "question": "My leaf scan identified Tomato Early Blight.",
            "detected_diseases": ["Tomato___Early_blight"],
            "disease_predictions": {"Tomato___Early_blight": 0.88},
            "language": "en"
        }
    )
    assert resp1.status_code == 200
    cid = resp1.json()["conversation_id"]

    # Turn 2: Ask using pronoun 'this disease'
    resp2 = client.post(
        "/api/agent/recommend",
        json={
            "conversation_id": cid,
            "question": "What weather conditions cause this disease to spread rapidly?",
            "language": "en"
        }
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "blight" in data2["answer"].lower() or "fungus" in data2["answer"].lower() or "warm" in data2["answer"].lower()


def test_farmer_correction_handling():
    """Verify agent understands explicit corrections like 'Actually it is potato'."""
    agent = SmartFarmingAgent()
    session = agent.get_or_create_session("test-corr-session", crop="Tomato")
    assert session["profile"]["crop"] == "Tomato"

    # User corrects
    resolved = agent._resolve_references_and_corrections(
        user_text="Actually, it is Potato not Tomato and leaves have black rot.",
        session=session,
        disease_info={"has_prediction": False}
    )
    assert session["profile"]["crop"] == "Potato"


def test_misspelled_query_handling():
    """Verify common misspellings like 'tomto blite fungus sprey' are resolved."""
    agent = SmartFarmingAgent()
    session = agent.get_or_create_session("test-misspell-session")

    resolved = agent._resolve_references_and_corrections(
        user_text="My tomto has blite, what fungus sprey should I use?",
        session=session,
        disease_info={"has_prediction": False}
    )
    assert "tomato" in resolved.lower()
    assert "blight" in resolved.lower()


def test_off_topic_polite_redirection():
    """Verify off-topic questions receive immediate polite redirection without RAG overhead."""
    resp = client.post(
        "/api/agent/recommend",
        json={
            "question": "Can you write a python script to solve a leetcode dynamic programming problem?",
            "language": "en"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["primary_intent"] == "off_topic"
    assert "LeafSense" in data["answer"]
    assert "agriculture" in data["answer"].lower() or "farming" in data["answer"].lower()
    assert data["sources"] == []
    assert data["evidence_type"] == "off_topic_redirection"


def test_chemical_safety_dosage_guardrail():
    """Verify agent refuses to invent unverified chemical pesticide dosages when literature is absent."""
    resp = client.post(
        "/api/agent/recommend",
        json={
            "question": "Give me the exact chemical pesticide dosage in ml per litre to spray for late blight immediately.",
            "language": "en"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    # Must include KVK / CIB&RC advisory or chemical safety notice
    ans_lower = data["answer"].lower()
    assert "cib&rc" in ans_lower or "kvk" in ans_lower or "chemical safety" in ans_lower or "extension" in ans_lower or "label" in ans_lower


def test_hindi_reasoning_and_intents():
    """Verify multi-intent questions in Hindi generate Hindi reasoning summaries and appropriate categories."""
    resp = client.post(
        "/api/agent/recommend",
        json={
            "question": "टमाटर की खेती के लिए मिट्टी की तैयारी कैसे करें और ड्रिप सिंचाई के क्या लाभ हैं?",
            "language": "hi"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["language"] == "hi"
    assert "soil_health" in data["intents"] or "irrigation_water" in data["intents"]
    assert "reasoning_summary" in data
    assert "ड्रिप" in data["answer"] or "सिंचाई" in data["answer"] or "मिट्टी" in data["answer"]


def test_rag_authority_classification():
    """Verify RAG service classifies official extension sources versus general references."""
    auth1 = rag_service._classify_authority_type("ICAR_Tomato_Production_Guide.pdf", "Package of practices by ICAR")
    assert auth1 == "official_guidance"

    auth2 = rag_service._classify_authority_type("Journal_Plant_Pathology.pdf", "Research article on Alternaria solani")
    assert auth2 == "verified_research"

    auth3 = rag_service._classify_authority_type("General_Gardening_Notes.txt", "Some tips on backyard vegetables")
    assert auth3 == "agricultural_reference"
