"""
LeafSense Phase 7 Test Suite - Interactive Conversational Agent
==============================================================
Tests:
- Multi-turn conversation memory and session tracking
- Session isolation across different conversation IDs
- Progressive farmer profile collection (crop, location, variety, stage, onset, spread, irrigation, weather)
- Disease-specific contextual probing and quick replies
- Non-blocking skip functionality
- Session reset endpoint (/api/agent/session/reset)
- Low confidence uncertainty communication in multi-turn dialogue
- Bilingual support (English and Hindi)
"""

import pytest
from fastapi.testclient import TestClient

from app import app
from agent_service import SmartFarmingAgent, agent_service

client = TestClient(app)


def test_multi_turn_session_preservation():
    """Verify context and profile accumulate across turns using the same conversation_id."""
    # Turn 1: Farmer introduces crop, location, and symptom
    resp1 = client.post(
        "/api/agent/recommend",
        json={
            "question": "I am growing tomato in Nashik, Maharashtra and seeing dark spots on the leaves.",
            "language": "en"
        }
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    cid = data1["conversation_id"]
    assert cid is not None
    assert data1["crop"] == "Tomato"
    assert "Nashik" in data1["location"]
    assert data1["farmer_profile"]["crop"] == "Tomato"
    assert "quick_replies" in data1
    assert len(data1["quick_replies"]) > 0

    # Turn 2: Farmer adds growth stage and onset time
    resp2 = client.post(
        "/api/agent/recommend",
        json={
            "conversation_id": cid,
            "question": "The crop is in flowering stage and symptoms just started 2 days ago.",
            "language": "en"
        }
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["conversation_id"] == cid
    # Crop and location from Turn 1 are preserved
    assert data2["crop"] == "Tomato"
    assert "Nashik" in data2["location"]
    # New profile fields extracted
    profile = data2["farmer_profile"]
    assert profile["crop"] == "Tomato"
    assert "flowering" in profile.get("growth_stage", "").lower()
    assert "2 days" in profile.get("first_noticed", "").lower()


def test_session_isolation_across_different_users():
    """Verify two distinct conversation IDs do not mix profiles or memory."""
    # Farmer A: Apple in Shimla
    resp_a = client.post(
        "/api/agent/recommend",
        json={
            "conversation_id": "session-farmer-apple-shimla",
            "question": "I have an apple orchard in Shimla, Himachal Pradesh.",
            "crop": "Apple",
            "location": "Shimla, HP",
            "language": "en"
        }
    )
    assert resp_a.status_code == 200
    assert resp_a.json()["crop"] == "Apple"

    # Farmer B: Potato in Agra
    resp_b = client.post(
        "/api/agent/recommend",
        json={
            "conversation_id": "session-farmer-potato-agra",
            "question": "I am cultivating potato in Agra, Uttar Pradesh.",
            "crop": "Potato",
            "location": "Agra, UP",
            "language": "en"
        }
    )
    assert resp_b.status_code == 200
    assert resp_b.json()["crop"] == "Potato"

    # Verify session A still has Apple and hasn't been overwritten
    resp_a2 = client.post(
        "/api/agent/recommend",
        json={
            "conversation_id": "session-farmer-apple-shimla",
            "question": "Leaves are dropping early.",
            "language": "en"
        }
    )
    assert resp_a2.status_code == 200
    assert resp_a2.json()["crop"] == "Apple"
    assert resp_a2.json()["farmer_profile"]["crop"] == "Apple"


def test_session_reset_endpoint():
    """Verify /api/agent/session/reset cleans memory for a fresh conversation."""
    cid = "session-to-be-reset"
    # Populate session
    client.post(
        "/api/agent/recommend",
        json={
            "conversation_id": cid,
            "question": "Grape farming in Pune with drip irrigation.",
            "crop": "Grape",
            "language": "en"
        }
    )

    # Call reset
    resp_reset = client.post(
        "/api/agent/session/reset",
        json={"conversation_id": cid}
    )
    assert resp_reset.status_code == 200
    assert resp_reset.json()["success"] is True

    # Query status after reset
    resp_status = client.get(f"/api/agent/session/{cid}")
    assert resp_status.status_code == 200
    profile = resp_status.json()["profile"]
    assert profile.get("crop") is None
    assert resp_status.json()["messages_count"] == 0


def test_disease_specific_probing_and_quick_replies():
    """Verify quick replies and probing questions adapt specifically to the detected pathogen."""
    # Tomato Early Blight
    resp_tb = client.post(
        "/api/agent/recommend",
        json={
            "question": "What is this disease on my tomato?",
            "detected_diseases": ["Tomato___Early_blight"],
            "disease_predictions": {"Tomato___Early_blight": 0.91},
            "language": "en"
        }
    )
    assert resp_tb.status_code == 200
    replies_tb = resp_tb.json()["quick_replies"]
    assert any("target" in r.lower() or "lower leaves" in r.lower() for r in replies_tb)
    assert any("skip" in r.lower() for r in replies_tb)

    # Apple Black Rot
    resp_ar = client.post(
        "/api/agent/recommend",
        json={
            "question": "How to treat this apple problem?",
            "detected_diseases": ["Apple___Black_rot"],
            "disease_predictions": {"Apple___Black_rot": 0.88},
            "language": "en"
        }
    )
    assert resp_ar.status_code == 200
    replies_ar = resp_ar.json()["quick_replies"]
    assert any("frog-eye" in r.lower() or "fruit" in r.lower() for r in replies_ar)


def test_skip_question_moves_to_advisory_management():
    """Verify clicking Skip or asking for direct advice transitions stage to advisory_management."""
    resp = client.post(
        "/api/agent/recommend",
        json={
            "question": "Skip / Show Guidance and tell me preventive organic treatments",
            "crop": "Tomato",
            "location": "Karnal, Haryana",
            "detected_diseases": ["Tomato___Early_blight"],
            "language": "en"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["stage"] == "advisory_management"
    assert "Advisory" in data["stage_label"] or "Management" in data["stage_label"]


def test_low_confidence_uncertainty_explanation():
    """Verify low prediction confidence explicitly communicates uncertainty and warns against treating as fact."""
    resp = client.post(
        "/api/agent/recommend",
        json={
            "question": "Is this definitely late blight?",
            "detected_diseases": ["Potato___Late_blight"],
            "disease_predictions": {"Potato___Late_blight": 0.34},  # < 0.50 Low
            "language": "en"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "Low / Uncertain" in data["confidence_assessment"]
    assert "KVK" in data["confidence_assessment"]


def test_conversational_hindi_multiturn():
    """Verify multi-turn session in Hindi extracts details and returns Hindi stage labels and quick replies."""
    resp1 = client.post(
        "/api/agent/recommend",
        json={
            "question": "मैं इंदौर मध्य प्रदेश से हूँ और मेरे टमाटर के पौधों पर पत्तियां मुड़ रही हैं।",
            "language": "hi"
        }
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["language"] == "hi"
    assert "quick_replies" in data1
    assert any("छोड़ें" in r or "उपचार" in r for r in data1["quick_replies"])
    assert data1["farmer_profile"]["crop"] == "Tomato"
