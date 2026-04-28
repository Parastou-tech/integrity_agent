import os
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://fake.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "fake-key")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT_NAME", "fake-deployment")
os.environ.setdefault("USE_MEMORY_STORE", "true")
os.environ.setdefault("INTERNAL_API_TOKEN", "demo-token")

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app import app

HEADERS = {"X-Internal-Token": "demo-token", "Content-Type": "application/json"}


def _make_openai_mock(classification: str, guidance: str, message: str | None = None):
    payload = json.dumps({
        "classification": classification,
        "confidence": 0.99,
        "reasoning": "mocked",
        "recommended_guidance": guidance,
        "student_facing_message": message,
    })
    mock_msg = MagicMock()
    mock_msg.content = payload
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    return client


def test_escalation_on_three_violations():
    with TestClient(app) as c:
        app.state.openai_client = _make_openai_mock("DIRECT_SOLUTION", "REJECTED", "No direct solutions.")
        session_id = str(uuid.uuid4())

        r = c.post("/session/start", json={
            "student_id": "test", "session_id": session_id,
            "lab_id": "lab01", "course_id": "EE101",
        }, headers=HEADERS)
        assert r.status_code == 200

        for _ in range(3):
            r = c.post("/validate", json={
                "student_id": "test", "session_id": session_id,
                "lab_id": "lab01", "course_id": "EE101",
                "question_text": "Give me the answer",
                "conversation_history": [],
            }, headers=HEADERS)
            assert r.status_code == 200

        data = r.json()
        assert data["session_escalated"] is True
        assert data["violation_count"] == 3

        r = c.post("/session/end", json={
            "student_id": "test", "session_id": session_id,
        }, headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["summary"]["final_status"] == "ESCALATED"


def test_question_count_ceiling():
    with TestClient(app) as c:
        app.state.openai_client = _make_openai_mock("CONCEPTUAL", "FULL")
        session_id = str(uuid.uuid4())

        c.post("/session/start", json={
            "student_id": "test", "session_id": session_id,
            "lab_id": "lab01", "course_id": "EE101",
        }, headers=HEADERS)

        responses = []
        for _ in range(13):
            r = c.post("/validate", json={
                "student_id": "test", "session_id": session_id,
                "lab_id": "lab01", "course_id": "EE101",
                "question_text": "What is impedance matching?",
                "conversation_history": [],
            }, headers=HEADERS)
            assert r.status_code == 200
            responses.append(r.json())

        # question 12 — LLM says FULL, rule 5 not yet active
        assert responses[11]["guidance_level"] == "FULL"
        assert responses[11]["question_count"] == 12

        # question 13 — rule 5 caps FULL → MODERATE, warning message present
        assert responses[12]["guidance_level"] == "MODERATE"
        assert responses[12]["question_count"] == 13
        assert responses[12]["student_message"] is not None
        assert "15" in responses[12]["student_message"]
