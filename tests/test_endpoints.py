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
from cosmos_client_memory import MemoryIntegrityClient

HEADERS = {"X-Internal-Token": "demo-token", "Content-Type": "application/json"}


def _make_openai_mock():
    payload = json.dumps({
        "classification": "CONCEPTUAL",
        "confidence": 0.99,
        "reasoning": "mocked",
        "recommended_guidance": "FULL",
        "student_facing_message": None,
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


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_check():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert "timestamp" in r.json()


def test_health_requires_no_auth():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# GET /session/{session_id}
# ---------------------------------------------------------------------------

def test_get_session():
    with TestClient(app) as c:
        session_id = str(uuid.uuid4())
        c.post("/session/start", json={
            "student_id": "test", "session_id": session_id,
            "lab_id": "lab01", "course_id": "EE101",
        }, headers=HEADERS)

        r = c.get(f"/session/{session_id}", params={"student_id": "test"}, headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["session_id"] == session_id


def test_get_session_not_found():
    with TestClient(app) as c:
        r = c.get("/session/nonexistent", params={"student_id": "test"}, headers=HEADERS)
        assert r.status_code == 404


def test_get_session_wrong_student_returns_404():
    with TestClient(app) as c:
        session_id = str(uuid.uuid4())
        c.post("/session/start", json={
            "student_id": "alice", "session_id": session_id,
            "lab_id": "lab01", "course_id": "EE101",
        }, headers=HEADERS)

        r = c.get(f"/session/{session_id}", params={"student_id": "bob"}, headers=HEADERS)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /report/generate
# ---------------------------------------------------------------------------

def test_report_generate():
    with TestClient(app) as c:
        session_id = str(uuid.uuid4())
        c.post("/session/start", json={
            "student_id": "test", "session_id": session_id,
            "lab_id": "lab01", "course_id": "EE101",
        }, headers=HEADERS)

        r = c.post("/report/generate", json={
            "student_id": "test", "session_id": session_id,
        }, headers=HEADERS)
        assert r.status_code == 200
        assert "report_id" in r.json()
        assert "report" in r.json()


def test_report_generate_not_found():
    with TestClient(app) as c:
        r = c.post("/report/generate", json={
            "student_id": "test", "session_id": "nonexistent",
        }, headers=HEADERS)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /report/post-lab
# ---------------------------------------------------------------------------

def test_post_lab_report():
    with TestClient(app) as c:
        sid1, sid2 = str(uuid.uuid4()), str(uuid.uuid4())
        for sid in [sid1, sid2]:
            c.post("/session/start", json={
                "student_id": "test", "session_id": sid,
                "lab_id": "lab01", "course_id": "EE101",
            }, headers=HEADERS)
            c.post("/session/end", json={
                "student_id": "test", "session_id": sid,
            }, headers=HEADERS)

        r = c.post("/report/post-lab", json={
            "student_id": "test",
            "session_ids": [sid1, sid2],
            "lab_id": "lab01",
            "course_id": "EE101",
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "report_id" in data
        assert "over_reliance_indicators" in data
        assert "summary" in data


def test_post_lab_report_no_sessions_returns_404():
    with TestClient(app) as c:
        r = c.post("/report/post-lab", json={
            "student_id": "test",
            "session_ids": ["nonexistent-1", "nonexistent-2"],
            "lab_id": "lab01",
            "course_id": "EE101",
        }, headers=HEADERS)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_start_session_duplicate_returns_409():
    with TestClient(app) as c:
        session_id = str(uuid.uuid4())
        c.post("/session/start", json={
            "student_id": "test", "session_id": session_id,
            "lab_id": "lab01", "course_id": "EE101",
        }, headers=HEADERS)

        r = c.post("/session/start", json={
            "student_id": "test", "session_id": session_id,
            "lab_id": "lab01", "course_id": "EE101",
        }, headers=HEADERS)
        assert r.status_code == 409


def test_validate_closed_session_returns_400():
    with TestClient(app) as c:
        app.state.openai_client = _make_openai_mock()
        session_id = str(uuid.uuid4())
        c.post("/session/start", json={
            "student_id": "test", "session_id": session_id,
            "lab_id": "lab01", "course_id": "EE101",
        }, headers=HEADERS)
        c.post("/session/end", json={
            "student_id": "test", "session_id": session_id,
        }, headers=HEADERS)

        r = c.post("/validate", json={
            "student_id": "test", "session_id": session_id,
            "lab_id": "lab01", "course_id": "EE101",
            "question_text": "test", "conversation_history": [],
        }, headers=HEADERS)
        assert r.status_code == 400
        assert "closed" in r.json()["detail"].lower()


def test_start_session_cosmos_error_returns_503():
    class _FailingCosmos(MemoryIntegrityClient):
        async def create_session(self, doc):
            raise Exception("Connection refused")

    with TestClient(app) as c:
        app.state.cosmos = _FailingCosmos()
        r = c.post("/session/start", json={
            "student_id": "test", "session_id": str(uuid.uuid4()),
            "lab_id": "lab01", "course_id": "EE101",
        }, headers=HEADERS)
        assert r.status_code == 503
