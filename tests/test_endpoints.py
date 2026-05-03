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
        "concept_tags": [],
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


## ---------------------------------------------------------------------------
# GET / and GET /health — AIEIC interface contract shape
# ---------------------------------------------------------------------------

def test_root_endpoint():
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body == {"status": "ok", "agent": "integrity", "version": "0.1.0"}


def test_health_check():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["agent"] == "integrity"
        assert body["version"] == "0.1.0"
        assert "timestamp" in body


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


# ---------------------------------------------------------------------------
# GET /analytics/lab/{lab_id}
# ---------------------------------------------------------------------------

def test_lab_analytics_empty_lab():
    with TestClient(app) as c:
        r = c.get("/analytics/lab/lab-no-sessions", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["lab_id"] == "lab-no-sessions"
        assert data["session_stats"]["total_sessions"] == 0
        assert data["question_stats"]["total_questions"] == 0
        assert data["per_student"] == []
        assert data["concept_struggle_summary"] == []


def test_lab_analytics_with_sessions():
    with TestClient(app) as c:
        app.state.openai_client = _make_openai_mock()
        lab_id = f"lab-analytics-{uuid.uuid4()}"

        # Start two sessions for different students, same lab
        for student in ["alice", "bob"]:
            sid = str(uuid.uuid4())
            c.post("/session/start", json={
                "student_id": student, "session_id": sid,
                "lab_id": lab_id, "course_id": "EE101",
            }, headers=HEADERS)
            c.post("/session/end", json={
                "student_id": student, "session_id": sid,
            }, headers=HEADERS)

        r = c.get(f"/analytics/lab/{lab_id}", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()

        assert data["lab_id"] == lab_id
        assert data["session_stats"]["total_sessions"] == 2
        assert data["session_stats"]["closed_sessions"] == 2
        assert "total_questions" in data["question_stats"]
        assert "direct_solution_attempts" in data["question_stats"]
        assert "CONCEPTUAL" in data["classification_distribution"]
        assert len(data["per_student"]) == 2
        statuses = {p["student_id"]: p["status"] for p in data["per_student"]}
        assert "alice" in statuses
        assert "bob" in statuses


def test_lab_analytics_course_id_filter():
    with TestClient(app) as c:
        app.state.openai_client = _make_openai_mock()
        lab_id = f"lab-filter-{uuid.uuid4()}"

        # Two sessions — different course IDs
        for course in ["EE101", "EE202"]:
            sid = str(uuid.uuid4())
            c.post("/session/start", json={
                "student_id": f"student-{course}", "session_id": sid,
                "lab_id": lab_id, "course_id": course,
            }, headers=HEADERS)

        # Filter to EE101 only
        r = c.get(f"/analytics/lab/{lab_id}", params={"course_id": "EE101"}, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["session_stats"]["total_sessions"] == 1
        assert data["per_student"][0]["student_id"] == "student-EE101"


def test_lab_analytics_requires_auth():
    with TestClient(app) as c:
        r = c.get("/analytics/lab/lab01", headers={"X-Internal-Token": "wrong"})
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# POST /integrity/log — AIEIC interface contract endpoint
# ---------------------------------------------------------------------------

def test_integrity_log_auto_creates_session():
    """First log call for an unknown session creates the session document."""
    with TestClient(app) as c:
        app.state.openai_client = _make_openai_mock()
        session_id = str(uuid.uuid4())

        r = c.post("/integrity/log", json={
            "student_id": "alex_m",
            "session_id": session_id,
            "message": "What does impedance matching mean?",
            "response_time_ms": 1234,
            "lab_id": "lab4",
            "course_id": "CSC580",
        }, headers=HEADERS)

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert isinstance(body["interaction_id"], str)
        # Session is now retrievable
        s = c.get(f"/session/{session_id}", params={"student_id": "alex_m"}, headers=HEADERS)
        assert s.status_code == 200
        assert s.json()["question_count"] == 1
        assert s.json()["lab_id"] == "lab4"


def test_integrity_log_appends_to_existing_session():
    with TestClient(app) as c:
        app.state.openai_client = _make_openai_mock()
        session_id = str(uuid.uuid4())
        c.post("/session/start", json={
            "student_id": "alex_m", "session_id": session_id,
            "lab_id": "lab4", "course_id": "CSC580",
        }, headers=HEADERS)

        for msg in ["q1", "q2", "q3"]:
            r = c.post("/integrity/log", json={
                "student_id": "alex_m",
                "session_id": session_id,
                "message": msg,
            }, headers=HEADERS)
            assert r.status_code == 200

        s = c.get(f"/session/{session_id}", params={"student_id": "alex_m"}, headers=HEADERS)
        assert s.json()["question_count"] == 3
        assert len(s.json()["questions"]) == 3
        # hint_level was persisted on each question
        for q in s.json()["questions"]:
            assert q["hint_level"] in (1, 2, 3)


def test_integrity_log_rejects_closed_session():
    with TestClient(app) as c:
        app.state.openai_client = _make_openai_mock()
        session_id = str(uuid.uuid4())
        c.post("/session/start", json={
            "student_id": "alex_m", "session_id": session_id,
            "lab_id": "lab4", "course_id": "CSC580",
        }, headers=HEADERS)
        c.post("/session/end", json={
            "student_id": "alex_m", "session_id": session_id,
        }, headers=HEADERS)

        r = c.post("/integrity/log", json={
            "student_id": "alex_m", "session_id": session_id, "message": "hi",
        }, headers=HEADERS)
        assert r.status_code == 400


def test_integrity_log_requires_auth():
    with TestClient(app) as c:
        r = c.post("/integrity/log", json={
            "student_id": "alex_m",
            "session_id": str(uuid.uuid4()),
            "message": "hi",
        }, headers={"X-Internal-Token": "wrong"})
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# GET /integrity/context/{student_id} — AIEIC interface contract endpoint
# ---------------------------------------------------------------------------

def test_integrity_context_empty_student():
    with TestClient(app) as c:
        r = c.get("/integrity/context/no-such-student", headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["total_questions"] == 0
        assert body["sessions_count"] == 0
        assert body["avg_hint_level"] == 0.0
        assert body["avg_questions_per_session"] == 0.0
        assert body["session_help_frequency"] == {}
        # All seven contract fields present
        for field in [
            "total_questions", "question_type_distribution", "avg_hint_level",
            "sessions_count", "avg_questions_per_session", "session_help_frequency",
            "summary",
        ]:
            assert field in body


def test_integrity_context_aggregates_across_sessions():
    with TestClient(app) as c:
        app.state.openai_client = _make_openai_mock()
        student_id = f"alex-{uuid.uuid4()}"

        # Two sessions, 3 messages in the first, 2 in the second
        sid1, sid2 = str(uuid.uuid4()), str(uuid.uuid4())
        for sid, count in [(sid1, 3), (sid2, 2)]:
            for i in range(count):
                c.post("/integrity/log", json={
                    "student_id": student_id,
                    "session_id": sid,
                    "message": f"question {i}",
                    "lab_id": "lab4",
                }, headers=HEADERS)

        r = c.get(f"/integrity/context/{student_id}", headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["total_questions"] == 5
        assert body["sessions_count"] == 2
        assert body["avg_questions_per_session"] == 2.5
        # The mock always returns CONCEPTUAL → hint_level falls back to 1 (NUDGE)
        assert body["question_type_distribution"]["CONCEPTUAL"] == 5
        assert body["avg_hint_level"] == 1.0
        assert body["session_help_frequency"][sid1] == 3
        assert body["session_help_frequency"][sid2] == 2
        assert "5 questions across 2 sessions" in body["summary"]


def test_integrity_context_requires_auth():
    with TestClient(app) as c:
        r = c.get("/integrity/context/alex_m", headers={"X-Internal-Token": "wrong"})
        assert r.status_code == 403
