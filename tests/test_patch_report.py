import os
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://fake.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "fake-key")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT_NAME", "fake-deployment")
os.environ.setdefault("USE_MEMORY_STORE", "true")
os.environ.setdefault("INTERNAL_API_TOKEN", "demo-token")

import uuid
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app import app, get_openai

HEADERS = {"X-Internal-Token": "demo-token", "Content-Type": "application/json"}
BAD_HEADERS = {"X-Internal-Token": "wrong-token", "Content-Type": "application/json"}


def _make_openai_mock():
    from unittest.mock import MagicMock, AsyncMock
    import json
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


def _start_session_and_get_report_id(c) -> tuple[str, str]:
    """Helper: start + immediately end a session, return (student_id, report_id)."""
    student_id = "patch-test"
    session_id = str(uuid.uuid4())
    c.post("/session/start", json={
        "student_id": student_id, "session_id": session_id,
        "lab_id": "lab01", "course_id": "EE101",
    }, headers=HEADERS)
    r = c.post("/session/end", json={
        "student_id": student_id, "session_id": session_id,
    }, headers=HEADERS)
    return student_id, r.json()["report_id"]


def test_patch_happy_path():
    app.dependency_overrides[get_openai] = lambda: _make_openai_mock()
    try:
        with TestClient(app) as c:
            student_id, report_id = _start_session_and_get_report_id(c)

            r = c.patch(f"/report/{report_id}", json={
                "student_id": student_id,
                "instructor_notes": {"flagged": True, "note": "test note"},
            }, headers=HEADERS)
            assert r.status_code == 200
            assert r.json()["instructor_notes"] == {"flagged": True, "note": "test note"}

            r = c.get(f"/report/{report_id}", params={"student_id": student_id}, headers=HEADERS)
            assert r.json()["instructor_notes"] == {"flagged": True, "note": "test note"}
    finally:
        app.dependency_overrides.clear()


def test_patch_overwrites_existing_notes():
    app.dependency_overrides[get_openai] = lambda: _make_openai_mock()
    try:
        with TestClient(app) as c:
            student_id, report_id = _start_session_and_get_report_id(c)

            c.patch(f"/report/{report_id}", json={
                "student_id": student_id,
                "instructor_notes": {"flagged": True, "note": "first"},
            }, headers=HEADERS)

            r = c.patch(f"/report/{report_id}", json={
                "student_id": student_id,
                "instructor_notes": {"flagged": False, "note": "overwritten"},
            }, headers=HEADERS)
            assert r.status_code == 200

            r = c.get(f"/report/{report_id}", params={"student_id": student_id}, headers=HEADERS)
            assert r.json()["instructor_notes"] == {"flagged": False, "note": "overwritten"}
    finally:
        app.dependency_overrides.clear()


def test_patch_wrong_student_id_returns_404():
    app.dependency_overrides[get_openai] = lambda: _make_openai_mock()
    try:
        with TestClient(app) as c:
            student_id, report_id = _start_session_and_get_report_id(c)

            r = c.patch(f"/report/{report_id}", json={
                "student_id": "different-student",
                "instructor_notes": {"flagged": True},
            }, headers=HEADERS)
            assert r.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_patch_nonexistent_report_returns_404():
    app.dependency_overrides[get_openai] = lambda: _make_openai_mock()
    try:
        with TestClient(app) as c:
            r = c.patch("/report/nonexistent-id", json={
                "student_id": "test",
                "instructor_notes": {"flagged": True},
            }, headers=HEADERS)
            assert r.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_patch_bad_token_returns_403():
    app.dependency_overrides[get_openai] = lambda: _make_openai_mock()
    try:
        with TestClient(app) as c:
            student_id, report_id = _start_session_and_get_report_id(c)

            r = c.patch(f"/report/{report_id}", json={
                "student_id": student_id,
                "instructor_notes": {"flagged": True},
            }, headers=BAD_HEADERS)
            assert r.status_code == 403
    finally:
        app.dependency_overrides.clear()
