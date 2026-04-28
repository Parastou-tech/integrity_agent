from datetime import datetime, timedelta, timezone
import uuid

from cosmos_client_memory import MemoryIntegrityClient
from report_generator import generate_post_lab_report


def _ts(offset_seconds: int = 0) -> str:
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return (base + timedelta(seconds=offset_seconds)).isoformat()


def _question(guidance_level: str, timestamp: str | None = None) -> dict:
    return {
        "question_id": str(uuid.uuid4()),
        "guidance_level": guidance_level,
        "timestamp": timestamp or _ts(),
    }


def _violation(violation_type: str) -> dict:
    return {
        "violation_id": str(uuid.uuid4()),
        "violation_type": violation_type,
    }


def _session(escalated: bool = False, questions: list | None = None, violations: list | None = None) -> dict:
    qs = questions or []
    vs = violations or []
    return {
        "id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "student_id": "test",
        "lab_id": "lab01",
        "escalated": escalated,
        "question_count": len(qs),
        "violation_count": len(vs),
        "questions": qs,
        "violations": vs,
    }


async def _report(sessions: list) -> dict:
    cosmos = MemoryIntegrityClient()
    await cosmos.initialize()
    return await generate_post_lab_report(
        student_id="test",
        session_docs=sessions,
        lab_id="lab01",
        cosmos=cosmos,
    )


# ---------------------------------------------------------------------------
# No indicators — clean session
# ---------------------------------------------------------------------------

async def test_clean_session_no_indicators():
    s = _session(questions=[
        _question("FULL", _ts(0)),
        _question("FULL", _ts(60)),
        _question("FULL", _ts(120)),
    ])
    report = await _report([s])
    ind = report["over_reliance_indicators"]
    assert ind["high_rejection_ratio"] is False
    assert ind["rapid_successive_questions"] is False
    assert ind["escalated_any_session"] is False
    assert not ind["repeated_violation_types"]
    assert ind["low_full_guidance_ratio"] is False
    assert "No over-reliance" in report["summary_text"]


# ---------------------------------------------------------------------------
# high_rejection_ratio
# ---------------------------------------------------------------------------

async def test_high_rejection_ratio_fires():
    # 3 rejected out of 5 = 60% > 20%
    questions = [_question("REJECTED", _ts(i * 60)) for i in range(3)] + \
                [_question("FULL", _ts(i * 60 + 300)) for i in range(2)]
    report = await _report([_session(questions=questions)])
    assert report["over_reliance_indicators"]["high_rejection_ratio"] is True


async def test_high_rejection_ratio_does_not_fire_at_boundary():
    # exactly 20% rejections — threshold is > 0.2, so 20% should not fire
    questions = [_question("REJECTED", _ts(0))] + \
                [_question("FULL", _ts(i * 60 + 60)) for i in range(4)]
    report = await _report([_session(questions=questions)])
    assert report["over_reliance_indicators"]["high_rejection_ratio"] is False


# ---------------------------------------------------------------------------
# escalated_any_session
# ---------------------------------------------------------------------------

async def test_escalated_any_session_fires():
    report = await _report([_session(escalated=True)])
    assert report["over_reliance_indicators"]["escalated_any_session"] is True


async def test_escalated_any_session_false_when_none_escalated():
    report = await _report([_session(escalated=False), _session(escalated=False)])
    assert report["over_reliance_indicators"]["escalated_any_session"] is False


async def test_escalated_any_session_fires_if_one_of_two_escalated():
    report = await _report([_session(escalated=False), _session(escalated=True)])
    assert report["over_reliance_indicators"]["escalated_any_session"] is True


# ---------------------------------------------------------------------------
# repeated_violation_types
# ---------------------------------------------------------------------------

async def test_repeated_violation_types_fires_at_three():
    violations = [_violation("DIRECT_SOLUTION_REQUEST") for _ in range(3)]
    report = await _report([_session(violations=violations)])
    assert "DIRECT_SOLUTION_REQUEST" in report["over_reliance_indicators"]["repeated_violation_types"]


async def test_repeated_violation_types_does_not_fire_at_two():
    violations = [_violation("DIRECT_SOLUTION_REQUEST") for _ in range(2)]
    report = await _report([_session(violations=violations)])
    assert not report["over_reliance_indicators"]["repeated_violation_types"]


# ---------------------------------------------------------------------------
# low_full_guidance_ratio
# ---------------------------------------------------------------------------

async def test_low_full_guidance_ratio_fires():
    # 1 FULL out of 5 = 20% < 50%
    questions = [_question("FULL", _ts(0))] + \
                [_question("MODERATE", _ts(i * 60 + 60)) for i in range(4)]
    report = await _report([_session(questions=questions)])
    assert report["over_reliance_indicators"]["low_full_guidance_ratio"] is True


async def test_low_full_guidance_ratio_does_not_fire_at_50_percent():
    # exactly 50% — threshold is < 0.5, so 50% should not fire
    questions = [_question("FULL", _ts(i * 60)) for i in range(3)] + \
                [_question("MODERATE", _ts(i * 60 + 300)) for i in range(3)]
    report = await _report([_session(questions=questions)])
    assert report["over_reliance_indicators"]["low_full_guidance_ratio"] is False


# ---------------------------------------------------------------------------
# rapid_successive_questions
# ---------------------------------------------------------------------------

async def test_rapid_successive_questions_fires():
    questions = [
        _question("FULL", _ts(0)),
        _question("FULL", _ts(10)),   # 10s apart — under 30s threshold
        _question("FULL", _ts(120)),
    ]
    report = await _report([_session(questions=questions)])
    assert report["over_reliance_indicators"]["rapid_successive_questions"] is True


async def test_rapid_successive_questions_does_not_fire_when_spaced():
    questions = [
        _question("FULL", _ts(0)),
        _question("FULL", _ts(60)),
        _question("FULL", _ts(120)),
    ]
    report = await _report([_session(questions=questions)])
    assert report["over_reliance_indicators"]["rapid_successive_questions"] is False


async def test_rapid_successive_questions_does_not_fire_at_boundary():
    # exactly 30s apart — threshold is < 30, so 30s should not fire
    questions = [
        _question("FULL", _ts(0)),
        _question("FULL", _ts(30)),
    ]
    report = await _report([_session(questions=questions)])
    assert report["over_reliance_indicators"]["rapid_successive_questions"] is False


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

async def test_summary_one_flag():
    report = await _report([_session(escalated=True)])
    assert "1 over-reliance indicator" in report["summary_text"]


async def test_summary_multiple_flags():
    violations = [_violation("DIRECT_SOLUTION_REQUEST") for _ in range(3)]
    questions = [_question("REJECTED", _ts(i * 60)) for i in range(3)] + \
                [_question("FULL", _ts(300))]
    s = _session(escalated=True, questions=questions, violations=violations)
    report = await _report([s])
    assert "Strong recommendation" in report["summary_text"]


# ---------------------------------------------------------------------------
# Multi-session aggregation
# ---------------------------------------------------------------------------

async def test_multi_session_aggregates_questions():
    s1 = _session(questions=[_question("FULL", _ts(0)), _question("REJECTED", _ts(60))])
    s2 = _session(questions=[_question("FULL", _ts(0)), _question("REJECTED", _ts(60))])
    report = await _report([s1, s2])
    assert report["stats"]["total_questions"] == 4
    assert report["stats"]["sessions_analysed"] == 2
