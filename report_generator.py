"""
Report Generator — builds and persists integrity reports.

Two public functions:
  generate_session_report()   — end-of-session summary for Instructor Co-pilot
  generate_post_lab_report()  — cross-session over-reliance analysis
"""

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from models import FinalStatus, GuidanceLevel, ReportType

if TYPE_CHECKING:
    from cosmos_client import CosmosIntegrityClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session report
# ---------------------------------------------------------------------------

async def generate_session_report(
    session_doc: dict,
    cosmos: "CosmosIntegrityClient",
) -> dict:
    """
    Build and persist a SESSION integrity report in the Cosmos DB 'reports' container.

    Returns the stored report document.
    """
    violations = session_doc.get("violations", [])
    questions = session_doc.get("questions", [])
    total_questions = session_doc.get("question_count", len(questions))
    violation_count = session_doc.get("violation_count", len(violations))
    escalated = session_doc.get("escalated", False)

    # guidance distribution
    distribution = {level.value: 0 for level in GuidanceLevel}
    for q in questions:
        level = q.get("guidance_level", GuidanceLevel.FULL.value)
        if level in distribution:
            distribution[level] += 1

    # final status
    if escalated:
        final_status = FinalStatus.ESCALATED.value
    elif violation_count > 0 or session_doc.get("warning_issued", False):
        final_status = FinalStatus.WARNING.value
    else:
        final_status = FinalStatus.CLEAN.value

    # escalation log
    escalation_ts = None
    escalation_reason = None
    if escalated and violations:
        # timestamp of the 3rd violation
        third = violations[2] if len(violations) >= 3 else violations[-1]
        escalation_ts = third.get("timestamp")
        escalation_reason = (
            f"Violation threshold (3) exceeded. "
            f"Final violation type: {third.get('violation_type', 'unknown')}"
        )

    report_id = str(uuid.uuid4())
    report = {
        "id": report_id,
        "report_id": report_id,
        "student_id": session_doc["student_id"],
        "session_id": session_doc["session_id"],
        "lab_id": session_doc.get("lab_id", ""),
        "course_id": session_doc.get("course_id", "CSC580"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "report_type": ReportType.SESSION.value,
        "summary": {
            "total_questions": total_questions,
            "violation_count": violation_count,
            "escalated": escalated,
            "final_status": final_status,
            "guidance_distribution": distribution,
            "question_frequency_warning_triggered": session_doc.get("warning_issued", False),
        },
        "violations_detail": violations,
        "escalation_log": {
            "escalated": escalated,
            "escalation_timestamp": escalation_ts,
            "reason": escalation_reason,
        },
        "instructor_notes": "",
        "raw_session_snapshot": session_doc,
    }

    await cosmos.create_report(report)
    logger.info(
        "Session report %s generated for student=%s session=%s status=%s",
        report_id,
        session_doc["student_id"],
        session_doc["session_id"],
        final_status,
    )
    return report


# ---------------------------------------------------------------------------
# Post-lab over-reliance report
# ---------------------------------------------------------------------------

async def generate_post_lab_report(
    student_id: str,
    session_docs: list[dict],
    lab_id: str,
    cosmos: "CosmosIntegrityClient",
) -> dict:
    """
    Analyse multiple sessions for a single lab assignment and produce a
    POST_LAB over-reliance report.

    Returns the stored report document.
    """
    if not session_docs:
        raise ValueError("session_docs must not be empty.")

    all_questions: list[dict] = []
    all_violations: list[dict] = []
    any_escalated = False

    for s in session_docs:
        all_questions.extend(s.get("questions", []))
        all_violations.extend(s.get("violations", []))
        if s.get("escalated", False):
            any_escalated = True

    total_q = len(all_questions)
    total_rejections = sum(
        1 for q in all_questions if q.get("guidance_level") == GuidanceLevel.REJECTED.value
    )
    total_full = sum(
        1 for q in all_questions if q.get("guidance_level") == GuidanceLevel.FULL.value
    )

    # Indicator: high rejection ratio
    high_rejection_ratio = (
        (total_rejections / total_q) > 0.2 if total_q > 0 else False
    )

    # Indicator: rapid successive questions (< 30 s apart)
    rapid_successive = _check_rapid_successive(all_questions)

    # Indicator: same violation type repeated 3+ times
    violation_type_counts: dict[str, int] = {}
    for v in all_violations:
        vt = v.get("violation_type", "UNKNOWN")
        violation_type_counts[vt] = violation_type_counts.get(vt, 0) + 1
    repeated_violation_types = {
        vt: count for vt, count in violation_type_counts.items() if count >= 3
    }

    # Indicator: low FULL guidance ratio
    low_full_ratio = (total_full / total_q) < 0.5 if total_q > 0 else False

    indicators = {
        "high_rejection_ratio": high_rejection_ratio,
        "rapid_successive_questions": rapid_successive,
        "escalated_any_session": any_escalated,
        "repeated_violation_types": repeated_violation_types,
        "low_full_guidance_ratio": low_full_ratio,
    }

    flagged_count = sum(
        1 for v in indicators.values() if (v is True or (isinstance(v, dict) and v))
    )
    if flagged_count == 0:
        summary = "No over-reliance indicators detected."
    elif flagged_count == 1:
        summary = "1 over-reliance indicator detected. Recommend instructor review."
    else:
        summary = (
            f"{flagged_count} over-reliance indicators detected. "
            "Strong recommendation for instructor review."
        )

    report_id = str(uuid.uuid4())
    report = {
        "id": report_id,
        "report_id": report_id,
        "student_id": student_id,
        "lab_id": lab_id,
        "session_ids": [s["session_id"] for s in session_docs],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "report_type": ReportType.POST_LAB.value,
        "summary_text": summary,
        "stats": {
            "total_questions": total_q,
            "total_violations": len(all_violations),
            "total_rejections": total_rejections,
            "sessions_analysed": len(session_docs),
        },
        "over_reliance_indicators": indicators,
        "violations_detail": all_violations,
        "instructor_notes": "",
    }

    await cosmos.create_report(report)
    logger.info(
        "Post-lab report %s generated for student=%s lab=%s flags=%d",
        report_id,
        student_id,
        lab_id,
        flagged_count,
    )
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_rapid_successive(questions: list[dict]) -> bool:
    """Return True if any two consecutive questions were asked < 30 seconds apart."""
    from datetime import timezone

    timestamps = []
    for q in questions:
        ts_str = q.get("timestamp")
        if ts_str:
            try:
                # Handle both "Z" suffix and offset-aware ISO strings
                ts_str_clean = ts_str.replace("Z", "+00:00")
                ts = datetime.fromisoformat(ts_str_clean)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                timestamps.append(ts)
            except ValueError:
                pass

    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if 0 < delta < 30:
            return True
    return False
