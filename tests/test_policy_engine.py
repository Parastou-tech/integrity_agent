import pytest

from models import GuidanceLevel, QuestionClassification, ViolationSeverity, ViolationType
from policy_engine import ClassificationResult, determine_guidance_level


def _make_llm_result(classification: str, guidance: str, message: str | None = None) -> ClassificationResult:
    return ClassificationResult(
        classification=QuestionClassification(classification),
        confidence=0.99,
        reasoning="test",
        recommended_guidance=GuidanceLevel(guidance),
        student_facing_message=message,
    )


# ---------------------------------------------------------------------------
# Rule 1 — hard frequency cap (question_count > 15)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question_count,expected_guidance,expected_violation", [
    (16, GuidanceLevel.REJECTED,  True),
    (15, GuidanceLevel.MODERATE, False),  # 15 is not > 15 so rule 1 does not fire, but rule 5 caps FULL→MODERATE
])
def test_rule1_frequency_cap(question_count, expected_guidance, expected_violation):
    llm = _make_llm_result("CONCEPTUAL", "FULL")
    guidance, is_violation, violation_type, _, _ = determine_guidance_level(
        QuestionClassification.CONCEPTUAL, question_count, 0, llm
    )
    assert guidance == expected_guidance
    assert is_violation == expected_violation
    if expected_violation:
        assert violation_type == ViolationType.FREQ_LIMIT_EXCEEDED


# ---------------------------------------------------------------------------
# Rule 2 — session already escalated (violation_count >= 3)
# ---------------------------------------------------------------------------

def test_rule2_escalated_blocks_without_new_violation():
    llm = _make_llm_result("CONCEPTUAL", "FULL")
    guidance, is_violation, violation_type, severity, _ = determine_guidance_level(
        QuestionClassification.CONCEPTUAL, 5, 3, llm
    )
    assert guidance == GuidanceLevel.REJECTED
    assert is_violation is False
    assert violation_type is None
    assert severity is None


def test_rule2_not_active_at_violation_count_2():
    # violation_count=2 means rule 2 does not fire — rule 3 fires for DIRECT_SOLUTION
    llm = _make_llm_result("DIRECT_SOLUTION", "REJECTED")
    guidance, is_violation, violation_type, _, _ = determine_guidance_level(
        QuestionClassification.DIRECT_SOLUTION, 5, 2, llm
    )
    assert guidance == GuidanceLevel.REJECTED
    assert is_violation is True
    assert violation_type == ViolationType.DIRECT_SOLUTION_REQUEST


# ---------------------------------------------------------------------------
# Rule priority conflicts
# ---------------------------------------------------------------------------

def test_rule1_beats_rule2():
    # q>15 AND violation_count>=3 — rule 1 must win (is_violation=True, FREQ_LIMIT_EXCEEDED)
    llm = _make_llm_result("DIRECT_SOLUTION", "REJECTED")
    guidance, is_violation, violation_type, _, _ = determine_guidance_level(
        QuestionClassification.DIRECT_SOLUTION, 16, 3, llm
    )
    assert guidance == GuidanceLevel.REJECTED
    assert is_violation is True
    assert violation_type == ViolationType.FREQ_LIMIT_EXCEEDED


def test_rule2_beats_rule3():
    # violation_count>=3 AND DIRECT_SOLUTION — rule 2 must win (is_violation=False)
    llm = _make_llm_result("DIRECT_SOLUTION", "REJECTED")
    guidance, is_violation, violation_type, _, _ = determine_guidance_level(
        QuestionClassification.DIRECT_SOLUTION, 5, 3, llm
    )
    assert guidance == GuidanceLevel.REJECTED
    assert is_violation is False
    assert violation_type is None


# ---------------------------------------------------------------------------
# Rule 3 — DIRECT_SOLUTION
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("violation_count", [0, 2])
def test_rule3_direct_solution(violation_count):
    llm = _make_llm_result("DIRECT_SOLUTION", "REJECTED", "No direct solutions.")
    guidance, is_violation, violation_type, severity, student_message = determine_guidance_level(
        QuestionClassification.DIRECT_SOLUTION, 5, violation_count, llm
    )
    assert guidance == GuidanceLevel.REJECTED
    assert is_violation is True
    assert violation_type == ViolationType.DIRECT_SOLUTION_REQUEST
    assert severity == ViolationSeverity.MAJOR
    assert student_message == "No direct solutions."


# ---------------------------------------------------------------------------
# Rule 4 — ANSWER_FARMING
# ---------------------------------------------------------------------------

def test_rule4_answer_farming():
    llm = _make_llm_result("ANSWER_FARMING", "MINIMAL", "Confirm your approach only.")
    guidance, is_violation, violation_type, severity, student_message = determine_guidance_level(
        QuestionClassification.ANSWER_FARMING, 5, 0, llm
    )
    assert guidance == GuidanceLevel.MINIMAL
    assert is_violation is True
    assert violation_type == ViolationType.ANSWER_FARMING
    assert severity == ViolationSeverity.MINOR
    assert student_message == "Confirm your approach only."


# ---------------------------------------------------------------------------
# Rule 5 — question count ceiling (question_count >= 13)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question_count,llm_guidance,expected_guidance", [
    (12, "FULL",     "FULL"),      # rule 5 not active
    (13, "FULL",     "MODERATE"),  # FULL capped to MODERATE
    (14, "FULL",     "MODERATE"),  # still capped
    (15, "FULL",     "MODERATE"),  # 15 is not > 15, so rule 1 doesn't fire; rule 5 caps
    (13, "MODERATE", "MODERATE"),  # MODERATE passes through unchanged
    (13, "MINIMAL",  "MINIMAL"),   # MINIMAL passes through unchanged
])
def test_rule5_question_ceiling(question_count, llm_guidance, expected_guidance):
    llm = _make_llm_result("CONCEPTUAL", llm_guidance)
    guidance, is_violation, violation_type, _, _ = determine_guidance_level(
        QuestionClassification.CONCEPTUAL, question_count, 0, llm
    )
    assert guidance == GuidanceLevel(expected_guidance)
    assert is_violation is False
    assert violation_type is None


def test_rule5_warning_message_no_llm_message():
    # When LLM student_facing_message is None, returned message is the warning string alone
    llm = _make_llm_result("CONCEPTUAL", "FULL", message=None)
    _, _, _, _, student_message = determine_guidance_level(
        QuestionClassification.CONCEPTUAL, 13, 0, llm
    )
    assert student_message is not None
    assert "15" in student_message  # warning references the hard limit


def test_rule5_warning_message_prepended_to_llm_message():
    # When LLM provides a message, warning is prepended
    llm = _make_llm_result("CONCEPTUAL", "FULL", message="extra guidance here")
    _, _, _, _, student_message = determine_guidance_level(
        QuestionClassification.CONCEPTUAL, 13, 0, llm
    )
    assert student_message is not None
    assert "15" in student_message
    assert "extra guidance here" in student_message


# ---------------------------------------------------------------------------
# Rule 4 beats Rule 5 — ANSWER_FARMING at q>=13 hits rule 4, not rule 5
# ---------------------------------------------------------------------------

def test_rule4_beats_rule5_at_high_question_count():
    # At q=13 rule 5 would cap FULL→MODERATE, but ANSWER_FARMING hits rule 4 first
    llm = _make_llm_result("ANSWER_FARMING", "MINIMAL")
    guidance, is_violation, violation_type, severity, _ = determine_guidance_level(
        QuestionClassification.ANSWER_FARMING, 13, 0, llm
    )
    assert guidance == GuidanceLevel.MINIMAL
    assert is_violation is True
    assert violation_type == ViolationType.ANSWER_FARMING
    assert severity == ViolationSeverity.MINOR


# ---------------------------------------------------------------------------
# Rule 6 — default: trust LLM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("classification,llm_guidance", [
    ("CONCEPTUAL",   "FULL"),
    ("PROCEDURAL",   "MODERATE"),
    ("CLARIFICATION","FULL"),
])
def test_rule6_default_trusts_llm(classification, llm_guidance):
    llm = _make_llm_result(classification, llm_guidance)
    guidance, is_violation, violation_type, severity, _ = determine_guidance_level(
        QuestionClassification(classification), 5, 0, llm
    )
    assert guidance == GuidanceLevel(llm_guidance)
    assert is_violation is False
    assert violation_type is None
    assert severity is None
