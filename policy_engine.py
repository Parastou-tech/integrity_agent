"""
Policy Engine — LLM-based question classification and guidance matrix.

Two public entry points:
  1. classify_question()  — calls Azure OpenAI to classify a student question
  2. determine_guidance_level() — applies the rule-based matrix to produce
     the final GuidanceLevel and violation metadata
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from openai import AsyncAzureOpenAI

from models import GuidanceLevel, QuestionClassification, ViolationSeverity, ViolationType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Student-facing messages (returned to Lab Companion for display)
# ---------------------------------------------------------------------------

_MSG_REJECTED_SOLUTION = (
    "I'm not able to provide direct solutions to lab problems. "
    "This wouldn't support your learning or comply with Cal Poly's academic integrity policy. "
    "I encourage you to review your notes, the lab handout, or visit office hours."
)

_MSG_REJECTED_FREQ = (
    "You have reached the 15-question limit for this session. "
    "Please visit office hours or contact your instructor for further help."
)

_MSG_REJECTED_ESCALATED = (
    "Your session has been flagged for academic integrity review. "
    "Please contact your instructor."
)

_MSG_MINIMAL = (
    "I can only confirm whether you're on the right track at this point. "
    "Please share your current approach and I'll tell you if your direction is correct."
)

_MSG_MODERATE_WARNING = (
    "You're approaching the session question limit ({count}/15). "
    "I can still help guide your thinking, but please make your remaining questions count."
)

# ---------------------------------------------------------------------------
# Classification system prompt
# ---------------------------------------------------------------------------

CLASSIFICATION_SYSTEM_PROMPT = """You are an academic integrity classifier for Cal Poly's STEM lab courses.
Your job is to determine whether a student's question is appropriate for a tutoring assistant to answer,
or whether it crosses into requesting direct solutions that would violate Cal Poly's academic integrity policy.

## Cal Poly Academic Integrity Context
Cal Poly's Honor Code requires that all submitted work be the student's own. In lab courses,
students must demonstrate genuine understanding by deriving their own solutions. Receiving
complete solutions, step-by-step worked answers, or code that directly solves an assignment
problem — even from an AI — constitutes a violation of the Honor Code.

## Classification Categories

Classify the question into exactly ONE of the following:

CONCEPTUAL
  Definition: Student seeks to understand a principle, theory, formula, or concept.
  Examples: "Can you explain what impedance matching means?",
            "Why does a bypass capacitor reduce noise?",
            "What is the difference between Thevenin and Norton equivalent circuits?"
  Guidance: FULL — Answer completely and pedagogically.

PROCEDURAL
  Definition: Student asks how to approach or set up a problem type, without asking for
  their specific numerical answer.
  Examples: "What steps do I take to analyze a BJT amplifier in small-signal mode?",
            "How do I generally set up a KVL equation for a mesh?",
            "What MATLAB function would I use to plot a Bode plot?"
  Guidance: MODERATE — Explain the general approach/method. Do not solve their specific problem.

CLARIFICATION
  Definition: Student asks about lab instructions, grading criteria, terminology in the
  handout, or logistical aspects of the assignment.
  Examples: "Does 'DC operating point' mean I should treat capacitors as open circuits?",
            "When the lab says 'verify', do I need a mathematical derivation or just simulation?",
            "Is R1 in Figure 2 connected to the non-inverting input?"
  Guidance: FULL — Clarify the instructions directly.

DIRECT_SOLUTION
  Definition: Student requests the specific numerical answer, complete derivation,
  finished code, or full circuit solution for their specific lab assignment, even if
  phrased indirectly.
  Key indicators: specific component values from their lab; phrases like "give me",
  "what is the answer", "solve this for me", "complete this", "write this function",
  "show me the full solution", "just tell me the value"; quoting a specific problem
  or figure and asking for the result.
  Guidance: REJECTED — Do not provide any part of the solution.

ANSWER_FARMING
  Definition: A sequence of incrementally specific questions that together would
  constitute solving the lab step-by-step without genuine student effort. A single
  question may appear benign, but in the context of the conversation history it
  reveals a pattern of extracting a complete solution piece by piece.
  Guidance: MINIMAL — Only confirm whether the student's own stated approach is
  on the right track. Do not introduce new information that advances their solution.

## Decision Rules

1. Read the current question carefully.
2. Review the conversation history for patterns.
3. If the question contains specific numerical values, component names, or an exact
   problem statement from the lab COMBINED with a request to solve/compute/find:
   classify as DIRECT_SOLUTION regardless of phrasing.
4. If the conversation history shows 3 or more questions that together are solving
   one lab problem step-by-step: classify the current question as ANSWER_FARMING
   even if in isolation it looks PROCEDURAL.
5. When in doubt between two categories, choose the more restrictive one.

## Output Format

Respond with ONLY valid JSON. No markdown, no explanation outside the JSON.

{
  "classification": "CONCEPTUAL|PROCEDURAL|CLARIFICATION|DIRECT_SOLUTION|ANSWER_FARMING",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<1-2 sentence explanation of the classification>",
  "recommended_guidance": "FULL|MODERATE|MINIMAL|REJECTED",
  "student_facing_message": <null or "message to show student when guidance is not FULL">
}
"""


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    classification: QuestionClassification
    confidence: float
    reasoning: str
    recommended_guidance: GuidanceLevel
    student_facing_message: Optional[str]


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------

async def classify_question(
    question_text: str,
    conversation_history: list[dict],
    session_context: dict,
    openai_client: AsyncAzureOpenAI,
    deployment_name: str,
) -> ClassificationResult:
    """
    Call Azure OpenAI to classify a student question.

    session_context keys: violation_count, question_count, lab_id
    conversation_history: [{"role": "user"|"assistant", "content": str}, ...]
    """
    history_text = ""
    if conversation_history:
        lines = []
        for turn in conversation_history[-6:]:  # last 3 exchanges
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)
    else:
        history_text = "(no prior conversation)"

    user_content = (
        f"Lab ID: {session_context.get('lab_id', 'unknown')}\n"
        f"Questions asked this session: {session_context.get('question_count', 0)}\n"
        f"Violations this session: {session_context.get('violation_count', 0)}\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Student's current question:\n{question_text}"
    )

    response = await openai_client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=400,
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    return ClassificationResult(
        classification=QuestionClassification(parsed["classification"]),
        confidence=float(parsed.get("confidence", 1.0)),
        reasoning=parsed.get("reasoning", ""),
        recommended_guidance=GuidanceLevel(parsed["recommended_guidance"]),
        student_facing_message=parsed.get("student_facing_message"),
    )


# ---------------------------------------------------------------------------
# Guidance matrix
# ---------------------------------------------------------------------------

def determine_guidance_level(
    classification: QuestionClassification,
    question_count: int,
    violation_count: int,
    llm_result: ClassificationResult,
) -> tuple[GuidanceLevel, bool, Optional[ViolationType], Optional[ViolationSeverity], Optional[str]]:
    """
    Apply the rule-based matrix on top of the LLM classification.

    Returns:
        (guidance_level, is_violation, violation_type, severity, student_message)

    Priority order — first matching rule wins:
      1. Frequency limit hard-block  (question_count > 15)
      2. Session already escalated   (violation_count >= 3)
      3. LLM: DIRECT_SOLUTION        → REJECTED, MAJOR
      4. LLM: ANSWER_FARMING         → MINIMAL,  MINOR
      5. Frequency warning ceiling   (question_count >= 13 → cap at MODERATE)
      6. Default: trust LLM
    """

    # Rule 1: hard frequency cap
    if question_count > 15:
        return (
            GuidanceLevel.REJECTED,
            True,
            ViolationType.FREQ_LIMIT_EXCEEDED,
            ViolationSeverity.MINOR,
            _MSG_REJECTED_FREQ,
        )

    # Rule 2: session already escalated — block but don't add another violation
    if violation_count >= 3:
        return (
            GuidanceLevel.REJECTED,
            False,
            None,
            None,
            _MSG_REJECTED_ESCALATED,
        )

    # Rule 3: direct solution request
    if classification == QuestionClassification.DIRECT_SOLUTION:
        msg = llm_result.student_facing_message or _MSG_REJECTED_SOLUTION
        return (
            GuidanceLevel.REJECTED,
            True,
            ViolationType.DIRECT_SOLUTION_REQUEST,
            ViolationSeverity.MAJOR,
            msg,
        )

    # Rule 4: answer farming pattern
    if classification == QuestionClassification.ANSWER_FARMING:
        msg = llm_result.student_facing_message or _MSG_MINIMAL
        return (
            GuidanceLevel.MINIMAL,
            True,
            ViolationType.ANSWER_FARMING,
            ViolationSeverity.MINOR,
            msg,
        )

    # Rule 5: approaching question limit — cap clean questions at MODERATE
    if question_count >= 13:
        ceiling = GuidanceLevel.MODERATE
        llm_level = llm_result.recommended_guidance
        final_level = (
            ceiling
            if llm_level == GuidanceLevel.FULL
            else llm_level
        )
        warning_msg = _MSG_MODERATE_WARNING.format(count=question_count)
        student_msg = (
            warning_msg
            if llm_result.student_facing_message is None
            else f"{warning_msg} {llm_result.student_facing_message}"
        )
        return (final_level, False, None, None, student_msg)

    # Rule 6: default — trust LLM
    return (
        llm_result.recommended_guidance,
        False,
        None,
        None,
        llm_result.student_facing_message,
    )
