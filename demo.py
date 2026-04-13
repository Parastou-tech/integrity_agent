"""
Integrity Guardian — Demo Script

Walks through the complete flow against a locally running server:
  1. Start a session
  2. Ask an appropriate conceptual question      → FULL guidance
  3. Ask a procedural question                   → MODERATE guidance
  4. Ask for a direct solution (violation #1)    → REJECTED
  5. Ask for a direct solution again (violation #2) → REJECTED
  6. Ask for a direct solution a third time (violation #3) → REJECTED + ESCALATION
  7. End the session and print the integrity report

Run the server first:
    USE_MEMORY_STORE=true uvicorn app:app --reload

Then in another terminal:
    python demo.py
"""

import asyncio
import json
import uuid
import httpx

BASE_URL = "http://127.0.0.1:8000"
HEADERS = {
    "X-Internal-Token": "demo-token",
    "Content-Type": "application/json",
}

STUDENT_ID = "demo_student"
SESSION_ID = str(uuid.uuid4())
LAB_ID = "lab3"

# ---------------------------------------------------------------------------
# Demo questions
# ---------------------------------------------------------------------------

QUESTIONS = [
    {
        "label": "✅ Conceptual question",
        "text": "Can you explain what a Thevenin equivalent circuit is and why we use it?",
    },
    {
        "label": "🟡 Procedural question",
        "text": "What are the general steps to find the small-signal model of a MOSFET amplifier?",
    },
    {
        "label": "🔴 Direct solution request #1 (VIOLATION)",
        "text": (
            "For Figure 3 in the lab, R1=10kΩ, R2=22kΩ, VDD=5V. "
            "Can you calculate the exact DC bias voltage at the drain for me?"
        ),
    },
    {
        "label": "🔴 Direct solution request #2 (VIOLATION)",
        "text": (
            "Just give me the complete transfer function for the circuit in Task 2. "
            "I need the answer to check my work."
        ),
    },
    {
        "label": "🔴 Direct solution request #3 (VIOLATION → ESCALATION)",
        "text": (
            "I'm running out of time. Can you just write the full MATLAB code "
            "that solves Part B of the lab for me?"
        ),
    },
]

# ---------------------------------------------------------------------------

def divider(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print(f"\n{'─' * width}")


def print_validation(q_label: str, response: dict) -> None:
    approved = response["approved"]
    level = response["guidance_level"]
    classification = response["classification"]
    violations = response["violation_count"]
    questions = response["question_count"]
    escalated = response["session_escalated"]

    status = "APPROVED" if approved else "REJECTED"
    icon = "✅" if approved else "❌"

    print(f"\n  {icon} Status:         {status}")
    print(f"     Classification: {classification}")
    print(f"     Guidance level: {level}")
    print(f"     Questions used: {questions}/15")
    print(f"     Violations:     {violations}/3")
    if escalated:
        print("     🚨 SESSION ESCALATED — instructor notified")
    if response.get("student_message"):
        print(f"\n  Message to student:\n    \"{response['student_message']}\"")


def print_report(report: dict) -> None:
    summary = report.get("summary", {})
    print(f"\n  Final status:    {summary.get('final_status')}")
    print(f"  Total questions: {summary.get('total_questions')}")
    print(f"  Violations:      {summary.get('violation_count')}")
    print(f"  Escalated:       {summary.get('escalated')}")
    print(f"\n  Guidance distribution:")
    for level, count in summary.get("guidance_distribution", {}).items():
        if count > 0:
            print(f"    {level:10s}: {count}")

    violations = report.get("violations_detail", [])
    if violations:
        print(f"\n  Violation log ({len(violations)} entries):")
        for v in violations:
            print(f"    [{v['sequence_number']}] {v['violation_type']} ({v['severity']})")
            print(f"         \"{v['question_text'][:70]}...\"")

    escalation = report.get("escalation_log", {})
    if escalation.get("escalated"):
        print(f"\n  🚨 Escalation reason: {escalation.get('reason')}")


async def run_demo() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS, timeout=30) as client:

        # Health check
        divider("HEALTH CHECK")
        r = await client.get("/health")
        print(f"  {r.json()}")

        # Start session
        divider("START SESSION")
        r = await client.post("/session/start", json={
            "student_id": STUDENT_ID,
            "session_id": SESSION_ID,
            "lab_id": LAB_ID,
            "course_id": "CSC580",
        })
        r.raise_for_status()
        print(f"  Session {SESSION_ID[:8]}... created for student '{STUDENT_ID}'")

        # Walk through each question
        for q in QUESTIONS:
            divider(q["label"])
            print(f"  Student asks: \"{q['text'][:80]}...\"" if len(q["text"]) > 80
                  else f"  Student asks: \"{q['text']}\"")

            r = await client.post("/validate", json={
                "student_id": STUDENT_ID,
                "session_id": SESSION_ID,
                "lab_id": LAB_ID,
                "course_id": "CSC580",
                "question_text": q["text"],
                "conversation_history": [],
            })
            r.raise_for_status()
            print_validation(q["label"], r.json())

        # End session and get report
        divider("END SESSION + INTEGRITY REPORT")
        r = await client.post("/session/end", json={
            "student_id": STUDENT_ID,
            "session_id": SESSION_ID,
        })
        r.raise_for_status()
        end_data = r.json()
        report_id = end_data["report_id"]
        print(f"  Report ID: {report_id}")

        # Fetch and display the full report
        r = await client.get(f"/report/{report_id}?student_id={STUDENT_ID}")
        r.raise_for_status()
        print_report(r.json())

        divider()
        print("\n  Demo complete. This report would be sent to the Instructor Co-pilot.\n")


if __name__ == "__main__":
    asyncio.run(run_demo())
