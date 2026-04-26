"""
AIEIC Integrity Guardian — FastAPI Service

Validates student questions for academic integrity compliance before the Lab
Companion responds. Tracks violations per session, escalates to the Instructor
Co-pilot when thresholds are exceeded, and generates end-of-session and
post-lab integrity reports.
"""

import json as _json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import AsyncAzureOpenAI, RateLimitError
from pydantic_settings import BaseSettings
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from cosmos_client import CosmosIntegrityClient
from cosmos_client_memory import MemoryIntegrityClient
from models import (
    EndSessionRequest,
    EndSessionResponse,
    GenerateReportRequest,
    GenerateReportResponse,
    PatchReportRequest,
    GuidanceLevel,
    PostLabCheckRequest,
    PostLabCheckResponse,
    QuestionRecord,
    SessionDocument,
    SessionStatus,
    StartSessionRequest,
    StartSessionResponse,
    ValidateQuestionRequest,
    ValidateQuestionResponse,
    ViolationRecord,
    ViolationSeverity,
)
from policy_engine import ClassificationResult, classify_question, determine_guidance_level
from report_generator import generate_post_lab_report, generate_session_report

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    AZURE_OPENAI_API_VERSION: str = "2024-12-01-preview"
    # Cosmos DB — not required when USE_MEMORY_STORE=true
    COSMOS_URL: str = ""
    COSMOS_KEY: str = ""
    COSMOS_DATABASE: str = "integrity_guardian"
    INTERNAL_API_TOKEN: str = "demo-token"
    USE_MEMORY_STORE: bool = False
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

def _key_by_student_id(request: Request) -> str:
    body = getattr(request.state, "_body", None)
    if body:
        try:
            return _json.loads(body).get("student_id", "unknown")
        except Exception:
            pass
    return "unknown"

limiter = Limiter(key_func=_key_by_student_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.USE_MEMORY_STORE:
        cosmos = MemoryIntegrityClient()
    else:
        cosmos = CosmosIntegrityClient(
            url=settings.COSMOS_URL,
            key=settings.COSMOS_KEY,
            database=settings.COSMOS_DATABASE,
        )
    await cosmos.initialize()

    openai_client = AsyncAzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )

    app.state.cosmos = cosmos
    app.state.openai_client = openai_client

    logger.info("Integrity Guardian started.")
    yield

    await cosmos.close()
    logger.info("Integrity Guardian shut down.")


app = FastAPI(
    title="AIEIC Integrity Guardian",
    version="1.0.0",
    description="Policy and integrity middleware for Cal Poly STEM lab AI tutoring.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def _cache_request_body(request: Request, call_next):
    body = await request.body()
    request.state._body = body
    return await call_next(request)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please slow down and try again shortly."},
    )


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

async def verify_internal_token(
    x_internal_token: str = Header(..., alias="X-Internal-Token"),
) -> None:
    if x_internal_token != settings.INTERNAL_API_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden.")


def get_cosmos(request: Request) -> CosmosIntegrityClient:
    return request.app.state.cosmos


def get_openai(request: Request) -> AsyncAzureOpenAI:
    return request.app.state.openai_client


# ---------------------------------------------------------------------------
# Health endpoint (no auth)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
async def health_check() -> dict:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/session/start",
    response_model=StartSessionResponse,
    tags=["session"],
    dependencies=[Depends(verify_internal_token)],
)
async def start_session(
    body: StartSessionRequest,
    cosmos: CosmosIntegrityClient = Depends(get_cosmos),
) -> StartSessionResponse:
    now = datetime.utcnow().isoformat() + "Z"
    doc = SessionDocument(
        id=body.session_id,
        student_id=body.student_id,
        session_id=body.session_id,
        lab_id=body.lab_id,
        course_id=body.course_id,
        started_at=now,
    ).model_dump()

    try:
        await cosmos.create_session(doc)
    except Exception as e:
        # Cosmos raises CosmosResourceExistsError (HTTP 409) on duplicate id
        if "409" in str(e) or "Conflict" in str(e):
            raise HTTPException(status_code=409, detail="Session already exists.")
        logger.error("Failed to create session: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail="Persistence layer unavailable.")

    logger.info("Session started: student=%s session=%s", body.student_id, body.session_id)
    return StartSessionResponse(
        session_id=body.session_id,
        started_at=datetime.utcnow(),
    )


@app.post(
    "/session/end",
    response_model=EndSessionResponse,
    tags=["session"],
    dependencies=[Depends(verify_internal_token)],
)
async def end_session(
    body: EndSessionRequest,
    cosmos: CosmosIntegrityClient = Depends(get_cosmos),
) -> EndSessionResponse:
    session = await cosmos.get_session(body.session_id, body.student_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    now = datetime.utcnow().isoformat() + "Z"
    session["status"] = SessionStatus.CLOSED.value
    session["ended_at"] = now
    await cosmos.upsert_session(session)

    report = await generate_session_report(session, cosmos)

    session["report_generated"] = True
    session["report_id"] = report["report_id"]
    await cosmos.upsert_session(session)

    logger.info(
        "Session ended: student=%s session=%s report=%s",
        body.student_id, body.session_id, report["report_id"],
    )
    return EndSessionResponse(
        session_id=body.session_id,
        report_id=report["report_id"],
        ended_at=datetime.utcnow(),
        summary=report["summary"],
    )


@app.get(
    "/session/{session_id}",
    tags=["session"],
    dependencies=[Depends(verify_internal_token)],
)
async def get_session(
    session_id: str,
    student_id: str,
    cosmos: CosmosIntegrityClient = Depends(get_cosmos),
) -> dict:
    session = await cosmos.get_session(session_id, student_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


# ---------------------------------------------------------------------------
# Validate endpoint (hot path)
# ---------------------------------------------------------------------------

@app.post(
    "/validate",
    response_model=ValidateQuestionResponse,
    tags=["validation"],
    dependencies=[Depends(verify_internal_token)],
)
@limiter.limit("60/minute")
async def validate_question(
    request: Request,
    body: ValidateQuestionRequest,
    cosmos: CosmosIntegrityClient = Depends(get_cosmos),
    openai_client: AsyncAzureOpenAI = Depends(get_openai),
) -> ValidateQuestionResponse:
    # 1. Retrieve session
    session = await cosmos.get_session(body.session_id, body.student_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.get("status") == SessionStatus.CLOSED.value:
        raise HTTPException(status_code=400, detail="Session is already closed.")

    # 2. Increment question counter
    session["question_count"] = session.get("question_count", 0) + 1
    question_count = session["question_count"]
    violation_count = session.get("violation_count", 0)

    # 3. Classify (with fail-safe fallback)
    llm_result: Optional[ClassificationResult] = None
    try:
        llm_result = await classify_question(
            question_text=body.question_text,
            conversation_history=body.conversation_history,
            session_context={
                "lab_id": body.lab_id,
                "question_count": question_count,
                "violation_count": violation_count,
            },
            openai_client=openai_client,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        )
    except RateLimitError:
        logger.warning("OpenAI rate limit hit during classification.")
        raise HTTPException(status_code=429, detail="Classification service rate limited.")
    except Exception as e:
        # Fail-safe: allow through at MODERATE to avoid blocking students on outages
        logger.error("Classifier error (fail-safe applied): %s", e, exc_info=True)
        from models import QuestionClassification
        llm_result = ClassificationResult(
            classification=QuestionClassification.PROCEDURAL,
            confidence=0.0,
            reasoning="Classifier unavailable — defaulting to MODERATE.",
            recommended_guidance=GuidanceLevel.MODERATE,
            student_facing_message="[Classifier temporarily unavailable — limited guidance in effect]",
        )

    # 4. Apply guidance matrix
    from models import QuestionClassification
    guidance_level, is_violation, violation_type, severity, student_message = (
        determine_guidance_level(
            classification=llm_result.classification,
            question_count=question_count,
            violation_count=violation_count,
            llm_result=llm_result,
        )
    )

    # 5. Set warning flag at question 13
    if question_count == 13 and not session.get("warning_issued", False):
        session["warning_issued"] = True

    # 6. Log violation if detected
    question_id = str(uuid.uuid4())
    if is_violation and violation_type is not None:
        violation_count += 1
        session["violation_count"] = violation_count

        v_record = ViolationRecord(
            question_id=question_id,
            sequence_number=question_count,
            violation_type=violation_type,
            severity=severity or ViolationSeverity.MINOR,
            question_text=body.question_text,
            guidance_level=guidance_level,
            student_message_sent=student_message or "",
        )
        session.setdefault("violations", []).append(v_record.model_dump())

        # 7. Escalate on 3rd violation
        if violation_count >= 3 and not session.get("escalated", False):
            session["escalated"] = True
            logger.critical(
                "INTEGRITY ESCALATION: student=%s session=%s lab=%s "
                "violation_count=%d last_violation_type=%s",
                body.student_id,
                body.session_id,
                body.lab_id,
                violation_count,
                violation_type.value,
            )

    # 8. Append question record
    q_record = QuestionRecord(
        question_id=question_id,
        sequence_number=question_count,
        text=body.question_text,
        classification=llm_result.classification,
        guidance_level=guidance_level,
        violation=is_violation,
        violation_type=violation_type,
        student_message_sent=student_message,
    )
    session.setdefault("questions", []).append(q_record.model_dump())

    # 9. Persist session
    await cosmos.upsert_session(session)

    approved = guidance_level not in (GuidanceLevel.REJECTED,)

    logger.info(
        "validate: student=%s q=%d guidance=%s violation=%s",
        body.student_id,
        question_count,
        guidance_level.value,
        violation_type.value if violation_type else "none",
    )

    return ValidateQuestionResponse(
        approved=approved,
        guidance_level=guidance_level,
        student_message=student_message,
        violation_detected=is_violation,
        violation_type=violation_type,
        violation_count=violation_count,
        question_count=question_count,
        session_escalated=session.get("escalated", False),
        classification=llm_result.classification,
    )


# ---------------------------------------------------------------------------
# Report endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/report/generate",
    response_model=GenerateReportResponse,
    tags=["reports"],
    dependencies=[Depends(verify_internal_token)],
)
async def generate_report(
    body: GenerateReportRequest,
    cosmos: CosmosIntegrityClient = Depends(get_cosmos),
) -> GenerateReportResponse:
    session = await cosmos.get_session(body.session_id, body.student_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    report = await generate_session_report(session, cosmos)

    session["report_generated"] = True
    session["report_id"] = report["report_id"]
    await cosmos.upsert_session(session)

    return GenerateReportResponse(report_id=report["report_id"], report=report)


@app.post(
    "/report/post-lab",
    response_model=PostLabCheckResponse,
    tags=["reports"],
    dependencies=[Depends(verify_internal_token)],
)
async def post_lab_check(
    body: PostLabCheckRequest,
    cosmos: CosmosIntegrityClient = Depends(get_cosmos),
) -> PostLabCheckResponse:
    session_docs = []
    for sid in body.session_ids:
        doc = await cosmos.get_session(sid, body.student_id)
        if doc is not None:
            session_docs.append(doc)

    if not session_docs:
        raise HTTPException(
            status_code=404, detail="No sessions found for the provided IDs."
        )

    report = await generate_post_lab_report(
        student_id=body.student_id,
        session_docs=session_docs,
        lab_id=body.lab_id,
        cosmos=cosmos,
    )

    return PostLabCheckResponse(
        report_id=report["report_id"],
        over_reliance_indicators=report["over_reliance_indicators"],
        summary=report["summary_text"],
    )


@app.get(
    "/report/{report_id}",
    tags=["reports"],
    dependencies=[Depends(verify_internal_token)],
)
async def get_report(
    report_id: str,
    student_id: str,
    cosmos: CosmosIntegrityClient = Depends(get_cosmos),
) -> dict:
    report = await cosmos.get_report(report_id, student_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.patch(
    "/report/{report_id}",
    tags=["reports"],
    dependencies=[Depends(verify_internal_token)],
)
async def patch_report(
    report_id: str,
    body: PatchReportRequest,
    cosmos: CosmosIntegrityClient = Depends(get_cosmos),
) -> dict:
    report = await cosmos.get_report(report_id, body.student_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    report["instructor_notes"] = body.instructor_notes
    return await cosmos.upsert_report(report)
