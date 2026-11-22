"""FastAPI interface for the biomedical RAG and research dashboard services."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv
from app.ragguard_repair import (
    RepairCapabilities,
    RepairRequest,
    RepairResult,
    SUPPORTED_REPAIRS,
    audit_repair,
    authorize_repair,
)
from app.ragguard_observation import get_server_setting

from dashboard.audit_utils import configuration_version, summarize_answer_quality_reviews
from dashboard.rag_audit import audit_answer, summarize_audits, retrieval_summary
from dashboard.research_metrics import calculate_retrieval_metrics
from dashboard.responsible_ai import (
    build_counterfactual_prompts,
    collect_risk_signals,
    compare_responses,
    ethical_risk_score,
    fairness_report,
    operator_ethical_risk_score,
)
from dashboard.security_controls import (
    find_pii,
    guard_question,
    redact_pii,
    scan_sources,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SUPPORTED_PROVIDERS = {"cohere", "openai", "huggingface"}
DEFAULT_MODELS = {
    "cohere": "command-a-plus-05-2026",
    "openai": "gpt-4o",
    "huggingface": "google/flan-t5-large",
}


class SessionResponse(BaseModel):
    session_id: str
    created_at: str


class ConfigureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "cohere"
    model: str | None = Field(default=None, max_length=200)


class QueryRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=15)
    retrieval_mode: str = "hybrid"


class RetrieveRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=15)
    retrieval_mode: str = "hybrid"


class RetrievalEvaluationRequest(BaseModel):
    retrieved_sources: list[str]
    relevant_sources: list[str]
    k: int = Field(default=5, ge=1, le=100)


class AnswerEvaluationRequest(BaseModel):
    answer: str
    evidence_texts: list[str]


class FairnessRequest(BaseModel):
    session_id: str
    template: str = Field(min_length=1, max_length=1000)
    groups: list[str] = Field(min_length=2, max_length=4)
    top_k: int = Field(default=5, ge=1, le=15)


class OperatorRiskRequest(BaseModel):
    impact: int = Field(ge=1, le=5)
    likelihood: int = Field(ge=1, le=5)
    title: str = Field(min_length=1, max_length=200)
    rationale: str = Field(default="", max_length=2000)


class ReviewRequest(BaseModel):
    query_event_id: str
    ratings: dict[str, int]
    notes: str = Field(default="", max_length=2000)


class FindingRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    severity: str = Field(pattern="^(low|moderate|high|critical)$")
    status: str = Field(default="open", pattern="^(open|mitigated|accepted)$")
    rationale: str = Field(default="", max_length=2000)


@dataclass
class DashboardSession:
    created_at: str
    owner_id: str = ""
    provider: str = "cohere"
    model: str = DEFAULT_MODELS["cohere"]
    retrieval_mode: str = "hybrid"
    top_k: int = 5
    service: Any = None
    documents: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    evaluation_runs: list[dict] = field(default_factory=list)
    answer_quality_reviews: list[dict] = field(default_factory=list)
    fairness_reports: list[dict] = field(default_factory=list)
    risk_register: list[dict] = field(default_factory=list)


_sessions: dict[str, DashboardSession] = {}
_sessions_lock = threading.RLock()
_STATE_DB = Path(os.getenv(
    "BIOMEDRAG_STATE_DB",
    str(PROJECT_ROOT / "data" / "biomedrag_state.sqlite3"),
))


def _init_state_store() -> None:
    _STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_STATE_DB) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS dashboard_sessions "
            "(session_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        rows = connection.execute("SELECT session_id, payload FROM dashboard_sessions").fetchall()
    with _sessions_lock:
        for session_id, payload in rows:
            try:
                state = json.loads(payload)
                _sessions[session_id] = DashboardSession(**state)
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.exception("Could not restore dashboard session %s", session_id)


def _persist_sessions() -> None:
    with _sessions_lock:
        snapshot = list(_sessions.items())
    with sqlite3.connect(_STATE_DB, timeout=10) as connection:
        connection.executemany(
            "INSERT INTO dashboard_sessions(session_id, payload) VALUES(?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET payload=excluded.payload",
            [(key, json.dumps({
                "created_at": session.created_at,
                "owner_id": session.owner_id,
                "provider": session.provider,
                "model": session.model,
                "retrieval_mode": session.retrieval_mode,
                "top_k": session.top_k,
                "documents": session.documents,
                "events": session.events,
                "evaluation_runs": session.evaluation_runs,
                "answer_quality_reviews": session.answer_quality_reviews,
                "fairness_reports": session.fairness_reports,
                "risk_register": session.risk_register,
            }, ensure_ascii=False)) for key, session in snapshot],
        )


_init_state_store()


def _new_service(provider: str, model: str, api_key: str):
    from app.services import RAGService

    return RAGService(provider=provider, model=model, api_key=api_key)


def _session(session_id: str, owner_id: str) -> DashboardSession:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        try:
            with sqlite3.connect(_STATE_DB, timeout=10) as connection:
                row = connection.execute(
                    "SELECT payload FROM dashboard_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            if row:
                session = DashboardSession(**json.loads(row[0]))
                with _sessions_lock:
                    _sessions[session_id] = session
        except (sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
            logger.exception("Could not load dashboard session %s", session_id)
    if session is None or session.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Dashboard session not found.")
    return session


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_payload(value: Any) -> Any:
    if isinstance(value, str):
        return redact_pii(value)[0]
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_payload(item) for key, item in value.items()}
    return value


def _configuration(session: DashboardSession) -> dict:
    settings = {
        "provider": session.provider,
        "model": session.model,
        "retrieval_mode": session.retrieval_mode,
        "top_k": session.top_k,
        "documents": [
            {key: record[key] for key in ("source", "size_bytes", "sha256")}
            for record in session.documents
        ],
    }
    return {"version": configuration_version(settings), "settings": settings}


def _require_service(session: DashboardSession, *, generation: bool = False):
    if session.service is None:
        message = (
            "Configure an LLM before querying."
            if generation
            else "Configure an LLM to initialize the knowledge index."
        )
        raise HTTPException(status_code=409, detail=message)
    return session.service


def _source_attack_indices(service: Any, question: str, top_k: int, mode: str) -> list[int]:
    evidence = service.retrieve(question, top_k=top_k, mode=mode)
    return scan_sources([
        str(item.get("text", item.get("content", ""))) for item in evidence
    ])


app = FastAPI(
    title="Biomedical RAG API",
    description="API for grounded biomedical Q&A, evaluation, audit, and governance.",
    version="1.0.0",
)
origins = [
    origin.strip()
    for origin in os.getenv(
        "BIOMEDRAG_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

_rate_limit_lock = threading.Lock()
_rate_limit_windows: dict[tuple[str, str], deque[float]] = {}
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_DEFAULT_RATE_LIMIT = 60
_DEFAULT_QUERY_RATE_LIMIT = 10
_DEFAULT_AUTH_FAILURE_RATE_LIMIT = 20
_DEFAULT_INTERNAL_RATE_LIMIT = 60


def _configured_api_tokens() -> dict[str, tuple[str, ...]]:
    raw_tokens = os.getenv("BIOMEDRAG_API_TOKENS", "")
    if not raw_tokens:
        raise ValueError("BIOMEDRAG_API_TOKENS is not configured.")
    try:
        token_map = json.loads(raw_tokens)
    except json.JSONDecodeError as error:
        raise ValueError("BIOMEDRAG_API_TOKENS must be a JSON object.") from error
    if not isinstance(token_map, dict) or not token_map:
        raise ValueError("BIOMEDRAG_API_TOKENS must be a non-empty JSON object.")

    tokens: dict[str, tuple[str, ...]] = {}
    seen_tokens: set[str] = set()
    for principal_id, configured in token_map.items():
        if (
            not isinstance(principal_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.@-]{1,128}", principal_id) is None
        ):
            raise ValueError(
                "Each API principal ID must contain 1-128 letters, digits, underscores, periods, @, or hyphens."
            )
        principal_tokens = [configured] if isinstance(configured, str) else configured
        if (
            not isinstance(principal_tokens, list)
            or not principal_tokens
            or any(
                not isinstance(token, str)
                or len(token) < 32
                or not token.isascii()
                for token in principal_tokens
            )
        ):
            raise ValueError(
                "Each principal must have a token or a non-empty list of tokens, each at least 32 ASCII characters."
            )
        if len(set(principal_tokens)) != len(principal_tokens):
            raise ValueError("A principal cannot repeat an API token.")
        if any(token in seen_tokens for token in principal_tokens):
            raise ValueError("API tokens must be unique across all principals.")
        tokens[principal_id] = tuple(principal_tokens)
        seen_tokens.update(principal_tokens)
    return tokens


def _authenticate_api_request(
    authorization: str,
    tokens: dict[str, tuple[str, ...]],
) -> str | None:
    scheme, separator, supplied_token = authorization.partition(" ")
    if (
        scheme.lower() != "bearer"
        or not separator
        or not supplied_token
        or not supplied_token.isascii()
    ):
        return None

    principal_id = None
    for candidate_id, expected_tokens in tokens.items():
        for expected_token in expected_tokens:
            if hmac.compare_digest(supplied_token, expected_token):
                principal_id = candidate_id
    return principal_id


def _standalone_local_principal(client_host: str, origin: str | None) -> str | None:
    """Allow the local standalone UI without issuing a browser bearer token."""
    if os.getenv("BIOMEDRAG_LOCAL_STANDALONE_ACCESS", "true").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        return None
    try:
        if not ipaddress.ip_address(client_host).is_loopback:
            return None
    except ValueError:
        return None
    if not origin or origin not in origins:
        return None
    return "standalone-local"


def _rate_limits() -> tuple[int, int]:
    values = []
    for name, default in (
        ("BIOMEDRAG_RATE_LIMIT_PER_MINUTE", _DEFAULT_RATE_LIMIT),
        ("BIOMEDRAG_QUERY_RATE_LIMIT_PER_MINUTE", _DEFAULT_QUERY_RATE_LIMIT),
    ):
        try:
            limit = int(os.getenv(name, str(default)))
        except ValueError as error:
            raise ValueError(f"{name} must be a positive integer.") from error
        if limit < 1:
            raise ValueError(f"{name} must be a positive integer.")
        values.append(limit)
    return values[0], values[1]


def _rate_limit_retry_after(
    principal_id: str,
    path: str,
    now: float,
    overall_limit: int,
    query_limit: int,
) -> int | None:
    buckets = [((principal_id, "all"), overall_limit)]
    if path in {"/api/query", "/api/governance/fairness"}:
        buckets.append(((principal_id, "generation"), query_limit))

    with _rate_limit_lock:
        active_windows = []
        for key, limit in buckets:
            window = _rate_limit_windows.setdefault(key, deque())
            while window and now - window[0] >= _RATE_LIMIT_WINDOW_SECONDS:
                window.popleft()
            if len(window) >= limit:
                return max(
                    1,
                    int(_RATE_LIMIT_WINDOW_SECONDS - (now - window[0]) + 0.999),
                )
            active_windows.append(window)
        for window in active_windows:
            window.append(now)
    return None


@app.middleware("http")
async def protect_api(request: Request, call_next):
    path = request.url.path
    client_host = request.client.host if request.client is not None else "unknown"
    if path.startswith("/internal/ragguard/repairs"):
        retry_after = _rate_limit_retry_after(
            f"internal:{client_host}",
            "/internal",
            time.monotonic(),
            _DEFAULT_INTERNAL_RATE_LIMIT,
            _DEFAULT_INTERNAL_RATE_LIMIT,
        )
        if retry_after is not None:
            return JSONResponse(
                status_code=429,
                content={"detail": "API rate limit exceeded. Retry later."},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)
    if (
        request.method == "OPTIONS"
        or not path.startswith("/api/")
        or path == "/api/health"
    ):
        return await call_next(request)

    try:
        overall_limit, query_limit = _rate_limits()
    except ValueError as error:
        logger.error("API security configuration error: %s", error)
        return JSONResponse(
            status_code=503,
            content={"detail": "API authentication or rate limiting is not configured correctly."},
        )

    authorization = request.headers.get("authorization", "")
    principal_id = None
    if not authorization:
        principal_id = _standalone_local_principal(client_host, request.headers.get("origin"))
    if principal_id is None:
        try:
            tokens = _configured_api_tokens()
        except ValueError as error:
            logger.error("API security configuration error: %s", error)
            return JSONResponse(
                status_code=503,
                content={"detail": "API authentication is not configured correctly."},
            )
        principal_id = _authenticate_api_request(authorization, tokens)
    if principal_id is None:
        retry_after = _rate_limit_retry_after(
            f"unauthenticated:{client_host}",
            "/api/auth",
            time.monotonic(),
            _DEFAULT_AUTH_FAILURE_RATE_LIMIT,
            _DEFAULT_AUTH_FAILURE_RATE_LIMIT,
        )
        if retry_after is not None:
            return JSONResponse(
                status_code=429,
                content={"detail": "API authentication rate limit exceeded. Retry later."},
                headers={"Retry-After": str(retry_after)},
            )
        return JSONResponse(
            status_code=401,
            content={"detail": "A valid bearer token is required."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    request.state.principal_id = principal_id
    retry_after = _rate_limit_retry_after(
        principal_id,
        path,
        time.monotonic(),
        overall_limit,
        query_limit,
    )
    if retry_after is not None:
        return JSONResponse(
            status_code=429,
            content={"detail": "API rate limit exceeded. Retry later."},
            headers={"Retry-After": str(retry_after)},
        )
    response = await call_next(request)
    try:
        await run_in_threadpool(_persist_sessions)
    except Exception:
        logger.exception("Could not persist dashboard state")
    return response


@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": app.version}


def _require_ragguard_repair_auth(
    authorization: str | None,
    *,
    request_id: str,
    strategy: str,
) -> None:
    if not get_server_setting("RAGGUARD_REPAIR_TOKEN"):
        audit_repair(
            request_id=request_id,
            strategy=strategy,
            authorized=False,
            outcome="repair_credential_not_configured",
            validated=None,
            context_verified=None,
        )
        raise HTTPException(status_code=503, detail="RAGGuard repair adapter is not configured.")
    if not authorize_repair(authorization):
        audit_repair(
            request_id=request_id,
            strategy=strategy,
            authorized=False,
            outcome="authorization_denied",
            validated=None,
            context_verified=None,
        )
        raise HTTPException(status_code=401, detail="Invalid repair adapter credential.")


@app.get(
    "/internal/ragguard/repairs/capabilities",
    response_model=RepairCapabilities,
)
def ragguard_repair_capabilities(
    authorization: str | None = Header(default=None),
):
    _require_ragguard_repair_auth(
        authorization,
        request_id="capability-discovery",
        strategy="capability_discovery",
    )
    return RepairCapabilities(supported_repairs=list(SUPPORTED_REPAIRS))


@app.post(
    "/internal/ragguard/repairs",
    response_model=RepairResult,
)
def ragguard_execute_repair(
    request: RepairRequest,
    authorization: str | None = Header(default=None),
):
    _require_ragguard_repair_auth(
        authorization,
        request_id=request.request_id,
        strategy=request.strategy,
    )
    expected_application = get_server_setting("RAGGUARD_APPLICATION_ID", "biomedicalrag")
    expected_environment = get_server_setting("RAGGUARD_ENVIRONMENT", "development")
    if (
        request.application_id != expected_application
        or request.environment != expected_environment
    ):
        audit_repair(
            request_id=request.request_id,
            strategy=request.strategy,
            authorized=False,
            outcome="application_identity_mismatch",
            validated=None,
            context_verified=None,
        )
        raise HTTPException(status_code=403, detail="Repair request application identity mismatch.")

    with _sessions_lock:
        sessions = list(_sessions.values())
    for session in sessions:
        service = session.service
        if service is None or not hasattr(service, "execute_ragguard_repair"):
            continue
        with service._ragguard_pending_lock:
            is_pending = request.request_id in service._ragguard_pending_repairs
        if is_pending:
            return service.execute_ragguard_repair(request)

    audit_repair(
        request_id=request.request_id,
        strategy=request.strategy,
        authorized=True,
        outcome="request_context_not_found",
        validated=False,
        context_verified=False,
    )
    raise HTTPException(status_code=409, detail="No active query context exists for this repair request.")


@app.post("/api/sessions", response_model=SessionResponse)
def create_session(request: Request):
    session_id = str(uuid4())
    created_at = _now()
    with _sessions_lock:
        _sessions[session_id] = DashboardSession(
            created_at=created_at,
            owner_id=request.state.principal_id,
        )
    return SessionResponse(session_id=session_id, created_at=created_at)


@app.get("/api/sessions/{session_id}")
def get_session_state(session_id: str, request: Request):
    session = _session(session_id, request.state.principal_id)
    return _session_state(session_id, session)


def _session_state(session_id: str, session: DashboardSession) -> dict[str, Any]:
    events = [event for event in session.events if event.get("event") == "query"]
    successful = [event for event in events if event.get("success")]
    latencies = [
        event["latency_ms"]
        for event in successful
        if isinstance(event.get("latency_ms"), (int, float))
    ]
    latency_mean = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "session_id": session_id,
        "created_at": session.created_at,
        "configuration": _configuration(session),
        "configured": session.service is not None,
        "documents": session.documents,
        "document_count": getattr(session.service, "total_documents", 0),
        "chunk_count": getattr(session.service, "total_chunks", 0),
        "source_names": getattr(session.service, "source_names", []),
        "events": session.events,
        "evaluation_runs": session.evaluation_runs,
        "answer_quality_reviews": session.answer_quality_reviews,
        "fairness_reports": session.fairness_reports,
        "risk_register": session.risk_register,
        "summary": {
            "successful_queries": len(successful),
            "failed_queries": len(events) - len(successful),
            "grounded_queries": sum(event.get("is_grounded") is True for event in successful),
            "mean_latency_ms": latency_mean,
            "answer_audits": summarize_audits(
                [event["audit"] for event in session.events if event.get("audit")]
            ),
            "retrieval_metrics": retrieval_summary(session.evaluation_runs),
            "answer_quality_reviews": summarize_answer_quality_reviews(session.answer_quality_reviews),
        },
    }


@app.post("/api/sessions/{session_id}/configure")
def configure_session(session_id: str, request: ConfigureRequest, http_request: Request):
    session = _session(session_id, http_request.state.principal_id)
    provider = request.provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=422, detail="Unsupported LLM provider.")
    model = (request.model or DEFAULT_MODELS[provider]).strip()
    if not model:
        raise HTTPException(status_code=422, detail="Model name must not be empty.")
    api_key = os.getenv(f"{provider.upper()}_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=f"Set {provider.upper()}_API_KEY on the API server before configuring this provider.",
        )
    try:
        service = _new_service(provider, model, api_key)
    except Exception as error:
        logger.exception("Could not configure RAG service for provider %s", provider)
        raise HTTPException(
            status_code=503,
            detail="The RAG service could not be initialized. Check API logs for details.",
        ) from error
    session.provider = provider
    session.model = model
    session.service = service
    session.events.append({
        "event": "configuration",
        "id": str(uuid4()),
        "timestamp": _now(),
        "success": True,
        "provider": provider,
        "model": model,
        "configuration": _configuration(session),
    })
    return _session_state(session_id, session)


@app.post("/api/documents")
async def upload_document(
    request: Request,
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    session = _session(session_id, request.state.principal_id)
    service = _require_service(session)
    filename = (file.filename or "").replace("\\", "/").split("/")[-1].strip()
    if not filename:
        raise HTTPException(status_code=422, detail="A filename is required.")
    if Path(filename).suffix.lower() not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=422,
            detail="Supported document formats are PDF, DOCX, and TXT.",
        )
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded documents must be 20 MB or smaller.")
    digest = hashlib.sha256(content).hexdigest()
    started = time.perf_counter()
    try:
        chunks = await run_in_threadpool(service.ingest, content, filename)
    except (ValueError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    record = {
        "filename": filename,
        "source": filename,
        "file_type": filename.rsplit(".", 1)[-1].lower() if "." in filename else "",
        "size_bytes": len(content),
        "sha256": digest,
        "chunks": chunks,
        "uploaded_at": _now(),
    }
    session.documents = [
        item for item in session.documents if item.get("filename") != filename
    ]
    session.documents.append(record)
    session.events.append({
        "event": "ingestion",
        "id": str(uuid4()),
        "timestamp": record["uploaded_at"],
        "success": True,
        "source": filename,
        "size_bytes": len(content),
        "sha256": digest,
        "latency_ms": (time.perf_counter() - started) * 1000,
    })
    return {"document": record, "chunk_count": chunks}


@app.post("/api/query")
def query(request: QueryRequest, http_request: Request):
    session = _session(request.session_id, http_request.state.principal_id)
    service = _require_service(session, generation=True)
    if request.retrieval_mode not in {"hybrid", "dense", "sparse"}:
        raise HTTPException(status_code=422, detail="Unsupported retrieval mode.")
    session.retrieval_mode = request.retrieval_mode
    session.top_k = request.top_k
    guard = guard_question(request.question)
    event_id = str(uuid4())
    event: dict[str, Any] = {
        "event": "query",
        "id": event_id,
        "timestamp": _now(),
        "success": False,
        "query": guard.sanitized,
        "security": {
            "pii_detected": len(guard.pii),
            "attack_score": guard.attack.score,
            "attack_patterns": guard.attack.patterns,
            "attack_blocked": guard.blocked,
            "source_attack_indices": [],
            "pii_exposed_in_answer": 0,
        },
    }
    started = time.perf_counter()
    if guard.blocked:
        event["error"] = "This query was blocked by the heuristic prompt-attack screen."
        session.events.append(event)
        raise HTTPException(status_code=400, detail=event["error"])
    try:
        # Compatibility for host adapters that do not expose the canonical RAG
        # pipeline. The built-in RAGService screens the exact prompt evidence.
        if not hasattr(service, "pipeline") and hasattr(service, "retrieve"):
            blocked_sources = _source_attack_indices(
                service, guard.sanitized, request.top_k, request.retrieval_mode,
            )
            if blocked_sources:
                event["security"]["source_attack_indices"] = blocked_sources
                event["error"] = "Potential prompt-injection content was found in retrieved evidence; generation was blocked."
                session.events.append(event)
                raise HTTPException(status_code=400, detail=event["error"])
        stream, sources, details = service.query_with_details(
            question=guard.sanitized,
            chat_history=[
                {"query": event["query"], "answer": event["answer"]}
                for event in session.events
                if event.get("event") == "query"
                and event.get("success")
                and event.get("query")
                and event.get("answer")
            ][-6:],
            top_k=request.top_k,
            mode=request.retrieval_mode,
        )
        raw_answer = "".join(stream)
        pii_count = len(find_pii(raw_answer))
        answer = redact_pii(raw_answer)[0]
        safe_sources = _redact_payload(sources)
        source_texts = [str(source.get("text", "")) for source in safe_sources]
        generation_success = bool(details.get("generation_success", True))
        answer_abstained = bool(details.get("answer_abstained", False))
        answer_audit = (
            audit_answer(answer, source_texts)
            if generation_success and not answer_abstained
            else None
        )
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as error:
        event["error"] = str(error)
        event["latency_ms"] = (time.perf_counter() - started) * 1000
        session.events.append(event)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Biomedical query failed")
        event["error"] = "The RAG query failed. Check API logs for details."
        event["latency_ms"] = (time.perf_counter() - started) * 1000
        session.events.append(event)
        raise HTTPException(status_code=502, detail=event["error"]) from error

    latency_ms = (time.perf_counter() - started) * 1000
    event.update({
        "success": generation_success,
        "answer": answer,
        "sources": [_redact_payload(source.get("source", "Unknown")) for source in safe_sources],
        "source_count": len(safe_sources),
        "is_grounded": details.get("is_grounded"),
        "retrieval_grounded": details.get("retrieval_grounded"),
        "generation_success": details.get("generation_success"),
        "answer_abstained": details.get("answer_abstained", False),
        "validation_passed": details.get("validation_passed"),
        "warnings": details.get("warnings", []),
        "latency_ms": latency_ms,
        "audit": answer_audit,
        "configuration": _configuration(session),
        "provider": session.provider,
        "model": session.model,
        "retrieval_mode": request.retrieval_mode,
        "top_k": request.top_k,
        "security": {**event["security"], "pii_exposed_in_answer": pii_count},
    })
    if not generation_success:
        event["error"] = "Generation failed. See the returned answer and warnings for provider details."
    session.events.append(event)
    return {
        "event_id": event_id,
        "answer": answer,
        "sources": safe_sources,
        "details": details,
        "audit": answer_audit,
        "security": event["security"],
        "latency_ms": latency_ms,
    }


@app.post("/api/retrieve")
def retrieve(request: RetrieveRequest, http_request: Request):
    session = _session(request.session_id, http_request.state.principal_id)
    service = _require_service(session)
    if request.retrieval_mode not in {"hybrid", "dense", "sparse"}:
        raise HTTPException(status_code=422, detail="Unsupported retrieval mode.")
    session.retrieval_mode = request.retrieval_mode
    session.top_k = request.top_k
    guard = guard_question(request.question)
    if guard.blocked:
        raise HTTPException(
            status_code=400,
            detail="This query was blocked by the heuristic prompt-attack screen.",
        )
    try:
        sources = service.retrieve(
            guard.sanitized,
            top_k=request.top_k,
            mode=request.retrieval_mode,
        )
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"sources": _redact_payload(sources), "security": {
        "pii_detected": len(guard.pii),
        "attack_score": guard.attack.score,
        "attack_blocked": guard.blocked,
    }}


@app.post("/api/evaluation/retrieval")
def evaluate_retrieval(request: RetrievalEvaluationRequest, session_id: str, http_request: Request):
    session = _session(session_id, http_request.state.principal_id)
    if not any(source.strip() for source in request.retrieved_sources):
        raise HTTPException(status_code=422, detail="Retrieve evidence before evaluating retrieval.")
    if not any(source.strip() for source in request.relevant_sources):
        raise HTTPException(status_code=422, detail="Add at least one relevant source label before evaluating retrieval.")
    metrics = calculate_retrieval_metrics(
        request.retrieved_sources,
        request.relevant_sources,
        request.k,
    )
    run = {
        "id": str(uuid4()),
        "timestamp": _now(),
        "k": request.k,
        "retrieved_sources": request.retrieved_sources,
        "relevant_sources": request.relevant_sources,
        "metrics": metrics,
        "configuration": _configuration(session),
    }
    session.evaluation_runs.append(run)
    session.events.append({"event": "retrieval_evaluation", **run})
    return run


@app.post("/api/evaluation/answer")
def evaluate_answer(request: AnswerEvaluationRequest, session_id: str, http_request: Request):
    session = _session(session_id, http_request.state.principal_id)
    safe_answer = redact_pii(request.answer)[0]
    safe_evidence = _redact_payload(request.evidence_texts)
    report = audit_answer(safe_answer, safe_evidence)
    session.events.append({
        "event": "answer_evaluation",
        "id": str(uuid4()),
        "timestamp": _now(),
        "success": True,
        "answer": safe_answer,
        "source_count": len(safe_evidence),
        "evidence_texts": safe_evidence,
        "audit": report,
        "configuration": _configuration(session),
    })
    return report


@app.post("/api/reviews")
def create_review(request: ReviewRequest, session_id: str, http_request: Request):
    session = _session(session_id, http_request.state.principal_id)
    query_event = next(
        (
            event for event in session.events
            if event.get("id") == request.query_event_id
            and event.get("event") == "query"
            and event.get("success")
        ),
        None,
    )
    if query_event is None:
        raise HTTPException(status_code=404, detail="Successful query event not found.")
    from dashboard.audit_utils import QUALITY_DIMENSIONS

    unknown_dimensions = set(request.ratings) - set(QUALITY_DIMENSIONS)
    if unknown_dimensions or any(
        type(value) is not int or value not in range(1, 6)
        for value in request.ratings.values()
    ):
        raise HTTPException(
            status_code=422,
            detail="Ratings must use recognized quality dimensions with integer values from 1 to 5.",
        )
    review = {
        "id": str(uuid4()),
        "timestamp": _now(),
        "query_event_id": request.query_event_id,
        "ratings": request.ratings,
        "notes": redact_pii(request.notes)[0],
    }
    session.answer_quality_reviews.append(review)
    session.events.append({"event": "answer_review", **review, "success": True})
    return review


@app.post("/api/governance/security-scan")
def security_scan(text: str = Form(...)):
    guard = guard_question(text)
    return {
        "pii": [{"kind": item.kind, "start": item.start, "end": item.end} for item in guard.pii],
        "redacted_text": guard.sanitized,
        "attack": {
            "score": guard.attack.score,
            "patterns": guard.attack.patterns,
            "blocked": guard.blocked,
        },
        "disclaimer": "Heuristic screening only; this is not a guarantee of PII or attack detection.",
    }


@app.post("/api/governance/fairness")
def run_fairness_screen(request: FairnessRequest, http_request: Request):
    session = _session(request.session_id, http_request.state.principal_id)
    service = _require_service(session, generation=True)
    safe_template = redact_pii(request.template)[0]
    safe_groups = [redact_pii(group)[0] for group in request.groups]
    try:
        variants = build_counterfactual_prompts(safe_template, safe_groups)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    responses = {}
    safe_template = redact_pii(request.template)[0]
    for group, question in variants:
        guard = guard_question(question)
        if guard.blocked:
            raise HTTPException(status_code=400, detail="A fairness variant was blocked by the prompt-attack screen.")
        try:
            if not hasattr(service, "pipeline") and hasattr(service, "retrieve"):
                blocked_sources = _source_attack_indices(
                    service, guard.sanitized, request.top_k, session.retrieval_mode,
                )
                if blocked_sources:
                    raise HTTPException(
                        status_code=400,
                        detail="Potential prompt-injection content was found in retrieved evidence; fairness generation was blocked.",
                    )
            stream, sources, details = service.query_with_details(
                question=guard.sanitized,
                top_k=request.top_k,
                mode=session.retrieval_mode,
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        responses[group] = {
            "answer": redact_pii("".join(stream))[0],
            "sources": _redact_payload([
                source.get("source", "Unknown")
                for source in sources
            ]),
            "warnings": details.get("warnings", []),
        }
    pairs = compare_responses(responses, safe_groups)
    report = fairness_report(pairs)
    report_record = {
        **report,
        "id": str(uuid4()),
        "timestamp": _now(),
        "template": safe_template,
        "groups": safe_groups,
        "configuration_version": _configuration(session)["version"],
        "responses": responses,
        "limits": "Lexical comparison is a screening signal for qualified human review, not proof of bias or fairness.",
    }
    session.fairness_reports.append(report_record)
    return report_record


@app.post("/api/governance/risk")
def record_operator_risk(request: OperatorRiskRequest, session_id: str, http_request: Request):
    session = _session(session_id, http_request.state.principal_id)
    score = operator_ethical_risk_score(request.impact, request.likelihood)
    record = {
        "id": str(uuid4()),
        "timestamp": _now(),
        "title": redact_pii(request.title)[0],
        "rationale": redact_pii(request.rationale)[0],
        "impact": request.impact,
        "likelihood": request.likelihood,
        **score,
    }
    session.risk_register.append(record)
    return record


@app.post("/api/governance/findings")
def add_governance_finding(request: FindingRequest, session_id: str, http_request: Request):
    session = _session(session_id, http_request.state.principal_id)
    finding = {
        "id": str(uuid4()),
        "timestamp": _now(),
        **{
            **request.model_dump(),
            "title": redact_pii(request.title)[0],
            "rationale": redact_pii(request.rationale)[0],
        },
    }
    session.risk_register.append(finding)
    return finding


@app.get("/api/governance/summary")
def governance_summary(session_id: str, request: Request):
    session = _session(session_id, request.state.principal_id)
    signals = collect_risk_signals(
        session.events,
        session.answer_quality_reviews,
        session.fairness_reports,
    )
    return {
        "risk_signals": signals,
        "observed_risk": ethical_risk_score(signals),
        "fairness": session.fairness_reports[-1] if session.fairness_reports else None,
        "findings": session.risk_register,
        "limitations": "These are research and heuristic indicators for human review, not clinical validation or security certification.",
    }


@app.get("/api/audit/export")
def export_audit(session_id: str, request: Request):
    session = _session(session_id, request.state.principal_id)
    return {
        "session_only": True,
        "session_id": session_id,
        "created_at": session.created_at,
        "exported_at": _now(),
        "current_configuration": _configuration(session),
        "documents": session.documents,
        "events": session.events,
        "evaluations": session.evaluation_runs,
        "answer_quality_reviews": session.answer_quality_reviews,
        "fairness_reports": session.fairness_reports,
        "governance_findings": session.risk_register,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
