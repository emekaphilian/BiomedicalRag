"""FastAPI interface for the biomedical RAG and research dashboard services."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

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
    provider: str = "cohere"
    model: str | None = None


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


def _new_service(provider: str, model: str, api_key: str):
    from app.services import RAGService

    return RAGService(provider=provider, model=model, api_key=api_key)


def _session(session_id: str) -> DashboardSession:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
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
        str(item.get("text", item.get("content", "")))
        for item in evidence
    ])


app = FastAPI(
    title="Biomedical RAG API",
    description="API for grounded biomedical Q&A, evaluation, audit, and governance.",
    version="1.0.0",
)
origins = [
    origin.strip()
    for origin in os.getenv("BIOMEDRAG_CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": app.version}


@app.post("/api/sessions", response_model=SessionResponse)
def create_session():
    session_id = str(uuid4())
    created_at = _now()
    with _sessions_lock:
        _sessions[session_id] = DashboardSession(created_at=created_at)
    return SessionResponse(session_id=session_id, created_at=created_at)


@app.get("/api/sessions/{session_id}")
def get_session_state(session_id: str):
    session = _session(session_id)
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
def configure_session(session_id: str, request: ConfigureRequest):
    session = _session(session_id)
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
    return get_session_state(session_id)


@app.post("/api/documents")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    session = _session(session_id)
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
        chunks = service.ingest(content, filename)
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
def query(request: QueryRequest):
    session = _session(request.session_id)
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
        blocked_sources = _source_attack_indices(
            service,
            guard.sanitized,
            request.top_k,
            request.retrieval_mode,
        )
        if blocked_sources:
            event["security"]["source_attack_indices"] = blocked_sources
            event["error"] = "Potential prompt-injection content was found in retrieved evidence; generation was blocked."
            session.events.append(event)
            raise HTTPException(status_code=400, detail=event["error"])
        stream, sources, details = service.query_with_details(
            question=guard.sanitized,
            top_k=request.top_k,
            mode=request.retrieval_mode,
        )
        raw_answer = "".join(stream)
        pii_count = len(find_pii(raw_answer))
        answer = redact_pii(raw_answer)[0]
        safe_sources = _redact_payload(sources)
        source_texts = [str(source.get("text", "")) for source in safe_sources]
        answer_audit = audit_answer(answer, source_texts)
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
        "success": True,
        "answer": answer,
        "sources": [_redact_payload(source.get("source", "Unknown")) for source in safe_sources],
        "source_count": len(safe_sources),
        "is_grounded": details.get("is_grounded"),
        "retrieval_grounded": details.get("retrieval_grounded"),
        "generation_success": details.get("generation_success"),
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
def retrieve(request: RetrieveRequest):
    session = _session(request.session_id)
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
def evaluate_retrieval(request: RetrievalEvaluationRequest, session_id: str):
    session = _session(session_id)
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
def evaluate_answer(request: AnswerEvaluationRequest, session_id: str):
    session = _session(session_id)
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
def create_review(request: ReviewRequest, session_id: str):
    session = _session(session_id)
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
        "notes": request.notes,
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
def run_fairness_screen(request: FairnessRequest):
    session = _session(request.session_id)
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
            blocked_sources = _source_attack_indices(
                service,
                guard.sanitized,
                request.top_k,
                session.retrieval_mode,
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
def record_operator_risk(request: OperatorRiskRequest, session_id: str):
    session = _session(session_id)
    score = operator_ethical_risk_score(request.impact, request.likelihood)
    record = {
        "id": str(uuid4()),
        "timestamp": _now(),
        "title": request.title,
        "rationale": request.rationale,
        "impact": request.impact,
        "likelihood": request.likelihood,
        **score,
    }
    session.risk_register.append(record)
    return record


@app.post("/api/governance/findings")
def add_governance_finding(request: FindingRequest, session_id: str):
    session = _session(session_id)
    finding = {
        "id": str(uuid4()),
        "timestamp": _now(),
        **request.model_dump(),
    }
    session.risk_register.append(finding)
    return finding


@app.get("/api/governance/summary")
def governance_summary(session_id: str):
    session = _session(session_id)
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
def export_audit(session_id: str):
    session = _session(session_id)
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
