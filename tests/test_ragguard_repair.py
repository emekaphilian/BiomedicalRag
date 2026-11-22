from types import SimpleNamespace
from uuid import uuid4
import logging

from fastapi.testclient import TestClient

import app.services as services
import app.api as api
from app.api import app
from app.ragguard_repair import (
    REPAIR_CONTEXT_TTL_SECONDS,
    PendingRepair,
    RepairRequest,
)


class FakePipeline:
    def __init__(self, config, response):
        self.config = config
        self.response = response
        self.calls = []
        self.documents = []
        self.chunks = []
        self.vectorstore = object()

    def query(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return self.response


def make_service(monkeypatch, response, original_details):
    monkeypatch.setenv("RAGGUARD_REPAIR_TOKEN", "repair-secret")
    monkeypatch.setattr(services, "load_default_corpus", lambda: [])
    monkeypatch.setattr(
        services,
        "BiomedicalRAGPipeline",
        lambda config: FakePipeline(config, response),
    )
    service = services.RAGService("cohere", "command-r-plus", "test-key")
    service.retrieve = lambda *_args, **_kwargs: []
    request_id = str(uuid4())
    service._remember_ragguard_request(
        request_id,
        PendingRepair(
            created_at=services.monotonic(),
            question="local private question",
            mode="sparse",
            top_k=5,
            chat_history=None,
            retrieval_query=None,
            original_details=original_details,
        ),
    )
    request = RepairRequest(
        request_id=request_id,
        application_id="biomedicalrag",
        environment="development",
        query="[redacted]",
        failure_type="LOW_CONTEXT_PRECISION",
        strategy="RERANK",
        retrieved_chunks=[],
    )
    return service, request


def test_rerank_repair_validates_request_candidate_without_retaining_its_content(monkeypatch):
    response = SimpleNamespace(
        answer="candidate answer",
        sources=[{"chunk_id": "candidate-1", "text": "local source text"}],
        is_grounded=True,
        retrieval_grounded=True,
        generation_success=True,
        validation_passed=True,
        answer_abstained=False,
        warnings=[],
        retrieval_time_ms=12.0,
        generation_time_ms=20.0,
    )
    service, request = make_service(
        monkeypatch,
        response,
        {
            "retrieval_grounded": False,
            "generation_success": True,
            "validation_passed": False,
            "answer_abstained": False,
        },
    )

    result = service.execute_ragguard_repair(request)

    assert result.status == "promoted"
    assert result.validated is True
    assert result.context_verified is True
    assert result.improvement > 0
    question, options = service.pipeline.calls[0]
    assert question == "local private question"
    assert options["use_reranking"] is True
    assert "candidate answer" not in result.model_dump_json()
    pending = service._ragguard_pending_repairs[request.request_id]
    assert pending.result == result
    assert pending.question == ""
    assert pending.chat_history is None
    assert pending.retrieval_query is None
    assert pending.original_details == {}
    assert not hasattr(pending, "candidate")
    assert not hasattr(pending, "original_answer")
    assert not hasattr(pending, "original_sources")

    service.query_with_details("next question", mode="sparse")
    _, future_options = service.pipeline.calls[-1]
    assert future_options["retrieval_mode"] == "sparse"
    assert future_options["use_reranking"] is False


def test_authenticated_repair_endpoint_dispatches_pending_query(monkeypatch):
    response = SimpleNamespace(
        answer="candidate answer",
        sources=[{"chunk_id": "candidate-1", "text": "local source text"}],
        is_grounded=True,
        retrieval_grounded=True,
        generation_success=True,
        validation_passed=True,
        answer_abstained=False,
        warnings=[],
        retrieval_time_ms=12.0,
        generation_time_ms=20.0,
    )
    service, request = make_service(
        monkeypatch,
        response,
        {
            "retrieval_grounded": False,
            "generation_success": True,
            "validation_passed": False,
            "answer_abstained": False,
        },
    )
    monkeypatch.setenv("RAGGUARD_APPLICATION_ID", "biomedicalrag")
    monkeypatch.setenv("RAGGUARD_ENVIRONMENT", "development")
    monkeypatch.setattr(
        api,
        "_sessions",
        {"active-query": api.DashboardSession(
            created_at="2026-10-23T00:00:00Z",
            service=service,
        )},
    )

    response = TestClient(app).post(
        "/internal/ragguard/repairs",
        json=request.model_dump(mode="json"),
        headers={"Authorization": "Bearer repair-secret"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "promoted"
    assert response.json()["context_verified"] is True
    assert "candidate answer" not in response.text


def test_unvalidated_repair_rolls_back_to_original_result(monkeypatch):
    response = SimpleNamespace(
        answer="unsafe candidate",
        sources=[],
        is_grounded=False,
        retrieval_grounded=False,
        generation_success=True,
        validation_passed=False,
        answer_abstained=True,
        warnings=["validation failed"],
        retrieval_time_ms=12.0,
        generation_time_ms=20.0,
    )
    service, request = make_service(
        monkeypatch,
        response,
        {
            "retrieval_grounded": False,
            "generation_success": False,
            "validation_passed": False,
            "answer_abstained": True,
        },
    )

    result = service.execute_ragguard_repair(request)

    assert result.status == "rolled_back"
    assert result.validated is False
    assert service._ragguard_pending_repairs[request.request_id].result == result


def test_unsupported_repair_is_rejected_without_pipeline_execution(monkeypatch):
    service, request = make_service(
        monkeypatch,
        SimpleNamespace(),
        {
            "retrieval_grounded": False,
            "generation_success": False,
            "validation_passed": False,
            "answer_abstained": True,
        },
    )
    request = request.model_copy(update={"strategy": "DEDUPLICATE"})

    result = service.execute_ragguard_repair(request)

    assert result.status == "rejected"
    assert service.pipeline.calls == []


def test_expired_repair_context_is_removed_and_rejected(monkeypatch):
    service, request = make_service(
        monkeypatch,
        SimpleNamespace(),
        {
            "retrieval_grounded": False,
            "generation_success": False,
            "validation_passed": False,
            "answer_abstained": True,
        },
    )
    service._ragguard_pending_repairs[request.request_id].created_at -= (
        REPAIR_CONTEXT_TTL_SECONDS + 1
    )

    result = service.execute_ragguard_repair(request)

    assert result.status == "rejected"
    assert request.request_id not in service._ragguard_pending_repairs
    assert service.pipeline.calls == []


def test_repair_endpoints_fail_closed_and_require_bearer(monkeypatch, caplog):
    client = TestClient(app)
    monkeypatch.delenv("RAGGUARD_REPAIR_TOKEN", raising=False)
    assert client.get("/internal/ragguard/repairs/capabilities").status_code == 503

    monkeypatch.setenv("RAGGUARD_REPAIR_TOKEN", "repair-secret")
    with caplog.at_level(logging.INFO, logger="biomedicalrag.ragguard_repair"):
        denied = client.get(
            "/internal/ragguard/repairs/capabilities",
            headers={"Authorization": "Bearer wrong-token"},
        )
        denied_operation = client.post(
            "/internal/ragguard/repairs",
            json={
                "request_id": str(uuid4()),
                "application_id": "biomedicalrag",
                "environment": "development",
                "query": "[redacted]",
                "failure_type": "NO_RETRIEVAL",
                "strategy": "RERANK",
                "retrieved_chunks": [],
            },
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert denied.status_code == 401
    assert denied_operation.status_code == 401
    assert '"authorized":false' in caplog.text
    assert "authorization_denied" in caplog.text

    allowed = client.get(
        "/internal/ragguard/repairs/capabilities",
        headers={"Authorization": "Bearer repair-secret"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["supported_repairs"] == ["HYBRID_RETRIEVAL", "RERANK"]
