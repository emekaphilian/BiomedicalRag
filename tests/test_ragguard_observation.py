from types import SimpleNamespace

import app.services as services
from app.ragguard_observation import RAGGuardObserver


class FakePipeline:
    def __init__(self, config):
        self.config = config
        self.documents = []
        self.chunks = []
        self.vectorstore = object()

    def query(self, question, **kwargs):
        return SimpleNamespace(
            answer="Private grounded answer",
            sources=[
                {
                    "chunk_id": "chunk-123",
                    "rerank_score": 0.83,
                    "text": "Private source excerpt",
                }
            ],
            is_grounded=True,
            retrieval_grounded=True,
            generation_success=True,
            validation_passed=True,
            answer_abstained=False,
            warnings=[],
            retrieval_time_ms=18.0,
            generation_time_ms=31.0,
        )


def test_service_sends_minimal_observation_and_preserves_result(monkeypatch):
    monkeypatch.setattr(services, "load_default_corpus", lambda: [])
    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)
    observer = SimpleNamespace(observations=[])

    def observe(**kwargs):
        observer.observations.append(kwargs)
        return SimpleNamespace(delivered=True)

    observer.observe = observe
    service = services.RAGService("cohere", "command-r-plus", "test-key")
    service.ragguard_observer = observer

    stream, sources, details = service.query_with_details("Sensitive medical question")

    assert "".join(stream) == "Private grounded answer"
    assert sources[0]["text"] == "Private source excerpt"
    assert details["generation_success"] is True
    assert observer.observations == [{
        "query": "Sensitive medical question",
        "retrieved_chunks": [("chunk-123", 0.83)],
        "retrieval_method": "hybrid",
        "retrieval_latency_ms": 18.0,
        "generation_status": "succeeded",
    }]


def test_observer_uses_bearer_token_and_redacts_query(monkeypatch):
    captured = {}

    class Response:
        is_success = True
        status_code = 201

    def fake_post(url, *, json, headers, timeout):
        captured.update(url=url, payload=json, headers=headers, timeout=timeout)
        return Response()

    monkeypatch.setattr("app.ragguard_observation.httpx.post", fake_post)
    observer = RAGGuardObserver(
        base_url="https://ragguard.example/api/v1",
        application_id="biomedicalrag",
        environment="development",
        token="observation-secret",
    )

    delivery = observer.observe(
        query="Sensitive medical question",
        retrieved_chunks=[("chunk-123", 0.83)],
        retrieval_method="hybrid",
        retrieval_latency_ms=18.0,
        generation_status="succeeded",
    )

    assert delivery.delivered is True
    assert captured["headers"] == {"Authorization": "Bearer observation-secret"}
    assert captured["payload"]["query"] == "[redacted]"
    assert captured["payload"]["retrieved_chunks"] == [{"id": "chunk-123", "score": 0.83}]
    assert "Sensitive medical question" not in str(captured["payload"])


def test_observer_is_disabled_without_required_environment(monkeypatch):
    for key in (
        "RAGGUARD_URL",
        "RAGGUARD_APPLICATION_ID",
        "RAGGUARD_ENVIRONMENT",
        "RAGGUARD_OBSERVATION_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    assert RAGGuardObserver.from_environment() is None


def test_observer_delivery_exception_does_not_escape_service(monkeypatch):
    monkeypatch.setattr(services, "load_default_corpus", lambda: [])
    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)
    service = services.RAGService("cohere", "command-r-plus", "test-key")
    service.ragguard_observer = SimpleNamespace(
        observe=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("network issue"))
    )

    stream, _, details = service.query_with_details("A question")

    assert "".join(stream) == "Private grounded answer"
    assert details["generation_success"] is True
