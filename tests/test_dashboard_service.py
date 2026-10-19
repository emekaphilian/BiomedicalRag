from types import SimpleNamespace

import pytest

import app.services as services


@pytest.fixture(autouse=True)
def empty_default_corpus(monkeypatch):
    monkeypatch.setattr(services, "load_default_corpus", lambda: [])


class FakePipeline:
    def __init__(self, config):
        self.config = config
        self.documents = []
        self.chunks = []
        self.queries = []
        self.vectorstore = object()

    def index_documents(self, documents):
        self.documents = list(documents)
        self.chunks = list(documents)

    def query(self, question, top_k, retrieval_mode, use_reranking):
        self.queries.append((question, top_k, retrieval_mode, use_reranking))
        return SimpleNamespace(
            answer="Grounded answer. MEDICAL DISCLAIMER",
            sources=[
                {"source": "guide.txt", "text": "Evidence excerpt", "rerank_score": 0.8}
            ],
            is_grounded=True,
            warnings=["example warning"],
            retrieval_time_ms=12.5,
            generation_time_ms=34.0,
        )


def test_dashboard_service_uses_grounded_pipeline_without_langchain_adapter(
    monkeypatch,
):
    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)

    service = services.RAGService(
        provider="cohere",
        model="command-r-plus",
        api_key="test-key",
    )

    assert service.pipeline.config["generator"] == {
        "provider": "cohere",
        "model_name": "command-r-plus",
        "api_key": "test-key",
        "use_mock": False,
    }


def test_dashboard_service_ingests_text_with_source_metadata(monkeypatch):
    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)
    service = services.RAGService("openai", "gpt-4o", "test-key")

    chunks = service.ingest(b"Diabetes information from a clinical guide.", "guide.txt")

    assert chunks == 1
    assert service.total_chunks == 1
    assert service.pipeline.documents[0].content == (
        "Diabetes information from a clinical guide."
    )
    assert service.pipeline.documents[0].metadata["source"] == "guide.txt"


def test_dashboard_service_indexes_default_corpus_and_keeps_it_on_upload(
    monkeypatch,
):
    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)
    base_documents = [
        services.BiomedicalDocument(
            content="Preloaded biomedical guidance.",
            metadata={"source": "MedlinePlus"},
        )
    ]
    monkeypatch.setattr(
        services,
        "load_default_corpus",
        lambda: list(base_documents),
    )
    service = services.RAGService("openai", "gpt-4o", "test-key")

    assert service.total_documents == 1
    assert service.total_chunks == 1
    assert service.source_names == ["MedlinePlus"]

    service.ingest(b"Additional uploaded guidance.", "extra.txt")

    assert service.total_documents == 2
    assert {document.metadata["source"] for document in service.pipeline.documents} == {
        "MedlinePlus",
        "extra.txt",
    }


def test_dashboard_service_replaces_same_filename_when_reindexed(monkeypatch):
    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)
    service = services.RAGService("openai", "gpt-4o", "test-key")

    service.ingest(b"First version of the guide.", "guide.txt")
    service.ingest(b"Updated version of the guide.", "guide.txt")

    assert len(service.pipeline.documents) == 1
    assert service.pipeline.documents[0].content == "Updated version of the guide."


def test_dashboard_service_returns_answer_chunks_and_provenance(monkeypatch):
    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)
    service = services.RAGService("huggingface", "google/flan-t5-large", "test-key")

    stream, sources = service.query_stream(
        "What does the guide say?",
        chat_history=[{"role": "user", "content": "ignored for safety"}],
        top_k=4,
        mode="sparse",
    )

    assert "".join(stream) == "Grounded answer. MEDICAL DISCLAIMER"
    assert sources[0]["text"] == "Evidence excerpt"
    assert service.pipeline.queries == [
        ("What does the guide say?", 4, "sparse", False)
    ]


def test_dashboard_service_exposes_pipeline_governance_details(monkeypatch):
    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)
    service = services.RAGService("cohere", "command-r-plus", "test-key")

    _, _, details = service.query_with_details("What does the guide say?")

    assert details == {
        "is_grounded": True,
        "retrieval_grounded": True,
        "generation_success": True,
        "validation_passed": False,
        "warnings": ["example warning"],
        "retrieval_time_ms": 12.5,
        "generation_time_ms": 34.0,
    }


def test_dashboard_service_retrieves_sparse_evidence_without_generation(
    monkeypatch,
):
    from rag import bm25_retriever

    class FakeBM25Retriever:
        def __init__(self, documents):
            self.documents = documents

        def search(self, query, k):
            assert query == "what does the guide say?"
            assert k == 3
            return [
                {"document": self.documents[0], "score": 0.75},
            ]

    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)
    monkeypatch.setattr(bm25_retriever, "BM25Retriever", FakeBM25Retriever)
    service = services.RAGService("cohere", "command-r-plus", "test-key")
    service.ingest(b"Diabetes information from a clinical guide.", "guide.txt")

    results = service.retrieve("What does the guide say?", top_k=3, mode="sparse")

    assert results[0]["metadata"]["source"] == "guide.txt"
    assert results[0]["score"] == 0.75
    assert results[0]["rank"] == 1
    assert service.pipeline.queries == []


def test_dashboard_service_sparse_retrieval_preserves_all_ranked_results(
    monkeypatch,
):
    from rag import bm25_retriever

    class FakeBM25Retriever:
        def __init__(self, documents):
            self.documents = documents

        def search(self, query, k):
            assert k == 3
            return [
                {"document": document, "score": score}
                for document, score in zip(self.documents, (0.9, 0.6))
            ]

    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)
    monkeypatch.setattr(bm25_retriever, "BM25Retriever", FakeBM25Retriever)
    service = services.RAGService("cohere", "command-r-plus", "test-key")
    service.ingest(b"First source with diabetes evidence.", "first.txt")
    service.ingest(b"Second source with diabetes evidence.", "second.txt")

    results = service.retrieve("Diabetes evidence", top_k=3, mode="sparse")

    assert [result["metadata"]["source"] for result in results] == [
        "first.txt",
        "second.txt",
    ]
    assert [result["score"] for result in results] == [0.9, 0.6]
    assert [result["rank"] for result in results] == [1, 2]
    assert service.pipeline.queries == []


def test_dashboard_service_retrieves_hybrid_evidence_with_chunk_provenance(
    monkeypatch,
):
    from rag import hybrid_retriever

    class FakeHybridRetriever:
        def __init__(self, vectorstore, documents):
            self.documents = documents

        def retrieve(self, query, k):
            assert query == "what does the guide say?"
            assert k == 2
            chunk = self.documents[0]
            return {
                "dense_results": [
                    SimpleNamespace(content=chunk.content, metadata=chunk.metadata)
                ],
                "sparse_results": [
                    {"document": chunk, "score": 0.5},
                ],
                "fused_results": [(chunk.content, 0.03)],
            }

    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)
    monkeypatch.setattr(hybrid_retriever, "HybridRetriever", FakeHybridRetriever)
    service = services.RAGService("cohere", "command-r-plus", "test-key")
    service.ingest(b"Diabetes information from a clinical guide.", "guide.txt")

    results = service.retrieve("What does the guide say?", top_k=2, mode="hybrid")

    assert results[0]["metadata"]["source"] == "guide.txt"
    assert results[0]["content"] == service.pipeline.chunks[0].content
    assert results[0]["rank"] == 1
    assert service.pipeline.queries == []


def test_dashboard_service_rejects_empty_key_and_unsupported_file(monkeypatch):
    monkeypatch.setattr(services, "BiomedicalRAGPipeline", FakePipeline)

    with pytest.raises(ValueError, match="API key"):
        services.RAGService("cohere", "command-r-plus", " ")

    service = services.RAGService("cohere", "command-r-plus", "test-key")
    with pytest.raises(ValueError, match="Supported document formats"):
        service.ingest(b"content", "guide.rtf")
