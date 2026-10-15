from rag.document import BiomedicalDocument
from rag.pipeline import (
    BiomedicialRAGPipeline,
    BiomedicalRAGPipeline,
)


class _FakeGenerationPipeline:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, query, k, retrieval_k):
        self.calls.append((query, k, retrieval_k))
        return self.response


def _generation_response(
    *,
    answer="Evidence-based answer [Evidence 1].",
    valid=True,
    grounded=True,
    evidence=None,
    issues=None,
):
    return {
        "final_answer": answer,
        "retrieval": {"grounded": grounded},
        "compression": {"evidence": evidence or []},
        "validation": {"valid": valid, "issues": issues or []},
        "timings": {
            "retrieval_time_ms": 12.5,
            "generation_time_ms": 3.5,
        },
    }


def test_pipeline_indexes_and_queries_with_mock_generation(monkeypatch):
    vectorstore = object()
    indexed_documents = []
    index_calls = []
    generation_pipeline = _FakeGenerationPipeline(
        _generation_response(
            evidence=[
                {
                    "metadata": {
                        "source": "MedlinePlus",
                        "document_type": "patient_education",
                        "document_id": "topic-42",
                    }
                }
            ]
        )
    )
    generation_calls = []

    def fake_build_index(documents):
        index_calls.append(documents)
        indexed_documents.extend(documents)
        return vectorstore, documents

    def fake_build_generation_pipeline(
        actual_vectorstore,
        documents,
        use_mock,
        use_reranking,
        generator_config,
    ):
        generation_calls.append(
            (actual_vectorstore, documents, use_mock, use_reranking, generator_config)
        )
        return generation_pipeline

    monkeypatch.setattr("rag.pipeline._build_index", fake_build_index)
    monkeypatch.setattr(
        "rag.pipeline._build_generation_pipeline",
        fake_build_generation_pipeline,
    )

    pipeline = BiomedicalRAGPipeline(
        {"retriever": {"retrieval_k": 9}}
    )
    source_metadata = {
        "source": "MedlinePlus",
        "document_type": "patient_education",
    }
    pipeline.index_documents(
        [BiomedicalDocument("  Diabetes   information ", source_metadata)]
    )
    response = pipeline.query("What is diabetes?", top_k=3)
    pipeline.query("What is diabetes?", top_k=2, use_reranking=False)

    assert indexed_documents[0].content == "Diabetes information"
    assert indexed_documents[0].metadata == source_metadata
    assert pipeline.vectorstore is vectorstore
    assert generation_calls[0][2:] == (True, True, {})
    assert generation_calls[1][3] is False
    assert generation_pipeline.calls == [
        ("what is diabetes?", 3, 9),
        ("what is diabetes?", 2, 9),
    ]
    assert response.is_grounded
    assert response.confidence is None
    assert response.sources[0]["document_id"] == "topic-42"
    assert response.retrieval_time_ms == 12.5
    assert response.generation_time_ms == 3.5
    assert "MEDICAL DISCLAIMER" in response.answer
    assert BiomedicialRAGPipeline is BiomedicalRAGPipeline


def test_pipeline_rejects_queries_before_indexing():
    pipeline = BiomedicalRAGPipeline({})

    try:
        pipeline.query("What is diabetes?")
    except RuntimeError as exc:
        assert "Call index_documents first" in str(exc)
    else:
        raise AssertionError("Querying before indexing should fail.")


def test_pipeline_blocks_personalized_medical_advice(monkeypatch):
    pipeline = BiomedicalRAGPipeline({})
    pipeline.vectorstore = object()
    calls = []

    def unexpected_generation_pipeline(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "rag.pipeline._build_generation_pipeline",
        unexpected_generation_pipeline,
    )

    response = pipeline.query("Should I take this medication?")

    assert not response.is_grounded
    assert response.sources == []
    assert calls == []
    assert response.warnings
    assert "MEDICAL DISCLAIMER" in response.answer


def test_pipeline_does_not_return_generation_that_failed_validation(monkeypatch):
    pipeline = BiomedicalRAGPipeline({})
    pipeline.vectorstore = object()
    generation_pipeline = _FakeGenerationPipeline(
        _generation_response(
            answer="You should take this medication.",
            valid=False,
            grounded=False,
            issues=["Forbidden phrase detected"],
        )
    )

    def fake_build_generation_pipeline(*args, **kwargs):
        del args, kwargs
        return generation_pipeline

    monkeypatch.setattr(
        "rag.pipeline._build_generation_pipeline",
        fake_build_generation_pipeline,
    )

    response = pipeline.query("What does the evidence say?")

    assert not response.is_grounded
    assert response.warnings == ["Forbidden phrase detected"]
    assert "could not produce an answer" in response.answer.lower()
    assert "You should take" not in response.answer
