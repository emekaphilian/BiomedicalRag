from rag.document import BiomedicalDocument
from rag.pipeline import (
    BiomedicialRAGPipeline,
    BiomedicalRAGPipeline,
)


class _FakeGenerationPipeline:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, query, k, retrieval_k, **kwargs):
        self.calls.append((query, k, retrieval_k, kwargs))
        return self.response


def _generation_response(
    *,
    answer="Evidence-based answer [Evidence 1].",
    valid=True,
    grounded=True,
    evidence=None,
    issues=None,
    generation=None,
    learning_answer=None,
):
    response = {
        "final_answer": answer,
        "retrieval": {"grounded": grounded},
        "compression": {"evidence": evidence or []},
        "validation": {"valid": valid, "issues": issues or []},
        "learning_answer": learning_answer,
        "timings": {
            "retrieval_time_ms": 12.5,
            "generation_time_ms": 3.5,
        },
    }
    if generation is not None:
        response["generation"] = generation
    return response


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
        ("what is diabetes?", 3, 9, {"answer_mode": "research"}),
        ("what is diabetes?", 2, 9, {"answer_mode": "research"}),
    ]
    assert response.is_grounded
    assert response.confidence is None
    assert response.sources[0]["document_id"] == "topic-42"
    assert response.retrieval_time_ms == 12.5
    assert response.generation_time_ms == 3.5
    assert "MEDICAL DISCLAIMER" in response.answer
    assert BiomedicialRAGPipeline is BiomedicalRAGPipeline


def test_followup_is_reformulated_before_scope_classification_and_retrieval(monkeypatch, caplog):
    generation_pipeline = _FakeGenerationPipeline(_generation_response())
    monkeypatch.setattr("rag.pipeline._build_generation_pipeline", lambda *args, **kwargs: generation_pipeline)
    pipeline = BiomedicalRAGPipeline({})
    pipeline.vectorstore = object()
    history = [{
        "query": "Explain CRISPR gene editing",
        "answer": "CRISPR uses guide RNA and a Cas enzyme to target DNA.",
    }]

    with caplog.at_level("INFO", logger="rag.pipeline"):
        response = pipeline.query("explain further", chat_history=history)

    assert response.answer.startswith("Evidence-based answer")
    assert generation_pipeline.calls[0][0] == "Explain CRISPR gene editing further."
    event = next(record.rag_query_context for record in caplog.records if hasattr(record, "rag_query_context"))
    assert event["current_query"] == "explain further"
    assert event["prior_messages"] == history
    assert event["reformulated_query"] == "Explain CRISPR gene editing further."
    assert event["classifier_result"]["topic_detected"] is True


def test_supported_biomedical_question_topics_are_recognized_before_retrieval():
    from rag.medical_risk import is_medical_query

    questions = [
        "Explain CRISPR gene editing",
        "What is hypertension?",
        "What causes hypertension?",
        "What is a tumor?",
        "What causes a tumor?",
        "What is the function of dopamine?",
        "How do neurons communicate?",
        "What is Alzheimer's disease?",
        "What is anemia?",
        "What is glaucoma?",
        "What is rheumatoid arthritis?",
        "How does the immune system work?",
        "What does the liver do?",
        "How does blood circulate?",
        "Explain cellular respiration",
    ]
    assert all(is_medical_query(question) for question in questions)
    assert not is_medical_query("What is the best way to repair a bicycle?")


def test_pipeline_without_index_returns_no_relevant_evidence(monkeypatch):
    monkeypatch.setattr("rag.generation_pipeline.retrieve_pubmed", lambda *args, **kwargs: [])
    pipeline = BiomedicalRAGPipeline({})
    response = pipeline.query("What is diabetes?")
    assert response.decision == "no_relevant_evidence"
    assert response.answer_abstained is True


def test_pipeline_never_returns_an_empty_success_answer(monkeypatch):
    generation_pipeline = _FakeGenerationPipeline(_generation_response(answer=""))
    monkeypatch.setattr("rag.pipeline._build_generation_pipeline", lambda *args, **kwargs: generation_pipeline)
    pipeline = BiomedicalRAGPipeline({})
    pipeline.vectorstore = object()

    response = pipeline.query("What is dopamine?")

    assert response.answer.strip()
    assert response.decision == "generation_failed"
    assert response.generation_success is False
    assert response.validation_passed is False


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

    response = pipeline.query("What does the diabetes evidence say?")

    assert not response.is_grounded
    assert response.warnings == ["Forbidden phrase detected"]
    assert "could not find enough supporting evidence" in response.answer.lower()
    assert "You should take" not in response.answer


def test_pipeline_reports_provider_failure_without_mislabeling_validation(
    monkeypatch,
):
    pipeline = BiomedicalRAGPipeline({})
    pipeline.vectorstore = object()
    provider_error = "Cohere request failed (HTTP 404): model not found"
    generation_pipeline = _FakeGenerationPipeline(
        _generation_response(
            answer=f"Generation failed: {provider_error}",
            valid=False,
            grounded=True,
            issues=[provider_error],
            generation={
                "success": False,
                "answer": f"Generation failed: {provider_error}",
                "error": provider_error,
            },
        )
    )
    monkeypatch.setattr(
        "rag.pipeline._build_generation_pipeline",
        lambda *args, **kwargs: generation_pipeline,
    )

    response = pipeline.query("How does Parkinson's disease affect dopamine?")

    assert response.generation_success is False
    assert response.validation_passed is False
    assert response.retrieval_grounded is True
    assert response.is_grounded is False
    assert "Generation failed: " + provider_error in response.answer
    assert "could not produce an answer that passed" not in response.answer.lower()
    assert response.warnings == [provider_error]


def test_pipeline_does_not_mark_unvalidated_answer_as_grounded(monkeypatch):
    pipeline = BiomedicalRAGPipeline({})
    pipeline.vectorstore = object()
    generation_pipeline = _FakeGenerationPipeline(
        _generation_response(
            answer="Dopamine levels are affected [Evidence 1].",
            valid=False,
            grounded=True,
            issues=["No evidence citations found in answer"],
        )
    )
    monkeypatch.setattr(
        "rag.pipeline._build_generation_pipeline",
        lambda *args, **kwargs: generation_pipeline,
    )

    response = pipeline.query("How does Parkinson's disease affect dopamine?")

    assert response.retrieval_grounded is True
    assert response.validation_passed is False
    assert response.is_grounded is False


def test_learning_mode_keeps_evidence_draft_with_limited_confidence(monkeypatch):
    pipeline = BiomedicalRAGPipeline({})
    pipeline.vectorstore = object()
    generation_pipeline = _FakeGenerationPipeline(
        _generation_response(
            answer="Simple explanation from retrieved evidence.",
            valid=False,
            grounded=True,
            issues=["One or more sentences lack an exact evidence citation"],
            learning_answer="Simple explanation from retrieved evidence [Evidence 1].",
        )
    )
    monkeypatch.setattr(
        "rag.pipeline._build_generation_pipeline",
        lambda *args, **kwargs: generation_pipeline,
    )

    response = pipeline.query(
        "What is diabetes?",
        answer_mode="learning",
    )

    assert generation_pipeline.calls[0][3]["answer_mode"] == "learning"
    assert response.learning_answer == "Simple explanation from retrieved evidence [Evidence 1]."
