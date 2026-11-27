from types import SimpleNamespace

from rag.bm25_retriever import BM25Retriever, _tokenize
from rag.document import BiomedicalDocument, RetrievedChunk
from rag.grounding import has_topic_overlap
import rag.hybrid_retriever as hybrid_module
import rag.generation_pipeline as generation_module


def test_bm25_normalizes_punctuation_for_medical_terms():
    documents = [
        BiomedicalDocument(
            "H. pylori can affect the stomach and cause ulcers.",
            {"title": "Peptic ulcers"},
        ),
        BiomedicalDocument(
            "Parkinson's disease can affect dopamine-producing nerve cells.",
            {"title": "Parkinson's disease"},
        ),
    ]
    query = "How does Parkinson's disease affect dopamine?"

    results = BM25Retriever(documents).search(query, k=2)

    assert results[0]["document"].metadata["title"] == "Parkinson's disease"


def test_bm25_keeps_possessive_disease_names_as_single_terms():
    assert _tokenize("Alzheimer's disease") == ["alzheimer", "disease"]


def test_bm25_question_boilerplate_does_not_outrank_subject_match():
    documents = [
        BiomedicalDocument(
            "A study about definition quality and peer review in biomedical manuscripts.",
            {"title": "Manuscript peer review"},
        ),
        BiomedicalDocument(
            "Dopamine is a neurotransmitter that helps nerve cells send signals.",
            {"title": "Parkinson's disease"},
        ),
    ]

    results = BM25Retriever(documents).search("What is the function of dopamine?", k=2)

    assert results[0]["document"].metadata["title"] == "Parkinson's disease"


def test_treatment_query_is_rewritten_for_local_and_external_retrieval(monkeypatch):
    received_local_queries = []
    received_external_queries = []

    class FakeHybridRetriever:
        def __init__(self, vectorstore, documents):
            self.dense_retriever = SimpleNamespace(vectorstore=vectorstore)

        def retrieve(self, query, k):
            received_local_queries.append(query)
            return {"fused_results": [], "grounded": False}

    monkeypatch.setattr(generation_module, "is_medical_query", lambda query: True)
    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(
        generation_module,
        "retrieve_pubmed",
        lambda query, **kwargs: received_external_queries.append(query) or [],
    )
    pipeline = generation_module.GroundedGenerationPipeline(
        object(), [], use_mock=False, use_reranking=False,
    )

    pipeline.generate(
        "What evidence supports current understanding of Alzheimer's therapies work?"
    )

    assert received_local_queries == ["alzheimer disease treatment"]
    assert received_external_queries == ["alzheimer treatment therapies management"]


def test_topic_gate_excludes_unrelated_passages():
    query = "How does Parkinson's disease affect dopamine?"

    assert has_topic_overlap(query, ["Parkinson's disease affects dopamine levels."])
    assert not has_topic_overlap(query, ["H. pylori can cause peptic ulcers."])


def test_topic_gate_normalizes_parkinsons_without_apostrophe():
    query = "How does dopamine affect Parkinsons disease?"

    assert has_topic_overlap(
        query,
        ["Parkinson's disease can affect dopamine-producing nerve cells."],
    )


def test_ulcer_is_in_biomedical_scope_and_expands_to_peptic_ulcer_terms():
    from rag.medical_risk import is_medical_query
    from rag.medical_entities import expand_concepts

    query = "tell me about stomach ulcer"

    assert is_medical_query(query)
    expanded = expand_concepts(query)
    assert "peptic ulcer" in expanded
    assert "h pylori" in expanded


def test_topic_gate_rejects_greeting_instead_of_using_arbitrary_evidence():
    assert not has_topic_overlap("hi", ["Speech and language problems in children."])


def test_hybrid_retriever_filters_dense_neighbors_without_topic_match(monkeypatch):
    relevant = BiomedicalDocument(
        "Parkinson's disease can affect dopamine-producing nerve cells.",
        {"title": "Parkinson's disease"},
    )
    unrelated = RetrievedChunk(
        "H. pylori can affect the stomach and cause ulcers.",
        {"title": "Peptic ulcers"},
        0.2,
    )

    class FakeDenseRetriever:
        def __init__(self, vectorstore):
            pass

        def retrieve(self, query, k):
            return {"grounded": True, "results": [unrelated]}

    class FakeBM25Retriever:
        def __init__(self, documents):
            pass

        def search(self, query, k):
            return [{"document": relevant, "score": 1.0}]

    monkeypatch.setattr(hybrid_module, "BiomedicalRetriever", FakeDenseRetriever)
    monkeypatch.setattr(hybrid_module, "BM25Retriever", FakeBM25Retriever)

    results = hybrid_module.HybridRetriever(object(), [relevant]).retrieve(
        "How does Parkinson's disease affect dopamine?", k=3
    )

    assert results["grounded"] is True
    assert [item[0] for item in results["fused_results"]] == [relevant.content]


def test_generation_abstains_instead_of_calling_llm_without_topic_evidence(monkeypatch):
    unrelated = BiomedicalDocument(
        "H. pylori can affect the stomach and cause ulcers.",
        {"title": "Peptic ulcers"},
    )

    class FakeHybridRetriever:
        def __init__(self, vectorstore, documents):
            pass

        def retrieve(self, query, k):
            return {
                "fused_results": [(unrelated.content, 0.03)],
                "grounded": True,
            }

    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(generation_module, "retrieve_pubmed", lambda *args, **kwargs: [])
    pipeline = generation_module.GroundedGenerationPipeline(
        object(), [unrelated], use_mock=False, use_reranking=False
    )

    result = pipeline.generate("How does Parkinson's disease affect dopamine?")

    assert result["answer_abstained"] is True
    assert result["compression"]["compressed_count"] == 0
    assert result["generation"]["success"] is True


def test_generation_suppresses_a_provider_answer_about_another_topic(monkeypatch):
    relevant = BiomedicalDocument(
        "Parkinson's disease is associated with dopamine-producing neurons.",
        {"title": "Parkinson's disease"},
    )

    class FakeHybridRetriever:
        def __init__(self, vectorstore, documents):
            pass

        def retrieve(self, query, k):
            return {
                "fused_results": [(relevant.content, 0.03)],
                "grounded": True,
            }

    def off_topic_answer(self, prompt, max_tokens=500, temperature=0.3):
        return {
            "answer": "The main cause of peptic ulcers is H. pylori [Evidence 1].",
            "model": "command-a-plus-05-2026",
            "success": True,
        }

    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(generation_module, "retrieve_pubmed", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        generation_module.GroundedGenerator,
        "generate",
        off_topic_answer,
    )
    pipeline = generation_module.GroundedGenerationPipeline(
        object(), [relevant], use_mock=False, use_reranking=False
    )

    result = pipeline.generate("How does Parkinson's disease affect dopamine?")

    assert result["answer_abstained"] is True
    assert result["compression"]["compressed_count"] == 0
    assert "not contain enough relevant evidence" in result["final_answer"]


def test_diabetes_symptom_query_with_only_indirect_passages_explains_evidence_gap(monkeypatch):
    documents = [
        BiomedicalDocument(
            "Prediabetes occurs when blood glucose is higher than normal.",
            {"source": "MedlinePlus", "title": "Prediabetes"},
        ),
        BiomedicalDocument(
            "Diabetes-related eye problems include diabetic retinopathy and vision loss.",
            {"source": "MedlinePlus", "title": "Diabetic eye problems"},
        ),
        BiomedicalDocument(
            "Retinal disorders can lead to impaired vision.",
            {"source": "MedlinePlus", "title": "Retinal disorders"},
        ),
    ]

    class FakeHybridRetriever:
        def __init__(self, vectorstore, received_documents):
            assert received_documents == documents
            self.dense_retriever = SimpleNamespace(vectorstore=vectorstore)

        def retrieve(self, query, k):
            assert "manifestation" in query
            return {"fused_results": [(doc.content, 1.0) for doc in documents], "grounded": True}

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("Generation must not run without direct symptom evidence")

    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(generation_module, "retrieve_pubmed", lambda *args, **kwargs: [])
    monkeypatch.setattr(generation_module.GroundedGenerator, "generate", unexpected_generation)
    pipeline = generation_module.GroundedGenerationPipeline(
        object(), documents, use_mock=False, use_reranking=False,
    )

    result = pipeline.generate("What are common symptoms of diabetes?")

    assert result["decision"] == "informational"
    assert result["answer_abstained"] is False
    assert result["compression"]["compressed_count"] == 3
    classifications = [item["evidence_classification"]["category"] for item in result["compression"]["evidence"]]
    assert "related_indirect" in classifications
    assert "complication" in classifications
    assert result["audit_trace"]["evidence_audit"]["relevance_counts"]["direct"] == 0
    assert "do not directly establish" in result["final_answer"]
    assert result["rejection_reason"] == "evidence_coverage_limited"


def test_mixed_diabetes_retrieval_promotes_direct_passages_before_top_k(monkeypatch):
    documents = [
        BiomedicalDocument(
            "Prediabetes occurs when blood glucose is higher than normal.",
            {"source": "MedlinePlus", "title": "Prediabetes"},
        ),
        BiomedicalDocument(
            "Diabetes-related eye complications include diabetic retinopathy and vision loss.",
            {"source": "MedlinePlus", "title": "Diabetic eye complications"},
        ),
        BiomedicalDocument(
            "Common symptoms of diabetes include increased thirst and frequent urination.",
            {"source": "MedlinePlus", "title": "Diabetes symptoms"},
        ),
        BiomedicalDocument(
            "Symptoms of diabetes can include unexplained weight loss.",
            {"source": "MedlinePlus", "title": "Diabetes symptoms overview"},
        ),
    ]

    class FakeHybridRetriever:
        def __init__(self, vectorstore, received_documents):
            self.dense_retriever = SimpleNamespace(vectorstore=vectorstore)

        def retrieve(self, query, k):
            return {"fused_results": [(doc.content, 1.0 - index * 0.01) for index, doc in enumerate(documents)], "grounded": True}

    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(generation_module, "retrieve_pubmed", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        generation_module.GroundedGenerator,
        "generate",
        lambda *args, **kwargs: {
            "answer": "Common symptoms of diabetes include increased thirst and frequent urination [Evidence 1].",
            "success": True,
            "model": "controlled-test",
        },
    )
    pipeline = generation_module.GroundedGenerationPipeline(
        object(), documents, use_mock=False, use_reranking=False,
    )

    result = pipeline.generate("What are common symptoms of diabetes?", k=2, retrieval_k=4)

    assert result["decision"] in {"answer_approved", "informational"}
    assert len(result["compression"]["evidence"]) == 2
    assert all(item["evidence_classification"]["category"] == "direct" for item in result["compression"]["evidence"])
    assert "weight loss" not in result["final_answer"].lower()


def test_hybrid_ranking_prefers_systemic_hypertension_over_pulmonary(monkeypatch):
    high_bp = BiomedicalDocument(
        "High blood pressure develops over time and can have primary or secondary causes.",
        {"title": "High Blood Pressure", "source": "MedlinePlus"},
    )
    pulmonary = BiomedicalDocument(
        "Pulmonary hypertension affects blood vessels in the lungs.",
        {"title": "Pulmonary Hypertension", "source": "MedlinePlus"},
    )

    class FakeDenseRetriever:
        def __init__(self, vectorstore):
            pass

        def retrieve(self, query, k):
            return {"results": [
                RetrievedChunk(pulmonary.content, pulmonary.metadata, 0.01),
                RetrievedChunk(high_bp.content, high_bp.metadata, 0.2),
            ]}

    class FakeSparseRetriever:
        def __init__(self, documents):
            pass

        def search(self, query, k):
            return [
                {"document": pulmonary, "score": 8.0},
                {"document": high_bp, "score": 3.0},
            ]

    monkeypatch.setattr(hybrid_module, "BiomedicalRetriever", FakeDenseRetriever)
    monkeypatch.setattr(hybrid_module, "BM25Retriever", FakeSparseRetriever)
    response = hybrid_module.HybridRetriever(object(), [high_bp, pulmonary]).retrieve_many(
        ["What causes hypertension?", "high blood pressure causes etiology"],
        "What causes hypertension?",
        k=2,
    )

    assert response["fused_results"][0][0] == high_bp.content
    assert response["candidate_diagnostics"][high_bp.content]["entity_match"] in {"exact_concept", "synonym"}
    pulmonary_diagnostic = response["candidate_diagnostics"].get(pulmonary.content)
    assert pulmonary_diagnostic is None or pulmonary_diagnostic["entity_match"] == "related_condition"


def test_hypertension_question_returns_limited_context_when_only_risk_and_pulmonary_sources_exist(monkeypatch):
    documents = [
        BiomedicalDocument(
            "Pulmonary hypertension affects blood vessels in the lungs.",
            {"source": "MedlinePlus", "title": "Pulmonary Hypertension"},
        ),
        BiomedicalDocument(
            "The risk of high blood pressure increases as you get older.",
            {"source": "MedlinePlus", "title": "High Blood Pressure"},
        ),
        BiomedicalDocument(
            "Lifestyle changes can help prevent high blood pressure.",
            {"source": "MedlinePlus", "title": "How to Prevent High Blood Pressure"},
        ),
    ]

    class FakeHybridRetriever:
        def __init__(self, vectorstore, received_documents):
            self.dense_retriever = SimpleNamespace(vectorstore=vectorstore)

        def retrieve(self, query, k):
            return {"fused_results": [(doc.content, 1.0) for doc in documents], "grounded": True}

    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(generation_module, "retrieve_pubmed", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        generation_module.GroundedGenerator, "generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No direct cause evidence; do not invent an answer")),
    )
    pipeline = generation_module.GroundedGenerationPipeline(
        object(), documents, use_mock=False, use_reranking=False,
    )

    result = pipeline.generate("What causes hypertension?")

    assert result["decision"] == "informational"
    assert result["evidence_confidence"] == "limited"
    assert "risk" in result["final_answer"].lower()
    assert "related but distinct condition" in result["final_answer"]
    assert result["audit_trace"]["evidence_audit"]["coverage_assessment"]["status"] == "insufficient"


def test_generic_diabetes_context_does_not_trigger_a_complete_comparison_answer(monkeypatch):
    document = BiomedicalDocument(
        "Diabetes is a disease in which blood glucose levels are too high.",
        {"source": "MedlinePlus", "title": "Diabetes"},
    )

    class FakeHybridRetriever:
        def __init__(self, vectorstore, documents):
            self.dense_retriever = SimpleNamespace(vectorstore=vectorstore)

        def retrieve(self, query, k):
            return {"fused_results": [(document.content, 1.0)], "grounded": True}

    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(generation_module, "retrieve_pubmed", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        generation_module.GroundedGenerator, "generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Generic diabetes context cannot support a comparison.")
        ),
    )
    pipeline = generation_module.GroundedGenerationPipeline(
        object(), [document], use_mock=False, use_reranking=False,
    )

    result = pipeline.generate("Compare type 1 vs type 2 diabetes.", answer_mode="learning")

    assert result["decision"] == "informational"
    assert result["evidence_confidence"] == "limited"
    assert result["audit_trace"]["evidence_audit"]["coverage_assessment"]["status"] == "insufficient"
    assert "type 1 diabetes" in result["final_answer"]
    assert "type 2 diabetes" in result["final_answer"]


def test_direct_high_blood_pressure_cause_evidence_can_generate_cited_answer(monkeypatch):
    document = BiomedicalDocument(
        "Secondary high blood pressure is caused by another medical condition or medicine.",
        {"source": "MedlinePlus", "title": "High Blood Pressure"},
    )

    class FakeHybridRetriever:
        def __init__(self, vectorstore, documents):
            self.dense_retriever = SimpleNamespace(vectorstore=vectorstore)

        def retrieve(self, query, k):
            return {"fused_results": [(document.content, 1.0)], "grounded": True}

    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(generation_module, "retrieve_pubmed", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        generation_module.GroundedGenerator, "generate",
        lambda *args, **kwargs: {
            "answer": "Secondary high blood pressure can be caused by another medical condition or medicine [Evidence 1].",
            "success": True,
            "model": "controlled-test",
        },
    )
    pipeline = generation_module.GroundedGenerationPipeline(
        object(), [document], use_mock=False, use_reranking=False,
    )

    result = pipeline.generate("What causes hypertension?")

    assert result["decision"] in {"answer_approved", "informational"}
    assert "caused by another medical condition or medicine" in result["final_answer"]
    assert result["validation"]["claim_support"][0]["support_status"] == "supported"


def test_cancer_cause_answer_synthesizes_mechanism_inheritance_and_risk_factors(monkeypatch):
    content = (
        "Cancer is a genetic disease caused by changes (mutations) in genes that affect cell function. "
        "Some mutations are inherited, while others develop during a person's lifetime. "
        "Risk factors include ultraviolet radiation, smoking, diet, physical inactivity, and aging."
    )
    document = BiomedicalDocument(content, {"source": "MedlinePlus", "title": "Cancer"})
    captured = {}

    class FakeHybridRetriever:
        def __init__(self, vectorstore, documents):
            self.dense_retriever = SimpleNamespace(vectorstore=vectorstore)

        def retrieve(self, query, k):
            return {"fused_results": [(document.content, 1.0)], "grounded": True}

    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(generation_module, "retrieve_pubmed", lambda *args, **kwargs: [])

    def generate(self, prompt, **kwargs):
        captured["prompt"] = prompt
        return {
            "answer": (
                "Cancer is a genetic disease caused by changes (mutations) in genes that affect cell function [Evidence 1]. "
                "Some mutations are inherited, while others develop during a person's lifetime [Evidence 1]. "
                "Risk factors include ultraviolet radiation, smoking, diet, physical inactivity, and aging [Evidence 1]."
            ),
            "success": True,
            "model": "controlled-test",
        }

    monkeypatch.setattr(generation_module.GroundedGenerator, "generate", generate)
    pipeline = generation_module.GroundedGenerationPipeline(
        object(), [document], use_mock=False, use_reranking=False,
    )

    result = pipeline.generate("What is the cause of cancer?", answer_mode="learning")

    assert "2-4 clear sentences" in captured["prompt"]["system"]
    assert "risk factors" in captured["prompt"]["system"].lower()
    assert result["decision"] in {"answer_approved", "informational"}
    assert not result["answer_abstained"]
    assert "genetic disease caused by changes" in result["final_answer"]
    assert "Risk factors include" in result["final_answer"]
    assert all(claim["support_status"] == "supported" for claim in result["validation"]["claim_support"])


def test_primary_and_secondary_hypertension_evidence_supports_cited_causal_claims(monkeypatch):
    documents = [
        BiomedicalDocument(
            "Primary, or essential, high blood pressure is the most common type. "
            "For most people who get this kind, it develops over time as you get older.",
            {"source": "MedlinePlus", "title": "High Blood Pressure"},
        ),
        BiomedicalDocument(
            "Secondary high blood pressure is caused by another medical condition "
            "or use of certain medicines.",
            {"source": "MedlinePlus", "title": "High Blood Pressure"},
        ),
    ]

    class FakeHybridRetriever:
        def __init__(self, vectorstore, received_documents):
            self.dense_retriever = SimpleNamespace(vectorstore=vectorstore)

        def retrieve(self, query, k):
            return {
                "fused_results": [(document.content, 1.0) for document in documents],
                "grounded": True,
            }

    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(generation_module, "retrieve_pubmed", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        generation_module.GroundedGenerator, "generate",
        lambda *args, **kwargs: {
            "answer": (
                "Primary high blood pressure develops over time as people get older "
                "[Evidence 1]. Secondary high blood pressure is caused by another "
                "medical condition or use of certain medicines [Evidence 2]."
            ),
            "success": True,
            "model": "controlled-test",
        },
    )
    pipeline = generation_module.GroundedGenerationPipeline(
        object(), documents, use_mock=False, use_reranking=False,
    )

    result = pipeline.generate("What causes hypertension?")

    assert result["audit_trace"]["evidence_audit"]["coverage_assessment"]["status"] == "complete"
    assert result["decision"] in {"answer_approved", "informational"}
    assert len(result["validation"]["claim_support"]) == 2
    assert all(
        claim["support_status"] == "supported"
        for claim in result["validation"]["claim_support"]
    )
    assert "[Evidence 1]" in result["final_answer"]
    assert "[Evidence 2]" in result["final_answer"]


def test_incomplete_direct_passage_only_supports_the_reported_symptoms(monkeypatch):
    document = BiomedicalDocument(
        "Diabetes symptoms can include increased thirst and frequent urination.",
        {"source": "MedlinePlus", "title": "Diabetes symptoms"},
    )

    class FakeHybridRetriever:
        def __init__(self, vectorstore, documents):
            self.dense_retriever = SimpleNamespace(vectorstore=vectorstore)

        def retrieve(self, query, k):
            return {"fused_results": [(document.content, 1.0)], "grounded": True}

    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(generation_module, "retrieve_pubmed", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        generation_module.GroundedGenerator,
        "generate",
        lambda *args, **kwargs: {
            "answer": "The passage identifies increased thirst and frequent urination as possible diabetes symptoms [Evidence 1].",
            "success": True,
            "model": "controlled-test",
        },
    )
    pipeline = generation_module.GroundedGenerationPipeline(
        object(), [document], use_mock=False, use_reranking=False,
    )

    result = pipeline.generate("What are common symptoms of diabetes?")

    assert result["decision"] in {"answer_approved", "informational"}
    assert "increased thirst" in result["final_answer"].lower()
    assert "frequent urination" in result["final_answer"].lower()
    assert "weight loss" not in result["final_answer"].lower()
