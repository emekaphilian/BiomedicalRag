from types import SimpleNamespace

from rag.bm25_retriever import BM25Retriever
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


def test_topic_gate_excludes_unrelated_passages():
    query = "How does Parkinson's disease affect dopamine?"

    assert has_topic_overlap(query, ["Parkinson's disease affects dopamine levels."])
    assert not has_topic_overlap(query, ["H. pylori can cause peptic ulcers."])


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


def test_diabetes_symptom_query_with_only_indirect_passages_withholds_answer(monkeypatch):
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

    assert result["decision"] == "insufficient_evidence"
    assert result["answer_abstained"] is True
    assert result["compression"]["compressed_count"] == 3
    classifications = [item["evidence_classification"]["category"] for item in result["compression"]["evidence"]]
    assert "related_indirect" in classifications
    assert "complication" in classifications
    assert result["audit_trace"]["evidence_audit"]["relevance_counts"]["direct"] == 0
    assert "do not establish the central answer" in result["final_answer"]


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
