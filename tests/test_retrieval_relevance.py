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
    assert "could not produce an answer" in result["final_answer"]
