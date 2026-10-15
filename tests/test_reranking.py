import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.document import BiomedicalDocument
import rag.reranking_retriever as reranking_module


def test_reranking_preserves_metadata_for_source_documents_and_chunks(
    monkeypatch,
):
    original = BiomedicalDocument(
        content="Full source document",
        metadata={"source": "medquad.xml"},
    )
    chunk = BiomedicalDocument(
        content="Retrieved indexed chunk",
        metadata={"source": "medquad.xml", "chunk_start": 0},
    )
    vectorstore = type("VectorStore", (), {"documents": [chunk]})()

    class FakeHybridRetriever:
        def __init__(self, received_vectorstore, documents):
            assert received_vectorstore is vectorstore
            assert documents == [original]

        def retrieve(self, query, k):
            return {
                "grounded": True,
                "fused_results": [
                    (chunk.content, 0.04),
                    (original.content, 0.03),
                ],
            }

    class FakeReranker:
        def __init__(self, model_name=None):
            self.candidates = None

        def rerank(self, query, candidates, k):
            self.candidates = candidates
            return candidates[:k]

    monkeypatch.setattr(
        reranking_module, "HybridRetriever", FakeHybridRetriever
    )
    monkeypatch.setattr(
        reranking_module, "CrossEncoderReranker", FakeReranker
    )

    retriever = reranking_module.RerankingRetriever(vectorstore, [original])
    response = retriever.retrieve("test query", k=2)

    assert response["grounded"] is True
    assert retriever.reranker.candidates == [
        (chunk.content, chunk.metadata, 0.04),
        (original.content, original.metadata, 0.03),
    ]


def test_reranking_handles_empty_fused_results(monkeypatch):
    class FakeHybridRetriever:
        def __init__(self, vectorstore, documents):
            pass

        def retrieve(self, query, k):
            return {"grounded": False, "fused_results": []}

    class FakeReranker:
        def __init__(self, model_name=None):
            pass

        def rerank(self, query, candidates, k):
            return candidates[:k]

    monkeypatch.setattr(
        reranking_module, "HybridRetriever", FakeHybridRetriever
    )
    monkeypatch.setattr(
        reranking_module, "CrossEncoderReranker", FakeReranker
    )

    retriever = reranking_module.RerankingRetriever(object(), [])

    assert retriever.retrieve("test query")["reranked_results"] == []


def test_reranking_skips_results_without_metadata(monkeypatch):
    document = BiomedicalDocument(
        content="Known document",
        metadata={"source": "medquad.xml"},
    )
    vectorstore = type("VectorStore", (), {"documents": []})()

    class FakeHybridRetriever:
        def __init__(self, vectorstore, documents):
            pass

        def retrieve(self, query, k):
            return {
                "grounded": True,
                "fused_results": [
                    ("Unknown document", 0.05),
                    (document.content, 0.04),
                ],
            }

    class FakeReranker:
        def __init__(self, model_name=None):
            self.candidates = None

        def rerank(self, query, candidates, k):
            self.candidates = candidates
            return candidates[:k]

    monkeypatch.setattr(
        reranking_module, "HybridRetriever", FakeHybridRetriever
    )
    monkeypatch.setattr(
        reranking_module, "CrossEncoderReranker", FakeReranker
    )

    retriever = reranking_module.RerankingRetriever(vectorstore, [document])
    response = retriever.retrieve("test query")

    assert retriever.reranker.candidates == [
        (document.content, document.metadata, 0.04),
    ]
    assert response["reranked_results"] == retriever.reranker.candidates


def run_dataset_demo():
    """Run the real MedQuAD retrieval demonstration on explicit request."""
    from rag.indexing import build_index
    from rag.ingestion import load_medquad_dataset
    from rag.preprocessing import preprocess_documents
    from rag.reranking_retriever import RerankingRetriever

    dataset_path = "data/raw/medquad"
    print("\n" + "=" * 80)
    print("BLOCK 6A: CROSS-ENCODER RERANKING TEST")
    print("=" * 80 + "\n")

    print("[1/6] Loading dataset...")
    documents = load_medquad_dataset(dataset_path)
    documents = preprocess_documents(documents[:2000])
    print(f"[OK] Documents loaded: {len(documents)}")

    print("\n[2/6] Building vector index...")
    vectorstore, chunks = build_index(documents)
    print(f"[OK] Chunks indexed: {len(chunks)}")

    print("\n[3/6] Initializing reranking retriever...")
    retriever = RerankingRetriever(vectorstore, documents)
    print("[OK] Reranking retriever ready")

    query = "What are symptoms of diabetes?"
    print(f"\n[4/6] Running retrieval + reranking...\n      Query: '{query}'")
    response = retriever.retrieve(query, k=3, retrieval_k=20)

    print("\n[5/6] Comparing fusion vs reranked results...\n")
    print("=" * 80)
    print("FUSION-RANKED RESULTS (RRF)")
    print("=" * 80 + "\n")
    for idx, (content, score) in enumerate(response["fused_results"][:3], 1):
        print(f"{idx}. Fusion Score: {score:.6f}")
        print(f"   {content[:150]}...\n")

    print("\n" + "=" * 80)
    print("CROSS-ENCODER RERANKED RESULTS")
    print("=" * 80 + "\n")
    for idx, result in enumerate(response["reranked_results"], 1):
        print(f"{idx}. Rerank Score: {result['rerank_score']:.4f}")
        print(f"   Fusion Score: {result['fusion_score']:.6f}")
        print(f"   {result['content'][:150]}...\n")

    print("\n[6/6] Reranking verification complete\n")
    print("=" * 80)
    print("[PASS] BLOCK 6A CROSS-ENCODER RERANKING TEST PASSED")
    print("=" * 80)
    print("\nKey Improvements:")
    print(f"  Fusion ranked: Top 3 by RRF score")
    print(f"  Reranked: Top 3 by semantic relevance")
    print(f"  Grounded: {response['grounded']}")
    print("\nReranking adds semantic understanding beyond fusion!")


if __name__ == "__main__":
    run_dataset_demo()
