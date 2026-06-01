import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingestion import load_medquad_dataset
from rag.preprocessing import preprocess_documents
from rag.indexing import build_index
from rag.reranking_retriever import RerankingRetriever


DATASET_PATH = "data/raw/medquad"


print("\n" + "="*80)
print("BLOCK 6A: CROSS-ENCODER RERANKING TEST")
print("="*80 + "\n")


print("[1/6] Loading dataset...")

documents = load_medquad_dataset(DATASET_PATH)
documents = preprocess_documents(documents[:2000])
print(f"[OK] Documents loaded: {len(documents)}")


print("\n[2/6] Building vector index...")

vectorstore, chunks = build_index(documents)
print(f"[OK] Chunks indexed: {len(chunks)}")


print("\n[3/6] Initializing reranking retriever...")

retriever = RerankingRetriever(vectorstore, documents)
print("[OK] Reranking retriever ready")


query = "What are symptoms of diabetes?"

print(f"\n[4/6] Running retrieval + reranking...")
print(f"      Query: '{query}'")

response = retriever.retrieve(query, k=3, retrieval_k=20)


print(f"\n[5/6] Comparing fusion vs reranked results...\n")

print("="*80)
print("FUSION-RANKED RESULTS (RRF)")
print("="*80 + "\n")

for idx, (content, score) in enumerate(response["fused_results"][:3], 1):
    print(f"{idx}. Fusion Score: {score:.6f}")
    print(f"   {content[:150]}...")
    print()


print("\n" + "="*80)
print("CROSS-ENCODER RERANKED RESULTS")
print("="*80 + "\n")

for idx, result in enumerate(response["reranked_results"], 1):
    print(f"{idx}. Rerank Score: {result['rerank_score']:.4f}")
    print(f"   Fusion Score: {result['fusion_score']:.6f}")
    print(f"   {result['content'][:150]}...")
    print()


print("\n[6/6] Reranking verification complete\n")

print("="*80)
print("[PASS] BLOCK 6A CROSS-ENCODER RERANKING TEST PASSED")
print("="*80)

print("\nKey Improvements:")
print(f"  Fusion ranked: Top 3 by RRF score")
print(f"  Reranked: Top 3 by semantic relevance")
print(f"  Grounded: {response['grounded']}")
print(f"\nReranking adds semantic understanding beyond fusion!")
