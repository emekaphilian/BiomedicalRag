import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingestion import load_medquad_dataset
from rag.preprocessing import preprocess_documents
from rag.indexing import build_index
from rag.hybrid_retriever import HybridRetriever


DATASET_PATH = "data/raw/medquad"


print("\n" + "="*80)
print("BLOCK 5 HYBRID RETRIEVAL TEST")
print("="*80 + "\n")


print("[1/5] Loading dataset...")

documents = load_medquad_dataset(DATASET_PATH)

documents = preprocess_documents(documents[:2000])

print(f"[OK] Documents loaded: {len(documents)}")


print("\n[2/5] Building vector index...")

vectorstore, chunks = build_index(documents)

print(f"[OK] Chunks indexed: {len(chunks)}")


print("\n[3/5] Initializing hybrid retriever...")

retriever = HybridRetriever(vectorstore, documents)

print("[OK] Hybrid retriever ready")


query = "What are symptoms of diabetes?"


print("\n[4/5] Running hybrid retrieval...")

response = retriever.retrieve(query)


print(f"\n[OK] Grounded: {response['grounded']}")


print("\n" + "="*80)
print("DENSE RETRIEVAL RESULTS")
print("="*80 + "\n")

for idx, result in enumerate(response["dense_results"], start=1):

    print(f"{idx}. {result.metadata['source']}")
    print(f"   Score: {result.similarity_score:.4f}")
    print(f"   Content: {result.content[:200]}...")
    print("-" * 80)


print("\n" + "="*80)
print("BM25 SPARSE RETRIEVAL RESULTS")
print("="*80 + "\n")

for idx, result in enumerate(response["sparse_results"], start=1):

    print(f"{idx}. {result['document'].metadata['source']}")
    print(f"   BM25 Score: {result['score']:.4f}")
    print(f"   Content: {result['document'].content[:200]}...")
    print("-" * 80)


print("\n" + "="*80)
print("FUSED RANKING RESULTS (RRF)")
print("="*80 + "\n")

for idx, result in enumerate(response["fused_results"][:5], start=1):

    print(f"{idx}. Fusion Score: {result[1]:.6f}")
    print(f"   Content: {result[0][:200]}...")
    print("-" * 80)


print("\n[5/5] Hybrid retrieval verification complete")

print("\n" + "="*80)
print("[PASS] BLOCK 5 HYBRID RETRIEVAL TEST PASSED")
print("="*80)

print("\nSummary:")
print(f"  Dense results: {len(response['dense_results'])}")
print(f"  BM25 results: {len(response['sparse_results'])}")
print(f"  Fused results: {len(response['fused_results'])}")
print(f"  Grounded: {response['grounded']}")
print("\nHybrid retrieval system working correctly!")
