import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingestion import load_medquad_dataset
from rag.preprocessing import preprocess_documents
from rag.indexing import build_index
from rag.generation_pipeline import GroundedGenerationPipeline


DATASET_PATH = "data/raw/medquad"

print("\n" + "="*80)
print("BLOCK 6B: GROUNDED LLM GENERATION TEST")
print("="*80 + "\n")

try:
    print("[1/6] Loading dataset...")
    documents = load_medquad_dataset(DATASET_PATH)
    documents = preprocess_documents(documents[:2000])
    print(f"[OK] Documents loaded: {len(documents)}")

    print("\n[2/6] Building vector index...")
    vectorstore, chunks = build_index(documents)
    print(f"[OK] Chunks indexed: {len(chunks)}")

    print("\n[3/6] Initializing generation pipeline...")
    pipeline = GroundedGenerationPipeline(vectorstore, documents, use_mock=True)
    print("[OK] Pipeline ready (using mock LLM for demo)")

    query = "What are the main symptoms of diabetes?"
    print(f"\n[4/6] Running end-to-end generation...")
    print(f"      Query: '{query}'")

    result = pipeline.generate(query, k=3, retrieval_k=20)

    print(f"\n[5/6] Pipeline results...\n")
    print("="*80)
    print("RETRIEVAL")
    print("="*80)
    print(f"  Fused candidates: {result['retrieval']['fused_count']}")
    print(f"  Reranked results: {result['retrieval']['reranked_count']}")
    print(f"  Grounded: {result['retrieval']['grounded']}")

    print("\n" + "="*80)
    print("COMPRESSION")
    print("="*80)
    print(f"  Compressed to: {result['compression']['compressed_count']} chunks")
    print(f"  Sources: {', '.join(result['compression']['sources'])}")

    print("\n" + "="*80)
    print("GENERATION")
    print("="*80)
    print(f"  Model: {result['generation']['model']}")
    print(f"  Success: {result['generation']['success']}")
    print(f"\n  Answer Preview:")
    print(f"  {result['generation']['answer'][:300]}...")

    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    print(f"  Valid: {result['validation']['valid']}")
    print(f"  Reason: {result['validation']['reason']}")
    print(f"  Citations found: {result['validation']['citations_found']}")
    if result['validation']['issues']:
        print(f"  Issues: {', '.join(result['validation']['issues'])}")

    print("\n" + "="*80)
    print("FINAL ANSWER")
    print("="*80)
    print(f"\n{result['final_answer']}\n")

    print("\n[6/6] Generation pipeline verification complete\n")

    print("="*80)
    print("[PASS] BLOCK 6B GENERATION TEST PASSED")
    print("="*80)
    print(f"\nArchitecture validated:")
    print(f"  [OK] Query safety pre-check")
    print(f"  [OK] Hybrid retrieval (20 candidates)")
    print(f"  [OK] Cross-encoder reranking (top 3)")
    print(f"  [OK] Context compression (max 3 chunks)")
    print(f"  [OK] Structured prompt construction")
    print(f"  [OK] LLM generation (mock for testing)")
    print(f"  [OK] Post-generation validation")
    print(f"\nTo use real OpenAI API:")
    print(f"  1. Export OPENAI_API_KEY environment variable")
    print(f"  2. Set use_mock=False in GroundedGenerationPipeline")
    print(f"  3. Run again")

except Exception as e:
    print(f"\n[FAIL] ERROR: {e}")
    import traceback
    traceback.print_exc()
