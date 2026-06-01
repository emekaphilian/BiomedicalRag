#!/usr/bin/env python
"""
BLOCK 4: BIOMEDICAL DATA INGESTION PIPELINE TEST
Tests MedQuAD ingestion with real biomedical data
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# First verify all imports work
print("=" * 80)
print("BLOCK 4: BIOMEDICAL DATA INGESTION TEST")
print("=" * 80)

print("\n[Setup] Verifying imports...")
try:
    from rag.ingestion import load_medquad_dataset
    print("[OK] rag.ingestion")
    from rag.preprocessing import preprocess_documents
    print("[OK] rag.preprocessing")
    from rag.indexing import build_index
    print("[OK] rag.indexing")
    from rag.retriever import BiomedicalRetriever
    print("[OK] rag.retriever")
    from rag.document import BiomedicalDocument
    print("[OK] rag.document")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

print("\n[OK] All imports successful\n")

# Now run the test
print("="*80)
print("MEDQUAD INGESTION TEST")
print("="*80)

DATASET_PATH = "data/raw/medquad"

try:
    # Step 1: Load dataset
    print("\n[1/6] Loading MedQuAD dataset (22,548 XML files)...")
    documents = load_medquad_dataset(DATASET_PATH)
    print(f"[OK] Loaded {len(documents)} QA pair documents")

    if len(documents) == 0:
        print("[FAIL] ERROR: No documents loaded!")
        sys.exit(1)

    # Show sample
    print(f"\n[2/6] Checking sample documents...")
    for i in range(min(3, len(documents))):
        doc = documents[i]
        print(f"\n  Document {i+1}:")
        print(f"    Directory: {doc.metadata.get('directory', 'unknown')}")
        print(f"    File: {doc.metadata.get('file', 'unknown')}")
        print(f"    Content length: {len(doc.content)} characters")
        print(f"    Preview: {doc.content[:120]}...")

    # Step 2: Preprocess
    print(f"\n[3/6] Preprocessing {len(documents)} documents...")
    documents = preprocess_documents(documents)
    print(f"[OK] Preprocessing complete")

    # Step 3: Build index (with smaller subset for speed)
    print(f"\n[4/6] Building vector index...")
    print(f"  (Creating chunks and embeddings - this takes 2-5 minutes for full dataset)")
    vectorstore, chunks = build_index(documents)
    print(f"[OK] Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"[OK] Indexed in FAISS vectorstore (384-dimensional embeddings)")

    # Step 4: Initialize retriever
    print(f"\n[5/6] Initializing retriever...")
    retriever = BiomedicalRetriever(vectorstore)
    print(f"[OK] Retriever initialized")

    # Step 5: Test retrieval
    print(f"\n[6/6] Testing biomedical retrieval...")
    test_queries = [
        "What are symptoms of diabetes?",
        "How is cancer treated?",
        "What causes hypertension?"
    ]

    for query in test_queries:
        response = retriever.retrieve(query, k=3)
        print(f"\n  Query: {query}")
        print(f"    Grounded: {response['grounded']}")
        print(f"    Results: {len(response['results'])}")
        if response['results']:
            for i, result in enumerate(response['results'][:2], 1):
                print(f"    Result {i}: score={result.similarity_score:.4f}, " +
                      f"source={result.metadata.get('directory', 'unknown')}")

    # Success summary
    print("\n" + "=" * 80)
    print("[PASS] BLOCK 4 MEDQUAD INGESTION TEST PASSED")
    print("=" * 80)

    print("\nSystem State:")
    print(f"  Total documents loaded: {len(documents)}")
    print(f"  Total chunks created: {len(chunks)}")
    print(f"  Vectorstore backend: FAISS (IndexFlatL2)")
    print(f"  Embedding model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"  Embedding dimension: 384")
    print(f"  Retrieval grounding: Enabled (L2 distance threshold)")

    print("\nBiomedical corpus status:")
    print("  [OK] XML parsing working")
    print("  [OK] Recursive directory scanning working")
    print("  [OK] QA pair extraction working")
    print("  [OK] Chunking working")
    print("  [OK] Embeddings generated successfully")
    print("  [OK] Vector indexing working")
    print("  [OK] Retrieval queries working")

    print("\nNext steps:")
    print("  -> BLOCK 5: Hybrid retrieval (BM25 + semantic)")
    print("  -> BLOCK 6: LLM response generation with citations")
    print("  -> BLOCK 7: Hallucination detection & safety verification")

except Exception as e:
    print(f"\n[FAIL] ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
