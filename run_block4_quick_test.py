#!/usr/bin/env python
"""
BLOCK 4: QUICK VALIDATION TEST
Tests with a limited subset to verify pipeline works
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.ingestion import load_medquad_dataset
from rag.preprocessing import preprocess_documents
from rag.indexing import build_index
from rag.retriever import BiomedicalRetriever

print("="*80)
print("BLOCK 4: QUICK VALIDATION TEST")
print("="*80)

DATASET_PATH = "data/raw/medquad"

try:
    # Load dataset  
    print("\n[1/5] Loading MedQuAD dataset...")
    all_documents = load_medquad_dataset(DATASET_PATH)
    print(f"[OK] Total available documents: {len(all_documents)}")
    
    # Use only first 1000 documents for quick test
    documents = all_documents[:1000]
    print(f"[OK] Using first {len(documents)} documents for quick validation")

    # Preprocess
    print(f"\n[2/5] Preprocessing...")
    documents = preprocess_documents(documents)

    # Build index
    print(f"\n[3/5] Building vector index...")
    vectorstore, chunks = build_index(documents)
    print(f"[OK] Created {len(chunks)} chunks")

    # Initialize retriever
    print(f"\n[4/5] Initializing retriever...")
    retriever = BiomedicalRetriever(vectorstore)

    # Test queries
    print(f"\n[5/5] Testing retrieval...")
    queries = ["diabetes symptoms", "cancer treatment"]
    
    for query in queries:
        response = retriever.retrieve(query, k=2)
        print(f"  Query: '{query}' -> {len(response['results'])} results, grounded={response['grounded']}")

    print("\n" + "="*80)
    print("[PASS] BLOCK 4 VALIDATION SUCCESSFUL")
    print("="*80)
    print(f"\nResults:")
    print(f"  Full dataset size: {len(all_documents)} documents")
    print(f"  Test subset: {len(documents)} documents")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Ingestion: WORKING")
    print(f"  Preprocessing: WORKING")
    print(f"  Vectorization: WORKING")
    print(f"  Retrieval: WORKING")
    
except Exception as e:
    print(f"\n[FAIL] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
