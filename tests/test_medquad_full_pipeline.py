import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingestion import load_medquad_dataset
from rag.preprocessing import preprocess_documents
from rag.indexing import build_index
from rag.retriever import BiomedicalRetriever


DATASET_PATH = "data/raw/medquad"


print("\n" + "="*80)
print("BLOCK 4: MEDQUAD PIPELINE TEST")
print("="*80)

try:
    print("\n[1/7] Loading MedQuAD dataset...")
    documents = load_medquad_dataset(DATASET_PATH)
    print(f"✓ Loaded {len(documents)} documents\n")

    if len(documents) == 0:
        print("ERROR: No documents loaded!")
        sys.exit(1)

    print("[2/7] Checking document samples...")
    for i, doc in enumerate(documents[:3]):
        print(f"  Doc {i+1}:")
        print(f"    Source: {doc.metadata['source']}")
        print(f"    File: {doc.metadata['file']}")
        print(f"    Directory: {doc.metadata['directory']}")
        print(f"    Content length: {len(doc.content)} chars")

    print("\n[3/7] Preprocessing documents...")
    documents = preprocess_documents(documents)
    print(f"✓ Preprocessed {len(documents)} documents")

    print("\n[4/7] Building vector index...")
    print("(This may take a minute with 22k+ documents...)")
    vectorstore, chunks = build_index(documents)
    print(f"✓ Created {len(chunks)} chunks")
    print(f"✓ Indexed in FAISS vectorstore")

    print("\n[5/7] Initializing retriever...")
    retriever = BiomedicalRetriever(vectorstore)
    print("✓ Retriever ready")

    # Test multiple queries
    test_queries = [
        "What are symptoms of diabetes?",
        "How is cancer treated?",
        "What causes hypertension?"
    ]

    print("\n[6/7] Running biomedical retrieval queries...")
    for query in test_queries:
        print(f"\n  Query: {query}")
        response = retriever.retrieve(query, k=3)
        print(f"  Grounded: {response['grounded']}")
        print(f"  Results: {len(response['results'])}")
        
        if response['results']:
            best = response['results'][0]
            print(f"  Best match score: {best.similarity_score:.4f}")
            print(f"  Source: {best.metadata['directory']}")

    print("\n[7/7] Pipeline verification complete")

    print("\n" + "="*80)
    print("SUCCESS: BLOCK 4 MEDQUAD INGESTION PIPELINE WORKING")
    print("="*80)
    print("\nSummary:")
    print(f"  Documents loaded: {len(documents)}")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Vectorstore ready: Yes")
    print(f"  Retrieval working: Yes")
    print("\nReady for next phase: Block 5 (Hybrid Retrieval) or Block 3 Phase 2 (LLM)")

except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
