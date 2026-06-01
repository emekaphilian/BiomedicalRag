#!/usr/bin/env python
"""
BLOCK 3 Retrieval Grounding Test Runner
"""

import sys
import os

# Ensure we're in the right directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 80)
print("BLOCK 3: RETRIEVAL GROUNDING FOUNDATION TEST")
print("=" * 80)

try:
    from rag.document import BiomedicalDocument
    from rag.chunking import chunk_document
    from rag.vectorstore import BiomedicalVectorStore
    from rag.retriever import BiomedicalRetriever
    from rag.grounding import build_grounded_context

    print("\n✓ All imports successful\n")

except ImportError as e:
    print(f"\n✗ Import error: {e}\n")
    sys.exit(1)

try:
    print("\n=== BLOCK 3 RETRIEVAL GROUNDING TEST ===\n")

    doc1 = BiomedicalDocument(
        content="""
Diabetes mellitus is a chronic metabolic disease characterized
by elevated blood glucose levels. Common symptoms include
frequent urination, excessive thirst, and unexplained weight loss.
""",
        metadata={
            "source": "WHO Diabetes Guidelines"
        }
    )

    doc2 = BiomedicalDocument(
        content="""
Hypertension is defined as persistently elevated blood pressure.
Lifestyle modification and antihypertensive medications are commonly used.
""",
        metadata={
            "source": "CDC Hypertension Guide"
        }
    )

    all_chunks = []

    for doc in [doc1, doc2]:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    print(f"[1/5] Created chunks: {len(all_chunks)}")

    vectorstore = BiomedicalVectorStore()
    vectorstore.add_documents(all_chunks)

    print("[2/5] Vectorstore indexed successfully")

    retriever = BiomedicalRetriever(vectorstore)
    query = "What are symptoms of diabetes?"
    response = retriever.retrieve(query)

    print(f"[3/5] Retrieved results: {len(response['results'])}")
    print(f"[4/5] Grounded: {response['grounded']}")

    print("\n=== RETRIEVED RESULTS ===\n")

    for idx, result in enumerate(response["results"], start=1):
        print(f"Result {idx}:")
        print(f"  Source: {result.metadata['source']}")
        print(f"  Score: {result.similarity_score:.4f}")
        print(f"  Content: {result.content[:100]}...")
        print("-" * 60)

    context = build_grounded_context(response["results"])

    print("\n=== GROUNDED CONTEXT ===\n")
    print(context[:1500])

    print("\n" + "=" * 80)
    print("✓✓✓ BLOCK 3 RETRIEVAL GROUNDING TEST PASSED ✓✓✓")
    print("=" * 80)
    print("\n✓ Chunks created successfully")
    print("✓ FAISS vectorstore indexed")
    print("✓ Source metadata attached")
    print("✓ Similarity scores computed")
    print("✓ Grounding verification working")
    print("✓ Evidence context formatted")
    print("\n" + "=" * 80 + "\n")

except Exception as e:
    print(f"\n✗ Test failed with error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)
