import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.document import BiomedicalDocument
from rag.chunking import chunk_document
from rag.vectorstore import BiomedicalVectorStore
from rag.retriever import BiomedicalRetriever
from rag.grounding import build_grounded_context


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


print(f"Created chunks: {len(all_chunks)}")


vectorstore = BiomedicalVectorStore()

vectorstore.add_documents(all_chunks)

print("Vectorstore indexed successfully")


retriever = BiomedicalRetriever(vectorstore)

query = "What are symptoms of diabetes?"

response = retriever.retrieve(query)

print(f"\nGrounded: {response['grounded']}")
print("\nRetrieved Results:\n")

for idx, result in enumerate(response["results"], start=1):

    print(f"Result {idx}")
    print(f"Source: {result.metadata['source']}")
    print(f"Score: {result.similarity_score:.4f}")
    print(f"Content: {result.content[:120]}...")
    print("-" * 50)


context = build_grounded_context(response["results"])

print("\n=== GROUNDED CONTEXT ===\n")

print(context[:1500])
