import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingestion import load_medquad_dataset
from rag.preprocessing import preprocess_documents
from rag.indexing import build_index
from rag.retriever import BiomedicalRetriever


DATASET_PATH = "data/raw/medquad"


print("\n=== BLOCK 4 MEDQUAD PIPELINE TEST ===\n")


print("[1/6] Loading MedQuAD dataset...")

documents = load_medquad_dataset(DATASET_PATH)

print(f"Loaded documents: {len(documents)}")


print("\n[2/6] Preprocessing documents...")

documents = preprocess_documents(documents)

print("Preprocessing complete")


print("\n[3/6] Building vector index...")

vectorstore, chunks = build_index(documents)

print(f"Indexed chunks: {len(chunks)}")


print("\n[4/6] Initializing retriever...")

retriever = BiomedicalRetriever(vectorstore)

print("Retriever ready")


query = "What are symptoms of diabetes?"


print("\n[5/6] Running biomedical retrieval query...")

response = retriever.retrieve(query)


print(f"Grounded: {response['grounded']}")

print("\nRetrieved Evidence:\n")


for idx, result in enumerate(response["results"], start=1):

    print(f"Result {idx}")
    print(f"Source: {result.metadata['source']}")
    print(f"File: {result.metadata['file']}")
    print(f"Score: {result.similarity_score:.4f}")
    print(f"Content: {result.content[:250]}...")
    print("-" * 60)


print("\n[6/6] Pipeline verification complete")

print("\n=== BLOCK 4 MEDQUAD INGESTION TEST PASSED ===")
