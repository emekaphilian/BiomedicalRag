from rag.chunking import chunk_document
from rag.vectorstore import BiomedicalVectorStore


def build_index(documents):
    """Chunk documents and build a biomedical vector index."""
    if not documents:
        raise ValueError(
            "Cannot build the index: no documents were provided."
        )

    all_chunks = []

    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(
            "Cannot build the index: document chunking produced no chunks."
        )

    vectorstore = BiomedicalVectorStore()
    vectorstore.add_documents(all_chunks)

    return vectorstore, all_chunks
