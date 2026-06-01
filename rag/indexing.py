from rag.chunking import chunk_document
from rag.vectorstore import BiomedicalVectorStore


def build_index(documents):

    all_chunks = []

    for doc in documents:

        chunks = chunk_document(doc)

        all_chunks.extend(chunks)

    vectorstore = BiomedicalVectorStore()

    vectorstore.add_documents(all_chunks)

    return vectorstore, all_chunks
