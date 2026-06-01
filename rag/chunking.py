from typing import List
from rag.document import BiomedicalDocument


def chunk_document(
    document: BiomedicalDocument,
    chunk_size: int = 300,
    overlap: int = 50
) -> List[BiomedicalDocument]:
    """
    Chunk biomedical text with overlap.
    """

    words = document.content.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words)

        chunk_metadata = document.metadata.copy()

        chunk_metadata["chunk_start"] = start
        chunk_metadata["chunk_end"] = end

        chunks.append(
            BiomedicalDocument(
                content=chunk_text,
                metadata=chunk_metadata
            )
        )

        start += chunk_size - overlap

    return chunks
