import re

from rag.document import BiomedicalDocument


def clean_text(text: str) -> str:
    """Normalize whitespace in a document."""
    if not isinstance(text, str):
        raise TypeError("Document content must be a string.")

    return re.sub(r"\s+", " ", text).strip()


def preprocess_documents(documents):
    """Clean documents while preserving their metadata."""
    processed = []

    for doc in documents:
        cleaned = clean_text(doc.content)

        if not cleaned:
            continue

        processed.append(
            BiomedicalDocument(
                content=cleaned,
                metadata=dict(doc.metadata),
            )
        )

    return processed
