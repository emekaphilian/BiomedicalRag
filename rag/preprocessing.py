import re

from rag.document import BiomedicalDocument


def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_documents(documents):

    processed = []

    for doc in documents:

        cleaned = clean_text(doc.content)

        processed.append(
            BiomedicalDocument(
                content=cleaned,
                metadata=doc.metadata
            )
        )

    return processed

