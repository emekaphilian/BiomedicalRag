import html
import re
from html.parser import HTMLParser

from rag.document import BiomedicalDocument


class _VisibleTextParser(HTMLParser):
    """Collect visible text while dropping HTML tags and hidden page content."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data):
        if not self._hidden_depth:
            self.parts.append(data)


def clean_text(text: str) -> str:
    """Strip embedded HTML and normalize whitespace in a document."""
    if not isinstance(text, str):
        raise TypeError("Document content must be a string.")

    if re.search(r"</?[A-Za-z][^>]*>", text):
        parser = _VisibleTextParser()
        parser.feed(text)
        parser.close()
        text = " ".join(parser.parts)

    text = re.sub(r"\s+", " ", html.unescape(text)).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


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
