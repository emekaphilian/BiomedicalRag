import numpy as np

from rag.document import BiomedicalDocument
from rag.vectorstore import BiomedicalVectorStore


class FakeModel:
    def encode(self, texts):
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)


class FakeIndex:
    def __init__(self, distances, indices):
        self.distances = np.asarray([distances], dtype=np.float32)
        self.indices = np.asarray([indices], dtype=np.int64)
        self.requested_k = None

    def search(self, query_embedding, k):
        assert query_embedding.shape == (1, 3)
        self.requested_k = k
        return self.distances[:, :k], self.indices[:, :k]


def make_store(documents, distances, indices):
    store = BiomedicalVectorStore.__new__(BiomedicalVectorStore)
    store.model = FakeModel()
    store.documents = documents
    store.index = FakeIndex(distances, indices)
    return store


def test_similarity_search_returns_document_and_metadata():
    document = BiomedicalDocument(
        content="Diabetes symptoms",
        metadata={"source": "medquad.xml"},
    )
    store = make_store([document], [0.25], [0])

    results = store.similarity_search("What are diabetes symptoms?", k=1)

    assert len(results) == 1
    assert results[0].content == document.content
    assert results[0].metadata == document.metadata
    assert results[0].similarity_score == 0.25
    assert store.index.requested_k == 1


def test_similarity_search_skips_non_finite_distances():
    documents = [
        BiomedicalDocument(f"Document {i}", {"id": i})
        for i in range(4)
    ]
    store = make_store(
        documents,
        [0.1, np.nan, np.inf, -np.inf],
        [0, 1, 2, 3],
    )

    results = store.similarity_search("query", k=4)

    assert [result.content for result in results] == ["Document 0"]


def test_similarity_search_skips_invalid_indices():
    document = BiomedicalDocument("Valid document", {"id": 1})
    store = make_store(
        [document],
        [0.1, 0.2, 0.3],
        [0, -1, 10],
    )

    results = store.similarity_search("query", k=3)

    assert len(results) == 1
    assert results[0].content == "Valid document"


def test_similarity_search_returns_empty_for_no_documents():
    store = make_store([], [], [])

    assert store.similarity_search("query") == []
    assert store.index.requested_k is None


def test_similarity_search_returns_empty_for_non_positive_k():
    store = make_store(
        [BiomedicalDocument("document", {})],
        [],
        [],
    )

    assert store.similarity_search("query", k=0) == []
    assert store.similarity_search("query", k=-1) == []
    assert store.index.requested_k is None
