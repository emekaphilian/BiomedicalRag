"""Smoke tests for the Sentence Transformers embedding provider."""

import sys
from types import SimpleNamespace

import numpy as np

from rag.embeddings import EmbeddingsProvider


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, **kwargs):
        del kwargs
        vectors = np.zeros((len(texts), 3), dtype=np.float32)
        vectors[:, 0] = 3.0
        vectors[:, 1] = 4.0
        return vectors


def test_embeddings_initialization_is_lazy_and_keeps_default_model(monkeypatch):
    models = []

    def create_model(model_name):
        model = FakeSentenceTransformer(model_name)
        models.append(model)
        return model

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=create_model),
    )

    provider = EmbeddingsProvider()

    assert provider.model_name == "all-MiniLM-L6-v2"
    assert provider.model is None
    assert models == []

    provider.embed_text("malaria symptoms")

    assert len(models) == 1
    assert models[0].model_name == "all-MiniLM-L6-v2"


def test_embed_single_text_returns_finite_normalized_vector(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    embedding = EmbeddingsProvider().embed_text("malaria symptoms")

    assert embedding.shape == (3,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    np.testing.assert_allclose(np.linalg.norm(embedding), 1.0, atol=1e-6)


def test_embed_batch_returns_one_normalized_vector_per_text(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    embeddings = EmbeddingsProvider().embed_batch(
        ["malaria symptoms", "hypertension causes"]
    )

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), [1.0, 1.0])
