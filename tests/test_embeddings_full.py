"""Unit and opt-in real-model tests for the embeddings provider."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rag.embeddings import EmbeddingsProvider


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = []

    def encode(
        self,
        texts,
        *,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        self.calls.append(
            (texts, convert_to_numpy, normalize_embeddings, show_progress_bar)
        )
        return np.tile(np.array([[3.0, 4.0]], dtype=np.float32), (len(texts), 1))


def install_fake_model(monkeypatch):
    instances = []

    def create_model(model_name):
        model = FakeSentenceTransformer(model_name)
        instances.append(model)
        return model

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=create_model),
    )
    return instances


def test_model_initialization_is_lazy_and_uses_configured_name(monkeypatch):
    instances = install_fake_model(monkeypatch)
    provider = EmbeddingsProvider("custom-biomedical-model")

    assert provider.model_name == "custom-biomedical-model"
    assert provider.model is None
    assert instances == []

    provider.embed_text("malaria symptoms")

    assert len(instances) == 1
    assert instances[0].model_name == "custom-biomedical-model"


def test_embed_text_returns_a_finite_normalized_vector(monkeypatch):
    instances = install_fake_model(monkeypatch)
    provider = EmbeddingsProvider()

    embedding = provider.embed_text("What are the symptoms of malaria?")

    assert embedding.shape == (2,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    assert instances[0].calls == [
        (
            ["What are the symptoms of malaria?"],
            True,
            True,
            False,
        )
    ]


def test_embed_batch_returns_normalized_vectors_and_reuses_model(monkeypatch):
    instances = install_fake_model(monkeypatch)
    provider = EmbeddingsProvider()
    texts = ["malaria symptoms", "hypertension causes"]

    embeddings = provider.embed_batch(texts)
    provider.embed_batch(["diabetes symptoms"])

    assert embeddings.shape == (2, 2)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), [1.0, 1.0])
    assert len(instances) == 1
    assert len(instances[0].calls) == 2


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_embed_text_rejects_empty_input(text):
    with pytest.raises(ValueError, match="must not be empty"):
        EmbeddingsProvider().embed_text(text)


def test_embed_text_rejects_non_string_input():
    with pytest.raises(TypeError, match="text must be a string"):
        EmbeddingsProvider().embed_text(None)


def test_embed_batch_rejects_empty_batch():
    with pytest.raises(ValueError, match="at least one string"):
        EmbeddingsProvider().embed_batch([])


@pytest.mark.parametrize("texts", ["malaria symptoms", [" ", "valid"]])
def test_embed_batch_rejects_invalid_items(texts):
    with pytest.raises((TypeError, ValueError)):
        EmbeddingsProvider().embed_batch(texts)


def test_embed_batch_rejects_non_finite_model_output(monkeypatch):
    class InvalidModel(FakeSentenceTransformer):
        def encode(self, texts, **kwargs):
            return np.array([[np.nan, 1.0]], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=InvalidModel),
    )

    with pytest.raises(ValueError, match="non-finite"):
        EmbeddingsProvider().embed_text("malaria symptoms")


def test_embed_batch_rejects_unexpected_model_output_shape(monkeypatch):
    class InvalidModel(FakeSentenceTransformer):
        def encode(self, texts, **kwargs):
            return np.array([1.0, 2.0], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=InvalidModel),
    )

    with pytest.raises(ValueError, match="unexpected output shape"):
        EmbeddingsProvider().embed_batch(["first", "second"])


@pytest.mark.skipif(
    os.environ.get("BIOMEDICAL_RAG_RUN_MODEL_TESTS") != "1",
    reason="Set BIOMEDICAL_RAG_RUN_MODEL_TESTS=1 to load the real model.",
)
def test_default_sentence_transformer_model_integration():
    provider = EmbeddingsProvider()
    embeddings = provider.embed_batch(
        [
            "What are the symptoms of malaria?",
            "What causes hypertension?",
        ]
    )

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    np.testing.assert_allclose(
        np.linalg.norm(embeddings, axis=1),
        np.ones(2),
        atol=1e-5,
    )
