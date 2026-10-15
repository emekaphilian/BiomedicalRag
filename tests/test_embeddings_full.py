"""Regression tests for the embeddings provider's current behavior."""

import pytest

from rag.embeddings import EmbeddingsProvider


def test_default_model_initialization():
    provider = EmbeddingsProvider()

    assert provider.model_name == "all-MiniLM-L6-v2"
    assert provider.model is None


def test_custom_model_initialization():
    provider = EmbeddingsProvider(model_name="custom-biomedical-model")

    assert provider.model_name == "custom-biomedical-model"
    assert provider.model is None


def test_embed_single_text_is_not_implemented():
    provider = EmbeddingsProvider()

    with pytest.raises(NotImplementedError):
        provider.embed_text("What are the symptoms of malaria?")


def test_embed_batch_is_not_implemented():
    provider = EmbeddingsProvider()

    with pytest.raises(NotImplementedError):
        provider.embed_batch(
            [
                "What are the symptoms of malaria?",
                "What causes hypertension?",
            ]
        )
