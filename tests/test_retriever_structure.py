"""Regression tests for the biomedical retriever."""

from unittest.mock import Mock

import pytest

import rag.retriever as retriever_module
from rag.retriever import BiomedicalRetriever


def test_constructor_stores_vectorstore():
    vectorstore = Mock()

    retriever = BiomedicalRetriever(vectorstore)

    assert retriever.vectorstore is vectorstore


def test_retrieve_forwards_query_and_k(monkeypatch):
    results = [{"id": "doc-1"}]
    vectorstore = Mock()
    vectorstore.similarity_search.return_value = results
    grounding_check = Mock(return_value=True)
    monkeypatch.setattr(retriever_module, "is_grounded", grounding_check)

    retriever = BiomedicalRetriever(vectorstore)
    retriever.retrieve("malaria symptoms", k=3)

    vectorstore.similarity_search.assert_called_once_with(
        "malaria symptoms", k=3
    )


def test_retrieve_uses_default_k(monkeypatch):
    vectorstore = Mock()
    vectorstore.similarity_search.return_value = []
    monkeypatch.setattr(retriever_module, "is_grounded", Mock(return_value=False))

    retriever = BiomedicalRetriever(vectorstore)
    retriever.retrieve("malaria symptoms")

    vectorstore.similarity_search.assert_called_once_with(
        "malaria symptoms", k=5
    )


def test_retrieve_returns_grounded_results(monkeypatch):
    results = [{"id": "doc-1"}, {"id": "doc-2"}]
    vectorstore = Mock()
    vectorstore.similarity_search.return_value = results
    grounding_check = Mock(return_value=True)
    monkeypatch.setattr(retriever_module, "is_grounded", grounding_check)

    retriever = BiomedicalRetriever(vectorstore)
    response = retriever.retrieve("malaria symptoms", k=2)

    assert response == {
        "grounded": True,
        "results": results,
    }
    grounding_check.assert_called_once_with(results)


def test_retrieve_preserves_false_grounding_result(monkeypatch):
    results = [{"id": "weak-match"}]
    vectorstore = Mock()
    vectorstore.similarity_search.return_value = results
    monkeypatch.setattr(
        retriever_module,
        "is_grounded",
        Mock(return_value=False),
    )

    retriever = BiomedicalRetriever(vectorstore)
    response = retriever.retrieve("unrelated question")

    assert response["grounded"] is False
    assert response["results"] == results


def test_empty_results_are_returned(monkeypatch):
    vectorstore = Mock()
    vectorstore.similarity_search.return_value = []
    monkeypatch.setattr(
        retriever_module,
        "is_grounded",
        Mock(return_value=False),
    )

    retriever = BiomedicalRetriever(vectorstore)
    response = retriever.retrieve("unknown topic")

    assert response == {
        "grounded": False,
        "results": [],
    }


def test_vectorstore_error_propagates(monkeypatch):
    vectorstore = Mock()
    vectorstore.similarity_search.side_effect = RuntimeError(
        "Vector store unavailable"
    )
    grounding_check = Mock()
    monkeypatch.setattr(retriever_module, "is_grounded", grounding_check)

    retriever = BiomedicalRetriever(vectorstore)

    with pytest.raises(RuntimeError, match="Vector store unavailable"):
        retriever.retrieve("malaria symptoms")

    grounding_check.assert_not_called()
