import pytest

from dashboard.research_metrics import calculate_retrieval_metrics


def test_retrieval_metrics_use_unique_case_insensitive_source_ranks():
    metrics = calculate_retrieval_metrics(
        ["Guide A", "guide a", "Guide B"],
        ["guide a", "guide b"],
        k=3,
    )

    assert metrics["recall@3"] == 1.0
    assert metrics["precision@3"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg@3"] == 1.0
    assert metrics["relevant_hits"] == 2


def test_retrieval_metrics_return_zero_when_no_relevant_sources_are_found():
    metrics = calculate_retrieval_metrics(["Guide A"], ["Guide B"], k=3)

    assert metrics["recall@3"] == 0.0
    assert metrics["precision@3"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["ndcg@3"] == 0.0


def test_retrieval_metrics_reject_non_positive_cutoff():
    with pytest.raises(ValueError, match="at least 1"):
        calculate_retrieval_metrics([], [], k=0)
