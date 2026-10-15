import pytest

from dashboard.audit_utils import (
    QUALITY_DIMENSIONS,
    configuration_version,
    ethical_risk_score,
    summarize_answer_quality_reviews,
)


def test_configuration_version_is_stable_and_changes_with_settings():
    first = {"provider": "cohere", "top_k": 5}
    same_settings = {"top_k": 5, "provider": "cohere"}

    assert configuration_version(first) == configuration_version(same_settings)
    assert configuration_version(first) != configuration_version(
        {"provider": "cohere", "top_k": 10}
    )


def test_answer_quality_summary_reports_human_rating_averages():
    reviews = [
        {
            "ratings": {
                "factual_correctness": 4,
                "evidence_support": 5,
                "completeness": 3,
                "safety": 5,
            }
        },
        {
            "ratings": {
                "factual_correctness": 2,
                "evidence_support": 3,
                "completeness": 5,
                "safety": 5,
            }
        },
    ]

    summary = summarize_answer_quality_reviews(reviews)

    assert summary["review_count"] == 2
    assert summary["factual_correctness"] == 3
    assert summary["evidence_support"] == 4
    assert summary["completeness"] == 4
    assert summary["safety"] == 5


def test_answer_quality_summary_handles_no_reviews():
    summary = summarize_answer_quality_reviews([])

    assert summary["review_count"] == 0
    assert all(summary[dimension] is None for dimension in QUALITY_DIMENSIONS)


def test_ethical_risk_score_is_operator_rated_impact_times_likelihood():
    assert ethical_risk_score(4, 5) == {"score": 20, "level": "Critical"}
    assert ethical_risk_score(2, 2) == {"score": 4, "level": "Low"}


@pytest.mark.parametrize(
    ("impact", "likelihood"),
    [(0, 2), (2, 6), (1.5, 2), (True, 2)],
)
def test_ethical_risk_score_rejects_values_outside_integer_scale(
    impact,
    likelihood,
):
    with pytest.raises(ValueError, match="integers from 1 to 5"):
        ethical_risk_score(impact, likelihood)
