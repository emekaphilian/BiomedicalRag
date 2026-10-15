"""Helpers for reproducible session audit records."""

import hashlib
import json
from typing import Mapping, Sequence


QUALITY_DIMENSIONS = (
    "factual_correctness",
    "hallucination_concern",
    "evidence_support",
    "faithfulness",
    "explainability",
    "completeness",
    "safety",
)


def ethical_risk_score(impact: int, likelihood: int) -> dict:
    """Compute a transparent operator-rated impact × likelihood risk indicator."""
    if (
        type(impact) is not int
        or type(likelihood) is not int
        or impact not in range(1, 6)
        or likelihood not in range(1, 6)
    ):
        raise ValueError("Impact and likelihood ratings must be integers from 1 to 5.")

    score = impact * likelihood
    if score >= 16:
        level = "Critical"
    elif score >= 10:
        level = "High"
    elif score >= 5:
        level = "Moderate"
    else:
        level = "Low"
    return {"score": score, "level": level}


def configuration_version(configuration: Mapping[str, object]) -> str:
    """Return a stable fingerprint for a non-secret configuration snapshot."""
    serialized = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"cfg-{hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:12]}"


def summarize_answer_quality_reviews(reviews: Sequence[Mapping]) -> dict:
    """Summarize human 1–5 quality ratings without interpreting them as validation."""
    summary = {"review_count": len(reviews)}
    for dimension in QUALITY_DIMENSIONS:
        values = [
            review["ratings"][dimension]
            for review in reviews
            if dimension in review.get("ratings", {})
        ]
        summary[dimension] = sum(values) / len(values) if values else None
    return summary
