"""Small source-level retrieval metrics used by the research dashboard."""

import math
from typing import Iterable


def calculate_retrieval_metrics(
    retrieved_sources: Iterable[str],
    relevant_sources: Iterable[str],
    k: int,
) -> dict:
    """Calculate source-level Recall@k, Precision@k, MRR, and binary NDCG@k."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    ranked = list(
        dict.fromkeys(
            str(source).strip().casefold()
            for source in retrieved_sources
            if str(source).strip()
        )
    )
    relevant = {
        str(source).strip().casefold()
        for source in relevant_sources
        if str(source).strip()
    }
    relevant_ranked = [source for source in ranked[:k] if source in relevant]
    hits = len(set(relevant_ranked))
    recall = hits / len(relevant) if relevant else 0.0
    precision = hits / k

    reciprocal_rank = next(
        (1.0 / rank for rank, source in enumerate(ranked[:k], 1) if source in relevant),
        0.0,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, source in enumerate(ranked[:k], 1)
        if source in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return {
        f"recall@{k}": recall,
        f"precision@{k}": precision,
        "mrr": reciprocal_rank,
        f"ndcg@{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
        "relevant_hits": hits,
        "relevant_source_count": len(relevant),
    }
