from typing import List

from rag.document import RetrievedChunk


MIN_GROUNDING_SCORE = 2.0


def is_grounded(results: List[RetrievedChunk]) -> bool:
    """
    Apply the current retrieval-distance threshold.

    Note: This is a heuristic retrieval check, not proof that
    an answer is medically accurate or fully supported by evidence.
    """
    if not results:
        return False

    best_score = results[0].distance

    return (
        best_score >= 0
        and best_score < MIN_GROUNDING_SCORE
    )


def build_grounded_context(results: List[RetrievedChunk]) -> str:
    """Format retrieved chunks as evidence for answer generation."""
    context_parts = []

    for idx, result in enumerate(results, start=1):
        source = result.metadata.get("source", "Unknown")
        score = result.distance

        context_parts.append(
            f"[Evidence {idx}]\n"
            f"Source: {source}\n"
            f"Distance: {score:.4f}\n\n"
            f"{result.content}"
        )

    return "\n\n".join(context_parts)
