from typing import List
from rag.document import RetrievedChunk


MIN_GROUNDING_SCORE = 2.0


def is_grounded(results: List[RetrievedChunk]) -> bool:
    """
    Determines whether retrieval is sufficiently grounded.
    """

    if not results:
        return False

    best_score = results[0].similarity_score

    return best_score < MIN_GROUNDING_SCORE


def build_grounded_context(results: List[RetrievedChunk]) -> str:
    """
    Builds formatted grounded evidence context.
    """

    context_parts = []

    for idx, result in enumerate(results, start=1):

        source = result.metadata.get("source", "Unknown")

        context_parts.append(
            f"""
[Evidence {idx}]
Source: {source}
Similarity Score: {result.similarity_score:.4f}

{result.content}
"""
        )

    return "\n".join(context_parts)
