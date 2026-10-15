import pytest

from rag.document import RetrievedChunk
from rag.grounding import MIN_GROUNDING_SCORE, build_grounded_context, is_grounded


def chunk(content="Evidence text", source="MedlinePlus", distance=0.5):
    return RetrievedChunk(
        content=content,
        metadata={"source": source},
        similarity_score=distance,
    )


def test_is_grounded_accepts_distance_below_threshold():
    assert is_grounded([chunk(distance=MIN_GROUNDING_SCORE - 0.01)])


def test_is_grounded_rejects_distance_at_threshold():
    assert not is_grounded([chunk(distance=MIN_GROUNDING_SCORE)])


@pytest.mark.parametrize("distance", [-0.1, float("nan"), float("inf")])
def test_is_grounded_rejects_invalid_distance(distance):
    assert not is_grounded([chunk(distance=distance)])


def test_is_grounded_rejects_empty_results():
    assert not is_grounded([])


def test_build_grounded_context_includes_source_distance_and_evidence():
    context = build_grounded_context([
        chunk(content="Diabetes education", source="MedlinePlus", distance=0.25),
        chunk(content="Research abstract", source="PubMed", distance=0.75),
    ])
    assert "[Evidence 1]" in context
    assert "Source: MedlinePlus" in context
    assert "Distance: 0.2500" in context
    assert "Diabetes education" in context
    assert "[Evidence 2]" in context
    assert "Source: PubMed" in context
    assert "Research abstract" in context
