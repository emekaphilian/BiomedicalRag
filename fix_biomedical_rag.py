from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(relative_path, old, new):
    path = ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        if new in text:
            print(f"Already fixed: {relative_path}")
            return
        raise RuntimeError(
            f"Could not find expected code block in {relative_path}; stopped safely."
        )
    if count != 1:
        raise RuntimeError(
            f"Expected one match in {relative_path}, found {count}; stopped safely."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
    print(f"Fixed: {relative_path}")


replace_once(
    "rag/vectorstore.py",
    "                    metadata=doc.metadata,\n"
    "                    similarity_score=float(distance),",
    "                    metadata=dict(doc.metadata),\n"
    "                    similarity_score=float(distance),",
)

replace_once(
    "rag/generation_pipeline.py",
    "            ranked_results = [\n"
    "                {\n"
    '                    "content": content,\n'
    '                    "metadata": metadata_by_content[content],\n'
    '                    "rerank_score": score,\n'
    "                }\n"
    '                for content, score in hybrid_result["fused_results"][:k]\n'
    "            ]",
    "            ranked_results = [\n"
    "                {\n"
    '                    "content": content,\n'
    '                    "metadata": metadata_by_content[content],\n'
    '                    "rerank_score": score,\n'
    "                }\n"
    '                for content, score in hybrid_result["fused_results"]\n'
    "                if content in metadata_by_content\n"
    "            ][:k]",
)


grounding_test = ROOT / "tests" / "test_grounding.py"
grounding_test.parent.mkdir(parents=True, exist_ok=True)
grounding_test.write_text(
    '''import pytest

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
''',
    encoding="utf-8",
    newline="\n",
)
print("Created tests/test_grounding.py")


hybrid_test = ROOT / "tests" / "test_generation_hybrid_mode.py"
hybrid_test.write_text(
    '''from rag.document import BiomedicalDocument
import rag.generation_pipeline as generation_module


class FakeHybridRetriever:
    def __init__(self, vectorstore, documents):
        self.documents = documents

    def retrieve(self, query, k):
        return {
            "fused_results": [
                ("known evidence", 0.9),
                ("unmapped phantom result", 0.8),
            ],
            "grounded": True,
        }


class FakeCompressor:
    def __init__(self, max_chunks=3):
        pass

    def compress(self, results):
        assert [item["content"] for item in results] == ["known evidence"]
        assert results[0]["metadata"]["source"] == "MedlinePlus"
        return {
            "context": [{"content": "known evidence", "metadata": results[0]["metadata"]}],
            "evidence_count": 1,
            "sources": ["MedlinePlus"],
            "formatted_text": "known evidence",
        }


class FakePromptConstructor:
    def construct(self, query, compressed_context):
        return "test prompt"


class FakeGenerator:
    def __init__(self, **kwargs):
        pass

    def generate_mock(self, prompt):
        return {
            "answer": "Grounded answer",
            "success": True,
            "model": "mock",
        }


class FakeValidator:
    def validate(self, answer, evidence_count):
        return {"valid": True, "issues": []}


def test_hybrid_mode_preserves_metadata_and_skips_unmapped_fused_results(monkeypatch):
    monkeypatch.setattr(generation_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(generation_module, "ContextCompressor", FakeCompressor)
    monkeypatch.setattr(
        generation_module, "GroundedPromptConstructor", FakePromptConstructor
    )
    monkeypatch.setattr(generation_module, "GroundedGenerator", FakeGenerator)
    monkeypatch.setattr(generation_module, "PostGenerationValidator", FakeValidator)

    document = BiomedicalDocument(
        content="known evidence",
        metadata={"source": "MedlinePlus", "document_type": "patient_education"},
    )
    pipeline = generation_module.GroundedGenerationPipeline(
        vectorstore=object(),
        documents=[document],
        use_mock=True,
        use_reranking=False,
    )
    result = pipeline.generate("What does the evidence say?", k=2, retrieval_k=5)
    assert result["generation"]["success"] is True
    assert result["validation"]["valid"] is True
''',
    encoding="utf-8",
    newline="\n",
)
print("Created tests/test_generation_hybrid_mode.py")
print("\nPatch applied.")
