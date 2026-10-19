from rag.document import BiomedicalDocument
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


class CitationRecoveryRetriever:
    def __init__(self, vectorstore, documents):
        self.documents = documents

    def retrieve(self, query, k):
        document = self.documents[0]
        return {
            "fused_results": [(document.content, 0.9)],
            "grounded": True,
        }


class CitationRecoveryCompressor:
    def __init__(self, max_chunks=3):
        pass

    def compress(self, results):
        item = results[0]
        return {
            "context": [{"content": item["content"], "metadata": item["metadata"]}],
            "evidence_count": 1,
            "sources": [item["metadata"]["source"]],
            "formatted_text": "[Evidence 1] known evidence",
        }


class CitationRecoveryPromptConstructor:
    def construct(self, query, compressed_context):
        return {
            "system": "Use only supplied evidence.",
            "user": f"Query: {query}\n{compressed_context['formatted_text']}",
            "query": query,
            "evidence_count": compressed_context["evidence_count"],
            "evidence_sources": compressed_context["sources"],
        }


class ScriptedGenerator:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        answer = next(self.answers)
        return {"answer": answer, "success": True, "model": "test-model"}


def make_citation_recovery_pipeline(monkeypatch, answers):
    monkeypatch.setattr(
        generation_module, "HybridRetriever", CitationRecoveryRetriever
    )
    monkeypatch.setattr(
        generation_module, "ContextCompressor", CitationRecoveryCompressor
    )
    monkeypatch.setattr(
        generation_module,
        "GroundedPromptConstructor",
        CitationRecoveryPromptConstructor,
    )
    pipeline = generation_module.GroundedGenerationPipeline(
        vectorstore=object(),
        documents=[
            BiomedicalDocument(
                content="known evidence",
                metadata={"source": "MedlinePlus"},
            )
        ],
        use_mock=False,
        use_reranking=False,
        generator_config={"max_tokens": 123, "temperature": 0.2},
    )
    generator = ScriptedGenerator(answers)
    pipeline.generator = generator
    return pipeline, generator


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


def test_missing_citations_retry_and_return_validated_retry(monkeypatch):
    pipeline, generator = make_citation_recovery_pipeline(
        monkeypatch,
        [
            "The evidence provides guidance.",
            "The evidence provides guidance [Evidence 1].",
        ],
    )

    result = pipeline.generate("What does the evidence say?")

    assert len(generator.prompts) == 2
    retry_prompt, retry_kwargs = generator.prompts[1]
    assert "CITATION RECOVERY REQUIREMENTS" in retry_prompt["system"]
    assert "Use only the original supplied evidence" in retry_prompt["user"]
    assert retry_kwargs == {"max_tokens": 123, "temperature": 0.2}
    assert result["generation"]["answer"] == (
        "The evidence provides guidance [Evidence 1]."
    )
    assert result["validation"]["valid"] is True
    assert result["final_answer"] == result["generation"]["answer"]


def test_missing_citations_remain_fail_closed_when_retry_also_fails(monkeypatch):
    pipeline, generator = make_citation_recovery_pipeline(
        monkeypatch,
        [
            "The evidence provides guidance.",
            "The evidence still provides guidance without citations.",
        ],
    )

    result = pipeline.generate("What does the evidence say?")

    assert len(generator.prompts) == 2
    assert result["generation"]["answer"] == (
        "The evidence still provides guidance without citations."
    )
    assert result["validation"]["valid"] is False
    assert "No evidence citations found in answer" in result["validation"]["issues"]
    assert result["final_answer"].startswith("[VALIDATION FAILED]")


def test_valid_initial_answer_does_not_retry(monkeypatch):
    pipeline, generator = make_citation_recovery_pipeline(
        monkeypatch,
        ["The evidence provides guidance [Evidence 1]."],
    )

    result = pipeline.generate("What does the evidence say?")

    assert len(generator.prompts) == 1
    assert result["validation"]["valid"] is True
    assert result["final_answer"] == "The evidence provides guidance [Evidence 1]."
