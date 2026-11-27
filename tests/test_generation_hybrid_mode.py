from types import SimpleNamespace

from rag.document import BiomedicalDocument
from rag.post_validation import PostGenerationValidator
import rag.generation_pipeline as generation_module


def test_comparison_extractive_fallback_uses_supported_evidence_for_each_side():
    query = "Compare type 1 vs type 2 diabetes."
    evidence = [
        {"content": "With Type 1 diabetes, the pancreas does not make insulin.", "metadata": {"title": "Type 1 Diabetes"}},
        {"content": "With Type 2 diabetes, the body does not make or use insulin well.", "metadata": {"title": "Type 2 Diabetes"}},
    ]
    for item in evidence:
        item["evidence_classification"] = generation_module.classify_evidence_set(query, [item])[0]

    answer = generation_module._comparison_extractive_fallback(query, evidence)

    assert answer is not None
    assert "Type 1 diabetes" in answer and "[Evidence 1]" in answer
    assert "Type 2 diabetes" in answer and "[Evidence 2]" in answer
    validation = PostGenerationValidator().validate(
        {"answer": answer, "success": True},
        len(evidence),
        [item["content"] for item in evidence],
        evidence_records=evidence,
    )
    assert validation["valid"]


class FakeHybridRetriever:
    def __init__(self, vectorstore, documents):
        self.documents = documents
        self.dense_retriever = SimpleNamespace(vectorstore=vectorstore)

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
    def validate(self, answer, evidence_count, evidence_texts=None, evidence_records=None):
        return {"valid": True, "issues": []}


class CitationRecoveryRetriever:
    def __init__(self, vectorstore, documents):
        self.documents = documents
        self.dense_retriever = SimpleNamespace(vectorstore=vectorstore)

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
            "context": [{
                "content": item["content"], "metadata": item["metadata"],
                **({"evidence_classification": item["evidence_classification"]}
                   if item.get("evidence_classification") else {}),
            }],
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
    monkeypatch.setattr(generation_module, "is_medical_query", lambda query: True)
    monkeypatch.setattr(generation_module, "has_topic_overlap", lambda query, passages: True)
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
                content="Diabetes is a condition in which blood glucose levels are high.",
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
    monkeypatch.setattr(generation_module, "is_medical_query", lambda query: True)
    monkeypatch.setattr(generation_module, "has_topic_overlap", lambda query, passages: True)
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


def test_uncited_draft_uses_a_cited_extractive_evidence_fallback(monkeypatch):
    pipeline, generator = make_citation_recovery_pipeline(
        monkeypatch,
        [
            "Diabetes involves elevated blood glucose.",
            "Diabetes involves elevated blood glucose [Evidence 1].",
        ],
    )

    result = pipeline.generate("What is diabetes?")

    assert len(generator.prompts) == 1
    assert "[Evidence 1]" in result["generation"]["answer"]
    assert "blood glucose" in result["generation"]["answer"]
    assert result["validation"]["valid"] is True
    assert result["final_answer"] == result["generation"]["answer"]


def test_uncited_answer_fallback_uses_only_passage_text(monkeypatch):
    pipeline, generator = make_citation_recovery_pipeline(
        monkeypatch,
        [
            "Diabetes involves elevated blood glucose.",
            "Diabetes involves elevated blood glucose without citations.",
        ],
    )

    result = pipeline.generate("What is diabetes?")

    assert len(generator.prompts) == 1
    assert result["generation"]["answer"] == result["final_answer"]
    assert "[Evidence 1]" in result["final_answer"]
    assert result["validation"]["valid"] is True
    assert result["answer_abstained"] is False
    assert "VALIDATION FAILED" not in result["final_answer"]
    assert "Original:" not in result["final_answer"]


def test_internal_reasoning_is_replaced_with_safe_refusal(monkeypatch):
    pipeline, _ = make_citation_recovery_pipeline(
        monkeypatch,
        ["We need to answer the query: Parkinson's disease affects dopamine. "
         "Answer: It affects dopamine-producing neurons [Evidence 1].",
         "We need to answer the query: Parkinson's disease affects dopamine. "
         "Answer: It affects dopamine-producing neurons [Evidence 1]."],
    )
    pipeline.documents[0].content = "Parkinson's disease affects dopamine."

    result = pipeline.generate("How does Parkinson's disease affect dopamine?")

    assert result["answer_abstained"] is True
    assert result["validation"]["valid"] is False
    assert "internal reasoning" in " ".join(result["validation"]["issues"]).lower()
    assert "We need to answer" not in result["final_answer"]
    assert result["generation"]["answer"] == result["final_answer"]


def test_valid_initial_answer_does_not_retry(monkeypatch):
    pipeline, generator = make_citation_recovery_pipeline(
        monkeypatch,
        ["Diabetes involves elevated blood glucose [Evidence 1]."],
    )

    result = pipeline.generate("What is diabetes?")

    assert len(generator.prompts) == 1
    assert result["validation"]["valid"] is True
    assert "Diabetes involves elevated blood glucose" in result["final_answer"]
    assert "[Evidence 1]" in result["final_answer"]


def test_relevant_low_risk_answer_is_preserved_with_validation_warning(monkeypatch):
    pipeline, _ = make_citation_recovery_pipeline(
        monkeypatch,
        [
            "Diabetes involves elevated blood glucose [Evidence 1].",
            "Diabetes involves elevated blood glucose [Evidence 1].",
        ],
    )
    pipeline.documents[0].content = "Diabetes refers to a condition involving high blood glucose."

    class LexicalReviewValidator:
        def validate(self, *args, **kwargs):
            return {
                "valid": False,
                "issues": ["One or more sentences failed the lexical claim-support screen"],
                "citation_references_validated": True,
                "claim_support_validated": "NOT ESTABLISHED",
                "claim_support": [{
                    "claim": "Diabetes involves elevated blood glucose [Evidence 1].",
                    "support_status": "unsupported", "citations": [1],
                    "lexical_support": 0.5, "semantic_similarity": 0.0,
                    "unsupported_numbers": [],
                }],
                "diagnostics": {
                    "citation_support_coverage": 1.0,
                    "citation_coverage": 1.0,
                    "lexical_support": 0.5,
                    "unsupported_number_count": 0,
                },
            }

    pipeline.validator = LexicalReviewValidator()
    result = pipeline.generate("What is diabetes?")

    assert result["decision"] == "informational"
    assert result["answer_abstained"] is False
    assert "Diabetes involves elevated blood glucose" in result["final_answer"]
    assert "could not be fully confirmed" in result["final_answer"]
    audit = result["audit_trace"]
    assert audit["question_classification"]["risk_tier"] == "R0"
    assert audit["evidence_audit"]["retrieved_evidence_count"] == 1
    assert audit["evidence_audit"]["generated_claims"]
    assert audit["evidence_audit"]["contradictory_evidence"]["status"] == "screened"
    assert audit["final_answer_status"] == "informational"


def test_hypertension_alias_supports_cited_fallback_when_generator_abstains(monkeypatch):
    pipeline, _ = make_citation_recovery_pipeline(
        monkeypatch,
        [
            "I do not have enough information to answer.",
            "I do not have enough information to answer.",
        ],
    )
    pipeline.documents[0].content = (
        "Primary high blood pressure develops over time as you get older. "
        "Secondary high blood pressure is caused by another medical condition "
        "or use of certain medicines."
    )

    result = pipeline.generate("What causes hypertension?")

    assert result["answer_abstained"] is False
    assert result["decision"] in {"answer_approved", "informational"}
    assert "high blood pressure develops over time" in result["final_answer"].lower()
    assert "[Evidence 1]" in result["final_answer"]


def test_empty_model_answer_uses_supported_evidence_instead_of_returning_blank(monkeypatch):
    pipeline, _ = make_citation_recovery_pipeline(monkeypatch, ["", ""])
    pipeline.documents[0].content = (
        "Dopamine is a neurotransmitter that helps control movement, motivation, and reward."
    )

    result = pipeline.generate("What is the function of dopamine?")

    assert result["final_answer"].strip()
    assert "dopamine" in result["final_answer"].lower()
    assert "[Evidence 1]" in result["final_answer"]
    assert result["answer_abstained"] is False


def test_direct_symptom_evidence_survives_question_wording_expansion(monkeypatch):
    pipeline, _ = make_citation_recovery_pipeline(
        monkeypatch,
        [
            "Common symptoms of diabetes include increased thirst and frequent urination [Evidence 1].",
        ],
    )
    pipeline.documents[0].content = (
        "Common symptoms of diabetes include increased thirst and frequent urination."
    )
    result = pipeline.generate("What are common symptoms of diabetes?")

    assert result["decision"] == "answer_approved"
    assert result["answer_abstained"] is False
    assert "increased thirst" in result["final_answer"]
    assert result["compression"]["evidence"][0]["evidence_classification"]["category"] == "direct"
