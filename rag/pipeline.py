"""End-to-end biomedical retrieval and grounded-answer pipeline."""

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence

from rag.document import BiomedicalDocument


@dataclass
class RAGResponse:
    """Response and provenance from a biomedical RAG query."""

    answer: str
    sources: List[Dict[str, Any]]
    confidence: Optional[float]
    is_grounded: bool
    warnings: List[str]
    retrieval_time_ms: float
    generation_time_ms: float
    retrieval_grounded: bool = False
    generation_success: bool = False
    validation_passed: bool = False
    answer_abstained: bool = False
    validation_states: Dict[str, Any] | None = None
    decision: str = "answer_approved"
    rejection_reason: str | None = None
    validation_diagnostics: Dict[str, Any] | None = None
    audit_trace: Dict[str, Any] | None = None
    risk_tier: str = "R0"
    question_category: str = "general_biomedical_education"
    claim_support: List[Dict[str, Any]] = field(default_factory=list)
    evidence_confidence: str | None = None
    confidence_basis: str | None = None


def _build_index(documents):
    from rag.indexing import build_index

    return build_index(documents)


def _build_generation_pipeline(
    vectorstore,
    documents,
    use_mock,
    use_reranking,
    generator_config,
):
    from rag.generation_pipeline import GroundedGenerationPipeline

    return GroundedGenerationPipeline(
        vectorstore,
        documents,
        use_mock=use_mock,
        use_reranking=use_reranking,
        generator_config=generator_config,
    )


class BiomedicalRAGPipeline:
    """Index biomedical documents and answer queries using retrieved evidence."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vectorstore = None
        self.documents: List[BiomedicalDocument] = []
        self.chunks: List[BiomedicalDocument] = []
        self._generation_pipelines = {}
        self._index_lock = RLock()

    def index_documents(
        self,
        documents: Sequence[BiomedicalDocument],
    ) -> None:
        """Preprocess and index biomedical documents with their metadata."""
        if not documents:
            raise ValueError("Cannot index an empty document collection.")

        if any(not isinstance(doc, BiomedicalDocument) for doc in documents):
            raise TypeError(
                "index_documents expects BiomedicalDocument instances."
            )

        from rag.preprocessing import preprocess_documents

        processed_documents = preprocess_documents(documents)
        if not processed_documents:
            raise ValueError(
                "Cannot index documents: preprocessing produced no usable content."
            )

        vectorstore, chunks = _build_index(processed_documents)
        with self._index_lock:
            self.vectorstore = vectorstore
            self.documents = processed_documents
            self.chunks = chunks
            self._generation_pipelines.clear()

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_reranking: bool = True,
        retrieval_mode: str = "hybrid",
        chat_history: Optional[list] = None,
        retrieval_query: Optional[str] = None,
        source_screen=None,
    ) -> RAGResponse:
        """Retrieve evidence and generate a safety-checked grounded answer."""
        with self._index_lock:
            return self._query_locked(
                question, top_k, use_reranking, retrieval_mode,
                chat_history, retrieval_query, source_screen,
            )

    def _query_locked(
        self, question, top_k, use_reranking, retrieval_mode,
        chat_history, retrieval_query, source_screen,
    ) -> RAGResponse:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if retrieval_mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError("retrieval_mode must be hybrid, dense, or sparse.")

        from rag.disclaimer import append_disclaimer
        from rag.medical_risk import classify_medical_risk, is_medical_followup, is_medical_query
        from rag.safety import process_safe_query

        medical_risk = classify_medical_risk(question)
        medical_scope = is_medical_query(question) or is_medical_followup(question, chat_history)
        if not medical_scope and medical_risk.tier not in {"R3", "R4"}:
            return RAGResponse(
                answer="BioMed RAG answers medical and biomedical questions only. Ask about a health condition, symptom, treatment evidence, anatomy, or biology topic.",
                sources=[], confidence=None, is_grounded=False,
                warnings=["The question is outside the medical and biomedical scope."],
                retrieval_time_ms=0.0, generation_time_ms=0.0,
                generation_success=False, validation_passed=False,
                answer_abstained=True, decision="out_of_scope",
                rejection_reason="non_medical_question", risk_tier=medical_risk.tier,
                question_category=medical_risk.category,
            )
        if medical_risk.tier in {"R2", "R3", "R4"}:
            emergency = medical_risk.tier == "R4"
            individualized = medical_risk.tier == "R2"
            response = (
                "If this may be a medical emergency, contact your local emergency "
                "service now. I can’t assess an individual emergency in chat."
                if emergency else
                "This may need urgent medical attention. Contact a qualified "
                "health professional or urgent care service now, especially if "
                "symptoms are severe or worsening."
            )
            if individualized:
                response = (
                    "I can share general medical information, but I can’t diagnose you "
                    "or recommend an individual treatment decision. Please discuss "
                    "your situation with a qualified healthcare professional."
                )
            return RAGResponse(
                answer=append_disclaimer(response), sources=[], confidence=None,
                is_grounded=False, warnings=["High-risk query safety override."],
                retrieval_time_ms=0.0, generation_time_ms=0.0,
                generation_success=False, validation_passed=False,
                answer_abstained=True, decision="safety_override",
                rejection_reason=f"medical_{medical_risk.label}_override",
                risk_tier=medical_risk.tier, question_category=medical_risk.category,
            )

        safety_result = process_safe_query(question)
        if not safety_result.allowed:
            return RAGResponse(
                answer=safety_result.response,
                sources=[],
                confidence=None,
                is_grounded=False,
                warnings=[safety_result.reason],
                retrieval_time_ms=0.0,
                generation_time_ms=0.0,
                retrieval_grounded=False,
                generation_success=False,
                validation_passed=False,
                answer_abstained=True,
                decision="safety_override",
                rejection_reason="query_safety_blocked",
                risk_tier=medical_risk.tier,
                question_category=medical_risk.category,
            )

        generation_config = self.config.get("generator", {})
        use_mock = generation_config.get(
            "use_mock",
            self.config.get("use_mock", True),
        )
        generation_pipeline = self._generation_pipelines.get(use_reranking)
        if generation_pipeline is None:
            generation_pipeline = _build_generation_pipeline(
                self.vectorstore,
                self.chunks,
                use_mock=use_mock,
                use_reranking=use_reranking,
                generator_config=generation_config,
            )
            self._generation_pipelines[use_reranking] = generation_pipeline

        generate_kwargs = {
            "k": top_k,
            "retrieval_k": max(
                top_k,
                self.config.get("retriever", {}).get("retrieval_k", 20),
            ),
        }
        if retrieval_mode != "hybrid":
            generate_kwargs["retrieval_mode"] = retrieval_mode
        if chat_history:
            generate_kwargs["chat_history"] = chat_history[-6:]
        if retrieval_query:
            generate_kwargs["retrieval_query"] = retrieval_query
        if source_screen:
            generate_kwargs["source_screen"] = source_screen
        result = generation_pipeline.generate(safety_result.query, **generate_kwargs)

        validation = result["validation"]
        warnings = list(validation.get("issues", []))
        generation_reported = "generation" in result
        generation = result.get("generation", {})
        generation_success = bool(
            generation.get("success", validation.get("valid", False))
        )

        explicit_decision = result.get("decision")
        decision = explicit_decision
        if decision is None:
            decision = (
                "generation_failed" if generation_reported and not generation_success
                else "answer_approved" if validation.get("valid")
                else "insufficient_evidence"
            )
        if decision in {"safety_override", "out_of_scope", "validation_review_required"}:
            answer = result.get("final_answer") or result.get("generation", {}).get("answer") or "This question needs urgent medical attention."
        elif decision == "no_relevant_evidence":
            answer = result.get("final_answer") or "No sufficiently relevant passages were found in the available knowledge base."
        elif decision == "generation_failed":
            answer = result.get("final_answer") or "Answer generation failed. Review any retrieved sources or retry the question."
        elif decision == "insufficient_evidence":
            answer = (
                result.get("final_answer")
                if explicit_decision == "insufficient_evidence"
                else "I could not find enough supporting evidence to answer confidently."
            )
        elif generation_reported and not generation_success:
            failure_detail = generation.get("error") or generation.get("answer")
            if not failure_detail:
                failure_detail = (
                    "The configured language model did not return an answer."
                )
            answer = str(failure_detail)
            if not answer.lower().startswith("generation failed:"):
                answer = f"Generation failed: {answer}"
        elif decision in {"answer_approved", "informational"}:
            answer = result["final_answer"]
        else:
            answer = "I could not find enough supporting evidence to answer confidently."
        if decision != "out_of_scope":
            answer = append_disclaimer(answer)

        evidence = result["compression"].get("evidence", [])
        sources = []
        for item in evidence:
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                continue
            sources.append(
                {
                    **metadata,
                    "evidence_id": item.get("evidence_id", f"evidence-{item.get('id', len(sources) + 1)}"),
                    "text": item.get("source_text") or item.get("content", ""),
                    "prompt_excerpt": item.get("prompt_excerpt", item.get("content", "")),
                    "rerank_score": item.get("rerank_score", 0.0),
                }
            )
        timings = result.get("timings", {})

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=None,
            is_grounded=bool(
                result["retrieval"]["grounded"]
                and generation_success
                and validation.get("valid")
            ),
            warnings=warnings,
            retrieval_time_ms=float(timings.get("retrieval_time_ms", 0.0)),
            generation_time_ms=float(timings.get("generation_time_ms", 0.0)),
            retrieval_grounded=bool(result["retrieval"]["grounded"]),
            generation_success=generation_success,
            validation_passed=bool(validation.get("valid")),
            answer_abstained=bool(result.get("answer_abstained", False)),
            decision=decision,
            rejection_reason=result.get("rejection_reason"),
            validation_diagnostics=validation.get("diagnostics"),
            audit_trace=result.get("audit_trace"),
            risk_tier=result.get("risk_tier", medical_risk.tier),
            question_category=medical_risk.category,
            claim_support=validation.get("claim_support", []),
            evidence_confidence=(
                "strong" if decision == "answer_approved"
                else "moderate" if decision == "informational"
                else "limited" if decision == "validation_review_required"
                else "low"
            ),
            confidence_basis=(
                "Heuristic summary of retrieval relevance, citation references, and lexical checks; not a calibrated probability or clinical accuracy assessment."
            ),
            validation_states={
                "RETRIEVAL_RELEVANCE_VALIDATED": bool(result["retrieval"].get("grounded")),
                "CITATION_REFERENCES_VALIDATED": bool(validation.get("citation_references_validated", False)),
                "CLAIM_SUPPORT_VALIDATED": "NOT ESTABLISHED",
                "CLAIM_SUPPORT_SCREEN_PASSED": bool(validation.get("claim_support_screen_passed", False)),
                "ANSWER_QUALITY_EVALUATED": None,
                "MEDICAL_ACCURACY_NOT_ESTABLISHED": "NOT ESTABLISHED",
                "claim_support_method": validation.get("claim_support_method", "not-evaluated"),
            },
        )


# Keep the original misspelled public name working for existing imports.
BiomedicialRAGPipeline = BiomedicalRAGPipeline
