"""End-to-end biomedical retrieval and grounded-answer pipeline."""

from dataclasses import dataclass
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
    ) -> RAGResponse:
        """Retrieve evidence and generate a safety-checked grounded answer."""
        if self.vectorstore is None:
            raise RuntimeError(
                "The RAG pipeline has no index. Call index_documents first."
            )
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if retrieval_mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError("retrieval_mode must be hybrid, dense, or sparse.")

        from rag.disclaimer import append_disclaimer
        from rag.safety import process_safe_query

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
        result = generation_pipeline.generate(safety_result.query, **generate_kwargs)

        validation = result["validation"]
        warnings = list(validation.get("issues", []))
        generation_reported = "generation" in result
        generation = result.get("generation", {})
        generation_success = bool(
            generation.get("success", validation.get("valid", False))
        )

        if generation_reported and not generation_success:
            failure_detail = generation.get("error") or generation.get("answer")
            if not failure_detail:
                failure_detail = (
                    "The configured language model did not return an answer."
                )
            answer = str(failure_detail)
            if not answer.lower().startswith("generation failed:"):
                answer = f"Generation failed: {answer}"
        elif validation.get("valid"):
            answer = result["final_answer"]
        else:
            answer = (
                "I could not produce an answer that passed the evidence and "
                "safety checks."
            )
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
                    "text": item.get("content", ""),
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
        )


# Keep the original misspelled public name working for existing imports.
BiomedicialRAGPipeline = BiomedicalRAGPipeline
