"""Dashboard adapter for the repository's grounded biomedical RAG pipeline."""

from io import BytesIO
import inspect
from contextlib import nullcontext
from pathlib import Path
import re
from dataclasses import dataclass
from collections import OrderedDict
from threading import RLock
from time import monotonic
from typing import Callable, Iterator, List, Optional, Tuple
from uuid import uuid4

from rag.document import BiomedicalDocument
from rag.pipeline import BiomedicalRAGPipeline
from rag.grounding import has_topic_overlap
from app.ragguard_observation import (
    RAGGuardObserver,
    enqueue_observation,
    get_server_setting,
)
from app.ragguard_repair import (
    MAX_REPAIR_CONTEXTS_PER_SERVICE,
    REPAIR_CONTEXT_TTL_SECONDS,
    PendingRepair,
    RepairRequest,
    RepairResult,
    audit_repair,
)

BASE_CORPUS_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "downloaded_sources"
)
@dataclass(frozen=True)
class RepairOutcome:
    """Public, non-sensitive result of an application-side repair request."""

    operation: str
    approved: bool
    executed: bool
    success: bool
    message: str


class ApplicationRepairAdapter:
    """Run a small allowlist of index repairs through the normal RAG service.

    Authorization is supplied by the hosting application because this project
    does not currently establish an authenticated principal. Missing or failing
    authorization always denies the request.
    """

    SUPPORTED_OPERATIONS = frozenset({"rebuild_index", "refresh_default_corpus"})

    def __init__(
        self,
        service: "RAGService",
        authorize: Optional[Callable[[object, str], bool]] = None,
    ):
        self._service = service
        self._authorize = authorize

    def execute(self, operation: str, *, principal: object = None) -> RepairOutcome:
        if not isinstance(operation, str) or operation not in self.SUPPORTED_OPERATIONS:
            return RepairOutcome(
                str(operation), False, False, False, "Unsupported repair operation."
            )

        if principal is None or self._authorize is None:
            return RepairOutcome(operation, False, False, False, "Authorization required.")
        try:
            authorized = bool(self._authorize(principal, operation))
        except Exception:
            authorized = False
        if not authorized:
            return RepairOutcome(operation, False, False, False, "Authorization denied.")

        try:
            if operation == "rebuild_index":
                if not self._service._documents:
                    return RepairOutcome(
                        operation, True, False, False, "No documents are available to index."
                    )
                self._service.pipeline.index_documents(self._service._documents)
                return RepairOutcome(operation, True, True, True, "Index rebuilt successfully.")

            changed = self._service.refresh_default_corpus()
            return RepairOutcome(
                operation,
                True,
                changed,
                True,
                "Default corpus loaded." if changed else "No corpus refresh was needed or available.",
            )
        except Exception:
            # Avoid returning source contents, model details, or internal errors.
            return RepairOutcome(operation, True, True, False, "Repair operation failed.")


def load_default_corpus() -> List[BiomedicalDocument]:
    """Load the repository's prepared biomedical corpus when it is available."""
    if not BASE_CORPUS_DIR.is_dir():
        return []

    from rag.ingestion import load_normalized_documents

    return load_normalized_documents(BASE_CORPUS_DIR)


class RAGService:
    """Expose the upload and query methods expected by the Streamlit UI."""

    def __init__(self, provider: str, model: str, api_key: str):
        provider = provider.lower()
        if provider not in {"openai", "cohere", "huggingface"}:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        if not api_key or not api_key.strip():
            raise ValueError("An API key is required to initialize the LLM.")

        self.provider = provider
        self.model = model
        # Optional telemetry; missing configuration leaves BiomedicalRag unchanged.
        self.ragguard_observer = RAGGuardObserver.from_environment()
        self._ragguard_pending_repairs: OrderedDict[str, PendingRepair] = OrderedDict()
        self._ragguard_pending_lock = RLock()
        self._ingest_lock = RLock()
        self._documents = load_default_corpus()
        self.pipeline = BiomedicalRAGPipeline(
            {
                "generator": {
                    "provider": provider,
                    "model_name": model,
                    "api_key": api_key,
                    "use_mock": False,
                },
                "retriever": {"retrieval_k": 20},
            }
        )
        if self._documents:
            self.pipeline.index_documents(self._documents)

    def create_repair_adapter(
        self,
        authorize: Optional[Callable[[object, str], bool]] = None,
    ) -> ApplicationRepairAdapter:
        """Create a fail-closed adapter using the host application's auth check."""
        return ApplicationRepairAdapter(self, authorize=authorize)

    @property
    def total_chunks(self) -> int:
        return len(self.pipeline.chunks)

    @property
    def total_documents(self) -> int:
        return len(self._documents)

    @property
    def source_names(self) -> List[str]:
        return sorted({
            str(document.metadata.get("source", "Unknown"))
            for document in self._documents
        })

    def refresh_default_corpus(self) -> bool:
        """Load the prepared corpus if this cached service has no index yet.

        Streamlit can keep a service in its session after the local corpus has
        been downloaded or generated. Refreshing an empty service here avoids
        making users upload a document just to unlock the prepared corpus.
        """
        if self.pipeline.vectorstore is not None:
            return False

        documents = load_default_corpus()
        if not documents:
            return False

        self._documents = documents
        self.pipeline.index_documents(self._documents)
        return True

    def ingest(self, file_bytes: bytes, filename: str) -> int:
        """Extract and index one uploaded PDF, DOCX, or UTF-8 text file."""
        with self._ingest_lock:
            return self._ingest_locked(file_bytes, filename)

    def _ingest_locked(self, file_bytes: bytes, filename: str) -> int:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            content = "\n".join(
                page.extract_text() or ""
                for page in PdfReader(BytesIO(file_bytes)).pages
            )
        elif suffix == ".docx":
            from docx import Document

            document = Document(BytesIO(file_bytes))
            content = "\n".join(paragraph.text for paragraph in document.paragraphs)
        elif suffix == ".txt":
            content = file_bytes.decode("utf-8-sig")
        else:
            raise ValueError("Supported document formats are PDF, DOCX, and TXT.")

        if not content.strip():
            raise ValueError(f"No readable text was found in {filename}.")

        replacement = BiomedicalDocument(
            content=content,
            metadata={"source": filename, "file": filename},
        )
        existing_index = next(
            (
                index for index, document in enumerate(self._documents)
                if document.metadata.get("source") == filename
            ),
            None,
        )
        if existing_index is None:
            self._documents.append(replacement)
        else:
            self._documents[existing_index] = replacement
        self.pipeline.index_documents(self._documents)
        return len(self.pipeline.chunks)

    def query_stream(
        self,
        question: str,
        chat_history: Optional[list] = None,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> Tuple[Iterator[str], list]:
        """Run a checked RAG query and return answer chunks plus provenance."""
        chunks, sources, _ = self.query_with_details(
            question,
            chat_history=chat_history,
            top_k=top_k,
            mode=mode,
        )
        return chunks, sources

    def query_with_details(
        self,
        question: str,
        chat_history: Optional[list] = None,
        top_k: int = 5,
        mode: str = "hybrid",
        answer_mode: str = "research",
        source_screen=None,
    ) -> Tuple[Iterator[str], list, dict]:
        """Return answer chunks, provenance, and checked pipeline details."""
        if mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        if source_screen is None:
            from dashboard.security_controls import scan_sources

            source_screen = lambda ranked: scan_sources([
                str(item.get("content", "")) for item in ranked
            ])

        query_options = {
            "top_k": top_k,
            "retrieval_mode": mode,
            "use_reranking": False,
        }
        query_parameters = inspect.signature(self.pipeline.query).parameters.values()
        if source_screen is not None and any(
            parameter.name == "source_screen"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in query_parameters
        ):
            query_options["source_screen"] = source_screen
        if any(
            parameter.name == "answer_mode"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in query_parameters
        ):
            query_options["answer_mode"] = answer_mode
        if chat_history:
            query_options["chat_history"] = chat_history[-6:]
        response = self.pipeline.query(question, **query_options)
        sources = [dict(source) for source in response.sources]
        chunks = re.findall(r"\S+\s*", response.answer)
        validation_states = getattr(response, "validation_states", None)
        details = {
            "is_grounded": bool(getattr(response, "is_grounded", False)),
            "retrieval_grounded": bool(
                getattr(response, "retrieval_grounded", getattr(response, "is_grounded", False))
            ),
            "generation_success": bool(
                getattr(response, "generation_success", True)
            ),
            "answer_abstained": bool(getattr(response, "answer_abstained", False)),
            "decision": str(getattr(response, "decision", "answer_approved")),
            "rejection_reason": getattr(response, "rejection_reason", None),
            "validation_diagnostics": getattr(response, "validation_diagnostics", None),
            "audit_trace": getattr(response, "audit_trace", None),
            "risk_tier": str(getattr(response, "risk_tier", "R0")),
            "question_category": str(getattr(response, "question_category", "general_biomedical_education")),
            "claim_support": getattr(response, "claim_support", []),
            "evidence_confidence": getattr(response, "evidence_confidence", None),
            "confidence_basis": getattr(response, "confidence_basis", None),
            "learning_answer": getattr(response, "learning_answer", None),
            "validation_passed": bool(
                getattr(response, "validation_passed", not getattr(response, "warnings", []))
            ),
            "warnings": list(getattr(response, "warnings", [])),
            "retrieval_time_ms": float(
                getattr(response, "retrieval_time_ms", 0.0)
            ),
            "generation_time_ms": float(
                getattr(response, "generation_time_ms", 0.0)
            ),
        }
        if validation_states:
            details["validation_states"] = validation_states

        request_id = str(uuid4())
        chat_history_for_repair = chat_history[-6:] if chat_history else None
        retrieval_query = query_options.get("retrieval_query")
        self._remember_ragguard_request(
            request_id,
            PendingRepair(
                created_at=monotonic(),
                question=question,
                mode=mode,
                top_k=top_k,
                chat_history=chat_history_for_repair,
                retrieval_query=retrieval_query,
                original_details=details,
            ),
        )

        # Queue telemetry after generation. It has its own bounded delivery
        # budget and cannot delay or replace the answer.
        observer = self.ragguard_observer
        if observer is not None:
            chunk_scores = []
            for source in sources:
                chunk_id = source.get("chunk_id") or source.get("id") or source.get("document_id")
                score = source.get("rerank_score", source.get("score"))
                if chunk_id is not None and score is not None:
                    try:
                        chunk_scores.append((str(chunk_id), float(score)))
                    except (TypeError, ValueError):
                        continue
            try:
                enqueued = enqueue_observation(
                    observer,
                    query=question,
                    request_id=request_id,
                    retrieved_chunks=chunk_scores,
                    retrieval_method=mode,
                    retrieval_latency_ms=details["retrieval_time_ms"],
                    generation_status=("succeeded" if details["generation_success"] else "failed"),
                )
            except Exception:
                enqueued = False
            if not enqueued:
                self._discard_ragguard_request(request_id)
        else:
            self._discard_ragguard_request(request_id)

        return iter(chunks), sources, details

    def _remember_ragguard_request(self, request_id: str, pending: PendingRepair) -> None:
        """Retain minimal query context for a delayed, correlated repair call."""
        if not get_server_setting("RAGGUARD_REPAIR_TOKEN"):
            return
        with self._ragguard_pending_lock:
            now = monotonic()
            expired = [
                key for key, item in self._ragguard_pending_repairs.items()
                if now - item.created_at > REPAIR_CONTEXT_TTL_SECONDS
            ]
            for key in expired:
                self._ragguard_pending_repairs.pop(key, None)
            self._ragguard_pending_repairs[request_id] = pending
            while len(self._ragguard_pending_repairs) > MAX_REPAIR_CONTEXTS_PER_SERVICE:
                self._ragguard_pending_repairs.popitem(last=False)

    def _discard_ragguard_request(self, request_id: str) -> None:
        with self._ragguard_pending_lock:
            self._ragguard_pending_repairs.pop(request_id, None)

    def execute_ragguard_repair(self, request: RepairRequest) -> RepairResult:
        """Validate one query-scoped candidate without changing future queries."""
        with self._ragguard_pending_lock:
            pending = self._ragguard_pending_repairs.get(request.request_id)
            if pending is None or monotonic() - pending.created_at > REPAIR_CONTEXT_TTL_SECONDS:
                self._ragguard_pending_repairs.pop(request.request_id, None)
                audit_repair(
                    request_id=request.request_id,
                    strategy=request.strategy,
                    authorized=True,
                    outcome="request_context_expired",
                    validated=False,
                    context_verified=False,
                )
                return RepairResult(
                    request_id=request.request_id,
                    strategy=request.strategy,
                    status="rejected",
                    validated=False,
                    context_verified=False,
                    improvement=None,
                )
            if pending.result is not None:
                if pending.strategy == request.strategy:
                    audit_repair(
                        request_id=request.request_id,
                        strategy=request.strategy,
                        authorized=True,
                        outcome="idempotent_replay",
                        validated=pending.result.validated,
                        context_verified=pending.result.context_verified,
                    )
                    return pending.result
                audit_repair(
                    request_id=request.request_id,
                    strategy=request.strategy,
                    authorized=True,
                    outcome="request_already_used",
                    validated=False,
                    context_verified=False,
                )
                return RepairResult(
                    request_id=request.request_id,
                    strategy=request.strategy,
                    status="rejected",
                    validated=False,
                    context_verified=False,
                    improvement=None,
                )
            if pending.attempted:
                audit_repair(
                    request_id=request.request_id,
                    strategy=request.strategy,
                    authorized=True,
                    outcome="request_already_used",
                    validated=False,
                    context_verified=False,
                )
                return RepairResult(
                    request_id=request.request_id,
                    strategy=request.strategy,
                    status="rejected",
                    validated=False,
                    context_verified=False,
                    improvement=None,
                )
            pending.attempted = True
            pending.strategy = request.strategy

        if request.strategy not in {"HYBRID_RETRIEVAL", "RERANK"}:
            result = RepairResult(
                request_id=request.request_id,
                strategy=request.strategy,
                status="rejected",
                validated=False,
                context_verified=False,
                improvement=None,
            )
            self._finish_ragguard_repair(request.request_id, result)
            return result

        target_mode = "hybrid" if request.strategy == "HYBRID_RETRIEVAL" else pending.mode
        target_reranking = request.strategy == "RERANK"
        if request.strategy == "HYBRID_RETRIEVAL" and pending.mode == "hybrid":
            result = RepairResult(
                request_id=request.request_id,
                strategy=request.strategy,
                status="rejected",
                validated=True,
                context_verified=True,
                improvement=0.0,
            )
            self._finish_ragguard_repair(request.request_id, result)
            return result

        try:
            # Check all likely candidate evidence before allowing it into the
            # generator. Repair execution cannot bypass the source attack scan.
            from dashboard.security_controls import scan_sources

            preflight = self.retrieve(
                pending.question,
                top_k=max(20, pending.top_k),
                mode=target_mode,
            )
            if scan_sources([
                str(item.get("text", item.get("content", "")))
                for item in preflight
            ]):
                raise ValueError("Candidate evidence failed source security screening.")

            response = self.pipeline.query(
                pending.question,
                top_k=pending.top_k,
                retrieval_mode=target_mode,
                use_reranking=target_reranking,
                **({"chat_history": pending.chat_history} if pending.chat_history else {}),
                **({"retrieval_query": pending.retrieval_query} if pending.retrieval_query else {}),
            )
            candidate_details = {
                "is_grounded": bool(getattr(response, "is_grounded", False)),
                "retrieval_grounded": bool(getattr(response, "retrieval_grounded", False)),
                "generation_success": bool(getattr(response, "generation_success", False)),
                "answer_abstained": bool(getattr(response, "answer_abstained", False)),
                "validation_passed": bool(getattr(response, "validation_passed", False)),
                "warnings": list(getattr(response, "warnings", [])),
                "retrieval_time_ms": float(getattr(response, "retrieval_time_ms", 0.0)),
                "generation_time_ms": float(getattr(response, "generation_time_ms", 0.0)),
            }
            context_verified = bool(
                candidate_details["retrieval_grounded"]
                and candidate_details["is_grounded"]
            )
            validated = bool(
                context_verified
                and candidate_details["generation_success"]
                and candidate_details["validation_passed"]
                and not candidate_details["answer_abstained"]
            )
            original_score = self._repair_quality_score(pending.original_details)
            candidate_score = self._repair_quality_score(candidate_details)
            improvement = round(candidate_score - original_score, 10)
            promoted = validated and improvement > 0
            result = RepairResult(
                request_id=request.request_id,
                strategy=request.strategy,
                status="promoted" if promoted else "rolled_back" if not validated else "rejected",
                validated=validated,
                context_verified=context_verified,
                improvement=improvement,
            )
        except Exception:
            result = RepairResult(
                request_id=request.request_id,
                strategy=request.strategy,
                status="rolled_back",
                validated=False,
                context_verified=False,
                improvement=None,
            )

        self._finish_ragguard_repair(request.request_id, result)
        return result

    @staticmethod
    def _repair_quality_score(details: dict) -> float:
        checks = (
            bool(details.get("retrieval_grounded")),
            bool(details.get("generation_success")),
            bool(details.get("validation_passed")),
            not bool(details.get("answer_abstained")),
        )
        return sum(checks) / len(checks)

    def _finish_ragguard_repair(
        self,
        request_id: str,
        result: RepairResult,
    ) -> None:
        with self._ragguard_pending_lock:
            pending = self._ragguard_pending_repairs.get(request_id)
            if pending is not None:
                pending.result = result
                # Keep only the idempotency record after completion. The
                # biomedical query and conversation context are no longer
                # needed once the repair result has been decided.
                pending.question = ""
                pending.chat_history = None
                pending.retrieval_query = None
                pending.original_details = {}
        audit_repair(
            request_id=request_id,
            strategy=result.strategy,
            authorized=True,
            outcome=result.status,
            validated=result.validated,
            context_verified=result.context_verified,
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list:
        """Retrieve and return ranked source evidence without generating an answer."""
        with getattr(self.pipeline, "_index_lock", nullcontext()):
            return self._retrieve_locked(question, top_k, mode)

    def _retrieve_locked(self, question: str, top_k: int, mode: str) -> list:
        if mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The query must be a non-empty string.")
        from rag.medical_risk import is_medical_query

        if not is_medical_query(question):
            raise ValueError("BioMed RAG retrieves medical and biomedical evidence only.")
        if self.pipeline.vectorstore is None:
            raise RuntimeError("Index documents before running retrieval.")

        from rag.safety import process_safe_query

        safety_result = process_safe_query(question)
        if not safety_result.allowed:
            raise ValueError(safety_result.response)

        if mode == "hybrid":
            from rag.hybrid_retriever import HybridRetriever

            response = HybridRetriever(
                self.pipeline.vectorstore,
                self.pipeline.chunks,
            ).retrieve(safety_result.query, k=top_k)
            retrieved = response["fused_results"][:top_k]
            metadata_by_content = {}
            for result in response["dense_results"]:
                metadata_by_content.setdefault(result.content, result.metadata)
            for result in response["sparse_results"]:
                document = result["document"]
                metadata_by_content.setdefault(document.content, document.metadata)
            candidates = [
                (content, metadata_by_content[content], score)
                for content, score in retrieved
                if content in metadata_by_content
            ]
        elif mode == "dense":
            from rag.retriever import BiomedicalRetriever

            response = BiomedicalRetriever(self.pipeline.vectorstore).retrieve(
                safety_result.query,
                k=top_k,
            )
            candidates = [
                (
                    result.content,
                    result.metadata,
                    result.similarity_score,
                )
                for result in response["results"]
            ]
        else:
            from rag.bm25_retriever import BM25Retriever

            response = BM25Retriever(self.pipeline.documents).search(
                safety_result.query,
                k=top_k,
            )
            candidates = [
                (
                    result["document"].content,
                    result["document"].metadata,
                    result["score"],
                )
                for result in response
            ]

        candidates = [
            item for item in candidates
            if has_topic_overlap(safety_result.query, [item[0]])
        ]
        return [
            {
                "content": content,
                "metadata": dict(metadata),
                "score": float(score),
                "rank": rank,
            }
            for rank, (content, metadata, score) in enumerate(candidates, 1)
        ]
