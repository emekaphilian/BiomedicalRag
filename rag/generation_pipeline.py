"""
Full grounded generation pipeline.
Integrates retrieval → reranking → compression → generation → validation.
"""

import re
from time import perf_counter

from rag.hybrid_retriever import HybridRetriever
from rag.reranking_retriever import RerankingRetriever
from rag.compression import ContextCompressor
from rag.prompt_constructor import GroundedPromptConstructor
from rag.generator import GroundedGenerator
from rag.grounding import has_topic_overlap
from rag.medical_risk import classify_medical_risk, is_medical_followup, is_medical_query
from rag.external_sources import retrieve_pubmed
from rag.post_validation import PostGenerationValidator
from dashboard.rag_audit import content_tokens, split_sentences


def _extractive_fallback(query, evidence_items):
    """Return a relevant, verbatim source span with evidence citations."""
    query_tokens = content_tokens(query)
    cause_terms = {"cause", "causation", "etiology", "etiologies"}
    asks_about_cause = bool(query_tokens & cause_terms)
    # Ignore relational wording when matching the disease/mechanism itself.
    # For example, a source may say "does not make insulin" rather than repeat
    # the question's word "cause".
    relation_terms = {
        "cause", "causation", "etiology", "etiologies", "affect", "effect",
        "impact", "influence", "work", "function", "mechanism", "role",
        "explain", "describe", "show", "tell", "say", "disease", "disorder",
    }
    topic_tokens = query_tokens - relation_terms
    if topic_tokens:
        query_tokens = topic_tokens
    if not query_tokens:
        return None

    cause_support = re.compile(
        r"\b(?:because|due to|result(?:s|ed|ing)? in|lead(?:s|ing)? to|"
        r"without|cause(?:s|d)?|does not make|doesn't make|do not make|"
        r"don't make|did not make|cannot make|can't make|not make|"
        r"does not produce|doesn't produce|do not produce|lack of|"
        r"impaired|dysfunction)\b",
        re.IGNORECASE,
    )

    best = None
    for evidence_index, item in enumerate(evidence_items, start=1):
        content = item.get("content", "")
        sentences = split_sentences(content)
        for start in range(len(sentences)):
            for width in (1, 2):
                span = sentences[start:start + width]
                if len(span) != width or any(sentence.rstrip().endswith("?") for sentence in span):
                    continue
                span_text = " ".join(span)
                if asks_about_cause and not cause_support.search(span_text):
                    continue
                span_tokens = set().union(*(content_tokens(sentence) for sentence in span))
                overlap = query_tokens & span_tokens
                coverage = len(overlap) / len(query_tokens)
                if len(overlap) < min(2, len(query_tokens)) or coverage < 0.3:
                    continue
                rank = (coverage, -width)
                if best is None or rank > best[0]:
                    cited_span = " ".join(
                        f"{sentence.strip()} [Evidence {evidence_index}]"
                        for sentence in span
                    )
                    best = (rank, cited_span)

    if best is None:
        return None
    return best[1]


class GroundedGenerationPipeline:
    """
    End-to-end pipeline for grounded biomedical generation.
    
    Architecture:
    Query → Safety Check → Retrieval → Reranking → Compression → 
    Prompt Construction → LLM Generation → Post-Validation
    """

    def __init__(
        self,
        vectorstore,
        documents,
        use_mock=True,
        use_reranking=True,
        generator_config=None,
    ):
        """
        Initialize generation pipeline.
        
        Args:
            vectorstore: FAISS vectorstore
            documents: Preprocessed documents
            use_mock: Use mock LLM (True) or OpenAI API (False)
        """
        self.documents = documents
        self.use_reranking = use_reranking
        if vectorstore is None:
            self.retriever = None
        elif use_reranking:
            self.retriever = RerankingRetriever(vectorstore, documents)
        else:
            self.retriever = HybridRetriever(vectorstore, documents)
        self.compressor = ContextCompressor(max_chunks=3)
        self.prompt_constructor = GroundedPromptConstructor()
        generator_config = generator_config or {}
        self.generator = GroundedGenerator(
            model_name=generator_config.get("model_name", "gpt-4o"),
            api_key=generator_config.get("api_key"),
            provider=generator_config.get("provider", "openai"),
        )
        self.generator_config = generator_config
        self.validator = PostGenerationValidator()
        self.use_mock = use_mock

    def generate(
        self,
        query,
        k=3,
        retrieval_k=20,
        retrieval_mode="hybrid",
        chat_history=None,
        retrieval_query=None,
        source_screen=None,
    ):
        """
        Generate grounded answer end-to-end.
        
        Args:
            query: User query
            k: Top K results after reranking
            retrieval_k: Number of candidates from hybrid retrieval
            
        Returns:
            Dictionary with full pipeline results
        """
        medical_risk = classify_medical_risk(query)
        if not (is_medical_query(query) or is_medical_followup(query, chat_history)):
            refusal = "BioMed RAG answers medical and biomedical questions only. Ask about a health condition, symptom, treatment evidence, anatomy, or biology topic."
            return {
                "query": query,
                "retrieval": {"fused_count": 0, "reranked_count": 0, "grounded": False},
                "compression": {"compressed_count": 0, "sources": [], "evidence": [], "formatted": "Out-of-scope query."},
                "generation": {"answer": refusal, "model": getattr(self.generator, "model_name", self.generator_config.get("model_name", "unknown")), "success": False},
                "validation": {"valid": False, "issues": ["The question is outside the medical and biomedical scope."]},
                "timings": {"retrieval_time_ms": 0.0, "generation_time_ms": 0.0},
                "final_answer": refusal,
                "answer_abstained": True,
                "decision": "out_of_scope",
                "rejection_reason": "non_medical_question",
                "risk_tier": medical_risk.tier,
            }

        retrieval_started = perf_counter()
        search_query = retrieval_query or query
        hybrid_retriever = getattr(self.retriever, "hybrid_retriever", self.retriever)
        dense_retriever = getattr(hybrid_retriever, "dense_retriever", None)
        if dense_retriever is None or getattr(dense_retriever, "vectorstore", None) is None:
            retrieval_result = {
                "grounded": False,
                "fused_results": [],
                "dense_results": [],
                "sparse_results": [],
                "reranked_results": [],
            }
            ranked_results = []
        elif self.use_reranking:
            retrieval_result = self.retriever.retrieve(
                search_query,
                k=k,
                retrieval_k=retrieval_k,
            )
            ranked_results = retrieval_result["reranked_results"]
        else:
            hybrid_result = self.retriever.retrieve(search_query, k=retrieval_k)
            metadata_by_content = {}
            for document in self.documents:
                metadata_by_content.setdefault(
                    document.content,
                    document.metadata,
                )
            if retrieval_mode == "dense":
                selected_results = [
                    (result.content, result.similarity_score)
                    for result in hybrid_result["dense_results"]
                ]
            elif retrieval_mode == "sparse":
                selected_results = [
                    (result["document"].content, result["score"])
                    for result in hybrid_result["sparse_results"]
                ]
            else:
                selected_results = hybrid_result["fused_results"]
            ranked_results = [
                {
                    "content": content,
                    "metadata": metadata_by_content[content],
                    "rerank_score": score,
                }
                for content, score in selected_results
                if content in metadata_by_content
            ][:k]
            retrieval_result = {
                **hybrid_result,
                "grounded": (
                    any(score > 0 for _, score in selected_results)
                    if retrieval_mode == "sparse"
                    else hybrid_result["grounded"]
                ),
                "reranked_results": ranked_results,
            }
        retrieval_time_ms = (perf_counter() - retrieval_started) * 1000
        external_count = 0

        local_topic_match = has_topic_overlap(
            query,
            [item.get("content", "") for item in ranked_results],
        )
        if not retrieval_result.get("grounded") or not local_topic_match:
            external_started = perf_counter()
            external_results = retrieve_pubmed(query, limit=min(max(k, 3), 5))
            retrieval_time_ms += (perf_counter() - external_started) * 1000
            relevant_external = [
                item for item in external_results
                if has_topic_overlap(query, [item.get("content", "")])
            ]
            if relevant_external:
                external_count = len(relevant_external)
                if local_topic_match and retrieval_result.get("grounded"):
                    ranked_results = (ranked_results + relevant_external)[:max(k, 3)]
                else:
                    ranked_results = (relevant_external + ranked_results[:max(0, k - len(relevant_external))])[:max(k, 3)]
                retrieval_result = {**retrieval_result, "grounded": True}

        if not ranked_results or not has_topic_overlap(
            search_query,
            [item.get("content", "") for item in ranked_results],
        ):
            refusal = "The available sources did not contain enough relevant evidence to answer this question reliably. Refine the question or add a source that directly covers the topic."
            evidence_reason = "evidence_too_weak" if ranked_results else "no_relevant_evidence"
            return {
                "query": query,
                "retrieval": {
                    "fused_count": len(retrieval_result.get("fused_results", [])) + external_count,
                    "reranked_count": len(ranked_results),
                    "grounded": False,
                },
                "compression": {
                    "compressed_count": 0,
                    "sources": [item.get("metadata", {}) for item in ranked_results],
                    "evidence": ranked_results,
                    "formatted": "Retrieved passages did not meet the evidence relevance threshold.",
                },
                "generation": {
                    "answer": refusal,
                    "model": getattr(self.generator, "model_name", self.generator_config.get("model_name", "unknown")),
                    "success": True,
                },
                "validation": {
                    "valid": False,
                    "issues": [
                        "No relevant evidence was retrieved."
                        if evidence_reason == "no_relevant_evidence"
                        else "Retrieved passages did not meet the evidence relevance threshold."
                    ],
                },
                "timings": {
                    "retrieval_time_ms": retrieval_time_ms,
                    "generation_time_ms": 0.0,
                },
                "final_answer": refusal,
                "answer_abstained": True,
                "decision": "no_relevant_evidence" if evidence_reason == "no_relevant_evidence" else "insufficient_evidence",
                "rejection_reason": evidence_reason,
                "risk_tier": medical_risk.tier,
            }

        # Step 2: Context Compression
        compressed_context = self.compressor.compress(
            ranked_results
        )

        # Screen the exact compressed evidence items that become the prompt.
        if source_screen is not None and source_screen(compressed_context["context"]):
            refusal = "Generation was blocked because retrieved evidence matched a prompt-injection screen."
            return {
                "query": query,
                "retrieval": {"fused_count": len(retrieval_result.get("fused_results", [])) + external_count, "reranked_count": len(ranked_results), "grounded": True},
                "compression": {"compressed_count": compressed_context["evidence_count"], "sources": compressed_context["sources"], "evidence": compressed_context["context"], "formatted": "Evidence blocked by security screen."},
                "generation": {"answer": refusal, "model": getattr(self.generator, "model_name", self.generator_config.get("model_name", "unknown")), "success": False},
                "validation": {"valid": False, "issues": ["Retrieved evidence matched the prompt-injection screen."]},
                "timings": {"retrieval_time_ms": retrieval_time_ms, "generation_time_ms": 0.0},
                "final_answer": refusal,
                "answer_abstained": True,
                "decision": "validation_review_required",
                "rejection_reason": "source_prompt_injection_blocked",
                "risk_tier": medical_risk.tier,
            }

        # Step 3: Prompt Construction
        prompt = (
            self.prompt_constructor.construct(query, compressed_context, chat_history)
            if chat_history
            else self.prompt_constructor.construct(query, compressed_context)
        )

        # Step 4: LLM Generation
        generation_started = perf_counter()
        if self.use_mock:
            generation_result = self.generator.generate_mock(prompt)
        else:
            generation_result = self.generator.generate(
                prompt,
                max_tokens=self.generator_config.get("max_tokens", 500),
                temperature=self.generator_config.get("temperature", 0.3),
            )
        generation_time_ms = (perf_counter() - generation_started) * 1000

        # Step 5: Post-Generation Validation
        validation_result = self.validator.validate(
            generation_result,
            compressed_context["evidence_count"],
            [item["content"] for item in compressed_context["context"]],
            evidence_records=compressed_context["context"],
        )
        if generation_result.get("success") and not has_topic_overlap(
            query, [generation_result.get("answer", "")]
        ):
            validation_result["valid"] = False
            validation_result.setdefault("issues", []).append(
                "Generated answer did not match the question topic."
            )

        # Retry once for a failed safety/grounding check. Model output can contain
        # reasoning text or omit citations even when the evidence is adequate. The
        # retry is revalidated with the same strict checks; it never bypasses them.
        should_retry_for_validation = (
            not self.use_mock
            and isinstance(prompt, dict)
            and compressed_context["evidence_count"] > 0
            and generation_result.get("success", False)
            and not validation_result.get("valid", False)
        )

        if should_retry_for_validation:
            retry_prompt = dict(prompt)
            retry_prompt["system"] = (
                f"{prompt.get('system', '')}\n\n"
                "CITATION RECOVERY REQUIREMENTS (AND OTHER VALIDATION FAILURES):\n"
                "- Answer using only the supplied retrieved evidence.\n"
                "- Cite every factual sentence using the exact evidence labels "
                "provided in the evidence text, such as [Evidence 1].\n"
                "- Never invent evidence labels or citations.\n"
                "- If the evidence is insufficient, use the required refusal.\n"
                "- Do not provide personalized medical advice or unsupported claims.\n"
                "- Return only the concise final answer. Never show reasoning, "
                "analysis, control tokens, or instructions.\n"
                "The previous draft failed these checks: "
                + "; ".join(str(issue) for issue in validation_result.get("issues", []))
            )
            retry_prompt["user"] = (
                f"{prompt.get('user', '')}\n\n"
                "Your previous response failed validation. Produce a corrected "
                "answer that satisfies all correction requirements above. Use "
                "only the original supplied evidence."
            )

            retry_started = perf_counter()
            retry_result = self.generator.generate(
                retry_prompt,
                max_tokens=self.generator_config.get("max_tokens", 500),
                temperature=self.generator_config.get("temperature", 0.3),
            )
            generation_time_ms += (
                perf_counter() - retry_started
            ) * 1000

            retry_validation = self.validator.validate(
                retry_result,
                compressed_context["evidence_count"],
                [item["content"] for item in compressed_context["context"]],
                evidence_records=compressed_context["context"],
            )
            if retry_result.get("success") and not has_topic_overlap(
                query, [retry_result.get("answer", "")]
            ):
                retry_validation["valid"] = False
                retry_validation.setdefault("issues", []).append(
                    "Generated answer did not match the question topic."
                )

            generation_result = retry_result
            validation_result = retry_validation

        # Refusals pass the validator's no-evidence-citation rule by design.
        # With relevant retrieved evidence, that can accidentally prevent the
        # extractive recovery path from running. For low-risk educational
        # questions, try a verbatim cited passage when the model abstains or its
        # draft fails validation. The extractive answer still has to pass the
        # normal citation, safety, and support checks.
        model_refusal_pattern = re.compile(
            r"\s*(?:i\s+(?:cannot|can't|could not|couldn't)\b|"
            r"there is insufficient evidence\b|not enough information\b)",
            re.IGNORECASE,
        )
        used_extractive_fallback = False
        should_try_extractive_fallback = (
            not validation_result.get("valid")
            or model_refusal_pattern.match(str(generation_result.get("answer", "")))
        )
        if (
            should_try_extractive_fallback
            and medical_risk.tier in {"R0", "R1"}
        ):
            extractive_answer = _extractive_fallback(
                query, compressed_context["context"],
            )
            if extractive_answer:
                extractive_result = {
                    **generation_result,
                    "answer": extractive_answer,
                    "success": True,
                }
                extractive_validation = self.validator.validate(
                    extractive_result,
                    compressed_context["evidence_count"],
                    [item["content"] for item in compressed_context["context"]],
                    evidence_records=compressed_context["context"],
                )
                if extractive_validation.get("valid"):
                    generation_result = extractive_result
                    validation_result = extractive_validation
                    used_extractive_fallback = True

        generated_draft = str(generation_result.get("answer", ""))
        issues = [str(issue) for issue in validation_result.get("issues", [])]
        model_refusal = bool(re.match(
            r"\s*(?:i\s+(?:cannot|can't|could not|couldn't)\b|"
            r"there is insufficient evidence\b|not enough information\b)",
            generated_draft,
            re.IGNORECASE,
        ))
        if model_refusal and validation_result.get("valid"):
            validation_result["valid"] = False
            validation_result.setdefault("issues", []).append(
                "The model abstained despite relevant evidence being retrieved."
            )
        has_external_source = any(
            (item.get("metadata") or {}).get("source_origin") == "external"
            for item in compressed_context.get("context", [])
        )
        has_relevant_evidence = bool(
            compressed_context["evidence_count"]
            and has_topic_overlap(
                query,
                [item.get("content", "") for item in compressed_context["context"]],
            )
        )
        diagnostics = validation_result.get("diagnostics") or {}
        informational_issues = {
            "One or more sentences failed the lexical claim-support screen",
            "One or more factual sentences lack a citation linked to overlapping evidence",
            "No evidence citations found in answer",
        }
        citation_support = diagnostics.get("citation_support_coverage") or 0
        informational_eligible = (
            not validation_result.get("valid", False)
            and medical_risk.tier in {"R0", "R1"}
            and generation_result.get("success", False)
            and bool(issues)
            and set(issues).issubset(informational_issues)
            and validation_result.get("citation_references_validated") is True
            and citation_support == 1.0
            and diagnostics.get("lexical_support") == 1.0
            and diagnostics.get("unsupported_number_count", 0) == 0
            and has_relevant_evidence
        )
        decision = (
            "validation_review_required" if model_refusal and has_relevant_evidence
            else "insufficient_evidence" if model_refusal
            else "informational" if used_extractive_fallback
            else "informational" if validation_result["valid"] and has_external_source
            else "informational" if validation_result["valid"] and not retrieval_result.get("grounded")
            else "answer_approved" if validation_result["valid"]
            else "informational" if informational_eligible
            else "validation_review_required" if generation_result.get("success") and has_relevant_evidence
            else "insufficient_evidence" if generation_result.get("success")
            else "generation_failed"
        )
        answer_abstained = decision in {
            "validation_review_required", "insufficient_evidence",
            "no_relevant_evidence", "generation_failed",
        }
        rejection_reason = None
        if decision == "informational":
            rejection_reason = (
                "extractive_evidence_fallback" if used_extractive_fallback
                else "external_source_quality_unverified" if has_external_source
                else "retrieval_relevance_limited" if not retrieval_result.get("grounded")
                else "validation_support_inconclusive"
            )
        elif decision == "validation_review_required":
            rejection_reason = "model_abstained" if model_refusal else "answer_validation_failed"
        elif answer_abstained:
            if not generation_result.get("success"):
                rejection_reason = "generation_failed"
            elif any("did not match the question topic" in str(issue).lower()
                     for issue in validation_result.get("issues", [])):
                rejection_reason = "generated_answer_off_topic"
            else:
                rejection_reason = "answer_validation_failed"
        if answer_abstained:
            # Never surface unvalidated provider output: it may contain reasoning,
            # control tokens, or unsupported medical claims.
            safe_answer = (
                "Relevant evidence was retrieved, but the generated answer could not be validated with sufficient confidence. No answer is shown; review the validation findings and sources."
                if decision == "validation_review_required"
                else "Answer generation failed. Review any retrieved sources or retry the question."
                if decision == "generation_failed"
                else "I could not find enough relevant evidence to answer this question reliably. Review the sources or refine the question and try again."
            )
            generation_result = {**generation_result, "answer": safe_answer}
            validation_result = {**validation_result, "answer": safe_answer}

        return {
            "query": query,
            "retrieval": {
                "fused_count": len(retrieval_result.get("fused_results", [])) + external_count,
                "reranked_count": len(ranked_results),
                "grounded": retrieval_result["grounded"]
            },
            "compression": {
                "compressed_count": compressed_context["evidence_count"],
                "sources": compressed_context["sources"],
                "evidence": compressed_context["context"],
                "formatted": compressed_context["formatted_text"][:200] + "..."
            },
            "generation": {
                "answer": generation_result["answer"],
                "model": generation_result["model"],
                "success": generation_result["success"]
            },
            "validation": validation_result,
            "timings": {
                "retrieval_time_ms": retrieval_time_ms,
                "generation_time_ms": generation_time_ms,
            },
            "final_answer": generation_result["answer"],
            "answer_abstained": answer_abstained,
            "decision": decision,
            "rejection_reason": rejection_reason,
            "risk_tier": medical_risk.tier,
            "question_category": medical_risk.category,
            "evidence_confidence": (
                "strong" if decision == "answer_approved"
                else "moderate" if decision == "informational"
                else "limited" if decision == "validation_review_required"
                else "low"
            ),
            "confidence_basis": (
                "Heuristic summary of retrieval relevance, citation references, and lexical checks; not a calibrated probability or clinical accuracy assessment."
            ),
            "audit_trace": {
                "model_prompt": prompt,
                "draft_status": (
                    "accepted" if not answer_abstained else "withheld_by_validation"
                ),
                "validation_issues": list(validation_result.get("issues", [])),
            },
        }
