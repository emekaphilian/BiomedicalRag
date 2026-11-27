"""
Full grounded generation pipeline.
Integrates retrieval → reranking → compression → generation → validation.
"""

import re
import logging
import json
from time import perf_counter

from rag.hybrid_retriever import HybridRetriever
from rag.reranking_retriever import RerankingRetriever
from rag.compression import ContextCompressor
from rag.prompt_constructor import GroundedPromptConstructor
from rag.generator import GroundedGenerator
from rag.grounding import has_topic_overlap
from rag.medical_risk import (
    classify_medical_risk, is_medical_followup, is_medical_query,
    reformulate_contextual_query,
)
from rag.external_sources import retrieve_pubmed
from rag.post_validation import PostGenerationValidator
from dashboard.rag_audit import content_tokens, split_sentences
from rag.evidence_relevance import assess_evidence_coverage, build_retrieval_queries, classify_evidence_set, interpret_query

logger = logging.getLogger(__name__)


def _external_search_query(query: str) -> str:
    """Remove question scaffolding before submitting a short PubMed query."""
    cleaned = re.sub(
        r"\b(?:summari[sz]e|what|how|does|do|is|are|the|a|an|of|about|for|with|work|works|evidence|supports?|current|understanding|please|explain|describe)\b",
        " ",
        query or "",
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split()).strip(" ?!.,:")
    return cleaned or query


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
    # Include normalized biomedical synonyms so a hypertension query can match
    # a source that consistently says "high blood pressure" (and similar pairs).
    from rag.medical_entities import expand_concepts
    query_tokens.update(content_tokens(expand_concepts(query)))
    query_tokens.difference_update(relation_terms)
    if not query_tokens:
        return None

    cause_support = re.compile(
        r"\b(?:because|due to|result(?:s|ed|ing)? in|lead(?:s|ing)? to|"
        r"without|cause(?:s|d)?|does not make|doesn't make|do not make|"
        r"don't make|did not make|cannot make|can't make|not make|"
        r"does not produce|doesn't produce|do not produce|lack of|"
        r"impaired|dysfunction|genetic disease|genetic changes?|mutations? in genes?|"
        r"develops? over time|cause(?: is| remains)? unknown|unknown cause|cause is not known)\b",
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


def _comparison_extractive_fallback(query, evidence_items):
    """Build a cited, source-only comparison from passages for each requested side."""
    intent = interpret_query(query)
    if intent.information_type != "comparison" or not intent.comparison_entities:
        return None

    from rag.evidence_relevance import classify_evidence

    lines = []
    for entity in intent.comparison_entities:
        entity_number = re.search(r"type\s*(1|2)", entity)
        if not entity_number:
            continue
        entity_pattern = re.compile(
            rf"\btype\s*{entity_number.group(1)}\b[^.!?]*\bdiabetes\b|"
            rf"\bdiabetes\b[^.!?]*\btype\s*{entity_number.group(1)}\b",
            re.IGNORECASE,
        )
        best = None
        for evidence_index, item in enumerate(evidence_items, start=1):
            classification = item.get("evidence_classification") or classify_evidence(
                query, item.get("content", ""), item.get("metadata", {}),
            )
            if classification.get("category") not in {"direct", "supporting_contextual"}:
                continue
            supported_entities = classification.get("comparison_entities_supported", [])
            if entity not in supported_entities:
                continue
            for sentence in split_sentences(str(item.get("content", ""))):
                sentence = sentence.strip()
                if len(sentence) < 30 or sentence.endswith("?") or not entity_pattern.search(sentence):
                    continue
                # Prefer the shortest complete sentence that states a fact about this side.
                rank = (len(sentence), -evidence_index)
                if best is None or rank < best[0]:
                    best = (rank, sentence, evidence_index)
        if best:
            lines.append(f"{best[1]} [Evidence {best[2]}]")

    return "\n".join(lines) if lines else None


def _learning_evidence_excerpt(evidence_items, query=None):
    """Build a short cited learning fallback from retained source text."""
    allowed_categories = {"direct", "supporting_contextual", "related_indirect", "complication"}
    for evidence_index, item in enumerate(evidence_items, start=1):
        category = (item.get("evidence_classification") or {}).get("category")
        if category not in allowed_categories:
            continue
        content = str(item.get("content", ""))
        for sentence in split_sentences(content):
            sentence = sentence.strip()
            if len(sentence) < 35 or sentence.endswith("?"):
                continue
            if query and not has_topic_overlap(query, [sentence]):
                continue
            return f"The retrieved passage says: {sentence} [Evidence {evidence_index}]"
    return None


def _accept_supported_partial(validator, result, validation, evidence_items, query, minimum_coverage=0.50):
    """Retain only semantically supported claims, repair their evidence links."""
    if not result.get("success") or validation.get("safety_issues"):
        return None
    claims = validation.get("claim_support") or []
    supported = sum(item.get("support_status") == "supported" for item in claims)
    if not supported:
        return None
    coverage = (validation.get("diagnostics") or {}).get("coverage_score")
    if coverage is None or coverage < minimum_coverage:
        return None
    original_answer = str(result.get("answer", ""))
    partial_answer = validator.retain_supported_claims(original_answer, claims)
    if not partial_answer or not has_topic_overlap(query, [partial_answer]):
        return None
    partial_result = {**result, "answer": partial_answer, "success": True}
    partial_validation = validator.validate(
        partial_result,
        len(evidence_items),
        [item.get("content", "") for item in evidence_items],
        evidence_records=evidence_items,
    )
    if not partial_validation.get("valid"):
        return None

    original_diagnostics = validation.get("diagnostics") or {}
    retained_count = sum(item.get("support_status") == "supported" for item in claims)
    original_count = len(claims)
    partial_validation["claim_support"] = claims
    partial_validation["partial_answer"] = retained_count < original_count
    partial_validation["issues"] = []
    partial_validation["diagnostics"] = {
        **(partial_validation.get("diagnostics") or {}),
        "generated_claims": original_diagnostics.get("generated_claims", original_count),
        "supported_claims": original_diagnostics.get("supported_claims", retained_count),
        "unsupported_claims": original_diagnostics.get("unsupported_claims", original_count - retained_count),
        "retained_claims": retained_count,
        "removed_claims": max(0, original_count - retained_count),
        "citation_repaired": partial_answer != original_answer and retained_count == original_count,
        "coverage_score": original_diagnostics.get("coverage_score"),
        "final_confidence": original_diagnostics.get("final_confidence"),
        "partial_answer": partial_validation["partial_answer"],
        "draft_diagnostics": original_diagnostics,
    }
    return partial_result, partial_validation


def _can_keep_relevant_answer_with_warning(
    generation_result,
    validation_result,
    evidence_items,
    *,
    has_direct_evidence,
    has_conflict,
    risk_tier,
    semantic_threshold,
):
    """Allow a cited, low-risk answer through when only support heuristics fail."""
    if (
        not generation_result.get("success")
        or not str(generation_result.get("answer", "")).strip()
        or not has_direct_evidence
        or has_conflict
        or risk_tier not in {"R0", "R1"}
        or validation_result.get("safety_issues")
        or not validation_result.get("citation_references_validated")
    ):
        return False

    claims = validation_result.get("claim_support") or []
    if not claims:
        return False

    for claim in claims:
        citations = {
            int(index) for index in claim.get("citations", [])
            if str(index).isdigit() and 1 <= int(index) <= len(evidence_items)
        }
        if not citations or claim.get("unsupported_numbers"):
            return False
        cited_direct_evidence = False
        for index in citations:
            classification = evidence_items[index - 1].get("evidence_classification") or {}
            category = classification.get("category")
            directly_relevant = category == "direct" or (
                category == "supporting_contextual"
                and classification.get("intent_match") is True
            )
            comparison_match = (
                category == "direct"
                and classification.get("information_type") == "comparison"
                and bool(classification.get("comparison_entities_supported"))
            )
            if (
                directly_relevant
                and (
                    classification.get("entity_match_type") != "related_condition"
                    or comparison_match
                )
            ):
                cited_direct_evidence = True
                break
        if not cited_direct_evidence:
            return False

        lexical_score = float(claim.get("lexical_support") or 0.0)
        semantic_score = float(claim.get("semantic_similarity") or 0.0)
        if max(lexical_score, semantic_score) < max(0.35, semantic_threshold - 0.10):
            return False
    return True


def _evidence_limited_answer(query_intent, evidence_items, coverage, learning=False):
    """Describe what retrieved passages cover without asserting missing facts."""
    subject = " ".join(query_intent.subject_terms) or "this topic"
    requested = query_intent.information_type.replace("_", " ")
    lines = [
        f"The retrieved passages do not directly establish the requested {requested} information about {subject}.",
        "\nWhat the available evidence covers:",
    ]
    shown = 0
    for index, item in enumerate(evidence_items, 1):
        classification = item.get("evidence_classification") or {}
        category = classification.get("category", "uncertain")
        metadata = item.get("metadata") or {}
        title = metadata.get("title") or metadata.get("document_title") or metadata.get("source") or "Retrieved passage"
        entity_type = classification.get("entity_match_type")
        if entity_type == "related_condition":
            description = f"{title} discusses a related but distinct condition; it does not establish facts about {subject}."
        elif query_intent.information_type == "causes" and category == "supporting_contextual":
            description = f"{title} provides risk or association context. Risk association alone does not establish a direct cause."
        elif classification.get("reason"):
            description = f"{title}: {classification['reason']}"
        else:
            description = f"{title} is related context, but its relevance to the requested {requested} information is uncertain."
        lines.append(f"• {description} [Evidence {index}]")
        shown += 1
        if shown >= 4:
            break
    if not shown:
        lines.append("No relevant passage details were available to summarize.")

    missing_components = [
        row.get("component") for row in coverage.get("components", [])
        if row.get("status") not in {"supported", "supporting_only"}
    ]
    lines.append("\nWhat remains uncertain:")
    lines.append(
        "The retrieved evidence does not establish " + ", ".join(missing_components) + "."
        if missing_components else coverage.get("reason", "Direct evidence for the requested answer was not retrieved.")
    )
    if learning:
        distinction = (
            "Risk associations and prevention advice are not the same as evidence of a direct cause."
            if query_intent.information_type == "causes"
            else "Related background is not the same as direct evidence for the requested answer."
        )
        lines.extend((
            "\nImportant distinction:",
            distinction,
            "\nFurther evidence needed:",
            "Retrieve passages that directly address the requested information and its relevant subtopics.",
        ))
    return "\n".join(lines)


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
        try:
            self.compressor = ContextCompressor(max_chunks=8, max_chunk_length=1500)
        except TypeError:
            # Keep lightweight injected compressors used by integrations/tests compatible.
            self.compressor = ContextCompressor(max_chunks=8)
        self.prompt_constructor = GroundedPromptConstructor()
        generator_config = generator_config or {}
        self.generator = GroundedGenerator(
            model_name=generator_config.get("model_name", "gpt-4o"),
            api_key=generator_config.get("api_key"),
            provider=generator_config.get("provider", "openai"),
        )
        self.generator_config = generator_config
        try:
            self.validator = PostGenerationValidator(
                embedding_model=getattr(vectorstore, "model", None),
                semantic_threshold=float(generator_config.get("semantic_support_threshold", 0.54)),
            )
        except TypeError:
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
        answer_mode="research",
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
        current_query = str(query or "").strip()
        reformulated_query = reformulate_contextual_query(current_query, chat_history)
        query = reformulated_query
        medical_risk = classify_medical_risk(query)
        topic_detected = is_medical_query(query)
        classifier_result = {
            "medical_scope": topic_detected or is_medical_followup(current_query, chat_history),
            "topic_detected": topic_detected,
            "risk_tier": medical_risk.tier,
            "question_category": medical_risk.category,
        }
        logger.info("rag_query_reformulation", extra={"rag_query_reformulation": {
            "current_query": current_query,
            "prior_messages": (chat_history or [])[-6:],
            "reformulated_query": reformulated_query,
            "classifier_result": classifier_result,
        }})
        if not classifier_result["medical_scope"]:
            refusal = "BioMed RAG answers medical and biomedical questions only. Ask about a health condition, symptom, treatment evidence, anatomy, or biology topic."
            return {
                "query": query,
                "current_query": current_query,
                "reformulated_query": reformulated_query,
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
        query_intent = interpret_query(search_query)
        retrieval_search_query = query_intent.retrieval_query
        retrieval_queries = build_retrieval_queries(search_query, query_intent)
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
            if hasattr(self.retriever, "retrieve_many"):
                retrieval_result = self.retriever.retrieve_many(
                    retrieval_queries,
                    retrieval_search_query,
                    k=retrieval_k,
                    retrieval_k=retrieval_k,
                )
            else:
                retrieval_result = self.retriever.retrieve(
                    retrieval_search_query, k=retrieval_k, retrieval_k=retrieval_k,
                )
            ranked_results = retrieval_result["reranked_results"]
        else:
            if hasattr(self.retriever, "retrieve_many"):
                hybrid_result = self.retriever.retrieve_many(
                    retrieval_queries, retrieval_search_query, k=retrieval_k,
                )
            else:
                hybrid_result = self.retriever.retrieve(retrieval_search_query, k=retrieval_k)
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
                    "retrieval_score": score,
                    **hybrid_result.get("candidate_diagnostics", {}).get(content, {}),
                }
                for content, score in selected_results[:retrieval_k]
                if content in metadata_by_content
            ]
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
            [
                " ".join(str((item.get("metadata") or {}).get(key) or "")
                          for key in ("title", "document_title", "section", "heading"))
                + " " + item.get("content", "")
                for item in ranked_results
            ],
        )
        if not retrieval_result.get("grounded") or not local_topic_match:
            external_started = perf_counter()
            external_results = retrieve_pubmed(
                _external_search_query(retrieval_queries[0] if retrieval_queries else retrieval_search_query),
                limit=min(max(k, 3), 5),
            )
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

        relevance_order = {
            "direct": 0, "supporting_contextual": 1,
            "related_indirect": 2, "complication": 3,
            "contradictory": 4, "irrelevant": 5, "uncertain": 6,
        }
        classifications = classify_evidence_set(query, ranked_results)
        for item, classification in zip(ranked_results, classifications):
            item["evidence_classification"] = classification
        comparison_entities = list(query_intent.comparison_entities)
        if comparison_entities:
            diverse = []
            for entity in comparison_entities:
                match = next((item for item in ranked_results if entity in
                              item["evidence_classification"].get("comparison_entities_supported", [])), None)
                if match is not None and match not in diverse:
                    diverse.append(match)
            ranked_results = diverse + [item for item in ranked_results if item not in diverse]
        candidate_ranking = [
            {
                "rank": index,
                "source": (item.get("metadata") or {}).get("source")
                    or (item.get("metadata") or {}).get("title")
                    or "Biomedical source",
                "score": float(item.get("rerank_score", item.get("fusion_score", 0.0)) or 0.0),
                "score_type": "cross_encoder" if self.use_reranking else "hybrid_fusion",
                "evidence_category": item.get("evidence_classification", {}).get("category"),
                "matched_subject_terms": item.get("evidence_classification", {}).get("matched_subject_terms", []),
                "semantic_score": item.get("semantic_score"),
                "keyword_score": item.get("keyword_score"),
                "entity_match": item.get("entity_match") or (item.get("evidence_classification") or {}).get("entity_match_type"),
                "entity_matches": item.get("entity_matches") or (item.get("evidence_classification") or {}).get("normalized_entity_matches", []),
                "retrieval_score": item.get("retrieval_score", item.get("rerank_score", item.get("fusion_score", 0.0))),
            }
            for index, item in enumerate(ranked_results[:10], 1)
        ]
        ranked_results.sort(key=lambda item: (
            relevance_order.get(item["evidence_classification"]["category"], 9),
            -{"exact_concept": 3, "synonym": 2, "subtype_match": 1, "related_condition": 0}.get(
                item.get("entity_match") or item["evidence_classification"].get("entity_match_type", "none"), 0
            ),
            -float(item.get("retrieval_score", item.get("rerank_score", item.get("fusion_score", 0.0))) or 0.0),
        ))
        ranked_results = ranked_results[:max(k, len(comparison_entities))]
        relevance_counts = {
            category: sum(
                item["evidence_classification"]["category"] == category
                for item in ranked_results
            )
            for category in relevance_order
        }
        intent_specific = query_intent.information_type != "general"
        lacks_direct_evidence = intent_specific and relevance_counts["direct"] == 0
        comparison_entities_covered = sorted({
            entity for item in ranked_results
            for entity in item["evidence_classification"].get("comparison_entities_supported", [])
        })
        comparison_coverage = (
            len(comparison_entities_covered) / len(comparison_entities)
            if comparison_entities else None
        )
        if comparison_entities and comparison_entities_covered:
            lacks_direct_evidence = False
        if ranked_results and query_intent.subject_terms and relevance_counts["contradictory"] > 0:
            refusal = (
                "Related biomedical sources were retrieved, but none directly addresses "
                f"the requested {query_intent.information_type} information. The available "
                "passages do not establish the central answer, so it has been withheld."
            )
            insufficient_context = self.compressor.compress(ranked_results)
            for evidence_index, evidence in enumerate(insufficient_context.get("context", []), 1):
                evidence["evidence_classification"] = ranked_results[evidence_index - 1].get("evidence_classification")
            evidence_audit = {
                "question_classification": {
                    "subject_terms": list(query_intent.subject_terms),
                    "information_type": query_intent.information_type,
                    "expected_evidence": list(query_intent.expected_evidence),
                },
                "retrieval_configuration": {
                    "mode": retrieval_mode, "top_k": k, "retrieval_k": retrieval_k,
                    "expanded_query": retrieval_search_query,
                    "expanded_queries": retrieval_queries,
                },
                "retrieved_evidence_count": len(ranked_results),
                "relevance_counts": relevance_counts,
                "top_ranked_candidates": candidate_ranking,
                "evidence": [
                    {"evidence_id": item.get("evidence_id"), **item.get("evidence_classification", {})}
                    for item in insufficient_context.get("context", [])
                ],
                "reason": (
                    "Retrieved passages contain explicit opposing claims."
                    if relevance_counts["contradictory"] > 0
                    else "No retrieved passage directly addresses the requested information type."
                ),
            }
            has_related_evidence = any(
                relevance_counts[key] for key in (
                    "direct", "supporting_contextual", "related_indirect", "complication", "contradictory",
                )
            )
            has_conflict = relevance_counts["contradictory"] > 0
            final_decision = (
                "conflicting_evidence" if has_conflict else
                "insufficient_evidence" if has_related_evidence else
                "no_relevant_evidence"
            )
            if has_conflict:
                refusal = "Retrieved passages contain conflicting information relevant to this question. The disagreement is unresolved, so a reliable answer has been withheld."
            elif not has_related_evidence:
                refusal = "No retrieved passages meaningfully addressed the requested biomedical subject or information type."
            learning_answer = None
            if answer_mode == "learning" and not has_conflict:
                learning_answer = _extractive_fallback(query, insufficient_context["context"])
                if learning_answer is None and has_related_evidence:
                    learning_answer = _learning_evidence_excerpt(
                        insufficient_context["context"],
                        query,
                    )
            return {
                "query": query,
                "retrieval": {
                    "fused_count": len(retrieval_result.get("fused_results", [])) + external_count,
                    "reranked_count": len(ranked_results), "grounded": False,
                },
                "compression": {
                    "compressed_count": len(insufficient_context.get("context", [])),
                    "sources": insufficient_context.get("sources", []),
                    "evidence": insufficient_context.get("context", []),
                    "formatted": insufficient_context.get("formatted_text", ""),
                },
                "generation": {
                    "answer": refusal,
                    "model": getattr(self.generator, "model_name", self.generator_config.get("model_name", "unknown")),
                    "success": True,
                },
                "validation": {
                    "valid": False,
                    "issues": [
                        "Retrieved evidence contains conflicting information."
                        if has_conflict else
                        "Retrieved evidence does not directly establish the requested information."
                    ],
                    "claim_support": [],
                },
                "timings": {"retrieval_time_ms": retrieval_time_ms, "generation_time_ms": 0.0},
                "final_answer": refusal,
                "learning_answer": learning_answer,
                "answer_abstained": True,
                "decision": final_decision,
                "rejection_reason": (
                    "conflicting_evidence" if has_conflict else
                    "evidence_too_weak" if has_related_evidence else
                    "no_relevant_evidence"
                ),
                "risk_tier": medical_risk.tier,
                "question_category": medical_risk.category,
                "evidence_confidence": "low",
                "audit_trace": {
                    "draft_status": "withheld_by_relevance_gate",
                    "question_classification": evidence_audit["question_classification"],
                    "evidence_audit": evidence_audit,
                    "final_answer_status": final_decision,
                    "final_decision_reason": evidence_audit["reason"],
                },
            }

        classified_relevance = any(
            item.get("evidence_classification", {}).get("category") in {
                "direct", "supporting_contextual", "related_indirect", "complication", "contradictory",
            }
            for item in ranked_results
        )
        if not ranked_results or (
            not classified_relevance
            and not has_topic_overlap(search_query, [item.get("content", "") for item in ranked_results])
        ):
            refusal = "The available sources did not contain enough relevant evidence to answer this question reliably. Refine the question or add a source that directly covers the topic."
            evidence_reason = "evidence_too_weak" if ranked_results else "no_relevant_evidence"
            learning_answer = (
                _extractive_fallback(query, ranked_results)
                if answer_mode == "learning" and ranked_results
                else None
            )
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
                "learning_answer": learning_answer,
                "answer_abstained": True,
                "decision": "no_relevant_evidence" if evidence_reason == "no_relevant_evidence" else "insufficient_evidence",
                "rejection_reason": evidence_reason,
                "risk_tier": medical_risk.tier,
                "audit_trace": {
                    "draft_status": "withheld_by_relevance_gate",
                    "question_classification": {
                        "subject_terms": list(query_intent.subject_terms),
                        "information_type": query_intent.information_type,
                        "expected_evidence": list(query_intent.expected_evidence),
                    },
                    "retrieval_configuration": {
                        "mode": retrieval_mode,
                        "top_k": k,
                        "retrieval_k": retrieval_k,
                        "expanded_query": retrieval_search_query,
                        "expanded_queries": retrieval_queries,
                        "top_ranked_candidates": candidate_ranking,
                    },
                    "evidence_audit": {
                        "relevance_counts": {"irrelevant": len(ranked_results)},
                        "reason": "No retrieved passage passed the entity and topic relevance screen.",
                        "evidence_classifications": [
                            {"evidence_id": item.get("evidence_id"), **(item.get("evidence_classification") or {})}
                            for item in ranked_results[:10]
                        ],
                    },
                    "validation_issues": [
                        "No relevant evidence was retrieved."
                        if evidence_reason == "no_relevant_evidence"
                        else "Retrieved passages did not meet the evidence relevance threshold."
                    ],
                    "final_answer_status": "no_relevant_evidence" if evidence_reason == "no_relevant_evidence" else "insufficient_evidence",
                },
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

        try:
            evidence_coverage = assess_evidence_coverage(
                query, compressed_context.get("context", []),
            )
        except Exception as error:
            evidence_coverage = {
                "status": "calculation_failed",
                "reason": f"Evidence coverage assessment failed: {type(error).__name__}.",
                "coverage_score": None,
                "components": [],
            }
        compressed_context["coverage_assessment"] = evidence_coverage
        has_direct_for_intent = any(
            (item.get("evidence_classification") or {}).get("category") == "direct"
            for item in compressed_context.get("context", [])
        )
        if query_intent.information_type != "general" and not has_direct_for_intent:
            limited_answer = _evidence_limited_answer(
                query_intent,
                compressed_context.get("context", []),
                {
                    "components": evidence_coverage.get("question_components", []),
                    "reason": evidence_coverage.get("reason"),
                },
                learning=answer_mode == "learning",
            )
            limitation = "Evidence coverage is limited: retrieved passages do not directly answer the requested information."
            diagnostics = {
                "coverage_status": evidence_coverage.get("status", "not_calculated"),
                "coverage_score": evidence_coverage.get("coverage_score"),
                "question_components": evidence_coverage.get("question_components", []),
                "evidence_groups": evidence_coverage.get("evidence_groups", {}),
                "generated_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "method": "question_aware_evidence_role_assessment",
                "final_confidence": "limited",
            }
            return {
                "query": query,
                "retrieval": {
                    "fused_count": len(retrieval_result.get("fused_results", [])) + external_count,
                    "reranked_count": len(ranked_results),
                    "grounded": bool(ranked_results),
                },
                "compression": {
                    **compressed_context,
                    "compressed_count": compressed_context.get("evidence_count", 0),
                    "evidence": compressed_context.get("context", []),
                    "formatted": compressed_context.get("formatted_text", ""),
                },
                "generation": {
                    "answer": limited_answer,
                    "model": "evidence_assessment",
                    "success": True,
                    "method": "evidence_assessment_template",
                },
                "validation": {
                    "valid": True,
                    "issues": [],
                    "safety_issues": [],
                    "claim_support": [],
                    "claim_support_validated": "NOT APPLICABLE: no answer claims generated",
                    "claim_support_method": "not_applicable",
                    "diagnostics": diagnostics,
                    "evidence_coverage": evidence_coverage,
                },
                "timings": {
                    "retrieval_time_ms": retrieval_time_ms,
                    "generation_time_ms": 0.0,
                },
                "final_answer": limited_answer,
                "learning_answer": limited_answer if answer_mode == "learning" else None,
                "answer_abstained": False,
                "decision": "informational",
                "rejection_reason": "evidence_coverage_limited",
                "risk_tier": medical_risk.tier,
                "question_category": medical_risk.category,
                "evidence_confidence": "limited",
                "confidence_basis": "Question-aware evidence roles show incomplete direct coverage; no answer claims were generated.",
                "audit_trace": {
                    "model_prompt": None,
                    "draft_status": "evidence_assessment_only",
                    "question_classification": {
                        "subject_terms": list(query_intent.subject_terms),
                        "information_type": query_intent.information_type,
                        "expected_evidence": list(query_intent.expected_evidence),
                        "comparison_entities": comparison_entities,
                    },
                    "retrieval_configuration": {
                        "mode": retrieval_mode,
                        "top_k": k,
                        "retrieval_k": retrieval_k,
                        "expanded_query": retrieval_search_query,
                        "expanded_queries": retrieval_queries,
                        "executed_queries": retrieval_result.get("query_diagnostics", retrieval_result.get("queries", retrieval_queries)),
                        "top_ranked_candidates": candidate_ranking,
                    },
                    "evidence_audit": {
                        "retrieved_evidence_count": len(ranked_results),
                        "relevance_counts": relevance_counts,
                        "coverage_assessment": evidence_coverage,
                        "reason": evidence_coverage.get("reason"),
                        "evidence_classifications": [
                            {"evidence_id": item.get("evidence_id"), **(item.get("evidence_classification") or {})}
                            for item in compressed_context.get("context", [])
                        ],
                        "generated_claims": [],
                        "citation_alignment": "No generated claims; citations identify passages discussed by the evidence assessment.",
                    },
                    "validation_issues": [limitation],
                    "final_answer_status": "informational",
                    "final_decision_reason": evidence_coverage.get("reason"),
                },
            }

        # Step 3: Prompt Construction
        prompt = (
            self.prompt_constructor.construct(query, compressed_context, chat_history)
            if chat_history
            else self.prompt_constructor.construct(query, compressed_context)
        )
        if isinstance(prompt, dict):
            prompt = dict(prompt)
            assessment_for_generation = {
                "coverage_status": evidence_coverage.get("status"),
                "coverage_reason": evidence_coverage.get("reason"),
                "question_components": evidence_coverage.get("question_components", []),
                "evidence_groups": evidence_coverage.get("evidence_groups", {}),
            }
            prompt["user"] = (
                f"{prompt.get('user', '')}\n\n"
                "APPROVED EVIDENCE ASSESSMENT (use this to constrain the answer):\n"
                + json.dumps(assessment_for_generation, ensure_ascii=False)
                + "\nDo not convert related, indirect, risk-association, or prevention passages into direct causal claims."
            )
        if answer_mode == "learning" and isinstance(prompt, dict):
            prompt = dict(prompt)
            prompt["system"] = """You are a biomedical educator for learners with no assumed medical background.

TEACHING GOAL:
- Answer the user's question directly in 2-4 clear sentences.
- Explain necessary medical terms briefly in plain English.
- Summarize supported mechanisms instead of copying an isolated passage when several passages fit together.
- For cause questions, distinguish mechanisms from risk factors; describe risk factors as increasing risk, not as direct causes.

EVIDENCE AND SAFETY RULES:
- Use only the supplied retrieved evidence for factual claims. Cite each factual sentence with its exact [Evidence N] label.
- Direct evidence may answer the question. Contextual evidence may clarify it. Related or complication evidence must not be presented as directly answering the question.
- If the evidence is insufficient, ambiguous, or unrelated, say so instead of filling gaps.
- Do not diagnose, recommend treatment, or give individualized medical advice.
- Return only the final answer. Do not include reasoning or commentary about these instructions."""
            prompt["user"] = (
                f"{prompt.get('user', '')}\n\n"
                "Give a concise, direct explanation in 2-4 sentences. Combine "
                "compatible facts from the supplied passages, retain citations, "
                "and state any important evidence limitation naturally."
            )
        elif answer_mode == "compare" and isinstance(prompt, dict):
            prompt = dict(prompt)
            prompt["system"] = """You are a biomedical evidence comparison assistant.

Compare the entities or approaches explicitly requested by the user. Use only the supplied retrieved evidence and cite factual statements with the exact [Evidence N] label. Organize the response with the headings Comparison, Shared characteristics, and Evidence gaps when applicable. Use a compact side-by-side structure when the evidence supports both sides. If one side lacks direct evidence, state that clearly and do not fill the gap from memory. Do not diagnose or give individualized medical advice. Return only the answer, without private reasoning."""
            prompt["user"] = (
                f"{prompt.get('user', '')}\n\n"
                "Present a clear evidence-grounded comparison. Include only "
                "shared characteristics supported by the retrieved passages, "
                "and identify missing or uneven evidence explicitly."
            )

        if (
            query_intent.information_type != "general"
            and relevance_counts["direct"] == 0
            and any(relevance_counts[key] for key in ("supporting_contextual", "related_indirect", "complication"))
            and isinstance(prompt, dict)
        ):
            prompt = dict(prompt)
            prompt["system"] = (
                f"{prompt.get('system', '')}\n\n"
                "PARTIAL EVIDENCE REQUIREMENT:\n"
                "No retrieved passage directly answers the requested information type. "
                "Do not present a complete answer or fill gaps from prior knowledge. "
                "Give a short, useful summary only of what the supplied passages do "
                "establish, cite each factual statement, and clearly say which requested "
                "detail was not found in the retrieved evidence. If the passages only "
                "provide background, name that background accurately and label the "
                "answer as limited evidence."
            )
            prompt["user"] = (
                f"{prompt.get('user', '')}\n\n"
                "Because no direct answer passage was retrieved, provide an evidence-"
                "grounded partial response and an explicit evidence gap. Do not answer "
                "the missing part from general knowledge."
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
        used_partial_answer = False
        if generation_result.get("success") and not has_topic_overlap(
            query, [generation_result.get("answer", "")]
        ):
            validation_result["valid"] = False
            validation_result.setdefault("issues", []).append(
                "Generated answer did not match the question topic."
            )

        partial_acceptance = _accept_supported_partial(
            self.validator, generation_result, validation_result,
            compressed_context["context"], query,
            minimum_coverage=float(self.generator_config.get("minimum_answer_coverage", 0.50)),
        )
        if partial_acceptance:
            generation_result, validation_result = partial_acceptance
            used_partial_answer = bool(validation_result.get("partial_answer"))

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
            partial_acceptance = _accept_supported_partial(
                self.validator, generation_result, validation_result,
                compressed_context["context"], query,
                minimum_coverage=float(self.generator_config.get("minimum_answer_coverage", 0.50)),
            )
            if partial_acceptance:
                generation_result, validation_result = partial_acceptance
                used_partial_answer = bool(validation_result.get("partial_answer"))

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
        if validation_result.get("safety_issues"):
            should_try_extractive_fallback = False
        if (
            should_try_extractive_fallback
            and medical_risk.tier in {"R0", "R1"}
        ):
            extractive_answer = _comparison_extractive_fallback(
                query, compressed_context["context"],
            ) or _extractive_fallback(query, compressed_context["context"])
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

        has_relevant_evidence = bool(
            compressed_context["evidence_count"]
            and has_topic_overlap(
                query,
                [item.get("content", "") for item in compressed_context["context"]],
            )
        )
        has_direct_evidence = (
            relevance_counts.get("direct", 0) > 0
            or (
                query_intent.information_type == "general"
                and has_relevant_evidence
            )
        )
        has_contextual_evidence = any(
            relevance_counts.get(category, 0)
            for category in ("supporting_contextual", "related_indirect", "complication")
        )
        if (
            not validation_result.get("valid")
            and not has_direct_evidence
            and has_contextual_evidence
            and not relevance_counts.get("contradictory", 0)
            and medical_risk.tier in {"R0", "R1"}
        ):
            partial_excerpt = _learning_evidence_excerpt(
                compressed_context["context"], query,
            )
            if partial_excerpt:
                partial_result = {**generation_result, "answer": partial_excerpt, "success": True}
                partial_validation = self.validator.validate(
                    partial_result,
                    compressed_context["evidence_count"],
                    [item["content"] for item in compressed_context["context"]],
                    evidence_records=compressed_context["context"],
                )
                if partial_validation.get("valid"):
                    generation_result = partial_result
                    validation_result = partial_validation
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
        accepted_with_warning = (
            not validation_result.get("valid")
            and _can_keep_relevant_answer_with_warning(
                generation_result,
                validation_result,
                compressed_context.get("context", []),
                has_direct_evidence=has_direct_evidence,
                has_conflict=bool(relevance_counts.get("contradictory", 0)),
                risk_tier=medical_risk.tier,
                semantic_threshold=getattr(self.validator, "semantic_threshold", 0.54),
            )
        )
        learning_answer = None
        if answer_mode == "learning" and compressed_context.get("evidence_count", 0) > 0:
            safe_to_show_learning_answer = (
                medical_risk.tier in {"R0", "R1"}
                and bool(generation_result.get("success"))
                and bool(generated_draft.strip())
                and not model_refusal
                and bool(validation_result.get("valid"))
                and not re.search(
                    r"(?:<analysis>|chain of thought|private reasoning|system prompt)",
                    generated_draft,
                    re.IGNORECASE,
                )
            )
            if used_extractive_fallback:
                learning_answer = generated_draft
            elif accepted_with_warning:
                learning_answer = generated_draft
            elif safe_to_show_learning_answer:
                learning_answer = generated_draft
            if learning_answer and not has_direct_evidence:
                learning_answer += (
                    "\n\nLimited evidence: the retrieved passage provides related context, "
                    "but it does not directly answer the requested information type."
                )
        diagnostics = validation_result.get("diagnostics") or {}
        decision = (
            "validation_review_required" if model_refusal and has_relevant_evidence
            else "insufficient_evidence" if model_refusal
            else "informational" if used_partial_answer
            else "informational" if used_extractive_fallback
            else "informational" if validation_result["valid"] and has_external_source
            else "informational" if validation_result["valid"] and not retrieval_result.get("grounded")
            else "informational" if validation_result["valid"] and not has_direct_evidence
            else "informational" if accepted_with_warning
            else "answer_approved" if validation_result["valid"]
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
                "retrieval_relevance_limited" if not has_direct_evidence
                else "partial_answer_accepted" if used_partial_answer
                else "extractive_evidence_fallback" if used_extractive_fallback
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

        final_answer = generation_result["answer"]
        if decision == "informational":
            final_answer += (
                "\n\nSome generated details were omitted because the retrieved evidence did "
                "not support them. The remaining claims cite their supporting passages."
                if used_partial_answer else
                "\n\nLimited evidence: the retrieved passage provides related context, "
                "but it does not directly answer the requested information type."
                if not has_direct_evidence else
                "\n\nSome wording could not be fully confirmed by automated support checks. "
                "The cited passages are directly relevant; please review them."
                if accepted_with_warning else
                "\n\nThe cited external source has not been independently assessed by this system."
                if has_external_source else
                "\n\nThe answer uses the retrieved passages, though retrieval confidence was limited."
                if not retrieval_result.get("grounded") else ""
            )
        if comparison_entities and comparison_coverage is not None and comparison_coverage < 1:
            missing_entities = [entity for entity in comparison_entities if entity not in comparison_entities_covered]
            final_answer += (
                "\n\nEvidence gap: the retrieved passages did not cover "
                + ", ".join(missing_entities)
                + ". Treat that part of the comparison as incomplete."
            )
        if answer_mode == "learning" and validation_result.get("valid") and not answer_abstained:
            learning_answer = final_answer

        coverage_score = diagnostics.get("coverage_score")
        if comparison_coverage is not None:
            diagnostics["claim_coverage_score"] = coverage_score
            diagnostics["comparison_entities"] = comparison_entities
            diagnostics["comparison_entities_covered"] = comparison_entities_covered
            diagnostics["comparison_coverage_score"] = comparison_coverage
            coverage_score = min(coverage_score, comparison_coverage) if coverage_score is not None else comparison_coverage
            diagnostics["coverage_score"] = coverage_score
            if comparison_coverage < 1 and validation_result.get("valid"):
                decision = "informational"
                answer_abstained = False
                rejection_reason = "comparison_evidence_incomplete"
        final_confidence = (
            "limited" if coverage_score is None or coverage_score < 0.50
            else "moderate" if coverage_score < 0.80
            else "high"
        )
        diagnostics["final_confidence"] = final_confidence
        diagnostics["partial_answer"] = used_partial_answer
        validation_result["diagnostics"] = diagnostics
        generated_claim_count = diagnostics.get("generated_claims", 0)
        supported_claim_count = diagnostics.get("supported_claims", 0)
        unsupported_claim_count = diagnostics.get("unsupported_claims", 0)

        logger.info(
            "rag_answer_validation",
            extra={"rag_validation": {
                "current_query": current_query,
                "prior_messages": (chat_history or [])[-6:],
                "query": query,
                "reformulated_query": query,
                "topic_detected": " ".join(query_intent.subject_terms),
                "classifier_result": classifier_result,
                "information_type": query_intent.information_type,
                "expanded_queries": retrieval_queries,
                "top_retrieved_documents": candidate_ranking[:5],
                "retrieval_scores": [
                    {
                        "source": (item.get("metadata") or {}).get("source"),
                        "score": item.get("retrieval_score", item.get("rerank_score")),
                    }
                    for item in ranked_results[:5]
                ],
                "retrieved_sources": len(set(compressed_context.get("sources", []))),
                "retrieved_chunks": len(compressed_context.get("context", [])),
                "retrieved_chunk_texts": [
                    item.get("content", "") for item in compressed_context.get("context", [])
                ],
                "answer_text": final_answer,
                "generated_claims": generated_claim_count,
                "supported_claims": supported_claim_count,
                "unsupported_claims": unsupported_claim_count,
                "coverage_score": coverage_score,
                "final_confidence": final_confidence,
                "partial_answer": used_partial_answer,
                "accepted_with_validation_warning": accepted_with_warning,
                "validation_valid": bool(validation_result.get("valid")),
                "validation_issues": list(validation_result.get("issues", [])),
                "validation_score": coverage_score,
                "validation_failure_reason": (
                    list(validation_result.get("issues", []))
                    if not validation_result.get("valid") else None
                ),
                "comparison_entities": comparison_entities,
                "comparison_entities_covered": comparison_entities_covered,
                "comparison_coverage_score": comparison_coverage,
                "rejected_claims": [
                    item.get("claim") for item in validation_result.get("claim_support", [])
                    if item.get("support_status") != "supported"
                ],
            }},
        )

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
            "final_answer": final_answer,
            "learning_answer": learning_answer,
            "answer_abstained": answer_abstained,
            "decision": decision,
            "rejection_reason": rejection_reason,
            "risk_tier": medical_risk.tier,
            "question_category": medical_risk.category,
            "evidence_confidence": (
                "strong" if final_confidence == "high"
                else "moderate" if final_confidence == "moderate"
                else "limited" if final_confidence == "limited"
                else "low"
            ),
            "confidence_basis": (
                "Evidence coverage estimated from semantic embedding similarity and claim/citation checks (lexical fallback if embeddings are unavailable); not a calibrated probability, entailment judgment, or clinical accuracy assessment."
            ),
            "audit_trace": {
                "model_prompt": prompt,
                "draft_status": (
                    "partial_context_fallback" if used_extractive_fallback and not has_direct_evidence
                    else "accepted" if not answer_abstained else "withheld_by_validation"
                ),
                "validation_issues": list(validation_result.get("issues", [])),
                "question_classification": {
                    "category": medical_risk.category,
                    "risk_tier": medical_risk.tier,
                    "subject_terms": list(query_intent.subject_terms),
                    "information_type": query_intent.information_type,
                    "expected_evidence": list(query_intent.expected_evidence),
                    "comparison_entities": comparison_entities,
                },
                "retrieval_configuration": {
                    "mode": retrieval_mode,
                    "top_k": k,
                    "retrieval_k": retrieval_k,
                    "expanded_query": retrieval_search_query,
                    "expanded_queries": retrieval_queries,
                    "executed_queries": retrieval_result.get("query_diagnostics", retrieval_result.get("queries", retrieval_queries)),
                    "top_ranked_candidates": candidate_ranking,
                },
                "evidence_audit": {
                    "retrieved_evidence_count": len(ranked_results),
                    "relevance_counts": relevance_counts,
                    "coverage_assessment": evidence_coverage,
                    "comparison_entities_covered": comparison_entities_covered,
                    "contradictory_evidence": {
                        "detected": relevance_counts["contradictory"] > 0,
                        "status": "screened",
                        "method": "explicit-opposing-terms heuristic; not clinical adjudication",
                    },
                    "evidence_classifications": [
                        {
                            "evidence_id": item.get("evidence_id"),
                            **(item.get("evidence_classification") or {}),
                        }
                        for item in compressed_context.get("context", [])
                    ],
                    "generated_claims": list(validation_result.get("claim_support", [])),
                    "citation_coverage": (diagnostics or {}).get("citation_coverage"),
                    "claim_support_status": validation_result.get("claim_support_validated", "NOT ESTABLISHED"),
                },
                "final_answer_status": decision,
                "final_decision_reason": rejection_reason or (
                    "answer passed configured checks" if decision == "answer_approved"
                    else "relevant evidence supports an informational answer with validation limitations"
                    if decision == "informational" else decision
                ),
            },
        }
