"""Safety checks and semantic, coverage-based support estimates for RAG answers.

Embedding similarity is a retrieval/support heuristic, not a biomedical entailment
model. Results are therefore exposed as evidence coverage, never as proof of truth.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np

from dashboard.rag_audit import audit_answer, content_tokens, split_sentences


_CITATION = re.compile(r"\s*\[Evidence\s+(\d+)\]", re.IGNORECASE)
_REASONING_MARKERS = (
    "<eos_token>", "<think>", "</think>", "chain of thought",
    "we need to answer the query", "the user says", "we should respond",
    "thus final answer:", "let me think",
)
_HALLUCINATION_INDICATORS = ("absolutely", "definitely", "certainly", "100% sure", "proven fact")
logger = logging.getLogger(__name__)


class PostGenerationValidator:
    """Validate output safety and estimate semantic support across all evidence."""

    def __init__(self, embedding_model=None, semantic_threshold: float = 0.54):
        self.embedding_model = embedding_model
        self.semantic_threshold = semantic_threshold
        self.forbidden_phrases = [
            "you should take", "i recommend", "start this medication",
            "stop this treatment", "this will cure", "guaranteed to", "will definitely",
        ]
        self.refusal_phrases = [
            "i do not have enough", "insufficient evidence", "cannot determine",
            "not enough information",
        ]

    def _log_validation(self, answer: str, evidence_texts, result: dict[str, Any]) -> None:
        """Write an auditable record of the exact validation input and result."""
        payload = {
            "retrieved_chunks": list(evidence_texts or []),
            "generated_answer": answer,
            "thresholds": {
                "semantic_support": self.semantic_threshold,
                "lexical_support": 0.6,
                "lexical_trace": 0.35,
                "citation_required_per_factual_sentence": True,
                "minimum_supporting_chunks": 1,
            },
            "validation": result,
            "failure_reason": result.get("issues") or result.get("reason"),
        }
        logger.info(
            "rag_post_generation_validation=%s",
            json.dumps(payload, ensure_ascii=False, default=str),
        )

    @staticmethod
    def _confidence(coverage: float | None) -> str:
        if coverage is None:
            return "limited"
        if coverage >= 0.80:
            return "high"
        if coverage >= 0.50:
            return "moderate"
        return "limited"

    def _embedding_support(self, claims: list[str], evidence_texts: list[str]):
        """Score claims against each complete passage and one joint evidence pool."""
        model = self.embedding_model
        if model is None or not hasattr(model, "encode") or not claims or not evidence_texts:
            return None

        candidates: list[tuple[str, int | None]] = []
        for source_index, passage in enumerate(evidence_texts, 1):
            normalized = " ".join(str(passage).split())
            if normalized:
                candidates.append((normalized[:2500], source_index))
            for sentence in split_sentences(normalized):
                if len(content_tokens(sentence)) >= 3:
                    candidates.append((sentence[:800], source_index))
        if not candidates:
            return None

        # The pooled text permits several retrieved passages to jointly support
        # a comparison claim even when no single chunk contains both sides.
        pooled = "\n".join(str(text) for text in evidence_texts if text)
        candidates.append((pooled[:5000], None))
        texts = claims + [item[0] for item in candidates]
        try:
            vectors = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            try:
                vectors = model.encode(texts)
            except Exception:
                return None
        except Exception:
            # Keep answer generation available if an optional embedding model
            # is unavailable; the explicitly labeled lexical fallback applies.
            return None
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
            return None
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            return None
        vectors = vectors / norms

        scores = []
        candidate_vectors = vectors[len(claims):]
        for claim_index in range(len(claims)):
            cosine = candidate_vectors @ vectors[claim_index]
            by_source: dict[int, float] = {}
            pooled_score = 0.0
            for candidate, score in zip(candidates, cosine):
                text, source_index = candidate
                del text
                value = float(score)
                if source_index is None:
                    pooled_score = max(pooled_score, value)
                else:
                    by_source[source_index] = max(by_source.get(source_index, 0.0), value)

            source_matches = [
                index for index, score in sorted(by_source.items(), key=lambda pair: pair[1], reverse=True)
                if score >= self.semantic_threshold - 0.04
            ]
            best_source_score = max(by_source.values(), default=0.0)
            best_score = max(best_source_score, pooled_score)
            if best_score >= self.semantic_threshold and not source_matches:
                source_matches = [index for index, _ in sorted(
                    by_source.items(), key=lambda pair: pair[1], reverse=True,
                )[:2]]
            scores.append({
                "score": round(best_score, 4),
                "source_scores": {str(index): round(score, 4) for index, score in by_source.items()},
                "evidence_indices": source_matches[:3],
                "supported": best_score >= self.semantic_threshold,
            })
        return scores

    def validate(
        self,
        generation_result: dict[str, Any],
        evidence_count: int,
        evidence_texts=None,
        evidence_records=None,
    ) -> dict[str, Any]:
        if not generation_result.get("success"):
            result = {
                "valid": False,
                "reason": "Generation failed",
                "answer": generation_result.get("answer", ""),
                "issues": [generation_result.get("error", "Generation failed")],
                "safety_issues": [],
                "claim_support": [],
                "diagnostics": None,
            }
            self._log_validation(
                str(generation_result.get("answer", "")), evidence_texts, result,
            )
            return result

        answer_text = str(generation_result.get("answer", ""))
        answer = answer_text.lower()
        issues: list[str] = []
        safety_issues: list[str] = []
        if not answer_text.strip():
            issues.append("Generated answer is empty")
        if any(marker in answer for marker in _REASONING_MARKERS):
            safety_issues.append("Internal reasoning or control tokens appeared in generated answer")
        for phrase in self.forbidden_phrases:
            if phrase in answer:
                safety_issues.append(f"Forbidden phrase detected: '{phrase}'")
        for match in re.finditer(r"\byou have\b", answer):
            clause_start = max(answer.rfind(mark, 0, match.start()) for mark in (".", "!", "?", ";", ":", ","))
            if not re.match(r"if\b", answer[clause_start + 1:match.start()].strip()):
                safety_issues.append("Forbidden phrase detected: direct diagnosis ('you have')")
                break
        issues.extend(safety_issues)

        evidence_texts = [str(text) for text in (evidence_texts or [])]
        evidence_records = evidence_records or []
        audit = audit_answer(answer_text, evidence_texts) if evidence_texts else None
        rows = audit["sentences"] if audit else []
        claims = [row["sentence"] for row in rows]
        semantic = self._embedding_support(claims, evidence_texts) if claims else None
        method = "embedding_similarity_heuristic_not_entailment" if semantic is not None else "lexical_overlap_fallback"

        claim_support = []
        unsupported_numbers = 0
        eligible_indexes = set()
        for index, record in enumerate(evidence_records, 1):
            if not isinstance(record, dict):
                continue
            classification = record.get("evidence_classification") or {}
            category = classification.get("category")
            if category and category not in {"direct", "supporting_contextual"}:
                continue
            if category and category != "direct" and classification.get("intent_match") is not True:
                continue
            comparison_match = (
                category == "direct"
                and classification.get("information_type") == "comparison"
                and bool(classification.get("comparison_entities_supported"))
            )
            if (
                classification.get("entity_match_type") == "related_condition"
                and not comparison_match
            ):
                continue
            eligible_indexes.add(index)
        for index, row in enumerate(rows):
            citations = [int(value) for value in _CITATION.findall(row["sentence"])]
            valid_citations = sorted({value for value in citations if 1 <= value <= evidence_count})
            invalid_citations = sorted({value for value in citations if value < 1 or value > evidence_count})
            semantic_row = semantic[index] if semantic is not None else None
            if semantic_row is None:
                supported = bool(row.get("supported"))
                evidence_indices = [row["source_index"]] if row.get("source_index") else []
                score = float(row.get("support") or 0.0)
            else:
                source_scores = semantic_row.get("source_scores", {})
                cited_scope = set(valid_citations) if valid_citations else None
                allowed = [
                    int(source_index) for source_index in source_scores
                    if int(source_index) in eligible_indexes
                    and (cited_scope is None or int(source_index) in cited_scope)
                ]
                evidence_indices = sorted(
                    allowed,
                    key=lambda source_index: source_scores.get(str(source_index), 0),
                    reverse=True,
                )[:3]
                score = max((float(source_scores.get(str(source_index), 0)) for source_index in evidence_indices), default=0.0)
                supported = score >= self.semantic_threshold

            # A retrieval similarity or lexical overlap cannot turn a passage
            # classified as unrelated/risk-only into support for a factual claim.
            if evidence_records:
                evidence_indices = [i for i in evidence_indices if i in eligible_indexes]
                supported = supported and bool(evidence_indices)
            if valid_citations and evidence_indices:
                cited_indices = set(valid_citations)
                evidence_indices = [i for i in evidence_indices if i in cited_indices]
                supported = supported and bool(evidence_indices)

            bad_numbers = list(row.get("unsupported_numbers", []))
            if bad_numbers:
                supported = False
                unsupported_numbers += len(bad_numbers)
            if invalid_citations:
                issues.append(f"Invalid evidence citation identifiers: {invalid_citations}")
            # Keep provenance only for passages that actually passed the support
            # screen. A citation on an unsupported sentence is not evidence.
            record_indexes = evidence_indices if supported else []
            source_ids = []
            evidence_ids = []
            for evidence_index in record_indexes:
                if 1 <= evidence_index <= len(evidence_records):
                    record = evidence_records[evidence_index - 1]
                    metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
                    source_id = (
                        metadata.get("document_id") or metadata.get("source_id")
                        or metadata.get("source") or record.get("source")
                    ) if isinstance(record, dict) else None
                    if source_id is not None:
                        source_ids.append(str(source_id))
                    if isinstance(record, dict):
                        evidence_ids.append(record.get("evidence_id", f"evidence-{evidence_index}"))
            claim_support.append({
                "claim_id": f"claim-{index + 1}",
                "claim": row["sentence"],
                "support_status": "supported" if supported else "unsupported",
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
                "source_ids": list(dict.fromkeys(source_ids)),
                "evidence_indices": evidence_indices,
                "citations": citations,
                "citation_status": "valid" if valid_citations and (not supported or set(evidence_indices) & set(valid_citations)) else "missing_or_invalid",
                "semantic_similarity": round(score, 4),
                "lexical_support": row.get("support"),
                "unsupported_numbers": bad_numbers,
                "method": method,
            })

        citation_refs = [int(value) for value in _CITATION.findall(answer_text)]
        invalid_refs = sorted({value for value in citation_refs if value < 1 or value > evidence_count})
        if invalid_refs:
            issues.append(f"Invalid evidence citation identifiers: {invalid_refs}")
        if evidence_count == 0 and not any(refusal in answer for refusal in self.refusal_phrases):
            issues.append("Should refuse when no evidence available")
        if not citation_refs and evidence_count > 0 and rows:
            issues.append("No evidence citations found in answer")
        if any(not claim["citations"] for claim in claim_support):
            issues.append("One or more factual claims lack citations")
        for indicator in _HALLUCINATION_INDICATORS:
            if indicator in answer and "[evidence" not in answer:
                issues.append("Possible hallucination: confident claim without citation")
                break

        supported_count = sum(claim["support_status"] == "supported" for claim in claim_support)
        claim_count = len(claim_support)
        coverage = supported_count / claim_count if claim_count else None
        confidence = self._confidence(coverage)
        if any(claim["support_status"] != "supported" for claim in claim_support):
            issues.append("One or more generated claims lack sufficient semantic support")
        valid = not issues
        diagnostics = {
            "generated_claims": claim_count,
            "supported_claims": supported_count,
            "unsupported_claims": claim_count - supported_count,
            "coverage_score": coverage,
            "final_confidence": confidence,
            "citation_coverage": audit.get("attribution_coverage") if audit else None,
            "citation_support_coverage": audit.get("citation_support_coverage") if audit else None,
            "lexical_support": audit.get("faithfulness") if audit else None,
            "unsupported_number_count": unsupported_numbers,
            "semantic_threshold": self.semantic_threshold if semantic is not None else None,
            "method": method,
            "similarity_is_entailment": False,
        }
        result = {
            "valid": valid,
            "reason": "Passed" if valid else "Partial support or validation checks failed",
            "answer": answer_text,
            "issues": list(dict.fromkeys(issues)),
            "safety_issues": list(dict.fromkeys(safety_issues)),
            "citations_found": bool(citation_refs),
            "citation_references_validated": not invalid_refs and bool(citation_refs),
            "claim_support_screen_passed": bool(claim_count and supported_count == claim_count),
            "claim_support_validated": "NOT ESTABLISHED",
            "claim_support_method": method,
            "diagnostics": diagnostics,
            "claim_support": claim_support,
            "evidence_used": evidence_count,
        }
        self._log_validation(answer_text, evidence_texts, result)
        return result

    def retain_supported_claims(self, answer: str, claim_support: list[dict]) -> str:
        """Drop unsupported factual sentences and attach citations to support."""
        by_sentence = {
            " ".join(_CITATION.sub("", str(claim.get("claim", ""))).split()).lower(): claim
            for claim in claim_support
        }
        kept = []
        for part in split_sentences(answer):
            key = " ".join(_CITATION.sub("", part).split()).lower()
            claim = by_sentence.get(key)
            if claim is None:
                # Keep headings and short transitions which are not factual claims.
                if len(content_tokens(part)) < 3:
                    kept.append(part)
                continue
            if claim.get("support_status") != "supported":
                continue
            indexes = [
                int(value) for value in claim.get("evidence_indices", [])
                if str(value).isdigit() and int(value) > 0
            ]
            if not indexes:
                indexes = [
                    int(value) for value in claim.get("citations", [])
                    if str(value).isdigit() and int(value) > 0
                ]
            cleaned = _CITATION.sub("", part).rstrip()
            if indexes:
                cleaned += " " + " ".join(f"[Evidence {value}]" for value in dict.fromkeys(indexes))
            kept.append(cleaned)
        return "\n".join(kept).strip()
