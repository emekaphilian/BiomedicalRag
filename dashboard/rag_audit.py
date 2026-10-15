"""Lexical RAG audit proxies for human review, not truth verification."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

_STOP = frozenset(
    "a an the and or but of in on at to for with by from as is are was were be been being this that "
    "these those it its into than then so such not no can may might will would should could has have "
    "had do does did which who whom whose what when where why how also more most other some any each "
    "their there they them we you your our".split()
)
_CITATION = re.compile(r"\[(\d+)\]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 4 and token.endswith("s") and not token.endswith("ss") else token


def content_tokens(text: str) -> set:
    tokens = (_stem(token) for token in _WORD.findall(_CITATION.sub(" ", text or "").lower()))
    return {token for token in tokens if token not in _STOP and (len(token) >= 3 or token.isdigit())}


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [part.strip() for part in parts if part and part.strip()]


def audit_answer(
    answer: str,
    evidence_texts: Sequence[str],
    support_threshold: float = 0.6,
    trace_threshold: float = 0.35,
    min_tokens: int = 3,
) -> dict:
    """Score sentence token overlap against evidence and report citation bounds."""
    evidence_tokens = [content_tokens(text) for text in evidence_texts]
    evidence_numbers = set().union(*(set(_NUMBER.findall(text)) for text in evidence_texts)) if evidence_texts else set()
    rows = []
    for sentence in split_sentences(answer):
        tokens = content_tokens(sentence)
        if len(tokens) < min_tokens:
            continue
        best_index, best = None, 0.0
        for index, evidence in enumerate(evidence_tokens):
            score = len(tokens & evidence) / len(tokens)
            if score > best:
                best_index, best = index, score
        bad_numbers = sorted(
            number for number in set(_NUMBER.findall(_CITATION.sub(" ", sentence)))
            if number not in evidence_numbers
        )
        rows.append({
            "sentence": sentence,
            "support": round(best, 3),
            "source_index": best_index + 1 if best_index is not None and best >= trace_threshold else None,
            "supported": best >= support_threshold and not bad_numbers,
            "unsupported_numbers": bad_numbers,
        })
    count = len(rows)
    supported = sum(row["supported"] for row in rows)
    markers = [int(match) for match in _CITATION.findall(answer or "")]
    return {
        "method": "lexical-overlap",
        "thresholds": {"support": support_threshold, "trace": trace_threshold},
        "sentence_count": count,
        "faithfulness": supported / count if count else None,
        "hallucination_rate": 1 - supported / count if count else None,
        "attribution_coverage": sum(row["source_index"] is not None for row in rows) / count if count else None,
        "flagged_sentences": [row for row in rows if not row["supported"]],
        "sentences": rows,
        "citation_markers": markers,
        "invalid_citations": sorted({marker for marker in markers if marker < 1 or marker > len(evidence_texts)}),
    }


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize_audits(audits: Sequence[Mapping]) -> dict:
    return {
        "answers_audited": len(audits),
        "faithfulness": _mean(audit.get("faithfulness") for audit in audits),
        "hallucination_rate": _mean(audit.get("hallucination_rate") for audit in audits),
        "attribution_coverage": _mean(audit.get("attribution_coverage") for audit in audits),
    }


def retrieval_summary(runs: Sequence[Mapping]) -> dict:
    """Average labeled retrieval metrics, choosing the first available @k value."""
    def pick(metrics, prefix):
        return next((value for key, value in metrics.items() if key.startswith(prefix)), None)

    metrics = [run.get("metrics", {}) for run in runs]
    return {
        "runs": len(runs),
        "recall": _mean(pick(item, "recall@") for item in metrics),
        "precision": _mean(pick(item, "precision@") for item in metrics),
        "ndcg": _mean(pick(item, "ndcg@") for item in metrics),
        "mrr": _mean(item.get("mrr") for item in metrics),
    }
