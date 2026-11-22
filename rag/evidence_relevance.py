"""Deterministic query intent and evidence relevance labels for biomedical RAG.

These labels are retrieval aids, not medical truth judgments. They use the
question, passage text, and available source metadata and are retained for audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP = frozenset(("a about also an and are as at be been being by can common do does for from had has have how i in is it its of on or please the their them there these this those to was what when where which who why with would "
                  "information evidence answer question explain describe tell me give provide list many much often usually typical say says").split())
_INTENT_WORDS = frozenset("symptom sign cause reason definition define meaning mechanism process function work effect impact affect complication consequence risk factor".split())
_INTENT_CUES = {
    "symptoms": ("symptom", "sign", "manifestation"),
    "causes": ("cause", "causes", "caused", "reason", "etiology", "risk factor", "develop"),
    "definition": ("define", "definition", "is a", "refers to", "characterized"),
    "mechanism": ("function", "work", "mechanism", "process", "regulate", "affect"),
    "complications": ("complication", "consequence", "damage", "effect"),
    "general": (),
}
_COMPLICATION_CUES = frozenset(("complication complication complications complication-related consequence consequences "
                               "retinopathy neuropathy nephropathy cardiovascular stroke blindness ulcer amputation damage "
                               "side effect adverse event sequela sequelae progression eye problem visual impairment").split())
_RELATED_PREFIXES = ("pre", "post", "non", "gestational")
_OPPOSITE_TERMS = (
    frozenset(("increas", "decreas")), frozenset(("higher", "lower")),
    frozenset(("more", "less")), frozenset(("present", "absent")),
    frozenset(("positive", "negative")),
)


def _stem(token: str) -> str:
    token = token.lower()
    for suffix in ("ies", "es", "ic", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def _tokens(text: str) -> set[str]:
    return {_stem(token) for token in _TOKEN.findall(text or "") if token.lower() not in _STOP}


@dataclass(frozen=True)
class QueryIntent:
    subject_terms: tuple[str, ...]
    information_type: str
    expected_evidence: tuple[str, ...]
    retrieval_query: str


def interpret_query(query: str) -> QueryIntent:
    """Derive a conservative, model-free representation of the information need."""
    lowered = (query or "").lower()
    if re.search(r"\b(symptoms?|signs?|manifestations?)\b", lowered):
        kind = "symptoms"
    elif re.search(r"\b(caus(?:e|es|ed|ing)|etiolog(?:y|ies)|why)\b", lowered):
        kind = "causes"
    elif re.search(r"\b(what is|define|definition|meaning of)\b", lowered):
        kind = "definition"
    elif re.search(r"\b(how (?:does|do|is|are)|mechanism|process|function|work)\b", lowered):
        kind = "mechanism"
    elif re.search(r"\b(complications?|consequences|effects?)\b", lowered):
        kind = "complications"
    else:
        kind = "general"
    subject = sorted({term for term in _tokens(lowered) if term not in _INTENT_WORDS})
    cues = _INTENT_CUES[kind]
    expanded = query.strip()
    if cues:
        expanded = f"{expanded} {' '.join(cues)}"
    return QueryIntent(tuple(subject), kind, cues, expanded)


def classify_evidence(query: str, passage: str, metadata: dict | None = None) -> dict:
    """Classify a passage relative to an information need; keep reasons explicit."""
    intent = interpret_query(query)
    metadata = metadata if isinstance(metadata, dict) else {}
    body = passage or ""
    title = str(metadata.get("title") or metadata.get("document_title") or "")
    section = str(metadata.get("section") or metadata.get("heading") or "")
    body_tokens = _tokens(body)
    subject = set(intent.subject_terms)
    matched = subject & body_tokens
    # Recognize a related prefixed form (e.g. pre-condition) without embedding
    # condition-specific aliases in production logic.
    related = {
        term for token in body_tokens for term in subject
        if any(token.startswith(prefix + term) for prefix in _RELATED_PREFIXES)
    }
    cues = _INTENT_CUES[intent.information_type]
    cue_tokens = set().union(*(_tokens(cue) for cue in cues)) if cues else set()
    cue_phrases = ("is a", "refers to", "characterized") if intent.information_type == "definition" else ()
    intent_in_body = bool(cue_tokens & body_tokens) or any(
        re.search(r"\b" + re.escape(phrase) + r"\b", body.lower())
        for phrase in cue_phrases
    )
    text_tokens = _tokens(" ".join((title, section)))
    title_subject = bool(subject & text_tokens)
    complication = bool(_COMPLICATION_CUES & (body_tokens | text_tokens))

    if complication and (matched or title_subject):
        category = "complication"
        reason = "Passage describes a complication or consequence rather than the requested general information."
    elif matched and (intent_in_body or (intent.information_type == "general")):
        category = "direct"
        reason = "Passage mentions the requested subject and addresses the requested information type."
    elif matched or related or title_subject:
        category = "related_indirect"
        reason = "Passage is about the requested or a related subject but does not establish the requested information type."
    elif not subject:
        category = "uncertain"
        reason = "The query did not yield reliable subject terms for deterministic relevance classification."
    else:
        category = "irrelevant"
        reason = "Passage does not meaningfully match the requested subject and information type."

    # Contextual support is deliberately conservative: the passage must mention
    # the subject and contain biomedical content, but not the requested relation.
    if category == "related_indirect" and matched and len(body_tokens) >= 8 and not related:
        category = "supporting_contextual"
        reason = "Passage provides subject-specific background but does not independently answer the requested information type."
    return {
        "category": category,
        "reason": reason,
        "subject_match": bool(matched),
        "intent_match": intent_in_body,
        "matched_subject_terms": sorted(matched),
        "information_type": intent.information_type,
    }

def classify_evidence_set(query: str, evidence: list[dict]) -> list[dict]:
    """Classify passages, then flag explicit opposing terms on the same topic."""
    rows = [
        classify_evidence(query, item.get("content", ""), item.get("metadata", {}))
        for item in evidence
    ]
    for left in range(len(evidence)):
        left_tokens = _tokens(evidence[left].get("content", ""))
        for right in range(left + 1, len(evidence)):
            right_tokens = _tokens(evidence[right].get("content", ""))
            shared = left_tokens & right_tokens
            subject_overlap = set(rows[left]["matched_subject_terms"]) & set(rows[right]["matched_subject_terms"])
            explicit_opposition = any(
                bool(left_tokens & pair) and bool(right_tokens & pair)
                and (left_tokens & pair) != (right_tokens & pair)
                for pair in _OPPOSITE_TERMS
            )
            if subject_overlap and len(shared) >= 3 and explicit_opposition:
                for index in (left, right):
                    rows[index] = {
                        **rows[index],
                        "category": "contradictory",
                        "reason": "Relevant passages use explicitly opposing factual terms; the disagreement is unresolved.",
                        "contradiction_pair": [left, right],
                    }
    return rows


def classify_contradictions(evidence: list[dict]) -> dict:
    """Report whether the conservative lexical contradiction screen found a pair."""
    pairs = sorted({
        tuple(row["contradiction_pair"])
        for row in evidence if row.get("contradiction_pair")
    })
    return {
        "detected": bool(pairs),
        "status": "screened" if evidence else "not_evaluated",
        "method": "explicit-opposing-terms heuristic; not clinical adjudication",
        "pairs": [list(pair) for pair in pairs],
    }
