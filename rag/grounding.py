import re
from typing import Iterable, List

from rag.document import RetrievedChunk


MIN_GROUNDING_SCORE = 2.0
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_FALLBACK_STOPWORDS = frozenset({
    "about", "and", "are", "can", "does", "explain", "for", "from", "how",
    "is", "me", "of", "on", "please", "say", "tell", "the", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "would",
})
_GENERIC_QUERY_TERMS = frozenset({
    "about", "affect", "affects", "cause", "causes", "change", "changes",
    "condition", "conditions", "disease", "diseases", "disorder", "disorders",
    "effect", "effects", "explain", "health", "impact", "information", "medical",
    "medicine", "problem", "problems", "provide", "related", "symptom", "symptoms",
    "treatment", "treatments", "what", "which", "who", "work", "works", "does",
    "doesn", "have", "many", "much", "most", "often", "show", "tell", "they",
    "this", "when", "where", "with", "your",
    "guide", "guidance", "evidence", "answer", "question", "query", "test", "say",
    "says", "based", "available", "include", "including", "information", "using",
    "common", "sign", "signs", "manifestation", "manifestations", "etiology",
    "reason", "reasons", "risk", "factor", "factors", "define", "definition",
    "meaning", "mechanism", "process", "function", "regulate", "regulated",
})
def _topic_terms(text: str) -> set[str]:
    terms = set()
    for raw in _TOKEN.findall(text or ""):
        token = raw.lower()
        # Preserve short biomedical acronyms (HIV, flu, IBS, RSV, etc.).
        is_acronym = len(raw) >= 2 and raw.isalpha() and raw.isupper()
        if token not in _GENERIC_QUERY_TERMS and (len(token) >= 4 or is_acronym):
            terms.add(token)
    return terms


def has_topic_overlap(query: str, passages: Iterable[str]) -> bool:
    """Require one or two specific query terms in evidence before generation."""
    anchors = _topic_terms(query)
    if not anchors:
        # Generic questions and greetings have no reliable topic anchor. Do not
        # allow arbitrary nearest-neighbor evidence to make them answerable;
        # require overlap on a meaningful fallback term (e.g. "guide").
        fallback_terms = {
            token for token in _TOKEN.findall((query or "").lower())
            if len(token) >= 3 and token not in _FALLBACK_STOPWORDS
        }
        if not fallback_terms:
            return False
        evidence_terms = set().union(*(
            set(_TOKEN.findall((passage or "").lower())) for passage in passages
        ))
        return bool(fallback_terms & evidence_terms)
    evidence_terms = set().union(*(_topic_terms(passage) for passage in passages))
    required_matches = min(2, len(anchors))
    return len(anchors & evidence_terms) >= required_matches


def is_grounded(results: List[RetrievedChunk]) -> bool:
    """
    Apply the current retrieval-distance threshold.

    Note: This is a heuristic retrieval check, not proof that
    an answer is medically accurate or fully supported by evidence.
    """
    if not results:
        return False

    best_score = results[0].distance

    return (
        best_score >= 0
        and best_score < MIN_GROUNDING_SCORE
    )


def build_grounded_context(results: List[RetrievedChunk]) -> str:
    """Format retrieved chunks as evidence for answer generation."""
    context_parts = []

    for idx, result in enumerate(results, start=1):
        source = result.metadata.get("source", "Unknown")
        score = result.distance

        context_parts.append(
            f"[Evidence {idx}]\n"
            f"Source: {source}\n"
            f"Distance: {score:.4f}\n\n"
            f"{result.content}"
        )

    return "\n\n".join(context_parts)
