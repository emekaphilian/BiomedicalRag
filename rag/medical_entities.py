"""Lightweight normalization for a few common biomedical concepts.

This is a conservative retrieval aid, not a complete clinical ontology.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_ALIASES = (
    ("pulmonary_hypertension", ("pulmonary hypertension", "pulmonary arterial hypertension")),
    ("systemic_hypertension", ("systemic hypertension", "high blood pressure", "hypertension", "blood pressure")),
    ("type_1_diabetes", ("type 1 diabetes mellitus", "type 1 diabetes", "type i diabetes")),
    ("type_2_diabetes", ("type 2 diabetes mellitus", "type 2 diabetes", "type ii diabetes")),
    ("alzheimer_disease", ("alzheimer disease", "alzheimer's disease", "alzheimers disease", "alzheimer")),
    ("diabetes", ("diabetes mellitus", "diabetes")),
    ("parkinson_disease", ("parkinson's disease", "parkinsons disease", "parkinson disease", "parkinson")),
    ("peptic_ulcer", ("stomach ulcer", "stomach ulcers", "gastric ulcer", "gastric ulcers", "peptic ulcer", "peptic ulcers", "ulcer")),
)


def normalize_concepts(text: str) -> tuple[str, ...]:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    found = []
    for concept, phrases in _ALIASES:
        if any(re.search(r"\b" + re.escape(re.sub(r"[^a-z0-9]+", " ", phrase)) + r"\b", normalized) for phrase in phrases):
            found.append(concept)
    if "pulmonary_hypertension" in found and "systemic_hypertension" in found:
        found.remove("systemic_hypertension")
    # A generic diabetes mention should not collapse a more specific subtype.
    if "diabetes" in found and any(item in found for item in ("type_1_diabetes", "type_2_diabetes")):
        found.remove("diabetes")
    return tuple(found)


def expand_concepts(query: str) -> str:
    concepts = normalize_concepts(query)
    additions = []
    for concept in concepts:
        if concept == "systemic_hypertension":
            additions.extend(("high blood pressure", "systemic hypertension"))
        elif concept == "pulmonary_hypertension":
            additions.append("pulmonary arterial hypertension")
        elif concept == "type_1_diabetes":
            additions.append("type 1 diabetes mellitus")
        elif concept == "type_2_diabetes":
            additions.append("type 2 diabetes mellitus")
        elif concept == "parkinson_disease":
            additions.extend(("parkinson's disease", "parkinson disease"))
        elif concept == "peptic_ulcer":
            additions.extend(("peptic ulcer", "stomach ulcer", "gastric ulcer", "h pylori"))
    return " ".join(dict.fromkeys([query.strip(), *additions])).strip()


def concept_match_score(query: str, passage: str, metadata: dict | None = None) -> tuple[float, str, list[str]]:
    """Return entity match score, match type, and matching concept IDs."""
    query_concepts = normalize_concepts(query)
    if not query_concepts:
        return 0.0, "none", []
    metadata = metadata if isinstance(metadata, dict) else {}
    title = " ".join(str(metadata.get(key) or "") for key in ("title", "document_title", "section", "heading"))
    context = f"{title} {passage or ''}"
    passage_concepts = normalize_concepts(context)

    # Specific pulmonary hypertension must never satisfy an unqualified or
    # systemic hypertension query just because both contain “hypertension”.
    if "systemic_hypertension" in query_concepts and "pulmonary_hypertension" in passage_concepts and "systemic_hypertension" not in passage_concepts:
        return 0.0, "related_condition", []
    if "pulmonary_hypertension" in query_concepts and "systemic_hypertension" in passage_concepts and "pulmonary_hypertension" not in passage_concepts:
        return 0.0, "related_condition", []

    opposite_diabetes_types = (
        ("type_1_diabetes", "type_2_diabetes"),
        ("type_2_diabetes", "type_1_diabetes"),
    )
    if any(query_entity in query_concepts and passage_entity in passage_concepts
           for query_entity, passage_entity in opposite_diabetes_types):
        return 0.0, "related_condition", []

    exact = [concept for concept in query_concepts if concept in passage_concepts]
    if exact:
        query_text = re.sub(r"[^a-z0-9]+", " ", (query or "").lower()).strip()
        passage_text = re.sub(r"[^a-z0-9]+", " ", context.lower()).strip()
        exact_phrase = []
        for concept in exact:
            for phrase in _ALIASES:
                if phrase[0] != concept:
                    continue
                for alias in phrase[1]:
                    normalized_alias = re.sub(r"[^a-z0-9]+", " ", alias).strip()
                    if re.search(r"\b" + re.escape(normalized_alias) + r"\b", query_text) and re.search(
                        r"\b" + re.escape(normalized_alias) + r"\b", passage_text,
                    ):
                        exact_phrase.append(concept)
                        break
                if concept in exact_phrase:
                    break
        if exact_phrase:
            return 1.0, "exact_concept", exact_phrase
        return 0.9, "synonym", exact
    if "diabetes" in query_concepts and any(x in passage_concepts for x in ("type_1_diabetes", "type_2_diabetes")):
        return 0.65, "subtype_match", [x for x in passage_concepts if x.endswith("_diabetes")]
    return 0.0, "no_match", []


def keyword_entity_overlap(query: str, passage: str, metadata: dict | None = None) -> tuple[float, dict]:
    """Combine normalized concept match and meaningful keyword overlap."""
    metadata = metadata if isinstance(metadata, dict) else {}
    title = " ".join(str(metadata.get(key) or "") for key in ("title", "document_title", "section", "heading"))
    text = f"{title} {passage or ''}".lower()
    ignored = {"what", "which", "who", "how", "does", "cause", "causes", "compare", "versus", "difference", "between", "the", "and", "for", "with", "from", "that", "this", "type"}
    query_terms = {token for token in _TOKEN.findall((query or "").lower()) if len(token) > 2 and token not in ignored}
    text_terms = set(_TOKEN.findall(text))
    keyword = len(query_terms & text_terms) / max(1, len(query_terms))
    entity, match_type, matched = concept_match_score(query, passage, metadata)
    score = max(keyword, entity)
    return min(1.0, score), {"keyword_overlap": round(keyword, 4), "entity_score": entity, "entity_match": match_type, "entity_matches": matched}


def normalized_scores(scores: Iterable[float]) -> list[float]:
    values = [float(value) for value in scores]
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]
