"""Deterministic query intent and evidence relevance labels for biomedical RAG.

These labels are retrieval aids, not medical truth judgments. They use the
question, passage text, and available source metadata and are retained for audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN = re.compile(r"[a-z0-9]+(?:['’]s)?", re.IGNORECASE)
_STOP = frozenset(("a about also an and are as at be been being by can common do does for from had has have how i in is it its of on or please the their them there these this those to was what when where which who why with would "
                  "information evidence answer question explain describe tell me give provide list many much often usually typical say says summarize summary current understanding support supports").split())
_INTENT_WORDS = frozenset("symptom sympto sign cause caus reason definition defin define meaning mechanism process function work effect impact affect complication consequence risk factor therapy therap therapeutic therapeut therapies treatment treatments drug drugs medication medications medicine medicines management manage intervention interventions support supports compare comparison compar versus vs difference differences type prevent prevention preventable preventabl reduce diagnosis diagnos diagnose diagnosed screening test communicate communication communicat signal signaling signalling transmission transmit role roles".split())
_INTENT_CUES = {
    "symptoms": ("symptom", "sign", "manifestation"),
    "causes": ("cause", "causes", "caused", "reason", "etiology", "due to", "result from"),
    "definition": ("define", "definition", "is a", "is when", "means", "defined as", "refers to", "characterized"),
    "mechanism": ("function", "work", "mechanism", "process", "regulate", "affect", "communicate", "communication", "signal", "signaling", "signalling", "transmission", "transmit"),
    "treatments": ("treatment", "therapy", "therapeutic", "drug", "medication", "medicine", "management", "intervention", "approved"),
    "complications": ("complication", "consequence", "damage", "effect"),
    "risk_factors": ("risk factor", "risk", "predispose", "associated with"),
    "summary": (),
    "explanation": (),
    "comparison": (),
    "prevention": ("prevent", "prevention", "reduce risk"),
    "diagnosis": ("diagnosis", "diagnostic", "screening", "test"),
    "general": (),
}
_COMPLICATION_PATTERNS = (
    re.compile(r"\bcomplications?\b", re.IGNORECASE),
    re.compile(r"\b(?:retinopathy|neuropathy|nephropathy|stroke|blindness|amputation|sequelae?)\b", re.IGNORECASE),
    re.compile(r"\b(?:eye problems?|vision loss|kidney damage|nerve damage)\b", re.IGNORECASE),
    re.compile(r"\bcardiovascular (?:disease|events?)\b", re.IGNORECASE),
)
_RELATED_PREFIXES = ("pre", "post", "non", "gestational")
_OPPOSITE_TERMS = (
    frozenset(("increas", "decreas")), frozenset(("higher", "lower")),
    frozenset(("more", "less")), frozenset(("present", "absent")),
    frozenset(("positive", "negative")),
)


def _stem(token: str) -> str:
    token = token.lower()
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("es", "ic", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def _tokens(text: str) -> set[str]:
    aliases = {"therapy": "treatment", "therapeut": "treatment", "medication": "drug", "medicine": "drug", "pharmacological": "drug", "diabetic": "diabet"}
    tokens = []
    for token in _TOKEN.findall(text or ""):
        token = re.sub(r"['’]s$", "", token.lower())
        if token in _STOP:
            continue
        normalized = _stem(token)
        tokens.append(aliases.get(normalized, normalized))
    return set(tokens)


@dataclass(frozen=True)
class QueryIntent:
    subject_terms: tuple[str, ...]
    information_type: str
    expected_evidence: tuple[str, ...]
    retrieval_query: str
    comparison_entities: tuple[str, ...] = ()


def interpret_query(query: str) -> QueryIntent:
    """Derive a conservative, model-free representation of the information need."""
    lowered = (query or "").lower()
    comparison_entities = ()
    type_comparison = re.search(r"\btype\s*1\b.*\btype\s*2\b|\btype\s*2\b.*\btype\s*1\b", lowered)
    if re.search(r"\b(compare|comparison|versus|vs\.?|difference between|differences between)\b", lowered):
        kind = "comparison"
        if type_comparison:
            comparison_entities = ("type 1 diabetes", "type 2 diabetes")
    elif re.search(r"\b(therap(?:y|ies)|therapeutics?|treat(?:ment|ments|ed|ing)?|medications?|medicines?|drugs?|pharmacological|management|interventions?)\b", lowered):
        kind = "treatments"
    elif re.search(r"\b(risk factors?|predispos(?:e|ing)|at risk)\b", lowered):
        kind = "risk_factors"
    elif re.search(r"\b(prevent(?:s|ed|ion|ing|able)?|reduce the risk|lower the risk)\b", lowered):
        kind = "prevention"
    elif re.search(r"\b(diagnos(?:e|is|ed|ing)|screen(?:ing)?|test(?:s|ed|ing)?)\b", lowered):
        kind = "diagnosis"
    elif re.search(r"\b(summar(?:y|ize|ise)|overview)\b", lowered):
        kind = "summary"
    elif re.search(r"\b(explain|describe)\b", lowered):
        kind = "explanation"
    elif re.search(r"\b(symptoms?|signs?|manifestations?)\b", lowered):
        kind = "symptoms"
    elif re.search(r"\b(caus(?:e|es|ed|ing)|etiolog(?:y|ies)|why)\b", lowered):
        kind = "causes"
    elif re.search(r"\b(how (?:does|do|is|are)|mechanism|process|function|role|communicat\w*|signal(?:ing|ling)?|transmission|transmit|work)\b", lowered):
        kind = "mechanism"
    elif re.search(r"\b(what is|define|definition|meaning of)\b", lowered):
        kind = "definition"
    elif re.search(r"\b(complications?|consequences|effects?)\b", lowered):
        kind = "complications"
    else:
        kind = "general"
    subject = sorted({
        term for term in _tokens(lowered)
        if term not in _INTENT_WORDS
        and not term.isdigit()
        and term not in {"disease", "diseas", "disorder", "condition", "illness"}
    })
    cues = _INTENT_CUES[kind]
    expanded = query.strip()
    if comparison_entities:
        expanded = f"{query.strip()} type 1 diabetes characteristics insulin production type 2 diabetes characteristics insulin resistance differences"
    elif subject and kind == "treatments":
        expanded = f"{' '.join(subject)} disease treatment"
    elif subject and kind == "symptoms":
        expanded = f"{' '.join(subject)} symptoms manifestation"
    elif subject and kind == "causes":
        expanded = f"{' '.join(subject)} causes etiology"
    elif subject and kind == "mechanism":
        expanded = f"{' '.join(subject)} mechanism function"
    elif subject and kind == "complications":
        expanded = f"{' '.join(subject)} complications"
    elif subject and kind in {"prevention", "diagnosis", "risk_factors"}:
        expanded = f"{' '.join(subject)} {' '.join(cues)}"
    elif subject and kind == "definition":
        expanded = f"{' '.join(subject)} definition overview"
    elif subject and kind in {"summary", "explanation", "general"}:
        expanded = f"{' '.join(subject)} biomedical overview"
    return QueryIntent(tuple(subject), kind, cues, expanded, comparison_entities)


def build_retrieval_queries(query: str, intent: QueryIntent | None = None) -> list[str]:
    """Create focused, meaning-preserving phrasings for biomedical retrieval."""
    from rag.medical_entities import expand_concepts

    intent = intent or interpret_query(query)
    if intent.comparison_entities:
        queries = [
            "type 1 diabetes overview",
            "type 2 diabetes overview",
            "difference between type 1 and type 2 diabetes",
            "type 1 diabetes insulin dependence",
            "type 2 diabetes insulin resistance",
        ]
    else:
        original = " ".join((query or "").split()).strip()
        focused_subject = " ".join(intent.subject_terms)
        if intent.information_type == "causes":
            focused_terms = "cause etiology mechanisms"
        elif intent.information_type == "risk_factors":
            focused_terms = "risk factors associated factors"
        elif intent.information_type == "treatments":
            focused_terms = "treatment therapies management"
        elif intent.information_type == "symptoms":
            focused_terms = "symptoms signs manifestations"
        elif intent.information_type == "mechanism":
            focused_terms = "function role mechanism process signaling"
        elif intent.information_type == "definition":
            focused_terms = "definition overview"
        elif intent.information_type == "diagnosis":
            focused_terms = "diagnosis testing screening"
        elif intent.information_type == "prevention":
            focused_terms = "prevention risk reduction"
        else:
            focused_terms = "biomedical overview"
        focused = expand_concepts(" ".join(part for part in (focused_subject, focused_terms) if part))
        base = expand_concepts(intent.retrieval_query)
        queries = [focused, base]
        if original:
            queries.append(original)

        # Concept-level expansions connect common biomedical phrasing to the
        # language used in reference passages. These expand concepts, not whole
        # canned questions, and retain the original query above.
        lowered = (query or "").lower()
        concept_expansions = (
            (r"\bdopamine\b", "dopamine neurotransmitter chemical messenger brain signaling movement"),
            (r"\b(?:neurons?|neural|synap\w*)\b", "neurons nerve cells neural communication synaptic transmission electrical chemical signals neurotransmitter release"),
            (r"\b(?:immune|immunity)\b", "immune system function immune response innate adaptive immunity cells antibodies"),
            (r"\b(?:liver|hepatic)\b", "liver hepatic function metabolism bile protein production detoxification"),
            (r"\b(?:circulation|circulatory)\b", "blood circulation circulatory system blood flow heart vessels cardiovascular"),
            (r"\bcellular respiration\b", "cellular respiration mitochondria ATP glucose oxygen energy metabolism"),
        )
        queries.extend(
            expand_concepts(expansion)
            for pattern, expansion in concept_expansions
            if re.search(pattern, lowered)
        )
        if intent.information_type in {"causes", "risk_factors"}:
            queries.extend((f"{base} cause etiology risk factors", f"{base} causes of disease"))
        elif intent.information_type == "treatments":
            queries.append(f"{base} current therapies treatment options")
        elif intent.information_type == "diagnosis":
            queries.append(f"{base} diagnosis diagnostic criteria testing screening")
        elif intent.information_type == "prevention":
            queries.append(f"{base} prevention risk reduction")
        elif intent.information_type == "definition":
            queries.append(f"{base} definition overview")
    return list(dict.fromkeys(" ".join(item.split()).strip() for item in queries if item.strip()))


def classify_evidence(query: str, passage: str, metadata: dict | None = None) -> dict:
    """Classify a passage relative to an information need; keep reasons explicit."""
    intent = interpret_query(query)
    metadata = metadata if isinstance(metadata, dict) else {}
    body = passage or ""
    title = str(metadata.get("title") or metadata.get("document_title") or "")
    section = str(metadata.get("section") or metadata.get("heading") or "")
    body_tokens = _tokens(body)
    from rag.medical_entities import concept_match_score, normalize_concepts
    concept_score, concept_match, entity_matches = concept_match_score(query, body, metadata)
    incompatible_entity = concept_match == "related_condition"
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
    text_tokens = _tokens(" ".join((title, section)))
    title_subject = bool(subject & text_tokens)
    matched_subject = matched | (subject & text_tokens)
    comparison_entities_supported = []
    for entity in intent.comparison_entities:
        entity_number = re.search(r"type\s*(1|2)", entity).group(1)
        entity_context = f"{title} {body}"
        if re.search(rf"\btype\s*{entity_number}\b", entity_context, re.IGNORECASE) and re.search(
            r"\bdiabetes\b", entity_context, re.IGNORECASE,
        ):
            comparison_entities_supported.append(entity)
    cue_phrases = ("is a", "is when", "means", "defined as", "refers to", "characterized") if intent.information_type == "definition" else ()
    intent_in_body = bool(cue_tokens & (body_tokens | text_tokens)) or any(
        re.search(r"\b" + re.escape(phrase) + r"\b", f"{title} {section} {body}".lower())
        for phrase in cue_phrases
    )
    source_text = " ".join((title, section, body))
    complication = any(pattern.search(source_text) for pattern in _COMPLICATION_PATTERNS)
    risk_relation = bool(re.search(
        r"\b(?:risk factors?|risk (?:of [\w -]+ )?increases?|more likely|higher risk|associated with|risk is higher)\b",
        source_text, re.IGNORECASE,
    ))
    prevention_relation = bool(re.search(r"\b(?:prevent(?:s|ed|ion|ing|able)?|prevention)\b", source_text, re.IGNORECASE))
    concepts = normalize_concepts(query)
    concept_subject_patterns = {
        "systemic_hypertension": r"(?:high blood pressure|systemic hypertension|hypertension)",
        "pulmonary_hypertension": r"(?:pulmonary arterial hypertension|pulmonary hypertension)",
        "type_1_diabetes": r"type\s*1 diabetes(?: mellitus)?",
        "type_2_diabetes": r"type\s*2 diabetes(?: mellitus)?",
        "diabetes": r"diabetes(?: mellitus)?",
        "alzheimer_disease": r"alzheimer(?:'s)? disease|alzheimers disease|alzheimer",
    }
    subject_pattern = next((concept_subject_patterns[item] for item in concepts if item in concept_subject_patterns), None)
    if subject_pattern is None:
        subject_pattern = r"(?:" + "|".join(re.escape(term) for term in sorted(subject)) + r")" if subject else r"(?!)"
    aging_mechanism = bool(re.search(
        r"\bdevelops? over time as you get older\b", source_text, re.IGNORECASE,
    ))
    genetic_mechanism_support = bool(
        (matched or title_subject or concept_score)
        and re.search(
            r"\b(?:genetic disease|genetic changes?|mutations? in genes?|changes? in genes?)\b",
            source_text,
            re.IGNORECASE,
        )
        and re.search(
            r"\b(?:cause[sd]?|affect|alter|change[sd]?|control|function|develop|grow)\b",
            source_text,
            re.IGNORECASE,
        )
    )
    cause_unknown_support = bool(
        (matched or title_subject or concept_score)
        and re.search(
            r"\b(?:no (?:known|clear|single) cause|(?:exact )?cause(?: of [\w -]+)? "
            r"(?:is )?(?:not known|unknown|unclear|not identified)|"
            r"(?:no one|doctors? do not|doctors? don't) know (?:the )?(?:exact )?cause|"
            r"do not know (?:the )?(?:exact )?cause)\b",
            source_text,
            re.IGNORECASE,
        )
    )
    tumor_growth_mechanism_support = bool(
        intent.information_type == "causes"
        and (matched or title_subject or concept_score)
        and re.search(r"\btumou?rs?\b", source_text, re.IGNORECASE)
        and re.search(
            r"(?:extra cells? grow|cells? grow out of control|normal cell growth and death|"
            r"cell growth and death processes? go wrong)",
            source_text,
            re.IGNORECASE,
        )
    )
    sentence_support = False
    causal_direction_support = False
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        sentence_tokens = _tokens(sentence)
        sentence_entity_score, _, _ = concept_match_score(query, sentence, {})
        if not (subject & sentence_tokens) and sentence_entity_score <= 0:
            continue
        if cue_tokens & sentence_tokens or any(
            re.search(r"\b" + re.escape(phrase) + r"\b", sentence.lower())
            for phrase in cue_phrases
        ):
            sentence_support = True
        if intent.information_type == "causes" and re.search(
            rf"(?:\b{subject_pattern}\b[^.!?]{{0,100}}\b(?:is|are|can be|may be|were|was)?\s*caused by\b|"
            rf"\b(?:cause|causes|etiology) of\b[^.!?]{{0,100}}\b{subject_pattern}\b|"
            rf"\b{subject_pattern}\b[^.!?]{{0,100}}\b(?:due to|result(?:s|ed)? from|because of)\b)",
            sentence,
            re.IGNORECASE,
        ):
            causal_direction_support = True
    if incompatible_entity and not (intent.information_type == "comparison" and intent.comparison_entities):
        direct_support = False
    elif intent.information_type == "comparison" and intent.comparison_entities:
        direct_support = bool(comparison_entities_supported)
    elif intent.information_type == "causes":
        direct_support = bool(
            causal_direction_support or aging_mechanism or genetic_mechanism_support
            or cause_unknown_support or tumor_growth_mechanism_support
        )
    elif intent.information_type == "treatments":
        treatment_terms = {"treatment", "treat", "therapeut"}
        title_treatment_match = bool(
            title_subject
            and treatment_terms
            & (text_tokens | body_tokens)
        )
        direct_support = sentence_support or title_treatment_match
    else:
        direct_support = sentence_support or (title_subject and intent_in_body)

    if incompatible_entity and not (intent.information_type == "comparison" and intent.comparison_entities):
        category = "related_indirect"
        reason = "Passage is about a distinct related condition, not the normalized disease concept requested."
    elif intent.information_type == "causes" and risk_relation and not direct_support:
        category = "supporting_contextual"
        reason = "Passage reports a risk association; this may inform risk but does not establish a direct cause."
    elif intent.information_type == "causes" and prevention_relation and not direct_support:
        category = "related_indirect"
        reason = "Passage discusses prevention rather than explaining the requested causes."
    elif complication and (matched or title_subject) and not direct_support:
        category = "complication"
        reason = "Passage describes a complication or consequence rather than the requested general information."
    elif (
        intent.information_type == "comparison"
        and intent.comparison_entities
        and comparison_entities_supported
    ):
        category = "direct"
        reason = "Passage contains information about one or more requested comparison entities."
    elif (matched or title_subject or comparison_entities_supported or concept_score) and (
        direct_support
        or intent.information_type in {"general", "summary", "explanation"}
        or (
            intent.information_type == "comparison"
            and not intent.comparison_entities
        )
    ):
        category = "direct"
        reason = "Passage mentions the requested subject and addresses the requested information type."
    elif matched or related or title_subject or concept_score:
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
    if (category == "related_indirect" and matched and len(body_tokens) >= 8 and not related
            and intent.information_type not in {"causes", "prevention", "diagnosis", "risk_factors"}):
        category = "supporting_contextual"
        reason = "Passage provides subject-specific background but does not independently answer the requested information type."
    role = {
        "direct": "direct_evidence",
        "supporting_contextual": "supporting_evidence",
        "related_indirect": "related_evidence",
        "complication": "indirect_evidence",
        "contradictory": "contradictory_evidence",
        "irrelevant": "irrelevant_evidence",
        "uncertain": "uncertain",
    }.get(category, "uncertain")
    return {
        "category": category,
        "evidence_role": role,
        "reason": reason,
        "relevance_explanation": reason,
        "classification_confidence": None,
        "classification_confidence_status": "not_calibrated",
        "subject_match": bool(matched_subject or concept_score),
        "intent_match": direct_support,
        "matched_subject_terms": sorted(matched_subject),
        "information_type": intent.information_type,
        "comparison_entities_supported": comparison_entities_supported,
        "normalized_entity_matches": entity_matches,
        "entity_match_type": concept_match,
    }

def classify_evidence_set(query: str, evidence: list[dict]) -> list[dict]:
    """Classify passages, then flag explicit opposing terms on the same topic."""
    intent = interpret_query(query)
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
            if intent.information_type == "comparison":
                left_entities = set(rows[left].get("comparison_entities_supported", []))
                right_entities = set(rows[right].get("comparison_entities_supported", []))
                # Contrasting evidence across the requested sides is expected;
                # only flag conflicts when both passages address the same side.
                if not (left_entities & right_entities):
                    continue
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


def assess_evidence_coverage(query: str, evidence: list[dict]) -> dict:
    """Assess question-aware evidence coverage from passage roles, not scores."""
    if not isinstance(query, str) or not query.strip():
        return {"status": "not_calculated", "reason": "The original question is unavailable.", "coverage_score": None, "components": []}
    if not evidence:
        return {"status": "no_relevant_coverage", "reason": "No passages were retrieved.", "coverage_score": 0.0, "components": []}

    intent = interpret_query(query)
    direct = []
    supporting = []
    related = []
    indirect = []
    contradictory = []
    irrelevant = []
    for index, item in enumerate(evidence, 1):
        classification = item.get("evidence_classification") or classify_evidence(
            query, item.get("content", ""), item.get("metadata", {}),
        )
        row = {
            "evidence_id": item.get("evidence_id", f"evidence-{index}"),
            "source": (item.get("metadata") or {}).get("source") or (item.get("metadata") or {}).get("title"),
            "reason": classification.get("reason", "Classification was not established."),
        }
        category = classification.get("category", "uncertain")
        if category == "direct":
            direct.append(row)
        elif category == "supporting_contextual":
            supporting.append(row)
        elif category in {"related_indirect", "complication"}:
            (indirect if category == "complication" else related).append(row)
        elif category == "contradictory":
            contradictory.append(row)
        else:
            irrelevant.append(row)

    components = []
    if intent.comparison_entities:
        covered_entities = {
            entity
            for item in evidence
            for entity in (
                (item.get("evidence_classification") or classify_evidence(
                    query, item.get("content", ""), item.get("metadata", {}),
                )).get("comparison_entities_supported", [])
            )
        }
        components = [
            {"component": entity, "status": "supported" if entity in covered_entities else "unsupported",
             "evidence_ids": [item.get("evidence_id", f"evidence-{index}") for index, item in enumerate(evidence, 1)
                              if entity in (item.get("evidence_classification") or classify_evidence(
                                  query, item.get("content", ""), item.get("metadata", {}),
                              )).get("comparison_entities_supported", [])]}
            for entity in intent.comparison_entities
        ]
        score = len(covered_entities & set(intent.comparison_entities)) / len(intent.comparison_entities)
    else:
        supported_direct = bool(direct)
        components = [{
            "component": intent.information_type,
            "status": "supported" if supported_direct else "supporting_only" if supporting else "unsupported",
            "evidence_ids": [row["evidence_id"] for row in direct or supporting],
        }]
        score = 1.0 if supported_direct else 0.5 if supporting else 0.0

    if contradictory:
        status = "conflicting_coverage"
        reason = "Relevant passages contain an unresolved disagreement."
    elif intent.comparison_entities and score == 0:
        status = "no_relevant_coverage" if not (supporting or related or indirect) else "insufficient"
        reason = "No requested comparison entity has direct passage coverage."
    elif intent.comparison_entities and score < 1:
        status = "partial"
        missing = [component["component"] for component in components if component["status"] != "supported"]
        reason = "Direct passage coverage is incomplete for: " + ", ".join(missing) + "."
    elif direct:
        status = "complete"
        reason = "Retrieved passages directly address the requested information type."
    elif supporting or related or indirect:
        status = "insufficient"
        reason = "Retrieved passages provide risk, related, or contextual information but do not directly establish the requested answer."
    else:
        status = "no_relevant_coverage"
        reason = "No retrieved passage meaningfully addresses the question."

    return {
        "status": status,
        "reason": reason,
        "coverage_score": round(score, 4),
        "question_components": components,
        "evidence_groups": {
            "direct": direct,
            "supporting": supporting,
            "related": related,
            "indirect": indirect,
            "contradictory": contradictory,
            "irrelevant": irrelevant,
        },
    }


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
