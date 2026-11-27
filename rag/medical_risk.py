"""Lightweight query risk tiers used to choose conservative answer states."""

import re
from dataclasses import dataclass

from rag.query_classifier import QueryClassifier, QueryType


@dataclass(frozen=True)
class MedicalRisk:
    tier: str
    label: str
    category: str = "general_biomedical_education"


_EMERGENCY_PATTERNS = (
    r"\b(?:am i having|could i be having|i think (?:i'm|i am) having)\b.{0,50}\b(?:heart attack|stroke|anaphylaxis|overdose)\b",
    r"\b(?:can't|cannot|unable to) breathe\b",
    r"\bnot breathing\b",
    r"\bunconscious\b",
    r"\bsevere chest pain (?:right now|now|currently)\b",
)
_URGENT_PATTERNS = (
    r"\b(?:urgent care|emergency room|emergency department)\b",
    r"\b(?:heavy|uncontrolled) bleeding\b",
    r"\b(?:rapidly worsening|sudden severe) symptoms\b",
    r"\bcan i wait until tomorrow\b",
)
_HEALTH_INFO_PATTERNS = (
    r"\b(?:symptom|side effect|prevention|complication|risk factor|screening)\w*\b",
)

_MEDICAL_TOPIC_PATTERN = re.compile(
    r"\b(?:medical|medicine|health|healthcare|clinical|patient|diagnos\w*|"
    r"treat\w*|therap\w*|medication|drug|dose|dosage|symptom\w*|disease\w*|"
    r"disorder\w*|condition\w*|syndrome\w*|infection\w*|virus\w*|bacteri\w*|"
    r"symptom\w*|side effect\w*|cancer|tumou?r|diabetes|parkinson\w*|ebola|alzheimer\w*|"
    r"hypertension|anemia|anaemia|obesity|stroke|seizure\w*|migraine\w*|arthritis|"
    r"eczema|allerg\w*|depression|anxiety|covid|influenza|flu|ulcer\w*|"
    r"gastric|peptic|stomach|digestive|intestinal|intestin\w*|colon|"
    r"heart|cardiac|blood|brain|lung|liver|kidney|skin|bone|muscle|nerve|"
    r"cell\w*|gene\w*|genetic\w*|protein\w*|hormone\w*|insulin|dopamine|"
    r"neuron\w*|synap\w*|neurotransmit\w*|serotonin|neuro\w*|"
    r"anatom\w*|physiolog\w*|biochem\w*|molecular|molecule\w*|enzyme\w*|"
    r"metabol\w*|receptor\w*|signal(?:ing|ling)?|circulat\w*|respirat\w*|"
    r"immune\w*|endocrine|skeletal|cardiovascular|nervous system|"
    r"vaccine\w*|immun\w*|patholog\w*|physiolog\w*|anatom\w*|pregnan\w*|"
    r"clinical trial\w*|biomedical|biolog\w*|pathogen\w*|"
    r"[a-z]+(?:itis|osis|emia|pathy|oma)\b)",
    re.IGNORECASE,
)
_MEDICAL_FOLLOWUP_PATTERN = re.compile(
    r"^\s*(?:and\s+)?(?:it|its|they|them|their|this|that|these|those|"
    r"what about|how about|how so|tell me more|explain more|explain further|"
    r"elaborate|go on|continue)\b",
    re.IGNORECASE,
)
_EXPLAIN_FURTHER = re.compile(
    r"^\s*(?:please\s+)?(?:explain|elaborate)(?:\s+on\s+(?:that|it))?\s+(?:further|more)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def reformulate_contextual_query(query: str, chat_history=None) -> str:
    """Resolve short follow-ups against the latest substantive user question."""
    current = " ".join(str(query or "").split())
    if not current or not chat_history:
        return current
    if not _MEDICAL_FOLLOWUP_PATTERN.search(current):
        return current

    previous_query = ""
    for turn in reversed(chat_history):
        if isinstance(turn, dict):
            candidate = " ".join(str(turn.get("query", "")).split()).strip()
            if candidate and is_medical_query(candidate):
                previous_query = candidate.rstrip(".!? ")
                break
    if not previous_query:
        return current

    if _EXPLAIN_FURTHER.fullmatch(current):
        return f"{previous_query} further."
    return f"{previous_query}. Follow-up: {current}"


def is_medical_query(query: str) -> bool:
    """Conservatively keep this assistant focused on medical/biomedical topics."""
    return bool(_MEDICAL_TOPIC_PATTERN.search(query or ""))


def is_medical_followup(query: str, chat_history=None) -> bool:
    """Allow a short contextual follow-up only after a medical question."""
    if not _MEDICAL_FOLLOWUP_PATTERN.search(query or ""):
        return False
    for turn in reversed(chat_history or []):
        if isinstance(turn, dict) and is_medical_query(str(turn.get("query", ""))):
            return True
    return False


def classify_medical_risk(query: str) -> MedicalRisk:
    """Classify broad interaction risk; this is a routing aid, not triage."""
    text = (query or "").lower()
    if any(re.search(pattern, text) for pattern in _EMERGENCY_PATTERNS):
        return MedicalRisk("R4", "emergency", "emergency_symptoms")
    if any(re.search(pattern, text) for pattern in _URGENT_PATTERNS):
        return MedicalRisk("R3", "urgent", "urgent_health_question")

    query_type, _ = QueryClassifier.classify(text)
    if query_type == QueryType.DIAGNOSTIC:
        category = "diagnostic_interpretation"
    elif query_type == QueryType.TREATMENT_ADVICE:
        category = "treatment_or_medication"
    elif re.search(r"\b(?:cause|causes|risk factor|etiolog)\w*\b", text):
        category = "causes_and_risk_factors"
    elif re.search(r"\b(?:symptom|signs?|complications?)\w*\b", text):
        category = "symptoms_and_characteristics"
    elif re.search(r"\b(?:prevent|prevention|screening)\w*\b", text):
        category = "prevention_and_general_health"
    elif query_type == QueryType.MECHANISM:
        category = "biological_mechanism"
    elif re.search(r"\b(?:what is|define|definition of)\b", text):
        category = "disease_or_biomedical_definition"
    else:
        category = "general_biomedical_education"
    if query_type in {QueryType.DIAGNOSTIC, QueryType.TREATMENT_ADVICE, QueryType.ADVERSARIAL}:
        return MedicalRisk("R2", "treatment_or_diagnosis", category)
    if any(re.search(pattern, text) for pattern in _HEALTH_INFO_PATTERNS):
        return MedicalRisk("R1", "health_information", category)
    return MedicalRisk("R0", "educational", category)
