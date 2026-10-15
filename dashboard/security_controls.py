"""Heuristic PII and prompt-attack screens; these do not guarantee protection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PIIMatch:
    kind: str
    start: int
    end: int


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_DOB = re.compile(r"\b(?:dob|date of birth|born(?: on)?)\s*[:\-]?\s*\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", re.I)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_IP = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{8,17}\d(?![\w.])")


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2:
            digit = digit * 2 - 9 if digit * 2 > 9 else digit * 2
        total += digit
    return total % 10 == 0


def find_pii(text: str) -> list:
    matches = []
    def add(kind, pattern, accept=lambda value: True):
        for found in pattern.finditer(text or ""):
            if accept(found.group()) and all(found.end() <= item.start or found.start() >= item.end for item in matches):
                matches.append(PIIMatch(kind, found.start(), found.end()))
    add("email", _EMAIL)
    add("ssn", _SSN)
    add("date_of_birth", _DOB)
    add("credit_card", _CARD, lambda value: 13 <= len(_digits(value)) <= 19 and _luhn_ok(_digits(value)))
    add("ip_address", _IP, lambda value: all(int(part) <= 255 for part in value.split(".")))
    add("phone", _PHONE, lambda value: 10 <= len(_digits(value)) <= 15)
    return sorted(matches, key=lambda item: item.start)


def redact_pii(text: str) -> tuple:
    matches = find_pii(text)
    redacted = text
    for match in reversed(matches):
        redacted = redacted[:match.start] + f"[REDACTED:{match.kind.upper()}]" + redacted[match.end:]
    return redacted, matches


_ATTACKS = (
    ("instruction_override", 2, re.compile(r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|any|your)\b[^.\n]{0,30}\b(?:instructions?|prompts?|rules?|guidelines?)\b", re.I)),
    ("system_prompt_exfiltration", 2, re.compile(r"\b(?:reveal|show|print|repeat|output|leak|display)\b[^.\n]{0,40}\b(?:system prompt|hidden (?:prompt|instructions?)|your (?:instructions?|prompt|rules))\b", re.I)),
    ("role_hijack", 1, re.compile(r"\b(?:you are now|act as|pretend (?:to be|you are)|developer mode|jailbreak|do anything now)\b", re.I)),
    ("delimiter_spoofing", 2, re.compile(r"(?:</?\s*(?:system|assistant)\s*>|\[/?INST\]|<\|(?:im_start|im_end|system)\|>|^\s*(?:system|assistant)\s*:)", re.I | re.M)),
    ("data_exfiltration", 2, re.compile(r"\b(?:send|post|upload|email|exfiltrate|forward)\b[^.\n]{0,60}(?:https?://|[\w.+-]+@[\w-]+\.\w+)", re.I)),
    ("safety_bypass", 1, re.compile(r"\b(?:without|bypass|disable|turn off)\b[^.\n]{0,20}\b(?:safety|filters?|restrictions?|guardrails?|disclaimers?)\b", re.I)),
)


@dataclass(frozen=True)
class AttackReport:
    score: int
    patterns: list = field(default_factory=list)
    blocked: bool = False


def detect_prompt_attack(text: str, block_threshold: int = 2) -> AttackReport:
    hits = [(name, weight) for name, weight, pattern in _ATTACKS if pattern.search(text or "")]
    score = sum(weight for _, weight in hits)
    return AttackReport(score, [name for name, _ in hits], score >= block_threshold)


def scan_sources(texts: Iterable[str]) -> list:
    return [index for index, text in enumerate(texts) if detect_prompt_attack(text).blocked]


@dataclass(frozen=True)
class GuardResult:
    sanitized: str
    pii: list
    attack: AttackReport
    blocked: bool


def guard_question(question: str, redact_pii_enabled: bool = True, block_attacks: bool = True) -> GuardResult:
    pii = find_pii(question)
    sanitized = redact_pii(question)[0] if redact_pii_enabled else question
    attack = detect_prompt_attack(question)
    return GuardResult(sanitized, pii, attack, block_attacks and attack.blocked)


CLEARANCE_LEVELS = ("public", "internal", "confidential", "restricted")
ROLE_CLEARANCE = {"guest": "public", "researcher": "internal", "clinical-reviewer": "confidential", "admin": "restricted"}
UNLABELED_SOURCE_LEVEL = "restricted"


def clearance_rank(level: str) -> int:
    return CLEARANCE_LEVELS.index(level) if level in CLEARANCE_LEVELS else len(CLEARANCE_LEVELS)


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    role: str
    role_clearance: str
    denied_sources: list
    reason: str


def authorize_sources(role: str, sources: Sequence[str], labels: Mapping[str, str]) -> AccessDecision:
    clearance = ROLE_CLEARANCE.get(role, "public")
    denied = sorted({source for source in sources if clearance_rank(labels.get(source, UNLABELED_SOURCE_LEVEL)) > clearance_rank(clearance)})
    if denied:
        return AccessDecision(False, role, clearance, denied,
                              f"Access denied: role '{role}' (clearance '{clearance}') may not see {', '.join(denied)}. The answer was withheld.")
    return AccessDecision(True, role, clearance, [], "Access granted.")
