"""Query classification and routing for specialized handling"""

import re
from enum import Enum
from typing import Dict, Tuple


class QueryType(Enum):
    """Types of biomedical queries"""

    FACTOID = "factoid"  # Simple facts (symptoms, definitions)
    MECHANISM = "mechanism"  # How/why questions (mechanisms, causes)
    MULTIHOP = "multihop"  # Multi-step reasoning (relationships, complications)
    AMBIGUOUS = "ambiguous"  # Requests needing clarification
    ADVERSARIAL = "adversarial"  # Misleading/harmful premises
    DIAGNOSTIC = "diagnostic"  # Self-diagnosis requests
    TREATMENT_ADVICE = "treatment_advice"  # Treatment requests


class QueryClassifier:
    """Classify biomedical queries for appropriate handling"""

    # Patterns that indicate adversarial/unsafe queries
    ADVERSARIAL_PATTERNS = [
        r"\bcure\b",
        r"\bpermanently reverse\b",
        r"\bFDA-approved cure\b",
        r"\bnatural[ly]* cure\b",
        r"\bcured completely\b",
        r"\bguaranteed treatment\b",
        r"\bproven to work\b",
        r"\bwork[s]? every time\b",
    ]

    # Patterns indicating diagnostic/self-diagnosis
    DIAGNOSTIC_PATTERNS = [
        r"(?:do\s+)?i\s+(?:have|have|suffer|got)\b",
        r"(?:what\s+)?(?:is\s+)?(?:wrong\s+)?with\s+me\b",
        r"why\s+am\s+i\s+(?:always\s+)?(?:tired|sick|dizzy)",
        r"could\s+i\s+have\b",
        r"am\s+i\s+(?:at\s+risk|susceptible|vulnerable)\b",
        r"do\s+(?:i|my)\s+(?:have\s+)?symptoms\b",
    ]

    # Patterns indicating treatment/medical advice requests
    TREATMENT_PATTERNS = [
        r"how\s+(?:do\s+)?i\s+(?:treat|cure|fix|manage|control)",
        r"(?:what\s+)?(?:is\s+)?(?:the\s+)?best\s+(?:treatment|cure|medicine|drug)",
        r"should\s+i\s+(?:take|use|try|do)\b",
        r"can\s+(?:i|this|that)\s+(?:cure|treat|fix)\b",
        r"(?:prescription|medication|medicine|drug)\s+(?:for|to)",
        r"(?:what\s+)?(?:medicine|pill|drug|medication)\s+should\b",
    ]

    # Patterns indicating ambiguous/unclear questions
    AMBIGUOUS_PATTERNS = [
        r"best\s+(?:treatment|cure|medicine|option)",
        r"(?:when|why)\s+(?:am|is)\s+i\b",
        r"how\s+do\s+i\s+(?:know|tell)\b",
        r"can\s+(?:.*\s+)?be\s+cured\b",
        r"(?:what\s+)?(?:is\s+)?the\s+(?:solution|answer|cure)\b",
    ]

    # Patterns indicating multi-hop reasoning
    MULTIHOP_PATTERNS = [
        r"(?:how|what)\s+(?:is\s+)?(?:the\s+)?(?:relationship|connection|link|relation)",
        r"(?:do|can)\s+(?:.*\s+)?(?:cause|lead to|result in|contribute to)\b",
        r"(?:what|which)\s+(?:complications|consequences|effects|risks)",
        r"(?:is|are)\s+(?:.*\s+)?(?:related|connected|linked)\b",
        r"(?:how)\s+(?:does|can)\s+(?:.*\s+)?(?:affect|impact|influence)\b",
        r"(?:what\s+)?(?:happens|occurs|results)\s+(?:when|if)",
    ]

    # Patterns indicating mechanism/how questions
    MECHANISM_PATTERNS = [
        r"how\s+(?:does|do|can)\s+(?:.*\s+)?(?:work|function|operate|act)",
        r"(?:what|how)\s+(?:is\s+)?(?:the\s+)?(?:mechanism|process|pathway)\b",
        r"(?:why|how)\s+(?:does|can)\s+(?:.*\s+)?(?:occur|happen|develop|start)",
        r"(?:explain|describe)\s+(?:.*\s+)?(?:process|mechanism|how)\b",
    ]

    @classmethod
    def classify(cls, query: str) -> Tuple[QueryType, Dict]:
        """
        Classify a biomedical query.

        Args:
            query: User query

        Returns:
            (query_type, metadata)
            metadata includes: strict_mode (bool), requires_refusal (bool), reason (str)
        """
        query_lower = query.lower()

        # Check adversarial patterns (highest priority)
        if cls._matches_patterns(query_lower, cls.ADVERSARIAL_PATTERNS):
            return QueryType.ADVERSARIAL, {
                "strict_mode": True,
                "requires_refusal": True,
                "reason": "adversarial_premise",
            }

        # Check diagnostic patterns
        if cls._matches_patterns(query_lower, cls.DIAGNOSTIC_PATTERNS):
            return QueryType.DIAGNOSTIC, {
                "strict_mode": True,
                "requires_refusal": True,
                "reason": "self_diagnosis_request",
            }

        # Check treatment advice patterns
        if cls._matches_patterns(query_lower, cls.TREATMENT_PATTERNS):
            return QueryType.TREATMENT_ADVICE, {
                "strict_mode": True,
                "requires_refusal": True,
                "reason": "medical_advice_request",
            }

        # Check multi-hop patterns (before ambiguous)
        if cls._matches_patterns(query_lower, cls.MULTIHOP_PATTERNS):
            return QueryType.MULTIHOP, {
                "strict_mode": False,
                "requires_refusal": False,
                "reason": "multihop_reasoning",
            }

        # Check mechanism patterns
        if cls._matches_patterns(query_lower, cls.MECHANISM_PATTERNS):
            return QueryType.MECHANISM, {
                "strict_mode": False,
                "requires_refusal": False,
                "reason": "mechanism_question",
            }

        # Check ambiguous patterns
        if cls._matches_patterns(query_lower, cls.AMBIGUOUS_PATTERNS):
            return QueryType.AMBIGUOUS, {
                "strict_mode": True,
                "requires_refusal": False,
                "reason": "ambiguous_request",
            }

        # Default to factoid
        return QueryType.FACTOID, {
            "strict_mode": False,
            "requires_refusal": False,
            "reason": "simple_factoid",
        }

    @staticmethod
    def _matches_patterns(text: str, patterns: list) -> bool:
        """Check if text matches any pattern in the list"""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
