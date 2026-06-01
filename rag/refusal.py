"""
Query Refusal Module
Detects unsafe medical diagnostic/treatment requests and refuses them.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Patterns that indicate diagnostic or treatment intent
DIAGNOSTIC_PATTERNS = [
    r"do i have",
    r"can you diagnose",
    r"what disease do i have",
    r"should i take",
    r"what medication should i take",
    r"what drug should i take",
    r"prescribe",
    r"am i sick",
    r"is this cancer",
    r"diagnose me",
    r"what do i have",
    r"what condition do i have",
    r"treatment plan for me",
    r"what medicine for",
    r"which drug for",
    r"tell me what i have",
    r"what is wrong with me",
    r"what do i suffer from",
]

REFUSAL_MESSAGE = (
    "I cannot provide medical diagnoses, treatment recommendations, "
    "or medication prescriptions. "
    "Please consult with a qualified healthcare professional "
    "for personalized medical advice."
)


def should_refuse(query: str) -> bool:
    """
    Detects unsafe medical diagnostic/treatment intent.
    
    Args:
        query: The user's query
        
    Returns:
        True if query should be refused, False otherwise
    """
    query = query.lower().strip()

    for pattern in DIAGNOSTIC_PATTERNS:
        if re.search(pattern, query):
            logger.warning(f"Diagnostic pattern detected, refusing query: {query[:50]}...")
            return True

    return False


def get_refusal_message() -> str:
    """Get the standard refusal message."""
    return REFUSAL_MESSAGE
