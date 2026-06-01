"""
Query Sanitization Module
Cleans and sanitizes user queries to prevent prompt injection and enforce safety.
"""

import re
import logging

logger = logging.getLogger(__name__)


def sanitize_query(query: str) -> str:
    """
    Comprehensive biomedical query sanitization.
    
    Performs:
    - Whitespace normalization
    - Prompt injection detection and blocking
    - Dangerous phrase removal
    
    Args:
        query: Raw user query
        
    Returns:
        Sanitized query string
    """
    query = query.strip()

    # Normalize whitespace
    query = re.sub(r"\s+", " ", query)

    # Phrases that indicate prompt injection attempts
    blocked_phrases = [
        "ignore previous instructions",
        "ignore your instructions",
        "disregard safety",
        "act as a doctor",
        "pretend you are a physician",
        "pretend you are a medical",
        "bypass safety",
        "bypass the safety",
        "override safety",
        "ignore the medical",
    ]

    lowered = query.lower()
    original_length = len(lowered)

    for phrase in blocked_phrases:
        if phrase in lowered:
            logger.warning(f"Blocked phrase detected: '{phrase}' in query")
            lowered = lowered.replace(phrase, "")

    if len(lowered) < original_length:
        logger.info(f"Query sanitized: removed {original_length - len(lowered)} characters")

    return lowered.strip()


def is_sanitized(query: str) -> bool:
    """
    Check if a query is safe and sanitized.
    
    Args:
        query: Query to validate
        
    Returns:
        True if query passed sanitization checks
    """
    sanitized = sanitize_query(query)
    return len(sanitized) > 0 and len(sanitized) > len(query) * 0.8
