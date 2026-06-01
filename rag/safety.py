"""
Safety Orchestrator Module
Main gateway for enforcing biomedical safety constraints.

Coordinates:
- Query sanitization
- Diagnostic request refusal
- Mandatory disclaimer enforcement
"""

import sys
import os
import logging
from typing import Dict, Any
from dataclasses import dataclass

# Ensure imports work from any location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.refusal import should_refuse, get_refusal_message
from rag.sanitizer import sanitize_query
from rag.disclaimer import append_disclaimer

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result from safety check."""
    allowed: bool
    query: str
    response: str = ""
    reason: str = ""
    safety_level: str = "info"  # "blocked", "warning", "safe"


def process_safe_query(query: str) -> SafetyCheckResult:
    """
    Main biomedical safety gateway.
    
    Performs comprehensive safety checks:
    1. Query sanitization
    2. Diagnostic/treatment request detection
    3. Disclaimer enforcement
    
    Args:
        query: User's original query
        
    Returns:
        SafetyCheckResult with:
        - allowed: Whether query can proceed
        - query: Sanitized query
        - response: Safety message if blocked
        - reason: Why it was blocked (if applicable)
    """
    logger.info(f"Processing query: {query[:50]}...")

    # Step 1: Sanitize query
    sanitized_query = sanitize_query(query)
    logger.debug(f"Sanitized query: {sanitized_query[:50]}...")

    # Step 2: Check for diagnostic intent
    if should_refuse(sanitized_query):
        response = append_disclaimer(get_refusal_message())
        logger.warning(f"Diagnostic request refused: {sanitized_query[:50]}...")

        return SafetyCheckResult(
            allowed=False,
            query=sanitized_query,
            response=response,
            reason="Diagnostic or treatment request detected",
            safety_level="blocked",
        )

    # Query is safe to proceed
    logger.info(f"Query passed safety checks: {sanitized_query[:50]}...")

    return SafetyCheckResult(
        allowed=True,
        query=sanitized_query,
        response="",
        reason="",
        safety_level="safe",
    )


def get_safety_info() -> Dict[str, Any]:
    """Get information about safety constraints."""
    return {
        "module": "rag.safety",
        "diagnostic_patterns": "10+ patterns detected",
        "sanitization": "prompt injection blocking",
        "disclaimer": "mandatory on all responses",
        "status": "active",
    }
