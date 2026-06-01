"""
Medical Disclaimer Module
Enforces mandatory medical disclaimers on all responses.
"""

import logging

logger = logging.getLogger(__name__)

MEDICAL_DISCLAIMER = (
    "\n\n⚠️ MEDICAL DISCLAIMER: This information is for educational purposes only "
    "and is not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult with a qualified healthcare professional before making any medical decisions."
)


def append_disclaimer(response: str) -> str:
    """
    Ensures every response contains the mandatory medical disclaimer.
    
    Args:
        response: The original response text
        
    Returns:
        Response with appended disclaimer (if not already present)
    """
    response = response.strip()

    if MEDICAL_DISCLAIMER not in response:
        response += MEDICAL_DISCLAIMER

    return response


def get_disclaimer() -> str:
    """Get the standard medical disclaimer."""
    return MEDICAL_DISCLAIMER
