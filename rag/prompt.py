"""
Prompt engineering module for biomedical QA.
"""

import logging
from typing import List
from enum import Enum

logger = logging.getLogger(__name__)


class PromptTemplate(Enum):
    """Predefined prompt templates for biomedical Q&A."""
    GROUNDED_QA = "grounded_qa"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    SYNTHESIS = "synthesis"


class PromptManager:
    """Manages prompts with biomedical safety constraints."""
    
    def __init__(self):
        self.medical_disclaimer = (
            "DISCLAIMER: This information is for educational purposes only "
            "and should not be used for medical diagnosis or treatment."
        )
    
    def build_qa_prompt(self, question: str, context: List[str], template: PromptTemplate = PromptTemplate.GROUNDED_QA) -> str:
        """Build a QA prompt with grounding requirements."""
        raise NotImplementedError
