"""
Generation evaluation metrics.
Measures quality of generated answers (BLEU, ROUGE, BERTScore, etc.)
"""

import logging

logger = logging.getLogger(__name__)


class GenerationMetrics:
    """Computes answer generation quality metrics."""
    
    @staticmethod
    def bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
        """BLEU score for answer similarity."""
        raise NotImplementedError
    
    @staticmethod
    def rouge_l(reference: str, hypothesis: str):
        """ROUGE-L score for answer similarity."""
        raise NotImplementedError
    
    @staticmethod
    def bert_score(references: list, hypothesis: str, model_name: str = "bert-base-uncased"):
        """BERTScore for semantic similarity."""
        raise NotImplementedError
