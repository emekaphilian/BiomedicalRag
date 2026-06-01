"""
Retrieval evaluation metrics.
Measures quality of document retrieval (NDCG, MRR, Recall@K, etc.)
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """Computes retrieval quality metrics."""
    
    @staticmethod
    def ndcg_at_k(relevant_ids: List[int], retrieved_ids: List[int], k: int = 5) -> float:
        """Normalized Discounted Cumulative Gain."""
        raise NotImplementedError
    
    @staticmethod
    def mrr(relevant_ids: List[int], retrieved_ids: List[int]) -> float:
        """Mean Reciprocal Rank."""
        raise NotImplementedError
    
    @staticmethod
    def recall_at_k(relevant_ids: List[int], retrieved_ids: List[int], k: int = 5) -> float:
        """Recall@K metric."""
        raise NotImplementedError
