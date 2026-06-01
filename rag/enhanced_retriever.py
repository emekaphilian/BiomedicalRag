"""Enhanced hybrid retriever with confidence metrics and grounding awareness"""

from typing import Dict, List
from rag.bm25_retriever import BM25Retriever
from rag.fusion import reciprocal_rank_fusion
from rag.retriever import BiomedicalRetriever
from rag.confidence import ConfidenceScorer


class EnhancedHybridRetriever:
    """Hybrid retriever with confidence thresholding and safety features"""

    def __init__(self, vectorstore, documents):
        self.dense_retriever = BiomedicalRetriever(vectorstore)
        self.sparse_retriever = BM25Retriever(documents)
        self.confidence_scorer = ConfidenceScorer()

    def retrieve(self, query: str, k: int = 5, return_scores: bool = True) -> Dict:
        """
        Retrieve grounded biomedical evidence with confidence metrics.

        Args:
            query: Biomedical query
            k: Number of top results
            return_scores: Include raw score details

        Returns:
            Dict with:
            - fused_results: List[(content, fused_score)]
            - confidence_metrics: Confidence scoring info
            - dense_results: Dense retrieval results
            - sparse_results: Sparse retrieval results
            - grounded: Boolean grounding check
        """
        # Dense retrieval (FAISS)
        dense_response = self.dense_retriever.retrieve(query, k=k)
        dense_results = dense_response["results"]

        # Extract L2 distances from dense results for confidence analysis
        dense_scores = [r.similarity_score for r in dense_results]

        # Sparse retrieval (BM25)
        sparse_results = self.sparse_retriever.search(query, k=k)
        sparse_scores = [r.get("score", 0.0) for r in sparse_results]

        # Fuse results
        fused_results = reciprocal_rank_fusion(dense_results, sparse_results)

        # Calculate confidence metrics
        confidence_metrics = self.confidence_scorer.calculate_retrieval_metrics(
            dense_scores=dense_scores,
            sparse_scores=sparse_scores,
            fused_scores=fused_results,
        )

        return {
            "grounded": dense_response["grounded"],
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "fused_results": fused_results,
            "confidence_metrics": confidence_metrics,
            "dense_scores": dense_scores if return_scores else None,
            "sparse_scores": sparse_scores if return_scores else None,
        }

    def retrieve_with_threshold(
        self,
        query: str,
        query_type: str = "factoid",
        strict_mode: bool = False,
        k: int = 5,
    ) -> Dict:
        """
        Retrieve with confidence thresholding and query-type awareness.

        Args:
            query: Biomedical query
            query_type: Type from QueryClassifier (factoid, mechanism, multihop, etc.)
            strict_mode: Enable strict thresholds
            k: Number of top results

        Returns:
            Dict with:
            - should_generate: Boolean (proceed with generation?)
            - refusal_reason: String (reason if should_generate is False)
            - fused_results: Retrieved evidence
            - confidence_metrics: Scoring details
            ... (all fields from retrieve())
        """
        retrieval_result = self.retrieve(query, k=k, return_scores=True)

        confidence_metrics = retrieval_result["confidence_metrics"]

        # Determine if generation should proceed
        should_generate, reason = ConfidenceScorer.should_generate(
            confidence_metrics,
            query_type=query_type,
            strict_mode=strict_mode,
        )

        return {
            **retrieval_result,
            "should_generate": should_generate,
            "refusal_reason": reason,
        }
