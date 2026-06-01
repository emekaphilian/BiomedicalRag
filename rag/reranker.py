from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Tuple


class CrossEncoderReranker:
    """
    Cross-encoder based reranking for medical retrieval.
    Dramatically improves ranking quality over fusion scores.
    
    Why it matters:
    - Fusion (RRF) uses mathematical rank combination
    - Cross-encoder understands semantic relevance deeply
    - Medical queries need semantic + keyword understanding
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder model.
        
        Models:
        - cross-encoder/ms-marco-MiniLM-L-6-v2 (general, fast, recommended start)
        - pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb (biomedical, better)
        - BAAI/bge-reranker-base (state-of-art, slower)
        """
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: List[Tuple],
        k: int = 5,
    ) -> List[Dict]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: User query string
            candidates: List of (content, metadata, fusion_score) tuples
            k: Number of top results to return
            
        Returns:
            List of reranked results with rerank scores
        """
        if not candidates:
            return []

        # Prepare query-document pairs for cross-encoder
        query_doc_pairs = [
            [query, candidate[0][:512]]  # Truncate to 512 chars for efficiency
            for candidate in candidates
        ]

        # Score all pairs using cross-encoder
        scores = self.model.predict(query_doc_pairs)

        # Create results with rerank scores
        reranked = []
        for idx, (content, metadata, fusion_score) in enumerate(candidates):
            reranked.append({
                "content": content,
                "metadata": metadata,
                "fusion_score": fusion_score,
                "rerank_score": float(scores[idx])
            })

        # Sort by rerank score (higher is better)
        reranked = sorted(
            reranked,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:k]

    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Rerank plain document list (simpler interface).
        
        Args:
            query: Search query
            documents: List of document texts
            k: Number of top results
            
        Returns:
            List of (document, score) tuples
        """
        if not documents:
            return []

        # Score query-document pairs
        pairs = [[query, doc[:512]] for doc in documents]
        scores = self.model.predict(pairs)

        # Sort and return
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]


class HybridReranker:
    """
    Combine multiple ranking signals:
    - Lexical similarity (BM25)
    - Dense similarity (embedding-based)
    - Cross-encoder relevance
    """

    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """Initialize hybrid reranker"""
        self.cross_encoder = CrossEncoderReranker(cross_encoder_model)

    def rerank_hybrid(
        self,
        query: str,
        documents: List[str],
        dense_scores: List[float] = None,
        sparse_scores: List[float] = None,
        top_k: int = 5,
        weights: Dict[str, float] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Rerank using combination of signals.
        
        Args:
            query: Search query
            documents: List of document texts
            dense_scores: Optional dense retrieval scores
            sparse_scores: Optional sparse (BM25) scores
            top_k: Number of top results to return
            weights: Signal weights (default: equal)
                - "dense": weight for dense scores
                - "sparse": weight for sparse scores
                - "cross": weight for cross-encoder
                
        Returns:
            List of (document, final_score, component_scores) tuples
        """
        if not documents:
            return []

        # Default weights: cross-encoder most important for semantic relevance
        if weights is None:
            weights = {"dense": 0.2, "sparse": 0.2, "cross": 0.6}

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Cross-encoder scores
        pairs = [[query, doc[:512]] for doc in documents]
        cross_scores_raw = self.cross_encoder.model.predict(pairs)
        cross_scores = self._normalize_scores(cross_scores_raw)

        # Normalize other scores if provided
        dense_norm = None
        if dense_scores is not None and len(dense_scores) > 0:
            dense_scores_arr = np.array(dense_scores)
            # Convert L2 distances to similarity (invert and normalize)
            dense_sim = 1.0 / (1.0 + dense_scores_arr)
            dense_norm = self._normalize_scores(dense_sim)

        sparse_norm = None
        if sparse_scores is not None and len(sparse_scores) > 0:
            sparse_scores_arr = np.array(sparse_scores)
            sparse_norm = self._normalize_scores(sparse_scores_arr)

        # Combine scores
        final_scores = np.zeros(len(documents))

        if weights.get("cross", 0) > 0:
            final_scores += weights["cross"] * cross_scores

        if dense_norm is not None and weights.get("dense", 0) > 0:
            final_scores += weights["dense"] * dense_norm

        if sparse_norm is not None and weights.get("sparse", 0) > 0:
            final_scores += weights["sparse"] * sparse_norm

        # Create result tuples with component scores
        results = []
        for i, (doc, final_score) in enumerate(zip(documents, final_scores)):
            components = {
                "cross_encoder": float(cross_scores[i]),
                "dense": float(dense_norm[i]) if dense_norm is not None else None,
                "sparse": float(sparse_norm[i]) if sparse_norm is not None else None,
            }
            results.append((doc, float(final_score), components))

        # Sort and return top-k
        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score)

