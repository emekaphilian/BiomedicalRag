"""Retrieval confidence scoring and thresholding for grounded generation"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class ConfidenceScorer:
    """Calculate retrieval confidence metrics from similarity scores"""

    @staticmethod
    def normalize_l2_distance(distance: float, max_distance: float = 2.0) -> float:
        """
        Normalize L2 distance to confidence score (0-1 range).
        L2 distance ranges 0-2 for normalized embeddings.
        Higher distance = lower confidence.
        """
        # Clamp to [0, max_distance]
        distance = min(max(distance, 0.0), max_distance)
        # Invert: lower distance -> higher confidence
        return 1.0 - (distance / max_distance)

    @staticmethod
    def calculate_retrieval_metrics(
        dense_scores: List[float],
        sparse_scores: List[float],
        fused_scores: List[Tuple[str, float]],
    ) -> Dict:
        """
        Calculate comprehensive retrieval confidence metrics.

        Args:
            dense_scores: L2 distances from FAISS
            sparse_scores: BM25 scores
            fused_scores: RRF fused results [(content, score)]

        Returns:
            Dict with metrics:
            - top_confidence: Best retrieval confidence (0-1)
            - avg_top3_confidence: Average of top 3 (0-1)
            - confidence_spread: Variance in top-3 scores
            - has_strong_evidence: Boolean (top >= threshold)
            - has_weak_evidence: Boolean (avg top-3 < 0.4)
        """
        if not fused_scores:
            return {
                "top_confidence": 0.0,
                "avg_top3_confidence": 0.0,
                "confidence_spread": 0.0,
                "has_strong_evidence": False,
                "has_weak_evidence": True,
                "retrieval_quality": "no_results",
            }

        # Normalize fused scores (they're RRF-fused, 0-1 range already)
        top_fused = [score for _, score in fused_scores[:3]]

        if not top_fused:
            return {
                "top_confidence": 0.0,
                "avg_top3_confidence": 0.0,
                "confidence_spread": 0.0,
                "has_strong_evidence": False,
                "has_weak_evidence": True,
                "retrieval_quality": "no_results",
            }

        top_score = top_fused[0]
        avg_top3 = np.mean(top_fused) if len(top_fused) >= 1 else 0.0
        spread = np.std(top_fused) if len(top_fused) > 1 else 0.0

        # Determine quality
        has_strong = top_score >= 0.5  # Strong top evidence
        has_weak = avg_top3 < 0.4  # Weak average evidence

        if top_score >= 0.6:
            quality = "strong"
        elif top_score >= 0.4:
            quality = "moderate"
        else:
            quality = "weak"

        return {
            "top_confidence": float(top_score),
            "avg_top3_confidence": float(avg_top3),
            "confidence_spread": float(spread),
            "has_strong_evidence": bool(has_strong),
            "has_weak_evidence": bool(has_weak),
            "retrieval_quality": quality,
        }

    @staticmethod
    def should_generate(
        confidence_metrics: Dict,
        query_type: str = "factoid",
        strict_mode: bool = False,
    ) -> Tuple[bool, str]:
        """
        Determine if answer generation should proceed based on retrieval confidence.

        Args:
            confidence_metrics: Output from calculate_retrieval_metrics()
            query_type: Type of query (factoid, mechanism, multihop, adversarial)
            strict_mode: Enable strict thresholds for medical/adversarial queries

        Returns:
            (should_generate: bool, reason: str)
        """
        top_conf = confidence_metrics["top_confidence"]
        avg_conf = confidence_metrics["avg_top3_confidence"]
        quality = confidence_metrics["retrieval_quality"]

        # Strict mode thresholds (adversarial/medical advice)
        if strict_mode:
            if top_conf < 0.6 or avg_conf < 0.5:
                return False, "insufficient_evidence_strict"
            if quality == "weak":
                return False, "weak_retrieval_strict"

        # Factoid questions (least strict)
        if query_type == "factoid":
            if top_conf < 0.3:
                return False, "no_relevant_evidence"
            return True, "acceptable"

        # Mechanism questions (medium strict)
        if query_type == "mechanism":
            if top_conf < 0.4 or avg_conf < 0.3:
                return False, "insufficient_mechanism_evidence"
            return True, "acceptable"

        # Multi-hop questions (stricter)
        if query_type == "multihop":
            if top_conf < 0.45 or avg_conf < 0.35:
                return False, "insufficient_multihop_evidence"
            return True, "acceptable"

        # Ambiguous questions (strictest)
        if query_type == "ambiguous":
            if top_conf < 0.5 or avg_conf < 0.4:
                return False, "insufficient_evidence_ambiguous"
            return True, "acceptable"

        # Default (conservative)
        if top_conf < 0.4 or avg_conf < 0.3:
            return False, "insufficient_evidence"
        return True, "acceptable"
