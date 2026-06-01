"""Refusal templates and strategies for unsafe/low-confidence queries"""

from typing import Optional
from enum import Enum


class RefusalReason(Enum):
    """Reasons for refusing to generate an answer"""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ADVERSARIAL_QUERY = "adversarial_query"
    MEDICAL_ADVICE = "medical_advice"
    DIAGNOSTIC_REQUEST = "diagnostic_request"
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS_REQUEST = "ambiguous_request"


class RefusalManager:
    """Manage consistent, deterministic refusal responses"""

    TEMPLATES = {
        RefusalReason.INSUFFICIENT_EVIDENCE: (
            "I don't have enough verified medical evidence in the retrieved context "
            "to answer this accurately. Please consult a healthcare professional "
            "or refer to authoritative medical sources."
        ),
        RefusalReason.ADVERSARIAL_QUERY: (
            "The retrieved medical evidence does not support the premise of this question. "
            "I cannot confirm the assumption made. For medical concerns, please consult "
            "a qualified healthcare professional."
        ),
        RefusalReason.MEDICAL_ADVICE: (
            "I cannot provide personalized medical advice or treatment recommendations. "
            "This requires professional clinical judgment. Please consult a qualified "
            "healthcare provider for diagnosis or treatment guidance."
        ),
        RefusalReason.DIAGNOSTIC_REQUEST: (
            "I cannot diagnose medical conditions or assess symptoms. "
            "If you're experiencing concerning symptoms, please see a healthcare professional "
            "who can properly examine and evaluate your condition."
        ),
        RefusalReason.LOW_CONFIDENCE: (
            "The retrieved evidence has low confidence and may not reliably support an answer. "
            "Please consult authoritative medical sources or a healthcare professional."
        ),
        RefusalReason.AMBIGUOUS_REQUEST: (
            "This question is too broad or ambiguous for a reliable answer. "
            "Could you provide more specific details? "
            "For personalized guidance, consult a healthcare professional."
        ),
    }

    @staticmethod
    def get_refusal(reason: RefusalReason, custom_message: Optional[str] = None) -> str:
        """
        Get a deterministic refusal response.

        Args:
            reason: RefusalReason enum value
            custom_message: Optional custom message to append

        Returns:
            Refusal response string
        """
        template = RefusalManager.TEMPLATES.get(
            reason,
            "I cannot provide a reliable answer based on the available information. "
            "Please consult a healthcare professional.",
        )

        if custom_message:
            template = f"{template}\n\n{custom_message}"

        return template

    @staticmethod
    def create_confidence_refusal(
        top_confidence: float,
        avg_confidence: float,
        retrieval_quality: str,
    ) -> str:
        """
        Create a refusal message based on confidence metrics.

        Args:
            top_confidence: Best retrieval confidence score
            avg_confidence: Average top-3 confidence
            retrieval_quality: Quality rating (strong/moderate/weak)

        Returns:
            Refusal message
        """
        if retrieval_quality == "weak":
            return RefusalManager.get_refusal(
                RefusalReason.LOW_CONFIDENCE,
                f"(Retrieval confidence: {top_confidence:.1%})",
            )
        elif retrieval_quality == "moderate" and avg_confidence < 0.35:
            return RefusalManager.get_refusal(
                RefusalReason.INSUFFICIENT_EVIDENCE,
                f"(Evidence coverage insufficient: {avg_confidence:.1%})",
            )
        else:
            return RefusalManager.get_refusal(RefusalReason.INSUFFICIENT_EVIDENCE)

    @staticmethod
    def create_query_type_refusal(query_type_metadata: dict) -> str:
        """
        Create a refusal based on query type.

        Args:
            query_type_metadata: Metadata from QueryClassifier.classify()

        Returns:
            Refusal message
        """
        reason_str = query_type_metadata.get("reason", "unknown")

        if reason_str == "adversarial_premise":
            return RefusalManager.get_refusal(RefusalReason.ADVERSARIAL_QUERY)
        elif reason_str == "medical_advice_request":
            return RefusalManager.get_refusal(RefusalReason.MEDICAL_ADVICE)
        elif reason_str == "self_diagnosis_request":
            return RefusalManager.get_refusal(RefusalReason.DIAGNOSTIC_REQUEST)
        elif reason_str == "ambiguous_request":
            return RefusalManager.get_refusal(RefusalReason.AMBIGUOUS_REQUEST)
        else:
            return RefusalManager.get_refusal(RefusalReason.INSUFFICIENT_EVIDENCE)
