"""Cohere LLM Generator for Biomedical Grounding"""

import os
from typing import Optional, List, Dict

import cohere


class CohereGenerator:
    """Generate grounded biomedical answers using Cohere API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "command-nightly",
    ):
        """
        Initialize Cohere generator

        Args:
            api_key: Cohere API key
            model: Cohere model name
        """

        self.api_key = api_key or os.getenv("COHERE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "COHERE_API_KEY not provided. "
                "Set environment variable before running."
            )

        self.client = cohere.ClientV2(api_key=self.api_key)
        self.model = model

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: str = "",
    ) -> str:
        """
        Generate grounded biomedical response

        Args:
            query: User biomedical query
            context: Retrieved evidence
            system_prompt: Optional override system prompt

        Returns:
            Generated grounded answer
        """

        system = system_prompt or self._default_system_prompt()

        user_prompt = f"""
Query:
{query}

Retrieved Evidence:
{context}

Instructions:
1. Answer ONLY using the provided evidence
2. Cite the evidence when possible
3. If evidence is insufficient, explicitly say so
4. Do NOT fabricate biomedical information
5. Be concise, medically accurate, and grounded
"""

        try:
            response = self.client.chat(
                model=self.model,
                max_tokens=500,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            # Cohere v2 response extraction
            answer = response.message.content[0].text

            return answer.strip()

        except Exception as e:
            raise RuntimeError(f"Cohere API error: {str(e)}")

    def _default_system_prompt(self) -> str:
        """Default biomedical grounding system prompt"""

        return """
You are a biomedical information assistant specializing in evidence-based medicine.

Rules:
1. Use ONLY the retrieved evidence
2. Never fabricate facts
3. Clearly state uncertainty
4. Refuse unsupported medical claims
5. Prioritize factual correctness over completeness
6. Keep responses concise and medically grounded
"""


class CohereGeneratorStreaming:
    """Streaming Cohere biomedical generator"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "command-r-plus",
    ):
        """
        Initialize streaming generator
        """

        self.api_key = api_key or os.getenv("COHERE_API_KEY")

        if not self.api_key:
            raise ValueError("COHERE_API_KEY not set")

        self.client = cohere.ClientV2(api_key=self.api_key)
        self.model = model

    def generate_stream(
        self,
        query: str,
        context: List[str],
    ):
        """
        Stream grounded biomedical answer

        Args:
            query: User query
            context: Evidence chunks

        Yields:
            Streamed text tokens
        """

        context_text = "\n\n".join(
            [
                f"[Evidence {i + 1}]\n{chunk}"
                for i, chunk in enumerate(context)
            ]
        )

        system_prompt = """
You are a biomedical assistant.

Use ONLY provided evidence.
Do NOT fabricate information.
If evidence is insufficient, explicitly state that.
"""

        user_prompt = f"""
Query:
{query}

Evidence:
{context_text}

Answer ONLY from the provided evidence.
"""

        try:
            response = self.client.chat(
                model=self.model,
                max_tokens=500,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            answer = response.message.content[0].text

            # Simulated token streaming
            for token in answer.split():
                yield token + " "

        except Exception as e:
            raise RuntimeError(f"Cohere streaming error: {str(e)}")


class EnhancedCohereGenerator:
    """
    Enhanced Cohere generator with confidence thresholding,
    query classification, and explicit refusal logic.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "command-nightly",
    ):
        """Initialize enhanced Cohere generator"""
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "COHERE_API_KEY not provided. "
                "Set environment variable before running."
            )
        self.client = cohere.ClientV2(api_key=self.api_key)
        self.model = model

    def generate_grounded(
        self,
        query: str,
        context: str,
        query_type: str = "factoid",
        confidence_metrics: Optional[Dict] = None,
        should_generate: bool = True,
        refusal_reason: Optional[str] = None,
        system_prompt: str = "",
    ) -> str:
        """
        Generate grounded answer with confidence awareness and refusal logic.

        Args:
            query: User biomedical query
            context: Retrieved evidence
            query_type: Type from QueryClassifier
            confidence_metrics: Output from ConfidenceScorer
            should_generate: Whether retrieval confidence permits generation
            refusal_reason: Specific refusal reason code
            system_prompt: Optional override system prompt

        Returns:
            Generated answer or refusal
        """
        # If retrieval confidence is insufficient, refuse
        if not should_generate:
            return self._get_refusal_message(refusal_reason, confidence_metrics)

        # If query type requires strict medical disclaimer, include it
        strict_mode = query_type in ["diagnostic", "treatment_advice", "adversarial"]

        system = system_prompt or self._system_prompt_for_type(query_type, strict_mode)

        user_prompt = self._build_user_prompt(
            query,
            context,
            query_type,
            confidence_metrics,
            strict_mode,
        )

        try:
            response = self.client.chat(
                model=self.model,
                max_tokens=500,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer = response.message.content[0].text
            return answer.strip()

        except Exception as e:
            raise RuntimeError(f"Cohere API error: {str(e)}")

    def _build_user_prompt(
        self,
        query: str,
        context: str,
        query_type: str,
        confidence_metrics: Optional[Dict],
        strict_mode: bool,
    ) -> str:
        """Build user prompt with query-type aware instructions"""

        base_prompt = f"""
Query:
{query}

Retrieved Evidence:
{context}
"""

        if strict_mode:
            instructions = """
STRICT MODE - Medical Safety Required:
1. Answer ONLY using the provided evidence
2. Cite ALL evidence when making claims
3. If evidence is insufficient or ambiguous, REFUSE explicitly
4. Do NOT speculate about medical outcomes or treatments
5. Do NOT confirm or deny diagnoses
6. Recommend consulting healthcare professionals for medical decisions
"""
        else:
            instructions = """
Instructions:
1. Answer ONLY using the provided evidence
2. Cite the evidence when possible
3. If evidence is insufficient, explicitly say so
4. Do NOT fabricate biomedical information
5. Be concise, medically accurate, and grounded
"""

        if confidence_metrics:
            retrieval_quality = confidence_metrics.get("retrieval_quality", "unknown")
            top_conf = confidence_metrics.get("top_confidence", 0)
            instructions += f"\nRetrieval Quality: {retrieval_quality} (confidence: {top_conf:.1%})"

        return base_prompt + instructions

    def _system_prompt_for_type(self, query_type: str, strict_mode: bool) -> str:
        """Generate system prompt tailored to query type"""

        base_system = """
You are a biomedical information assistant specializing in evidence-based medicine.

Core Rules:
1. Use ONLY the retrieved evidence provided
2. Never fabricate or hallucinate facts
3. Clearly state when evidence is insufficient
4. Refuse unsupported medical claims
5. Prioritize factual correctness over completeness
"""

        if query_type == "factoid":
            return base_system + """
Factoid Question Focus:
- Provide specific, well-cited answers
- Keep responses concise
- Use evidence directly
"""

        elif query_type in ["mechanism", "multihop"]:
            return base_system + """
Complex Reasoning Focus:
- Synthesize information across evidence chunks
- Explain causal relationships using evidence
- If evidence gaps exist, state them explicitly
- Ensure each claim is traceable to evidence
"""

        elif query_type == "ambiguous":
            return base_system + """
Ambiguous Question Focus:
- Identify ambiguities in the question
- Ask for clarification or interpret charitably
- Provide evidence-based answers to each interpretation
- Avoid speculation
"""

        elif strict_mode:
            return base_system + """
MEDICAL SAFETY CRITICAL:
- This query involves medical advice or diagnosis
- REFUSE any claims not fully supported by strong evidence
- Include explicit disclaimers about consulting healthcare professionals
- Prioritize safety over being helpful
- When in doubt, refuse
"""

        else:
            return base_system

    def _get_refusal_message(
        self,
        refusal_reason: Optional[str],
        confidence_metrics: Optional[Dict],
    ) -> str:
        """Get appropriate refusal message"""

        from rag.refusal_manager import RefusalManager, RefusalReason

        if not refusal_reason:
            refusal_reason = "insufficient_evidence"

        # Map reason codes to RefusalReason enum
        reason_map = {
            "insufficient_evidence_strict": RefusalReason.INSUFFICIENT_EVIDENCE,
            "weak_retrieval_strict": RefusalReason.LOW_CONFIDENCE,
            "no_relevant_evidence": RefusalReason.INSUFFICIENT_EVIDENCE,
            "insufficient_mechanism_evidence": RefusalReason.INSUFFICIENT_EVIDENCE,
            "insufficient_multihop_evidence": RefusalReason.INSUFFICIENT_EVIDENCE,
            "insufficient_evidence_ambiguous": RefusalReason.INSUFFICIENT_EVIDENCE,
            "insufficient_evidence": RefusalReason.INSUFFICIENT_EVIDENCE,
            "low_confidence": RefusalReason.LOW_CONFIDENCE,
        }

        refusal_enum = reason_map.get(
            refusal_reason,
            RefusalReason.INSUFFICIENT_EVIDENCE,
        )

        if confidence_metrics:
            return RefusalManager.create_confidence_refusal(
                top_confidence=confidence_metrics.get("top_confidence", 0),
                avg_confidence=confidence_metrics.get("avg_top3_confidence", 0),
                retrieval_quality=confidence_metrics.get("retrieval_quality", "weak"),
            )
        else:
            return RefusalManager.get_refusal(refusal_enum)