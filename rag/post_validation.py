"""
Post-generation safety validation.
Ensures answers meet medical safety standards.
"""

import re


from dashboard.rag_audit import audit_answer


class PostGenerationValidator:
    """
    Validates generated answers for safety and grounding.
    
    Checks:
    1. Response includes evidence citations
    2. No forbidden medical advice
    3. No hallucinated claims
    4. Appropriate refusals for insufficient evidence
    """

    def __init__(self):
        # Forbidden terms that indicate medical advice
        self.forbidden_phrases = [
            "you should take",
            "i recommend",
            "start this medication",
            "stop this treatment",
            "you have",  # diagnosis
            "this will cure",
            "guaranteed to",
            "will definitely",
        ]
        
        # Refusal indicators (good)
        self.refusal_phrases = [
            "i do not have enough",
            "insufficient evidence",
            "cannot determine",
            "not enough information",
        ]

    def validate(self, generation_result, evidence_count, evidence_texts=None):
        """
        Validate generated answer.
        
        Args:
            generation_result: Output from GroundedGenerator
            evidence_count: Number of evidence chunks used
            
        Returns:
            Dictionary with validation results
        """
        if not generation_result["success"]:
            return {
                "valid": False,
                "reason": "Generation failed",
                "answer": generation_result["answer"],
                "issues": [generation_result["error"]]
            }

        answer = generation_result["answer"].lower()
        issues = []

        reasoning_markers = (
            "<eos_token>", "<think>", "</think>", "chain of thought",
            "we need to answer the query", "the user says", "we should respond",
            "thus final answer:", "let me think",
        )
        if any(marker in answer for marker in reasoning_markers):
            issues.append("Internal reasoning or control tokens appeared in generated answer")

        # Check for forbidden medical advice
        for phrase in self.forbidden_phrases:
            if phrase in answer:
                issues.append(f"Forbidden phrase detected: '{phrase}'")

        # Check citation identifiers against the exact evidence shown to the model.
        citations = [int(value) for value in re.findall(
            r"\[Evidence\s+(\d+)\]", generation_result["answer"], re.IGNORECASE
        )]
        invalid_citations = sorted({value for value in citations if value < 1 or value > evidence_count})
        if not citations:
            if not any(refusal in answer for refusal in self.refusal_phrases):
                issues.append("No evidence citations found in answer")
        if invalid_citations:
            issues.append(f"Invalid evidence citation identifiers: {invalid_citations}")

        # Check for appropriate refusal if no evidence
        if evidence_count == 0:
            if not any(refusal in answer for refusal in self.refusal_phrases):
                issues.append("Should refuse when no evidence available")

        # Check for hallucinations (very confident claims without evidence)
        hallucination_indicators = [
            "absolutely",
            "definitely",
            "certainly",
            "100% sure",
            "proven fact"
        ]
        
        for indicator in hallucination_indicators:
            if indicator in answer and "[evidence" not in answer:
                issues.append(f"Possible hallucination: confident claim without citation")
                break

        support_audit = audit_answer(
            generation_result["answer"], evidence_texts or [],
        ) if evidence_texts else None
        if support_audit and support_audit["sentence_count"]:
            if support_audit["invalid_citations"]:
                issues.append("Evidence citation references do not match the supplied evidence")
            if support_audit["citation_support_coverage"] < 1.0:
                issues.append("One or more factual sentences lack a citation linked to overlapping evidence")
            if support_audit["faithfulness"] < 1.0:
                issues.append("One or more sentences failed the lexical claim-support screen")

        valid = len(issues) == 0

        return {
            "valid": valid,
            "reason": "Passed" if valid else "Failed validation",
            "answer": generation_result["answer"],
            "issues": issues,
            "citations_found": "[evidence" in answer or "source:" in answer,
            "citation_references_validated": not invalid_citations and bool(citations),
            "claim_support_screen_passed": bool(
                support_audit and support_audit.get("faithfulness") == 1.0
                and support_audit.get("citation_support_coverage") == 1.0
            ),
            "claim_support_validated": "NOT ESTABLISHED",
            "claim_support_method": "lexical-overlap-screen" if support_audit else "not-evaluated",
            "evidence_used": evidence_count
        }
