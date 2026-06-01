"""
Post-generation safety validation.
Ensures answers meet medical safety standards.
"""


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

    def validate(self, generation_result, evidence_count):
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

        # Check for forbidden medical advice
        for phrase in self.forbidden_phrases:
            if phrase in answer:
                issues.append(f"Forbidden phrase detected: '{phrase}'")

        # Check for citations (should have [Evidence])
        if "[evidence" not in answer and "source:" not in answer:
            if not any(refusal in answer for refusal in self.refusal_phrases):
                issues.append("No evidence citations found in answer")

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

        valid = len(issues) == 0

        return {
            "valid": valid,
            "reason": "Passed" if valid else "Failed validation",
            "answer": generation_result["answer"],
            "issues": issues,
            "citations_found": "[evidence" in answer or "source:" in answer,
            "evidence_used": evidence_count
        }
