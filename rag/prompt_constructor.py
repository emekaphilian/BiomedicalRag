"""
Prompt construction for grounded biomedical generation.
Enforces strict citation requirements.
"""


class GroundedPromptConstructor:
    """
    Constructs structured prompts for biomedical LLM generation.
    
    Key principles:
    - ONLY use provided evidence
    - Enforce citation for every claim
    - Refuse if evidence insufficient
    - No fabrication allowed
    """

    def __init__(self):
        self.system_prompt = """You are a biomedical information assistant.

CRITICAL REQUIREMENTS:
1. Answer ONLY using the provided retrieved evidence
2. If evidence is insufficient, say: "I do not have enough grounded evidence to answer this question reliably."
3. Do NOT fabricate, speculate, or provide unsupported medical information
4. CITE the evidence source for every medical claim
5. If you cannot find direct evidence, refuse to answer

OUTPUT FORMAT:
Answer: [Your answer with citations]
Sources: [List evidence items used]

DO NOT provide medical diagnoses, treatment plans, or medical advice.
Keep answers factual and evidence-grounded."""

    def construct(self, query, compressed_context):
        """
        Construct full prompt for LLM generation.
        
        Args:
            query: User query
            compressed_context: Output from ContextCompressor
            
        Returns:
            Dictionary with system prompt and user query
        """
        evidence_text = compressed_context.get("formatted_text", "")
        evidence_count = compressed_context.get("evidence_count", 0)

        user_prompt = f"""Query: {query}

{evidence_text}

Based ONLY on the evidence provided above:
1. Answer the query if sufficient evidence exists
2. Cite which evidence item(s) you used
3. Refuse if evidence is insufficient or contradictory

Remember: Do not generate information not in the evidence."""

        return {
            "system": self.system_prompt,
            "user": user_prompt,
            "query": query,
            "evidence_count": evidence_count,
            "evidence_sources": compressed_context.get("sources", [])
        }
