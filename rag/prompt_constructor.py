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
4. Cite every factual sentence inline using the exact format [Evidence 1],
   with the number matching the evidence item's label
5. Respect each evidence item's relationship label. Direct evidence may answer
   the requested information. Supporting/contextual evidence may clarify it.
   Related-indirect and complication evidence must not be presented as if they
   directly answer the question.
6. If you cannot find direct evidence, refuse to answer

OUTPUT FORMAT:
Write a concise answer in plain language with inline citations. Do not add a separate
Sources section; citations in the answer are the source references.

DO NOT provide medical diagnoses, treatment plans, or medical advice.
Keep answers factual and evidence-grounded. Return only the final answer. Never
include private reasoning, analysis, planning, control tokens, or commentary about
following these instructions."""

    def construct(self, query, compressed_context, chat_history=None):
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

        history_block = ""
        if chat_history:
            turns = []
            for turn in chat_history[-6:]:
                previous_query = str(turn.get("query", "")).strip()[:1000]
                previous_answer = str(turn.get("answer", "")).strip()[:1500]
                if previous_query and previous_answer:
                    turns.append(f"User: {previous_query}\nAssistant: {previous_answer}")
            if turns:
                history_block = (
                    "Prior conversation (use only to resolve follow-up references; "
                    "all factual claims in the answer must be supported by the current evidence):\n"
                    + "\n\n".join(turns)
                    + "\n\n"
                )

        user_prompt = f"""{history_block}Current question: {query}

{evidence_text}

Based ONLY on the evidence provided above:
1. Answer the query if sufficient evidence exists
2. Cite every factual sentence with the exact label, for example [Evidence 1]
3. Refuse if evidence is insufficient or contradictory

Remember: Do not generate information not in the evidence."""

        return {
            "system": self.system_prompt,
            "user": user_prompt,
            "query": query,
            "evidence_count": evidence_count,
            "evidence_sources": compressed_context.get("sources", [])
        }
