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
    - Return supported facts and explain evidence gaps when coverage is partial
    - No fabrication allowed
    """

    def __init__(self):
        self.system_prompt = """You are a biomedical information assistant.

CRITICAL REQUIREMENTS:
1. Answer ONLY using the provided retrieved evidence
2. Answer from supported facts only. If coverage is partial, give the supported portion and clearly identify what remains unanswered.
3. Do NOT fabricate, speculate, or provide unsupported medical information
4. Cite every factual sentence inline using the exact format [Evidence 1],
   with the number matching the evidence item's label
5. Respect each evidence item's relationship label. Direct evidence may answer
   the requested information. Supporting/contextual evidence may clarify it.
   Related-indirect and complication evidence must not be presented as if they
   directly answer the question.
6. Do not turn risk associations into causes, prevention advice into treatment evidence,
   or a related condition into the requested condition. When direct evidence is missing,
   explain what the retrieved passages cover and what they do not establish.

OUTPUT FORMAT:
Answer the user's question directly in 2-4 concise sentences, in plain language, with inline citations.
For cause questions, lead with the established biological mechanism and distinguish it
from factors that are associated with increased risk. Do not present risk factors as
direct causes. If the evidence does not identify one cause, say so briefly. Do not add a separate
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

Coverage assessment (a conservative evidence map, not a clinical judgment):
{compressed_context.get("coverage_assessment", "Not calculated")}

Based ONLY on the evidence provided above:
1. Lead with a direct answer in 2-4 concise sentences if sufficient evidence exists
2. Cite every factual sentence with the exact label, for example [Evidence 1]
3. If evidence is incomplete, state only supported facts and describe the missing coverage.
   Do not invent a general answer from outside the retrieved evidence. For cause questions,
   separate causal mechanisms from risk factors and describe risk factors only as risks.

Remember: Do not generate information not in the evidence."""

        return {
            "system": self.system_prompt,
            "user": user_prompt,
            "query": query,
            "evidence_count": evidence_count,
            "evidence_sources": compressed_context.get("sources", [])
        }
