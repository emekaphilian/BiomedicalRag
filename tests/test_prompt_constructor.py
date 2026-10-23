from rag.prompt_constructor import GroundedPromptConstructor


def test_prompt_requires_exact_inline_evidence_citations():
    prompt = GroundedPromptConstructor().construct(
        "What does the evidence say?",
        {
            "formatted_text": "[Evidence 1]\nContent: Example evidence.",
            "evidence_count": 1,
            "sources": ["MedlinePlus"],
        },
    )

    assert "every factual sentence inline" in prompt["system"]
    assert "[Evidence 1]" in prompt["system"]
    assert "every factual sentence with the exact label" in prompt["user"]


def test_prompt_uses_prior_turns_only_for_follow_up_context():
    prompt = GroundedPromptConstructor().construct(
        "What about its symptoms?",
        {
            "formatted_text": "[Evidence 1]\nContent: Parkinson's disease can affect movement.",
            "evidence_count": 1,
            "sources": ["MedlinePlus"],
        },
        [{"query": "What is Parkinson's disease?", "answer": "It is a movement disorder."}],
    )
    assert "What is Parkinson's disease?" in prompt["user"]
    assert "What about its symptoms?" in prompt["user"]
    assert "factual claims in the answer must be supported by the current evidence" in prompt["user"]
