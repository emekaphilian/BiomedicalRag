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
