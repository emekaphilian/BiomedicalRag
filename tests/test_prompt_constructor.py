from rag.prompt_constructor import GroundedPromptConstructor
from rag.compression import ContextCompressor


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


def test_prompt_requests_direct_causal_summary_and_separates_risk_factors():
    prompt = GroundedPromptConstructor().construct(
        "What is the cause of cancer?",
        {"formatted_text": "[Evidence 1]\nCancer is a genetic disease caused by mutations.", "evidence_count": 1},
    )

    assert "2-4 concise sentences" in prompt["system"]
    assert "distinguish it" in prompt["system"]
    assert "Do not present risk factors as" in prompt["system"]


def test_compression_ends_at_sentence_boundary_when_chunk_is_truncated():
    text = "Cancer is a genetic disease caused by mutations. Exposure to ultraviolet radiation may increase cancer risk."
    result = ContextCompressor(max_chunks=1, max_chunk_length=60).compress([{
        "content": text,
        "metadata": {"source": "MedlinePlus"},
        "rerank_score": 0.9,
    }])

    assert result["context"][0]["content"] == "Cancer is a genetic disease caused by mutations."
    assert result["context"][0]["source_text"] == text
