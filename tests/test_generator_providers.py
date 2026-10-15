import sys
from types import SimpleNamespace

from rag.generator import GroundedGenerator


def test_cohere_provider_receives_grounded_system_and_evidence_prompts(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "secret"

        def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                message=SimpleNamespace(
                    content=[SimpleNamespace(text="Answer [Evidence 1].")]
                )
            )

    monkeypatch.setitem(
        sys.modules,
        "cohere",
        SimpleNamespace(ClientV2=FakeClient),
    )
    generator = GroundedGenerator(
        model_name="command-r-plus",
        api_key="secret",
        provider="cohere",
    )
    prompt = {
        "system": "Use only retrieved evidence.",
        "user": "Evidence: clinical text",
        "query": "What does the evidence say?",
        "evidence_count": 1,
        "evidence_sources": ["guide.txt"],
    }

    result = generator.generate(prompt)

    assert result["success"] is True
    assert result["answer"] == "Answer [Evidence 1]."
    assert calls[0]["model"] == "command-r-plus"
    assert calls[0]["messages"] == [
        {"role": "system", "content": "Use only retrieved evidence."},
        {"role": "user", "content": "Evidence: clinical text"},
    ]
