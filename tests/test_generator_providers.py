import sys
from types import SimpleNamespace

from rag.generator import GroundedGenerator
from rag import generator_cohere


def test_streaming_cohere_generator_uses_versioned_default_model(monkeypatch):
    monkeypatch.setattr(
        generator_cohere.cohere,
        "ClientV2",
        lambda **kwargs: SimpleNamespace(api_key=kwargs["api_key"]),
    )

    generator = generator_cohere.CohereGeneratorStreaming(api_key="secret")

    assert generator.model == "command-r-plus-08-2024"


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


def test_cohere_provider_extracts_text_content_returned_as_string(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "secret"

        def chat(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                message=SimpleNamespace(
                    content="Answer returned as plain text [Evidence 1]."
                )
            )

    monkeypatch.setitem(
        sys.modules,
        "cohere",
        SimpleNamespace(ClientV2=FakeClient),
    )
    generator = GroundedGenerator(
        model_name="command-a-plus-05-2026",
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
    assert result["answer"] == "Answer returned as plain text [Evidence 1]."


def test_cohere_provider_rejects_response_without_text_content(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "secret"

        def chat(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                message=SimpleNamespace(content=None)
            )

    monkeypatch.setitem(
        sys.modules,
        "cohere",
        SimpleNamespace(ClientV2=FakeClient),
    )
    generator = GroundedGenerator(
        model_name="command-a-plus-05-2026",
        api_key="secret",
        provider="cohere",
    )
    prompt = {
        "system": "Use evidence.",
        "user": "Evidence: text",
        "query": "Question?",
        "evidence_count": 1,
        "evidence_sources": ["guide.txt"],
    }

    result = generator.generate(prompt)

    assert result["success"] is False
    assert "did not contain text content" in result["error"]


def test_provider_error_reports_status_and_message_without_headers(monkeypatch):
    class FakeProviderError(Exception):
        status_code = 401
        body = {"message": "Invalid API key"}
        headers = {"authorization": "must-not-be-shown"}

        def __str__(self):
            return f"headers: {self.headers}, status_code: {self.status_code}"

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

        def chat(self, **kwargs):
            raise FakeProviderError()

    monkeypatch.setitem(
        sys.modules,
        "cohere",
        SimpleNamespace(ClientV2=FakeClient),
    )
    generator = GroundedGenerator(
        model_name="command-r-plus",
        api_key="secret-key",
        provider="cohere",
    )
    prompt = {
        "system": "Use evidence.",
        "user": "Evidence: text",
        "query": "Question?",
        "evidence_count": 1,
        "evidence_sources": ["guide.txt"],
    }

    result = generator.generate(prompt)

    assert result["success"] is False
    assert "HTTP 401" in result["error"]
    assert "Invalid API key" in result["error"]
    assert "headers" not in result["error"]
    assert "must-not-be-shown" not in result["error"]


def test_unlabelled_attribute_error_reports_safe_source_location():
    generator = GroundedGenerator(provider="cohere")

    try:
        raise AttributeError()
    except AttributeError as error:
        message = generator._safe_error_message(error)

    assert "AttributeError at test_generator_providers.py:" in message
