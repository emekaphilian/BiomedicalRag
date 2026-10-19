from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest


def test_playground_accepts_and_preserves_question_before_setup() -> None:
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path)).run()

    model = next(widget for widget in app.selectbox if widget.label == "Model")
    assert model.value == "command-a-plus-05-2026"
    assert model.options == [
        "command-a-plus-05-2026",
        "command-a-03-2025",
    ]

    question = next(
        widget
        for widget in app.text_input
        if widget.key == "playground_question_input"
    )
    assert question.disabled is False

    draft = "How does dopamine affect movement?"
    question.set_value(draft)
    next(button for button in app.button if button.label == "Send").click().run()

    question = next(
        widget
        for widget in app.text_input
        if widget.key == "playground_question_input"
    )
    assert question.value == draft
    assert any(
        "Connect an LLM" in warning.value
        for warning in app.warning
    )
    assert not app.exception


def test_playground_uses_preloaded_chunks_without_session_uploads() -> None:
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path))
    app.session_state["rag_service"] = SimpleNamespace(
        provider="openai",
        model="gpt-4o",
        total_chunks=3,
        total_documents=2,
        source_names=["MedlinePlus", "PubMed"],
    )
    app.run()

    assert app.session_state["docs_indexed"] == []
    assert not any(
        "Index a document before sending" in item.value
        for item in app.caption
    )
    send_button = next(button for button in app.button if button.label == "Send")
    assert send_button.disabled is False


def test_playground_recovers_from_stale_service_instance() -> None:
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path))
    app.session_state["rag_service"] = SimpleNamespace(
        provider="openai",
        model="gpt-4o",
        total_chunks=3,
    )
    app.session_state["llm_ready"] = True
    app.run()

    assert not app.exception
    assert app.session_state["rag_service"] is None
    assert app.session_state["llm_ready"] is False
    assert any(
        "predates the current knowledge-base interface" in warning.value
        for warning in app.warning
    )


def test_playground_discloses_answer_validation_failure() -> None:
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    answer = "I could not produce an answer that passed the evidence and safety checks."
    app = AppTest.from_file(str(app_path))
    app.session_state["rag_service"] = SimpleNamespace(
        provider="cohere",
        model="command-r-plus",
        total_chunks=3,
        total_documents=2,
        source_names=["MedlinePlus"],
    )
    app.session_state["display_history"] = [
        {"role": "user", "content": "How does dopamine affect movement?"},
        {
            "role": "assistant",
            "content": answer,
            "sources": [{"source": "MedlinePlus"}],
            "details": {
                "is_grounded": True,
                "warnings": ["No evidence citations found in answer"],
            },
            "provider": "cohere",
            "retrieval_mode": "hybrid",
            "top_k": 5,
            "latency_ms": 700,
        },
    ]
    app.session_state["audit_events"] = [
        {
            "event": "query",
            "id": "query-validation-failure",
            "timestamp": "query-validation-failure",
            "success": True,
            "query": "How does dopamine affect movement?",
            "answer": answer,
            "sources": ["MedlinePlus"],
            "warnings": ["No evidence citations found in answer"],
            "is_grounded": True,
        }
    ]

    app.run()

    assert not app.exception
    assert any(
        "Evidence retrieved · answer withheld" in item.value
        for item in app.caption
    )
    assert any(
        "Generation / validation: No evidence citations found in answer"
        in item.value
        for item in app.warning
    )
    assert app.session_state["audit_events"][0]["warnings"] == [
        "No evidence citations found in answer"
    ]


@pytest.mark.parametrize("has_matching_review", [False, True])
def test_quick_review_renders_for_latest_answer(
    has_matching_review: bool,
) -> None:
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    answer = "Dopamine is involved in movement."
    app = AppTest.from_file(str(app_path))
    app.session_state["rag_service"] = SimpleNamespace(
        provider="openai",
        model="gpt-4o",
        total_chunks=3,
        total_documents=2,
        source_names=["MedlinePlus"],
    )
    app.session_state["display_history"] = [
        {
            "role": "user",
            "content": "How does dopamine affect movement?",
        },
        {
            "role": "assistant",
            "content": answer,
            "sources": [],
        },
    ]
    app.session_state["audit_events"] = [
        {
            "event": "query",
            "id": "query-1",
            "timestamp": "query-1",
            "success": True,
            "query": "How does dopamine affect movement?",
            "answer": answer,
        },
    ]
    if has_matching_review:
        app.session_state["answer_quality_reviews"] = [
            {
                "query_event_id": "query-1",
                "timestamp": "review-1",
                "query": "How does dopamine affect movement?",
                "answer": answer,
                "sources": [],
                "evaluator": "Reviewer",
                "ratings": {
                    "factual_correctness": 4,
                    "hallucination_concern": 1,
                    "evidence_support": 4,
                    "faithfulness": 4,
                    "explainability": 4,
                    "completeness": 4,
                    "safety": 5,
                },
                "notes": "",
            },
        ]

    app.run()

    assert not app.exception
    if has_matching_review:
        assert any(
            "Reviewed by a human" in item.value
            for item in app.success
        )
        assert not any(
            item.label == "Hallucination concern (1 = none, 5 = high)"
            for item in app.select_slider
        )
    else:
        assert any(
            item.label == "Hallucination concern (1 = none, 5 = high)"
            for item in app.select_slider
        )
