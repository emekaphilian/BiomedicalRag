from rag.document import BiomedicalDocument
from rag.preprocessing import clean_text, preprocess_documents


def test_clean_text_removes_html_markup_and_hidden_content():
    text = (
        "<p>Parkinson's disease affects <a href='/topic'>dopamine</a>.</p>"
        "<script>tracking()</script><style>.hidden { display: none; }</style>"
    )

    assert clean_text(text) == "Parkinson's disease affects dopamine."


def test_preprocessing_cleans_markup_and_preserves_metadata():
    document = BiomedicalDocument(
        "<p>Parkinson's disease affects dopamine.</p>",
        {"source": "medlineplus"},
    )

    processed = preprocess_documents([document])

    assert processed[0].content == "Parkinson's disease affects dopamine."
    assert processed[0].metadata == {"source": "medlineplus"}
