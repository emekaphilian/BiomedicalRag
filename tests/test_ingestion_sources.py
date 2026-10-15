import json

from rag.ingestion import (
    load_medlineplus_dataset,
    load_normalized_documents,
    load_pubmed_dataset,
)
from rag.chunking import chunk_document
from rag.preprocessing import preprocess_documents


def test_load_medlineplus_health_topic_xml(tmp_path):
    xml_path = tmp_path / "medlineplus.xml"
    xml_path.write_text(
        """\
<healthTopics>
  <healthTopic id="123" title="Diabetes" date="2025-03-04"
               url="https://medlineplus.gov/diabetes.html">
    <fullSummary><![CDATA[
      <p>Diabetes affects <b>blood sugar</b>.</p>
      <p>Learn about symptoms and treatment.</p>
    ]]></fullSummary>
  </healthTopic>
</healthTopics>
""",
        encoding="utf-8",
    )

    documents = load_medlineplus_dataset(xml_path)

    assert len(documents) == 1
    assert documents[0].content == (
        "Diabetes\n\nDiabetes affects blood sugar. "
        "Learn about symptoms and treatment."
    )
    assert documents[0].metadata == {
        "source": "MedlinePlus",
        "document_type": "patient_education",
        "file": "medlineplus.xml",
        "title": "Diabetes",
        "document_id": "123",
        "publication_date": "2025-03-04",
        "url": "https://medlineplus.gov/diabetes.html",
    }


def test_load_pubmed_abstracts_and_skip_invalid_records(tmp_path):
    xml_path = tmp_path / "pubmed.xml"
    (tmp_path / "malformed.xml").write_text(
        "<PubmedArticleSet>",
        encoding="utf-8",
    )
    xml_path.write_text(
        """\
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>456789</PMID>
      <Article>
        <ArticleTitle>Study of <i>diabetes</i>.</ArticleTitle>
        <Journal>
          <JournalIssue>
            <PubDate><Year>2024</Year><Month>Jan</Month><Day>12</Day></PubDate>
          </JournalIssue>
        </Journal>
        <Abstract>
          <AbstractText Label="BACKGROUND">A background finding.</AbstractText>
          <AbstractText Label="RESULTS">A results finding.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>999999</PMID>
      <Article><ArticleTitle>No abstract</ArticleTitle></Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
""",
        encoding="utf-8",
    )

    documents = load_pubmed_dataset(tmp_path)

    assert len(documents) == 1
    assert documents[0].content == (
        "Study of diabetes.\n\n"
        "BACKGROUND: A background finding.\n"
        "RESULTS: A results finding."
    )
    assert documents[0].metadata == {
        "source": "PubMed",
        "document_type": "research_abstract",
        "file": "pubmed.xml",
        "title": "Study of diabetes.",
        "document_id": "456789",
        "url": "https://pubmed.ncbi.nlm.nih.gov/456789/",
        "publication_date": "2024 Jan 12",
    }


def test_preprocessing_preserves_source_metadata_and_skips_empty_content():
    from rag.document import BiomedicalDocument

    source_metadata = {"source": "PubMed", "document_type": "research_abstract"}
    documents = preprocess_documents(
        [
            BiomedicalDocument(content="  useful   content ", metadata=source_metadata),
            BiomedicalDocument(content=" \n\t ", metadata={"source": "MedlinePlus"}),
        ]
    )

    assert len(documents) == 1
    assert documents[0].content == "useful content"
    assert documents[0].metadata == source_metadata
    assert documents[0].metadata is not source_metadata


def test_load_normalized_documents_from_directory_preserves_sources(tmp_path):
    pubmed_path = tmp_path / "pubmed.jsonl"
    medlineplus_path = tmp_path / "medlineplus" / "topics.jsonl"
    medlineplus_path.parent.mkdir()

    pubmed_record = {
        "source": "PubMed",
        "document_id": "PMID:456789",
        "title": "Study of diabetes",
        "document_type": "research_abstract",
        "content": "Title: Study of diabetes\n\nAbstract: A study result.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/456789/",
        "license_status": "Review applicable terms",
    }
    medlineplus_record = {
        "source": "MedlinePlus",
        "document_id": "123",
        "title": "Diabetes",
        "document_type": "patient_education",
        "content": "Diabetes affects blood sugar.",
        "url": "https://medlineplus.gov/diabetes.html",
    }
    pubmed_path.write_text(
        json.dumps(pubmed_record) + "\n",
        encoding="utf-8",
    )
    medlineplus_path.write_text(
        "\n" + json.dumps(medlineplus_record) + "\n",
        encoding="utf-8",
    )

    single_file_documents = load_normalized_documents(pubmed_path)
    assert len(single_file_documents) == 1
    assert single_file_documents[0].metadata["source"] == "PubMed"

    documents = load_normalized_documents(tmp_path)

    assert [document.metadata["source"] for document in documents] == [
        "MedlinePlus",
        "PubMed",
    ]
    assert [document.metadata["document_type"] for document in documents] == [
        "patient_education",
        "research_abstract",
    ]
    assert documents[0].content == medlineplus_record["content"]
    assert documents[0].metadata == {
        key: value
        for key, value in medlineplus_record.items()
        if key != "content"
    }
    assert documents[1].content == pubmed_record["content"]
    assert documents[1].metadata == {
        key: value for key, value in pubmed_record.items() if key != "content"
    }

    processed = preprocess_documents(documents)
    chunks = chunk_document(processed[0], chunk_size=3, overlap=1)

    assert len(chunks) == 2
    assert chunks[0].metadata["source"] == "MedlinePlus"
    assert chunks[0].metadata["document_type"] == "patient_education"


def test_load_normalized_documents_reports_invalid_record_location(tmp_path):
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        '{"source": "PubMed", "content": "valid"}\n'
        '{"source": "MedlinePlus", "content": ""}\n',
        encoding="utf-8",
    )

    try:
        load_normalized_documents(jsonl_path)
    except ValueError as exc:
        assert "records.jsonl at line 2" in str(exc)
    else:
        raise AssertionError("Invalid normalized records should be rejected.")
