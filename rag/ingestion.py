import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

from rag.document import BiomedicalDocument


class _HTMLTextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "li",
        "ol",
        "p",
        "section",
        "table",
        "tr",
        "ul",
    }

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, _attrs):
        del _attrs
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)


def _element_text(element):
    return " ".join("".join(element.itertext()).split())


def _xml_files(dataset_path, source):
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"{source} dataset path does not exist: {path}"
        )

    if path.is_file():
        if path.suffix.lower() != ".xml":
            raise ValueError(f"Expected an XML file, received: {path}")
        return [path]

    if not path.is_dir():
        raise NotADirectoryError(
            f"Expected an XML file or dataset directory, received: {path}"
        )

    return sorted(path.rglob("*.xml"))


def _parse_xml_files(dataset_path, source):
    xml_files = _xml_files(dataset_path, source)
    print(f"[Ingestion] Found {len(xml_files)} {source} XML files")

    for filepath in xml_files:
        try:
            yield filepath, ET.parse(filepath).getroot(), None
        except (ET.ParseError, OSError) as exc:
            yield filepath, None, exc


def load_normalized_documents(dataset_path):
    """
    Load normalized biomedical records from one JSONL file or a directory.

    Each record must be an object with non-empty string ``source`` and
    ``content`` fields. All other record fields are preserved as metadata.
    """
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Normalized dataset path does not exist: {path}"
        )

    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Expected a JSONL file, received: {path}")
        jsonl_files = [path]
    elif path.is_dir():
        jsonl_files = sorted(
            file
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() == ".jsonl"
        )
    else:
        raise NotADirectoryError(
            f"Expected a JSONL file or dataset directory, received: {path}"
        )

    print(f"[Ingestion] Found {len(jsonl_files)} normalized JSONL files")

    documents = []
    for filepath in jsonl_files:
        with filepath.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {filepath} at line {line_number}: "
                        f"{exc.msg}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected a JSON object in {filepath} "
                        f"at line {line_number}."
                    )

                content = record.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        f"Expected non-empty string content in {filepath} "
                        f"at line {line_number}."
                    )

                source = record.get("source")
                if not isinstance(source, str) or not source.strip():
                    raise ValueError(
                        f"Expected non-empty string source in {filepath} "
                        f"at line {line_number}."
                    )

                metadata = {
                    key: value for key, value in record.items()
                    if key != "content"
                }
                documents.append(
                    BiomedicalDocument(content=content, metadata=metadata)
                )

    print(f"[Ingestion] Loaded {len(documents)} normalized documents")

    return documents


def load_medquad_dataset(dataset_path):
    """
    Load MedQuAD question-answer pairs from XML files.

    Each valid question-answer pair becomes a BiomedicalDocument.
    """
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"MedQuAD dataset directory does not exist: {dataset_dir}"
        )

    if not dataset_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a dataset directory, received: {dataset_dir}"
        )

    xml_files = sorted(dataset_dir.rglob("*.xml"))

    print(f"[Ingestion] Found {len(xml_files)} XML files")

    documents = []
    parse_errors = 0

    for filepath in xml_files:
        try:
            root = ET.parse(filepath).getroot()
        except (ET.ParseError, OSError) as exc:
            parse_errors += 1
            print(f"[Ingestion] Could not parse {filepath}: {exc}")
            continue

        for qa_pair in root.findall(".//QAPair"):
            question = qa_pair.findtext("Question")
            answer = qa_pair.findtext("Answer")

            if not question or not answer:
                continue

            question = question.strip()
            answer = answer.strip()

            if not question or not answer:
                continue

            content = f"Question:\n{question}\n\nAnswer:\n{answer}"

            metadata = {
                "source": "MedQuAD",
                "file": filepath.name,
                "directory": filepath.parent.name,
                "question": question,
            }

            documents.append(
                BiomedicalDocument(
                    content=content,
                    metadata=metadata,
                )
            )

    print(f"[Ingestion] Extracted {len(documents)} QA pairs")
    print(f"[Ingestion] Files with parsing errors: {parse_errors}")

    return documents


def load_medlineplus_dataset(dataset_path):
    """
    Load MedlinePlus health-topic XML exports.

    The path may be a single XML file or a directory containing XML files.
    """
    documents = []
    parse_errors = 0

    for filepath, root, error in _parse_xml_files(dataset_path, "MedlinePlus"):
        if error is not None:
            parse_errors += 1
            print(f"[Ingestion] Could not parse {filepath}: {error}")
            continue

        for topic in root.iter("healthTopic"):
            title = topic.get("title", "").strip()
            summary_element = topic.find("fullSummary")
            if summary_element is None:
                continue

            summary_html = "".join(summary_element.itertext())
            extractor = _HTMLTextExtractor()
            extractor.feed(summary_html)
            summary = " ".join("".join(extractor.parts).split())

            if not title or not summary:
                continue

            metadata = {
                "source": "MedlinePlus",
                "document_type": "patient_education",
                "file": filepath.name,
                "title": title,
            }
            for field, attribute in (
                ("document_id", "id"),
                ("publication_date", "date"),
                ("url", "url"),
            ):
                value = topic.get(attribute)
                if value:
                    metadata[field] = value

            documents.append(
                BiomedicalDocument(
                    content=f"{title}\n\n{summary}",
                    metadata=metadata,
                )
            )

    print(f"[Ingestion] Extracted {len(documents)} MedlinePlus topics")
    print(f"[Ingestion] Files with parsing errors: {parse_errors}")

    return documents


def _pubmed_publication_date(article):
    article_date = article.find(
        "./MedlineCitation/Article/ArticleDate"
    )
    publication_date = article.find(
        "./MedlineCitation/Article/Journal/JournalIssue/PubDate"
    )

    for date_element in (article_date, publication_date):
        if date_element is None:
            continue

        date_parts = [
            _element_text(date_element.find(field))
            for field in ("Year", "Month", "Day")
            if date_element.find(field) is not None
        ]
        if date_parts:
            return " ".join(date_parts)

        medline_date = date_element.findtext("MedlineDate")
        if medline_date:
            return medline_date.strip()

    return None


def load_pubmed_dataset(dataset_path):
    """
    Load PubMed article abstracts from NCBI EFetch XML files.

    The path may be a single XML file or a directory containing XML files.
    Records without an abstract are skipped. License metadata is intentionally
    omitted because reuse terms vary by article.
    """
    documents = []
    parse_errors = 0

    for filepath, root, error in _parse_xml_files(dataset_path, "PubMed"):
        if error is not None:
            parse_errors += 1
            print(f"[Ingestion] Could not parse {filepath}: {error}")
            continue

        for article in root.iter("PubmedArticle"):
            medline_citation = article.find("MedlineCitation")
            if medline_citation is None:
                continue

            article_element = medline_citation.find("Article")
            if article_element is None:
                continue

            title_element = article_element.find("ArticleTitle")
            title = _element_text(title_element) if title_element is not None else ""
            abstract = article_element.find("Abstract")
            if abstract is None:
                continue

            abstract_parts = []
            for section in abstract.findall(".//AbstractText"):
                text = _element_text(section)
                if text:
                    label = section.get("Label")
                    abstract_parts.append(
                        f"{label}: {text}" if label else text
                    )

            if not abstract_parts:
                continue

            pmid_element = medline_citation.find("PMID")
            pmid = _element_text(pmid_element) if pmid_element is not None else ""

            metadata = {
                "source": "PubMed",
                "document_type": "research_abstract",
                "file": filepath.name,
            }
            if title:
                metadata["title"] = title
            if pmid:
                metadata["document_id"] = pmid
                metadata["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            publication_date = _pubmed_publication_date(article)
            if publication_date:
                metadata["publication_date"] = publication_date

            content = "\n\n".join(
                part for part in (title, "\n".join(abstract_parts)) if part
            )
            documents.append(
                BiomedicalDocument(
                    content=content,
                    metadata=metadata,
                )
            )

    print(f"[Ingestion] Extracted {len(documents)} PubMed abstracts")
    print(f"[Ingestion] Files with parsing errors: {parse_errors}")

    return documents
