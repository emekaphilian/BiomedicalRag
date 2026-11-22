"""Best-effort PubMed abstract retrieval for medical questions."""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def retrieve_pubmed(query: str, limit: int = 3, timeout: float = 3.5) -> list[dict]:
    """Fetch a few PubMed abstracts, returning RAG-shaped passages.

    PubMed is treated as an external research index, not as an endorsement or
    independent validation of an article's clinical quality.
    """
    if os.getenv("BIOMED_RAG_EXTERNAL_SOURCES", "true").strip().lower() in {
        "0", "false", "no", "off",
    }:
        return []

    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max(1, min(int(limit), 5)),
        "retmode": "xml",
        "tool": "biomedical-rag",
        "email": os.getenv("NCBI_EMAIL", ""),
    })
    request = urllib.request.Request(
        f"{_EUTILS}/esearch.fcgi?{params}",
        headers={"User-Agent": "BioMedRAG/2.0 (biomedical research assistant)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            search_root = ET.fromstring(response.read())
        ids = [node.text for node in search_root.findall(".//IdList/Id") if node.text]
        if not ids:
            return []

        fetch_params = urllib.parse.urlencode({
            "db": "pubmed", "id": ",".join(ids), "retmode": "xml",
        })
        fetch_request = urllib.request.Request(
            f"{_EUTILS}/efetch.fcgi?{fetch_params}",
            headers={"User-Agent": "BioMedRAG/2.0 (biomedical research assistant)"},
        )
        with urllib.request.urlopen(fetch_request, timeout=timeout) as response:
            records_root = ET.fromstring(response.read())
    except (OSError, TimeoutError, ET.ParseError, ValueError):
        return []

    passages = []
    for article in records_root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        title = " ".join("".join(article.find(".//ArticleTitle").itertext()).split()) if article.find(".//ArticleTitle") is not None else ""
        abstract = " ".join(
            "".join(node.itertext()).strip()
            for node in article.findall(".//Abstract/AbstractText")
            if "".join(node.itertext()).strip()
        )
        abstract = " ".join(abstract.split())
        if not pmid or not abstract:
            continue
        journal = (article.findtext(".//Journal/Title") or "PubMed indexed journal").strip()
        year = (article.findtext(".//PubDate/Year") or "").strip()
        passages.append({
            "content": f"{title}. {abstract}"[:6000],
            "metadata": {
                "source": f"{journal}{f' ({year})' if year else ''} · PMID {pmid}",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source_provider": "PubMed / NCBI",
                "source_quality": "external_research_unverified",
                "source_origin": "external",
                "pmid": pmid,
            },
            "rerank_score": 0.0,
        })
    return passages
