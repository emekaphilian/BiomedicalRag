import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed" / "downloaded_sources"

EMAIL = os.environ.get("NCBI_EMAIL", "").strip()
TOOL = "BiomedicalRag"

def request(url, timeout=90):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": f"{TOOL}/1.0 (contact: {EMAIL or 'not-provided'})"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()

def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def download_medlineplus():
    print("\n[MedlinePlus] Discovering the official XML download...")
    raw_dir = RAW / "medlineplus"
    raw_dir.mkdir(parents=True, exist_ok=True)

    page = request("https://medlineplus.gov/xml.html").decode("utf-8", "replace")
    links = re.findall(
        r"""href=["']([^"']*mplus_topics[^"']*\.xml(?:\.gz)?(?:\?[^"']*)?)["']""",
        page,
        re.IGNORECASE
    )

    if not links:
        raise RuntimeError(
            "Could not discover a MedlinePlus topics XML download from "
            "https://medlineplus.gov/xml.html. Check the page and retry."
        )

    urls = list(dict.fromkeys(
        urllib.parse.urljoin("https://medlineplus.gov/xml.html", link)
        for link in links
    ))
    url = urls[0]
    print(f"[MedlinePlus] Downloading {url}")

    data = request(url, timeout=180)
    if url.lower().split("?")[0].endswith(".gz"):
        data = gzip.decompress(data)

    raw_path = raw_dir / "medlineplus_topics.xml"
    raw_path.write_bytes(data)

    root = ET.fromstring(data)
    records = []

    for topic in root.iter():
        if topic.tag.split("}")[-1].lower() not in {
            "health-topic", "health_topic"
        }:
            continue

        def text_of(*names):
            wanted = {name.lower() for name in names}

            # MedlinePlus stores title, URL, and ID as topic attributes.
            for key, value in topic.attrib.items():
                if key.lower() in wanted and value.strip():
                    return " ".join(value.split())

            # Other fields, such as full-summary, are child elements.
            for child in topic.iter():
                if child is topic:
                    continue

                if child.tag.split("}")[-1].lower() in wanted:
                    value = " ".join("".join(child.itertext()).split())
                    if value:
                        return value
            return ""

        title = text_of("title")
        if not title:
            continue

        records.append({
            "id": text_of("health-topic-id", "id"),
            "title": title,
            "content": " ".join(filter(None, [
                title,
                text_of("full-summary", "summary"),
                text_of("also-called"),
            ])),
            "url": text_of("url"),
            "source": "medlineplus",
            "source_url": url,
        })

    if not records:
        raise RuntimeError(
            "Downloaded MedlinePlus XML, but no health topics were extracted. "
            "Inspect the XML format before proceeding."
        )

    out = OUTPUT / "medlineplus_topics.jsonl"
    write_jsonl(out, records)
    print(f"[MedlinePlus] Extracted {len(records)} topics -> {out}")
    return len(records)

def download_pubmed(count):
    if not EMAIL or "@" not in EMAIL or "example.com" in EMAIL.lower():
        raise RuntimeError(
            "Set NCBI_EMAIL to a real contact email before using PubMed."
        )

    raw_dir = RAW / "pubmed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print(f"\n[PubMed] Searching for up to {count} biomedical records...")
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": "biomedical[Title/Abstract]",
        "retmax": count,
        "retmode": "json",
        "tool": TOOL,
        "email": EMAIL,
    })
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + params
    result = json.loads(request(search_url).decode("utf-8"))
    ids = result.get("esearchresult", {}).get("idlist", [])

    if not ids:
        print("[PubMed] No records returned.")
        write_jsonl(OUTPUT / "pubmed_abstracts.jsonl", [])
        return 0

    records = []
    batch_size = 100

    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        params = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "tool": TOOL,
            "email": EMAIL,
        })
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + params
        data = request(url)
        raw_path = raw_dir / f"pubmed_batch_{start // batch_size + 1}.xml"
        raw_path.write_bytes(data)

        root = ET.fromstring(data)

        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//MedlineCitation/PMID", default="").strip()
            title_node = article.find(".//Article/ArticleTitle")
            title = " ".join("".join(title_node.itertext()).split()) if title_node is not None else ""

            abstract_parts = []
            for node in article.findall(".//Article/Abstract/AbstractText"):
                text = " ".join("".join(node.itertext()).split())
                label = node.attrib.get("Label", "")
                if text:
                    abstract_parts.append(f"{label}: {text}" if label else text)

            abstract = " ".join(abstract_parts)
            if not title and not abstract:
                continue

            journal = article.findtext(".//Article/Journal/Title", default="")
            year = article.findtext(".//Article/Journal/JournalIssue/PubDate/Year", default="")
            if not year:
                year = article.findtext(".//Article/Journal/JournalIssue/PubDate/MedlineDate", default="")

            records.append({
                "id": pmid,
                "title": title,
                "abstract": abstract,
                "content": " ".join(filter(None, [title, abstract])),
                "journal": journal,
                "publication_date": year,
                "source": "pubmed",
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            })

        if start + batch_size < len(ids):
            time.sleep(0.4)

    out = OUTPUT / "pubmed_abstracts.jsonl"
    write_jsonl(out, records)
    print(f"[PubMed] Extracted {len(records)} articles -> {out}")
    return len(records)

def main():
    parser = argparse.ArgumentParser(
        description="Download and normalize MedlinePlus topics and PubMed abstracts."
    )
    parser.add_argument(
        "--pubmed-count", type=int, default=20,
        help="Maximum PubMed records to retrieve (default: 20)."
    )
    parser.add_argument(
        "--skip-medlineplus", action="store_true",
        help="Skip the MedlinePlus download."
    )
    args = parser.parse_args()

    if args.pubmed_count < 0:
        parser.error("--pubmed-count cannot be negative")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    totals = {}

    try:
        if not args.skip_medlineplus:
            totals["MedlinePlus"] = download_medlineplus()
        if args.pubmed_count:
            totals["PubMed"] = download_pubmed(args.pubmed_count)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Check your internet connection and the source service.", file=sys.stderr)
        sys.exit(1)

    print("\nDownload summary:")
    for name, count in totals.items():
        print(f"  {name}: {count} records")
    print(f"Normalized files: {OUTPUT}")

if __name__ == "__main__":
    main()
