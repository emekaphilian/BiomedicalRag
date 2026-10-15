# Biomedical dataset ingestion

Place provider-supplied XML files in source-specific directories:

- `data/raw/medquad/` — MedQuAD question-answer XML files
- `data/raw/medlineplus/` — MedlinePlus Health Topics XML exports
- `data/raw/pubmed/` — PubMed EFetch XML files containing `PubmedArticle` records

The XML loaders accept a single XML file or a directory searched recursively:

```python
from rag.ingestion import (
    load_medlineplus_dataset,
    load_medquad_dataset,
    load_pubmed_dataset,
)

documents = (
    load_medquad_dataset("data/raw/medquad")
    + load_medlineplus_dataset("data/raw/medlineplus")
    + load_pubmed_dataset("data/raw/pubmed")
)
```

Each loader returns `BiomedicalDocument` instances. MedlinePlus records are
marked as `patient_education`, and PubMed records as `research_abstract`, so
their source types remain distinguishable during preprocessing and indexing.
The loaders do not download data or build an index.

For downloaded, normalized JSONL records (one JSON object per line, with
non-empty `source` and `content` fields), use the shared loader:

```python
from rag.ingestion import load_normalized_documents
from rag.indexing import build_index
from rag.preprocessing import preprocess_documents

documents = load_normalized_documents("data/processed/downloaded_sources")
documents = preprocess_documents(documents)
vectorstore, chunks = build_index(documents)
```

The JSONL loader accepts one `.jsonl` file or recursively reads `.jsonl` files
from a directory. All record fields except `content` are retained as document
metadata. Invalid JSON or records missing a usable `source` or `content` fail
with the file and line number rather than being silently ignored.

Obtain data from the original providers and check their current access,
redistribution, and attribution terms before using or sharing it. License
metadata is not assigned automatically because terms can vary by resource or
article.
