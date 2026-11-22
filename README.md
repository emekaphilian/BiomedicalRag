# 🧬 Biomedical RAG — Medical Question Answering System

> A domain-specific Retrieval-Augmented Generation system for grounded biomedical Q&A, built on 32,814 MedQuAD documents with hybrid retrieval, Cohere Command-R, and a research-grade evaluation framework.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Cohere](https://img.shields.io/badge/Cohere-Command--R-39594C?style=flat-square)](https://cohere.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0064BD?style=flat-square)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [Project Status](#project-status)
- [Quick Start](#quick-start)
- [Deploying to Streamlit Cloud](#deploying-to-streamlit-cloud)
- [Configuration](#configuration)
- [Evaluation Framework](#evaluation-framework)
- [System Design Decisions](#system-design-decisions)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Safety & Ethics](#safety--ethics)
- [Citation](#citation)

---

## Overview

This project implements a **research prototype for biomedical question answering** using Retrieval-Augmented Generation (RAG). Retrieval provides candidate evidence for answer generation; heuristic checks can reject some unsupported or poorly cited output. They do not establish medical accuracy or clinical safety.

**Research context:** MSc dissertation project — University of Aveiro | ACM Student Project Conference 2026

### What makes this different from standard RAG

| Feature | Standard RAG | This System |
|---|---|---|
| Retrieval | Dense-only | **Hybrid: FAISS + BM25 + RRF** |
| Reranking | None | **Cross-encoder (ms-marco-MiniLM)** |
| Safety | None | **Refusal logic + confidence thresholds** |
| Query handling | One-size-fits-all | **Query classification (factoid / multi-hop / adversarial)** |
| Evaluation | Evolving | **PubMedQA labeled adapter + expanded adversarial cases** |
| Corpus | Generic | **32,814 MedQuAD biomedical QA pairs** |

---

## Live Demo

> 🚀 **[Open the live app →](https://your-app-name.streamlit.app)** *(replace with your Streamlit Cloud URL)*

The demo requires a Cohere API key (free tier works). You can get one at [dashboard.cohere.com](https://dashboard.cohere.com/api-keys).

The Streamlit research dashboard provides Overview, Knowledge Base, RAG Playground,
Evaluation Lab, Safety & Governance, and Audit History sections. Its retrieval
evaluation is manually labeled and source-level; answer-quality assessments are
human rubric ratings rather than automated or clinical validation. Evaluation runs
capture the labels, ranked evidence, and a non-secret configuration fingerprint.
The Governance section includes an operator-authored bias/fairness report and an
impact × likelihood ethical-risk triage score. The security posture explicitly
identifies missing application login and PII/PHI redaction; it does not certify
enterprise security. Session activity, configuration snapshots, and remediation
history remain in memory until exported as JSON. This prototype is for research
use and is not a clinically validated system.

---

## Architecture

### Application boundary

The React dashboard calls a single FastAPI application. FastAPI delegates query
and ingestion work to the existing `RAGService` and Python RAG pipeline, and
exposes the research metrics, answer audit, responsible-AI, and security-control
helpers as API services. Audit and governance records are isolated by dashboard
session but currently remain in API-process memory.

```text
React + TypeScript + Vite
          │ JSON / multipart HTTP
          ▼
FastAPI (app.main:app)
   ├── session, query, upload, retrieval
   ├── audit, evaluation, governance, security
   └── RAGService (app.services)
               │
               ▼
     Existing Python RAG pipeline
```

The Streamlit dashboard remains available during migration; it uses the same
Python pipeline directly.

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    Query Classifier                      │
│       factoid │ multi-hop │ ambiguous │ adversarial      │
└──────────────────────────┬──────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
┌──────────────────┐             ┌──────────────────────┐
│  Dense Retrieval │             │  Sparse Retrieval    │
│  FAISS + MiniLM  │             │  BM25 (rank-bm25)    │
│  all-MiniLM-L6   │             │                      │
└────────┬─────────┘             └──────────┬───────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
            ┌────────────────────┐
            │ Reciprocal Rank    │
            │ Fusion (RRF)       │
            └────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Cross-Encoder        │
         │  Reranker             │
         │  ms-marco-MiniLM-L6   │
         └────────┬──────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Evidence-grounded prompt   │
    │  + Safety constraints       │
    └──────────────┬──────────────┘
                   │
                   ▼
      ┌────────────────────────┐
      │   Cohere Command-R+    │
      │   (streaming)          │
      └────────────┬───────────┘
                   │
                   ▼
    Grounded answer + source citations
```

### Corpus

| Statistic | Value |
|---|---|
| Source | [MedQuAD](https://github.com/abachaa/MedQuAD) — NLM/NIH |
| XML files processed | 22,548 |
| Biomedical QA pairs | 32,814 |
| Embedding model | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Index type | FAISS IndexFlatIP (cosine similarity) |
| Chunk size | 900 tokens / 150 overlap |

### File Structure

```
Biomedical-RAG/
├── frontend/
│   ├── src/                ← React + TypeScript research dashboard
│   └── package.json        ← Vite build and frontend tests
├── app/
│   ├── api.py              ← Canonical FastAPI endpoints and session services
│   ├── main.py             ← FastAPI entry point: app.main:app
│   └── services.py         ← Shared RAG service adapter
├── dashboard/
│   └── app.py              ← Existing Streamlit dashboard
├── rag/
│   ├── retriever.py        ← Hybrid FAISS + BM25 retrieval
│   ├── reranker.py         ← Cross-encoder reranking
│   └── pipeline.py         ← End-to-end RAG pipeline
├── evaluation/
│   ├── PubMedQA adapter    ← labeled biomedical test split
│   └── metrics.py          ← Recall@K, Precision@K, MRR
├── configs/
│   └── config.yaml         ← Legacy configuration (not loaded by FastAPI)
├── data/                   ← MedQuAD processed corpus (not committed)
├── .streamlit/
│   └── secrets.toml        ← API keys (gitignored)
├── requirements.txt
├── runtime.txt             ← python-3.11
└── README.md
```

---

## Project Status

| Phase | Description | Status | Completion |
|---|---|---|---|
| Phase 1 | Data Acquisition & Indexing | ✅ Complete | 95% |
| Phase 2 | RAG System Implementation | ✅ Largely Complete | 85% |
| Phase 3 | Evaluation & Optimisation | 🔄 In Progress | 30% |
| Phase 4 | Safety, Ethics & Finalisation | 📋 Planned | 5% |

**Overall: ~65–70% complete**

### Historical Phase 3 grounding results (18-query suite)

| Category | Pass Rate | Notes |
|---|---|---|
| Factoid questions | 5/5 (100%) | Strong on direct recall |
| Mechanism questions | 2/4 (50%) | Improving with reranker |
| Multi-hop reasoning | 1/3 (33%) | Cross-encoder helps |
| Ambiguous queries | 1/3 (33%) | Confidence thresholding added |
| Adversarial queries | 0/3 (0%) | Refusal logic implemented |
| **Overall** | **9/18 (50%)** | **Hallucination rate: 27.8%** |

Stabilisation improvements (confidence thresholding, query classification, refusal logic) have been implemented to address the failing categories.

---

## Quick Start

### React + FastAPI dashboard (recommended)

The React application is the new dashboard frontend. The biomedical RAG pipeline,
answer checks, and research/governance services remain in Python. Configure provider
credentials only on the API server; they are never sent to the browser.

```powershell
# Install Python dependencies in your active Python 3.11 environment
pip install -r requirements.txt
$env:COHERE_API_KEY = "your-key"
$env:BIOMEDRAG_API_TOKENS = '{"researcher":"replace-with-a-unique-random-token-of-at-least-32-characters"}'
uvicorn app.main:app --reload
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Set `VITE_API_URL` at frontend build/dev time if the
API is not at `http://127.0.0.1:8000`; configure `BIOMEDRAG_CORS_ORIGINS` on the
API server with a comma-separated allowlist when using another frontend origin.
`BIOMEDRAG_API_TOKENS` must be a JSON object mapping principal IDs to unique,
random bearer tokens of at least 32 ASCII characters. A principal may temporarily
have an array of old and new tokens during rotation; both still map to the same
principal. The standalone browser app uses loopback-only local access by default,
so it does not ask you for an API bearer token. Non-local API clients still need
one of the configured bearer tokens. Set `BIOMEDRAG_LOCAL_STANDALONE_ACCESS=false`
to require bearer authentication for local browser access too. RAGGuard
observation and repair use separate server-side credentials; do not put those
integration credentials in the browser.
The health endpoint is public; other `/api` endpoints require a bearer token
except for the restricted local standalone browser path described above.
Provider credentials remain server-side, and configuration requests cannot
override them. The local standalone exception is restricted to loopback clients
and loopback browser origins; it does not grant access to remote clients.

Generate API tokens with a cryptographically secure generator, for example
`python -c "import secrets; print(secrets.token_urlsafe(32))"`. Store tokens in
the API host's secret manager or environment, never in source control. To rotate
one user's API token without an authentication outage, first deploy both tokens
for that principal, have the user enter the new token, then remove the old token
from the secret and redeploy. Token values must remain unique across principals.
During the overlap, the JSON value for that principal is an array:

```json
{"researcher":["old-api-token-replace-with-32-plus-chars","new-api-token-replace-with-32-plus-chars"]}
```

Session metadata, query/audit events, evaluations, reviews, and governance records
are persisted in SQLite at `data/biomedrag_state.sqlite3` by default. Override
the path with `BIOMEDRAG_STATE_DB`. The active RAG service, uploaded document
contents/index, and request-scoped repair contexts remain process-local. After
an API restart, restored sessions require provider reconfiguration and uploaded
documents must be uploaded again. Run one API worker until those runtime objects
and cross-worker writes have shared storage. The default CORS origin is
`http://localhost:5173`.
If Vite runs on another port, add its exact origin to
`BIOMEDRAG_CORS_ORIGINS` (for example, `http://127.0.0.1:5174`).

The API limits each principal to 60 requests per minute and 10 generation or
fairness requests per minute by default. Override these with
`BIOMEDRAG_RATE_LIMIT_PER_MINUTE` and
`BIOMEDRAG_QUERY_RATE_LIMIT_PER_MINUTE`. These limits are process-local; for
multi-worker or multi-instance deployments, also enforce shared rate limits at
the gateway or load balancer. Failed API authentication attempts are limited to
20 per client IP per minute, and internal RAGGuard repair requests to 60 per
client IP per minute. Session ownership is tied to the configured principal,
not to the session ID.

### Multi-worker storage decision

Do not run multiple API workers against the current in-memory session and
repair stores. For a multi-worker deployment, use PostgreSQL as the source of
truth for sessions, audit state, and repair idempotency/context; keep uploaded
files in durable object storage and rebuild or share the session indexes from
those files. Use Redis for atomic shared rate-limit counters and any transient
observation queue. Protect stored biomedical query context with access controls,
encryption at rest, and short retention. A shared rate limiter by itself is not
enough: every worker must be able to restore the same session and repair context.

The API exposes `/api/health`, session configuration, document upload, query and
retrieval, answer/retrieval evaluation, audit export, human review, security scan,
fairness screening, and operator risk endpoints. Use OpenAI or Hugging Face by
setting `OPENAI_API_KEY` or `HUGGINGFACE_API_KEY` on the API server instead.

Dashboard sessions, uploaded documents, evaluations, and audit events are held in
API-process memory, matching the prototype's previous session-scoped behavior.
They are not durable and may be lost when the server restarts; use the audit JSON
export to retain a record. PII and prompt-attack checks are heuristic and do not
constitute comprehensive protection, enterprise security certification, or
clinical validation.

### Existing Streamlit dashboard

The Streamlit app remains available during migration and continues to use the same
Python RAG pipeline:

```bash
streamlit run dashboard/app.py
```

It is still the route for Streamlit Cloud deployments documented below.

### Prerequisites

- Python **3.11** (required — 3.14 has LangChain/Pydantic incompatibilities)
- A Cohere API key ([free tier](https://dashboard.cohere.com/api-keys) is sufficient)

### Installation

```bash
# 1. Clone
git clone https://github.com/emekaphilian/Biomedical-RAG.git
cd Biomedical-RAG

# 2. Create virtual environment with Python 3.11
py -3.11 -m venv venv311           # Windows
python3.11 -m venv venv311         # Linux/Mac

# Activate
venv311\Scripts\activate           # Windows
source venv311/bin/activate        # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys (optional — can paste in UI)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and add your keys
```

### Running locally

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

### First-time setup in the app

1. Select **Cohere** as provider, choose **command-r-plus**
2. Paste your Cohere API key → click **Initialise LLM**
3. Upload a PDF (e.g. a clinical guideline or MedQuAD export)
4. Click **Index file(s)**
5. Ask a biomedical question in the chat

---

## Deploying to Streamlit Cloud

### One-time setup

1. Push your code to GitHub (this repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Set:
   - **Repository:** `emekaphilian/Biomedical-RAG`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
5. Click **Advanced settings** → **Secrets** and add:

```toml
cohere_api_key = "your-cohere-api-key-here"
# Optional:
# openai_api_key = "sk-..."
# huggingface_api_key = "hf_..."
```

6. Click **Deploy**

Streamlit Cloud will install dependencies from `requirements.txt` and use Python 3.11 from `runtime.txt` automatically.

### After deployment

- Your app URL will be `https://your-app-name.streamlit.app`
- Share this URL with your supervisors — they can use it without any local setup
- To redeploy after code changes: push to `main` on GitHub, Streamlit redeploys automatically
- To force a cold restart: `share.streamlit.io` → your app → ⋮ → **Reboot app**

### Secrets for Streamlit Cloud

The app reads API keys from `st.secrets` automatically when deployed. The key names must match exactly:

| Provider | Secret key name |
|---|---|
| Cohere | `cohere_api_key` |
| OpenAI | `openai_api_key` |
| Hugging Face | `huggingface_api_key` |

For optional RAGGuard observation delivery, add these server-side secrets to
the deployment that runs queries (or set the matching environment variables on
another host):

| Secret | Purpose |
|---|---|
| `RAGGUARD_URL` | RAGGuard service base URL |
| `RAGGUARD_OBSERVATION_TOKEN` | Server-side bearer credential for observations |
| `RAGGUARD_APPLICATION_ID` | Source application ID (defaults to `biomedicalrag`) |
| `RAGGUARD_ENVIRONMENT` | Deployment environment (defaults to `development`) |

Observations are queued in process memory and delivered in the background. Each
delivery has a four-second budget, up to three attempts for transient failures,
and a bounded queue of 256; a full queue drops new telemetry. The observer sends
redacted queries and operational metadata only. Missing configuration, shutdown,
queue saturation, network errors, and rejected observations do not interrupt RAG
answers. This queue is best effort and is not durable across process restarts.

Observation and repair are separate capabilities. The observation token only
authorizes delivery to RAGGuard. The optional `RAGGUARD_REPAIR_TOKEN` belongs on
the FastAPI deployment that exposes `/internal/ragguard/repairs`; it must be a
dedicated credential matching the one configured in RAGGuard. Repair runs only
after that authenticated endpoint is called and never replaces the answer already
returned to the user. A Streamlit-only deployment can send observations, but it
does not expose the FastAPI repair endpoint. Keep tokens in deployment secrets;
do not commit them.

Repair callbacks are accepted for up to five minutes after the query. Context
contains the original question and recent conversation only while a repair is
pending; expired entries are cleaned on later access or insertion, with at most
256 entries per service. Once a result is recorded, that query context is
cleared. A `promoted` repair result means that a validated candidate scored
better for that correlated request. It does not alter the answer already
returned, session settings, or retrieval behavior for future queries. It is a
request-scoped evaluation, not a persistent retrieval-policy change. Context
remains tied to the worker that handled the query, so repairs are supported only
in a single-worker deployment until session and repair state are shared.

API user tokens support a staged overlap rotation as described above. The
observation and repair credentials are each single configured secrets in this
integration. Rotate those with a coordinated cutover: update the corresponding
credential on both services, redeploy, and verify a synthetic observation and
an authorized repair before resuming automated repair. No live RAGGuard
credentials should be placed in this repository.

---

## Configuration

The React + FastAPI application is the canonical deployment path. Provider defaults
and request controls are defined in `app/api.py`; the active `RAGService` pipeline
settings are defined in `app/services.py` and `rag/pipeline.py`. The legacy
`configs/config.yaml` is not loaded by that path and should not be treated as an
effective runtime configuration. Streamlit remains a separate legacy interface.

The following block is a historical example only; changing it does not configure
the canonical FastAPI service:

```yaml
retrieval:
  chunk_size: 900
  chunk_overlap: 150
  top_k: 5
  mode: hybrid          # hybrid | dense | sparse
  embed_model: all-MiniLM-L6-v2
  reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

generation:
  provider: cohere
  model: command-r-plus
  temperature: 0.3
  max_tokens: 1024

safety:
  confidence_threshold: 0.45
  enable_refusal: true
  refusal_phrases:
    - "I cannot answer this from the available documents."
    - "The provided documents do not contain sufficient evidence."
```

---

## Evaluation Framework

`evaluation/pubmedqa_dataset.py` adapts the locally supplied official PubMedQA
test labels and contexts for biomedical evaluation. The adversarial suite also
covers personalized dosing, unsafe medication changes, urgent symptoms, false
cure claims, and prompt injection. The previously reported 18-query results are
historical and should not be presented as results for these expanded suites.
The checks in this application are screening signals; PubMedQA task accuracy
does not establish clinical accuracy or safety.

Each generated response reports separate retrieval relevance, citation-reference,
lexical claim-support screen, answer-quality, and medical-accuracy states. A
passing lexical screen is not reported as `CLAIM_SUPPORT_VALIDATED`; that state
remains `NOT ESTABLISHED`. Medical accuracy is always reported as
`NOT ESTABLISHED` unless externally assessed through a separate clinical
validation process.

The evaluation framework lives in `evaluation/` and implements:

### Retrieval metrics (Tier 1)

```bash
python evaluation/run_retrieval_eval.py \
    --test_set data/test_set.jsonl \
    --configs dense sparse hybrid
```

| Metric | Target | Current |
|---|---|---|
| Recall@5 | ≥ 0.75 | TBD (Phase 3) |
| Recall@10 | ≥ 0.85 | TBD |
| Precision@5 | ≥ 0.60 | TBD |
| MRR | ≥ 0.65 | TBD |
| Latency p95 | < 300ms | ~198ms ✅ |

### Generation metrics (Tier 2)

| Metric | Target | Current |
|---|---|---|
| Exact Match | ≥ 35% | In progress |
| F1 Score | ≥ 0.55 | In progress |
| Grounding Rate | ≥ 80% | 50% (pre-stabilisation) |
| Hallucination Rate | < 10% | 27.8% (pre-stabilisation) |
| Refusal Accuracy | ≥ 90% | Implemented, measuring |

### Test set integrity

The test set is SHA-256 locked to prevent data leakage. Always run `python build_locked_test_set.py` before any evaluation, and `python evaluation/verify_test_set.py` before each run.

---

## System Design Decisions

### Why hybrid retrieval?

Dense retrieval (FAISS) excels at semantic similarity but misses exact medical terminology. BM25 catches specific drug names, disease codes, and clinical terms that embeddings may not weight highly. Reciprocal Rank Fusion combines both rankings without requiring score normalisation.

### Why Cohere Command-R?

Command-R has a grounded generation mode with native source attribution, making citation accuracy easier to validate. It also has a generous free tier suitable for academic research.

### Why cross-encoder reranking?

Bi-encoder retrieval (FAISS) optimises for recall at the cost of precision. The cross-encoder re-scores candidate chunks using the full query–passage interaction, significantly improving precision@3. The tradeoff is latency (~50–100ms additional), which is acceptable for this use case.

### Why MedQuAD?

MedQuAD provides structured biomedical QA pairs from authoritative NIH sources (MedlinePlus, Cancer.gov, GHR, etc.), making it ideal for a grounded Q&A system. The structured format also makes evaluation easier.

---

## Known Limitations

- **No real-time data:** Corpus is static. Recent clinical guidelines or drug approvals after the MedQuAD snapshot are not covered.
- **Hallucination on adversarial queries:** The system still generates plausible-sounding but unsupported answers for some adversarial queries. Refusal logic is implemented but not yet fully evaluated.
- **Multi-hop reasoning:** Questions requiring synthesis across multiple documents (e.g. "How does obesity contribute to diabetes via insulin resistance?") show lower accuracy than factoid questions.
- **MedQuAD coverage gaps:** MedQuAD covers a broad but not exhaustive set of conditions. Rare diseases and highly specialised clinical topics may return low-confidence results.
- **Not a diagnostic tool:** This system must never be used for clinical decision making. See [Safety & Ethics](#safety--ethics).

---

## Roadmap

### Phase 3 (current sprint)
- [ ] Formal retrieval evaluation (Recall@K, MRR) across three retriever configurations
- [ ] Cross-encoder reranker integration (in progress)
- [ ] Baseline comparisons: plain LLM vs BM25+LLM vs full hybrid RAG
- [ ] REST API: `POST /query`, `GET /health`, `POST /feedback`
- [ ] User evaluation study with medical students

### Phase 4
- [ ] Safety audit and bias analysis
- [ ] "Not a substitute for professional advice" enforcement in every response
- [ ] Dataset bias and limitations documentation
- [ ] Results suitable for ACM Student Project Conference 2026 submission

---

## Safety & Ethics

> **⚠️ Critical:** This system is a research prototype and **must not** be used for clinical diagnosis, treatment decisions, or any form of medical advice.

- All responses are AI-generated from a static biomedical corpus and may be incomplete, outdated, or incorrect
- The system implements refusal logic for queries requesting diagnoses or clinical recommendations
- Users are shown a disclaimer on every session
- Hallucination rate is tracked and minimisation is a core research objective
- The project follows the Pan-Africa Center for AI Ethics guidelines (PACFAIE, 2025)

---

## Development Challenges & Solutions

| Challenge | Solution |
|---|---|
| Cohere v2 API breaking change (`system=` kwarg removed) | Migrated to `messages=[{"role":"system",...}]` format |
| LangChain v0.2+ import restructuring | Updated all imports to `langchain-community` and `langchain-core` |
| Python 3.14 / Pydantic V1 incompatibility | Pinned `python-3.11` in `runtime.txt` |
| HuggingFace model deprecations | Curated working fallback model list |
| Hallucination on adversarial queries | Confidence thresholding + query classification + refusal logic |
| BM25 + FAISS score normalisation | Replaced normalisation with Reciprocal Rank Fusion |

---

## Citation

If you use this system or reference it in academic work, please cite:

```bibtex
@misc{ogbonna2025biomedrag,
  title   = {Biomedical Retrieval-Augmented Generation System for Medical Question Answering},
  author  = {Ogbonna, Emeka Philian},
  year    = {2025},
  url     = {https://github.com/emekaphilian/Biomedical-RAG},
  note    = {MSc Research Project, University of Aveiro}
}
```

---

## Author

**Emeka Philian Ogbonna** — ML/AI Engineer  
📍 Abuja, Nigeria · Remote  
🔗 [linkedin.com/in/emekaogbonna](https://linkedin.com/in/emekaogbonna) · 💻 [github.com/emekaphilian](https://github.com/emekaphilian)

---

*Built with 🧬 for the advancement of accessible biomedical AI.*
