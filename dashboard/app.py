"""
dashboard/app.py  — Biomedical RAG System
Streamlit MVP for supervisor review
Entry point: streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import time
import json
import hashlib
import html
import re
from datetime import datetime, timezone
from dashboard.session_logic import (
    SERVICE_LOADER,
    classify_upload,
    commit_turn,
    configuration_drift,
    get_saved_api_key,
    median_latency,
    review_coverage,
    unique_source_name,
)
from dashboard.rag_audit import audit_answer
from dashboard.responsible_ai import (
    build_counterfactual_prompts,
    collect_risk_signals,
    compare_responses,
    ethical_risk_score as observed_ethical_risk_score,
    fairness_report as build_fairness_report,
)
from dashboard.security_controls import find_pii, guard_question, redact_pii

# ── Page config (must be first) ──────────────────────────────────────────────
st.set_page_config(
    page_title="BioMed RAG — Medical Q&A System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme & CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:          #0B0F1A;
    --bg2:         #111827;
    --bg3:         #1A2235;
    --border:      #1E2D45;
    --accent:      #00C9A7;
    --accent2:     #3B82F6;
    --accent3:     #F59E0B;
    --danger:      #EF4444;
    --text:        #E2E8F0;
    --muted:       #64748B;
    --mono:        'IBM Plex Mono', monospace;
    --sans:        'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
}

/* Keep Streamlit's native sidebar toggle visible while hiding unrelated chrome. */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] {
    visibility: visible;
    background: transparent;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    color: var(--accent) !important;
}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
    color: var(--accent) !important;
}
.stDeployButton { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: var(--mono);
    color: var(--accent);
    letter-spacing: 0.05em;
    font-size: 1.1rem;
    margin-bottom: 2px;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--muted);
    font-size: 0.82em;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Buttons */
.stButton > button {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 0.82em;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(0,201,167,0.06);
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: var(--accent);
    border-color: var(--accent);
    color: #000;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    background: #00b396;
    color: #000;
}

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: var(--sans) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}

/* Chat bubbles */
.user-msg {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px 12px 2px 12px;
    padding: 13px 16px;
    margin: 10px 0 6px;
    font-size: 0.95em;
    line-height: 1.65;
    color: var(--text);
    position: relative;
}
.user-msg::before {
    content: "YOU";
    font-family: var(--mono);
    font-size: 0.65em;
    color: var(--muted);
    display: block;
    margin-bottom: 5px;
    letter-spacing: 0.1em;
}
.bot-msg {
    background: linear-gradient(135deg, rgba(0,201,167,0.05), rgba(59,130,246,0.04));
    border: 1px solid rgba(0,201,167,0.2);
    border-left: 3px solid var(--accent);
    border-radius: 2px 12px 12px 12px;
    padding: 13px 18px;
    margin: 6px 0 10px;
    font-size: 0.95em;
    line-height: 1.75;
    color: var(--text);
    position: relative;
}
.bot-msg::before {
    content: "BIOMED RAG";
    font-family: var(--mono);
    font-size: 0.65em;
    color: var(--accent);
    display: block;
    margin-bottom: 5px;
    letter-spacing: 0.1em;
}
.error-msg {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 3px solid var(--danger);
    border-radius: 2px 12px 12px 12px;
    padding: 13px 18px;
    margin: 6px 0 10px;
    font-size: 0.9em;
    color: #FCA5A5;
    font-family: var(--mono);
}

/* Source cards */
.src-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.82em;
    color: var(--muted);
    line-height: 1.6;
}
.src-card strong { color: var(--accent2); font-family: var(--mono); }
.src-excerpt { color: #94A3B8; font-style: italic; margin-top: 4px; }

/* Status pills */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: var(--mono);
    font-size: 0.72em;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.pill-green  { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
.pill-red    { background: rgba(239,68,68,0.12);  color: #FCA5A5; border: 1px solid rgba(239,68,68,0.3); }
.pill-amber  { background: rgba(245,158,11,0.12); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); }
.pill-blue   { background: rgba(59,130,246,0.12); color: #93C5FD; border: 1px solid rgba(59,130,246,0.3); }

/* Metric cards */
.m-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 6px 0; }
.m-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 10px;
    text-align: center;
}
.m-val { font-family: var(--mono); font-size: 1.5em; font-weight: 500; color: var(--accent); }
.m-lbl { font-size: 0.72em; color: var(--muted); margin-top: 3px; text-transform: uppercase; letter-spacing: 0.06em; }

/* Header band */
.hdr-band {
    background: linear-gradient(90deg, rgba(0,201,167,0.08), rgba(59,130,246,0.05), transparent);
    border: 1px solid rgba(0,201,167,0.15);
    border-radius: 10px;
    padding: 20px 28px;
    margin-bottom: 20px;
}
.hdr-title {
    font-family: var(--mono);
    font-size: 1.55em;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}
.hdr-sub {
    color: var(--muted);
    font-size: 0.88em;
    line-height: 1.5;
}
.accent-dot { color: var(--accent); }

/* Safety disclaimer */
.disclaimer {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8em;
    color: #FCD34D;
    line-height: 1.5;
    margin: 8px 0;
}

/* Phase badge */
.phase-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 6px;
    padding: 4px 10px;
    font-family: var(--mono);
    font-size: 0.75em;
    color: #93C5FD;
}

/* Progress bars */
.prog-wrap { margin: 6px 0; }
.prog-label { display: flex; justify-content: space-between; font-size: 0.78em; color: var(--muted); margin-bottom: 4px; }
.prog-track { height: 5px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 3px; transition: width 0.5s; }

/* Radio */
.stRadio > div { gap: 4px !important; }
.stRadio label { color: var(--text) !important; font-size: 0.88em !important; }
[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg3);
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 8px;
}

/* Slider */
.stSlider > div > div > div { background: var(--border) !important; }
.stSlider [data-testid="stThumbValue"] { color: var(--accent) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.82em !important;
}

/* Chat input */
.stChatInput textarea {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
.stChatInput textarea:focus {
    border-color: var(--accent) !important;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 50px 30px;
    color: var(--muted);
}
.empty-icon { font-size: 3.5em; margin-bottom: 16px; }
.empty-title { font-family: var(--mono); font-size: 0.95em; color: #94A3B8; margin-bottom: 8px; }
.empty-hint { font-size: 0.82em; line-height: 1.7; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* Info/warning overrides */
.stAlert { border-radius: 8px !important; }
a {
    text-decoration: none;
    color: #464feb;
}
tr th, tr td {
    border: 1px solid var(--border);
}
tr th {
    background-color: var(--bg3);
    color: var(--text);
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "rag_service":      None,
        "chat_history":     [],   # clean LLM format
        "display_history":  [],   # {role, content, sources, latency_ms}
        "docs_indexed":     [],
        "document_records": [],
        "audit_events":     [],
        "evaluation_runs":  [],
        "answer_quality_reviews": [],
        "risk_register":    [],
        "responsible_ai_assessments": [],
        "fairness_reports": [],
        "llm_ready":        False,
        "provider":         "cohere",
        "model":            "command-r-plus",
        "top_k":            5,
        "retrieval_mode":   "hybrid",
        "query_class":      "factoid",
        "settings_open":    False,
        "total_queries":    0,
        "avg_latency_ms":   0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


def _configuration_snapshot(overrides: dict = None) -> dict:
    from dashboard.audit_utils import configuration_version

    settings = {
        "provider": st.session_state.provider,
        "model": st.session_state.model,
        "retrieval_mode": st.session_state.retrieval_mode,
        "top_k": st.session_state.top_k,
        "documents": [
            {
                "source": record["source"],
                "file_type": record["file_type"],
                "size_bytes": record["size_bytes"],
                "chunks": record["chunks"],
                "sha256": record.get("sha256", ""),
            }
            for record in st.session_state.document_records
        ],
    }
    if overrides:
        settings.update(overrides)
    return {
        "version": configuration_version(settings),
        "settings": settings,
    }


# ── Lazy import of RAG service ────────────────────────────────────────────────
def _load_service(provider: str, model: str, api_key: str):
    return SERVICE_LOADER.load(provider, model, api_key)


def _get_saved_api_key(provider: str) -> str:
    """Read a provider key from Streamlit secrets or the environment."""
    return get_saved_api_key(provider, st.secrets)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 BioMed RAG")
    st.markdown("*Biomedical Q&A System — MSc Research MVP*")
    st.divider()

    # ── LLM Config ────────────────────────────────────────────────────────
    st.markdown("### 🤖 LLM Provider")

    provider_map = {
        "cohere":       "Cohere (Command-R)",
        "openai":       "OpenAI (GPT-4)",
        "huggingface":  "Hugging Face",
    }
    model_map = {
        "cohere":       ["command-r-plus", "command-r", "command-nightly"],
        "openai":       ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "huggingface":  ["microsoft/DialoGPT-medium", "google/flan-t5-large", "gpt2"],
    }

    prov_labels = list(provider_map.values())
    prov_keys   = list(provider_map.keys())
    prov_idx    = prov_keys.index(st.session_state.provider)
    sel_prov    = st.selectbox("Provider", prov_labels, index=prov_idx)
    new_prov    = prov_keys[prov_labels.index(sel_prov)]
    if new_prov != st.session_state.provider:
        st.session_state.provider  = new_prov
        st.session_state.model     = model_map[new_prov][0]
        st.session_state.llm_ready = False

    models  = model_map[st.session_state.provider]
    cur_idx = models.index(st.session_state.model) if st.session_state.model in models else 0
    st.session_state.model = st.selectbox("Model", models, index=cur_idx)

    # API key
    api_key_val = _get_saved_api_key(st.session_state.provider)

    if api_key_val:
        st.markdown('<span class="pill pill-green">✓ API key loaded from configuration</span>', unsafe_allow_html=True)
    else:
        st.caption("No saved API key found. Enter one below to initialize the provider.")
        api_key_val = st.text_input(
            "API Key",
            type="password",
            placeholder=f"Enter your {sel_prov} API key",
        )

    active_service = st.session_state.rag_service
    pending_model_change = bool(configuration_drift(
        {"provider": active_service.provider, "model": active_service.model}
        if active_service else None,
        {"provider": st.session_state.provider, "model": st.session_state.model},
    ))
    if st.button(
        "⚡ Re-initialise LLM" if pending_model_change else "⚡ Initialise LLM",
        use_container_width=True,
        type="primary",
    ):
        if not api_key_val:
            st.error("API key required.")
        else:
            with st.spinner("Connecting…"):
                svc, err = _load_service(
                    st.session_state.provider,
                    st.session_state.model,
                    api_key_val,
                )
            if err:
                st.error(f"Failed: {err}")
            else:
                st.session_state.rag_service = svc
                st.session_state.llm_ready   = True
                st.success("LLM ready!")

    if pending_model_change:
        st.warning(
            f"Model changed to {st.session_state.model}. Re-initialise to apply; "
            f"queries still use {active_service.model}."
        )

    st.divider()

    # ── Document Upload ────────────────────────────────────────────────────
    st.markdown("### 📄 Knowledge Base")

    uploaded = st.file_uploader(
        "Upload biomedical documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="MedQuAD exports, clinical guidelines, research papers, textbooks",
    )

    if uploaded:
        existing_records = {
            record["filename"]: record
            for record in st.session_state.document_records
        }
        uploaded_with_hashes = [
            (file, hashlib.sha256(file.getvalue()).hexdigest())
            for file in uploaded
        ]
        new_files = [
            (file, file_hash)
            for file, file_hash in uploaded_with_hashes
            if classify_upload(file.name, file_hash, st.session_state.document_records) != "duplicate"
        ]
        changed_names = [
            file.name for file, _ in new_files
            if file.name in existing_records
        ]
        if changed_names:
            st.warning(
                "Updated content detected for: "
                + ", ".join(changed_names)
                + ". Indexing will retain a distinct content-hash provenance key."
            )
        if new_files:
            if st.button(
                f"📥 Index or re-index {len(new_files)} file(s)",
                use_container_width=True,
            ):
                svc = st.session_state.rag_service
                if not svc:
                    st.warning("Initialise the LLM first.")
                else:
                    bar = st.progress(0)
                    total_chunks = 0
                    for i, (f, file_hash) in enumerate(new_files):
                        with st.spinner(f"Processing {f.name}…"):
                            try:
                                previous_chunks = svc.total_chunks
                                file_bytes = f.getvalue()
                                source_name = unique_source_name(
                                    f.name, file_hash, st.session_state.document_records
                                )
                                n = svc.ingest(file_bytes, source_name)
                                previous_record = existing_records.get(f.name)
                                file_chunks = (
                                    n - previous_chunks
                                    + (previous_record.get("chunks", 0) if previous_record else 0)
                                )
                                total_chunks += file_chunks
                                if f.name not in st.session_state.docs_indexed:
                                    st.session_state.docs_indexed.append(f.name)
                                new_record = {
                                    "filename": f.name,
                                    "file_type": f.name.rsplit(".", 1)[-1].upper(),
                                    "size_bytes": f.size,
                                    "chunks": file_chunks,
                                    "sha256": file_hash,
                                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                                    "source": source_name,
                                }
                                if previous_record:
                                    # Keep prior versions in the session manifest for lineage.
                                    new_record["filename"] = f.name
                                    new_record["source"] = source_name
                                    st.session_state.document_records.append(new_record)
                                    existing_records[f.name] = new_record
                                else:
                                    st.session_state.document_records.append(new_record)
                                    existing_records[f.name] = new_record
                                st.session_state.audit_events.append({
                                    "event": "document_indexed",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "filename": f.name,
                                    "chunks": file_chunks,
                                    "provider": svc.provider,
                                    "model": svc.model,
                                    "configuration": _configuration_snapshot({
                                        "provider": svc.provider,
                                        "model": svc.model,
                                    }),
                                })
                            except Exception as e:
                                st.error(f"{f.name}: {e}")
                                st.session_state.audit_events.append({
                                    "event": "document_indexing_failed",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "filename": f.name,
                                    "error": str(e),
                                    "provider": svc.provider,
                                    "model": svc.model,
                                    "configuration": _configuration_snapshot({
                                        "provider": svc.provider,
                                        "model": svc.model,
                                    }),
                                })
                        bar.progress((i + 1) / len(new_files))
                    bar.empty()
                    st.success(f"Indexed {total_chunks:,} chunks from {len(new_files)} file(s)")
        else:
            st.info("All files already indexed.")

    if st.session_state.document_records:
        for record in st.session_state.document_records:
            st.markdown(
                f"<div style='font-size:0.8em;padding:3px 0'>📄 "
                f"{html.escape(record['filename'])} "
                f"<span style='float:right;color:var(--accent)'>"
                f"{record['chunks']:,} chunks</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Retrieval controls ─────────────────────────────────────────────────
    st.markdown("### Retrieval")
    st.session_state.retrieval_mode = st.radio(
        "Retrieval mode",
        ["hybrid", "dense", "sparse"],
        index=["hybrid", "dense", "sparse"].index(st.session_state.retrieval_mode),
        horizontal=True,
        label_visibility="collapsed",
    )
    with st.expander("Advanced retrieval settings"):
        st.session_state.top_k = st.slider(
            "Top-K chunks", min_value=3, max_value=15, value=st.session_state.top_k,
            help="Number of chunks retrieved before reranking"
        )
    method = {
        "hybrid": "FAISS + BM25 + Reciprocal Rank Fusion",
        "dense": "FAISS vector similarity",
        "sparse": "BM25 keyword ranking",
    }[st.session_state.retrieval_mode]
    st.caption(f"Top-K {st.session_state.top_k} · {method}")

    st.divider()

    # ── Safety disclaimer ──────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Research Use Only.</strong><br>
        This system is an MSc research prototype. All responses are AI-generated
        from biomedical literature and must not be used for clinical diagnosis,
        treatment decisions, or medical advice. Always consult a qualified
        healthcare professional.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        SERVICE_LOADER.clear()
        st.cache_resource.clear()
        st.rerun()


# ── Main dashboard ────────────────────────────────────────────────────────────
svc = st.session_state.rag_service
llm_ok = svc is not None
docs_ok = bool(st.session_state.docs_indexed)
n_chunks = svc.total_chunks if svc else 0
query_events = [
    event for event in st.session_state.audit_events
    if event.get("event") == "query"
]
successful_queries = [event for event in query_events if event.get("success")]
failed_queries = [event for event in query_events if not event.get("success")]
grounded_queries = [
    event for event in successful_queries if event.get("is_grounded") is True
]
warning_count = sum(bool(event.get("warnings")) for event in successful_queries)
latency_samples = [
    event["latency_ms"]
    for event in successful_queries
    if isinstance(event.get("latency_ms"), (int, float))
]
reviewed_queries, _ = review_coverage(
    successful_queries, st.session_state.answer_quality_reviews
)
configuration = _configuration_snapshot(
    {
        "provider": svc.provider,
        "model": svc.model,
    }
    if svc else None
)

title_col, config_col, export_col = st.columns([3.5, 1.2, 1.1])
with title_col:
    st.markdown("### BioMed RAG")
    st.caption("Research & auditor workspace · inspect evidence, review answers, track safety")
with config_col:
    st.markdown(
        f"<div style='padding-top:14px;text-align:center'>"
        f"<span class='pill pill-green'>{configuration['version']}</span></div>",
        unsafe_allow_html=True,
    )
with export_col:
    st.download_button(
        "Export audit JSON",
        data=json.dumps(
            {
                "session_only": True,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "current_configuration": configuration,
                "documents": st.session_state.document_records,
                "events": st.session_state.audit_events,
                "evaluations": st.session_state.evaluation_runs,
                "answer_quality_reviews": st.session_state.answer_quality_reviews,
                "responsible_ai_assessments": st.session_state.responsible_ai_assessments,
                "fairness_reports": st.session_state.fairness_reports,
                "governance_findings": st.session_state.risk_register,
            },
            indent=2,
            ensure_ascii=False,
        ),
        file_name="biomedical-rag-session-audit.json",
        mime="application/json",
        use_container_width=True,
        key="playground_audit_export",
    )

summary_cols = st.columns(4)
summary_cols[0].metric("Successful queries", len(successful_queries))
summary_cols[1].metric(
    "Grounded",
    f"{len(grounded_queries)} / {len(successful_queries)}",
)
summary_cols[2].metric(
    "Median latency",
    f"{median_latency(latency_samples) / 1000:.1f} s"
    if median_latency(latency_samples) is not None else "—",
)
summary_cols[3].metric(
    "Reviewed by human",
    f"{reviewed_queries} / {len(successful_queries)}",
)

st.markdown(
    f"<span class='pill {'pill-green' if llm_ok else 'pill-red'}'>"
    f"{'Connected' if llm_ok else 'LLM not connected'}</span> &nbsp; "
    f"<span class='pill {'pill-green' if docs_ok else 'pill-blue'}'>"
    f"{len(st.session_state.document_records)} documents · {n_chunks:,} chunks</span> &nbsp; "
    f"<span class='pill pill-blue'>{st.session_state.retrieval_mode.upper()} · "
    f"k={st.session_state.top_k}</span> &nbsp; "
    f"<span class='phase-badge'>RESEARCH USE ONLY · SESSION-SCOPED</span>",
    unsafe_allow_html=True,
)
st.divider()

overview_tab, knowledge_tab, playground_tab, evaluation_tab, governance_tab, audit_tab = st.tabs(
    [
        "Overview",
        "Knowledge",
        "Playground",
        "Evaluation",
        "Governance",
        "Audit",
    ]
)

with overview_tab:
    st.subheader("System overview")
    st.caption(
        "Live values reflect this browser session only. The dashboard does not "
        "claim durable corpus storage, clinical validation, or production monitoring."
    )
    metric_cols = st.columns(5)
    metric_cols[0].metric("Indexed documents", len(st.session_state.docs_indexed))
    metric_cols[1].metric("Indexed chunks", f"{n_chunks:,}")
    metric_cols[2].metric("Successful queries", len(successful_queries))
    metric_cols[3].metric(
        "Mean query latency",
        f"{st.session_state.avg_latency_ms:.0f} ms"
        if successful_queries else "—",
    )
    metric_cols[4].metric("Grounded responses", len(grounded_queries))

    health_col, activity_col = st.columns([1, 2])
    with health_col:
        st.markdown("#### System health")
        st.write("LLM connection:", "Ready" if llm_ok else "Not connected")
        st.write("Retrieval index:", "Ready" if n_chunks else "No indexed chunks")
        st.write("Active provider:", svc.provider if svc else "Not connected")
        st.write("Active model:", svc.model if svc else "Not connected")
        if pending_model_change:
            st.caption(
                f"Pending selection: {st.session_state.provider} · "
                f"{st.session_state.model}"
            )
        st.write("Retrieval mode:", st.session_state.retrieval_mode)
        st.write("Failed queries:", len(failed_queries))
    with activity_col:
        st.markdown("#### Query latency")
        latencies = [
            event["latency_ms"]
            for event in successful_queries
            if isinstance(event.get("latency_ms"), (int, float))
        ]
        if latencies:
            st.line_chart({"Latency (ms)": latencies}, height=220)
        else:
            st.info("Latency will appear after a successful playground query.")

    st.markdown("#### Recent session activity")
    recent_events = list(reversed(st.session_state.audit_events[-10:]))
    if recent_events:
        st.dataframe(recent_events, use_container_width=True, hide_index=True)
    else:
        st.info("No ingestion or query activity has been recorded in this session.")

with knowledge_tab:
    st.subheader("Knowledge Base")
    st.info(
        "Documents are held in the active RAG service for this session. They are "
        "not durably stored by this dashboard; use the session reset only when "
        "you are ready to discard the in-memory index."
    )
    kb_metrics = st.columns(3)
    kb_metrics[0].metric("Documents", len(st.session_state.document_records))
    kb_metrics[1].metric("Chunks", f"{n_chunks:,}")
    kb_metrics[2].metric(
        "Indexed source bytes",
        f"{sum(item['size_bytes'] for item in st.session_state.document_records):,}",
    )
    if st.session_state.document_records:
        st.markdown("#### Document metadata and lineage")
        st.dataframe(
            st.session_state.document_records,
            use_container_width=True,
            hide_index=True,
            column_config={
                "filename": "Source document",
                "file_type": "Type",
                "size_bytes": st.column_config.NumberColumn("Size (bytes)", format="%d"),
                "chunks": st.column_config.NumberColumn("Indexed chunks", format="%d"),
                "sha256": "SHA-256",
                "indexed_at": "Indexed (UTC)",
                "source": "Provenance key",
            },
        )
        st.caption(
            "Per-file chunk counts are calculated from the index total before and "
            "after each upload; this session manifest is not persisted."
        )
    else:
        st.warning("No documents indexed. Upload PDF, DOCX, or TXT files from the sidebar.")

with playground_tab:
    st.subheader("RAG Playground")
    st.caption(
        "Ask, inspect the evidence, then rate the answer. Answers use the checked "
        "RAG pipeline and include a research-use disclaimer."
    )
    chat_col, evidence_col = st.columns([1.55, 0.95], gap="large")
    assistant_turns = [
        turn for turn in st.session_state.display_history
        if turn["role"] == "assistant"
    ]
    latest_answer = assistant_turns[-1] if assistant_turns else None

    with chat_col:
        if not st.session_state.display_history:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🔬</div>
                <div class="empty-title">READY FOR BIOMEDICAL QUERIES</div>
                <div class="empty-hint">
                    Connect an LLM and index documents in the sidebar to begin.
                    <br><br><strong>Example:</strong>
                    "How does Parkinson's disease affect dopamine?"
                </div>
            </div>
            """, unsafe_allow_html=True)

        for turn in st.session_state.display_history:
            if turn["role"] == "user":
                with st.chat_message("user"):
                    st.write(turn["content"])
            elif turn["role"] == "error":
                with st.chat_message("assistant"):
                    st.error(turn["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(turn["content"])
                    details = turn.get("details", {})
                    status = "Grounded" if details.get("is_grounded") else "Not grounded"
                    st.caption(
                        f"{status} · {len(turn.get('sources', []))} sources · "
                        f"{turn.get('provider', st.session_state.provider)} · "
                        f"{turn.get('retrieval_mode', st.session_state.retrieval_mode)} · "
                        f"k={turn.get('top_k', st.session_state.top_k)} · "
                        f"{turn.get('latency_ms', 0) / 1000:.1f} s"
                    )

        with st.form("playground_query_form", clear_on_submit=True):
            input_col, send_col = st.columns([5, 1])
            with input_col:
                question = st.text_input(
                    "Ask a biomedical question",
                    placeholder="Ask a biomedical question…",
                    label_visibility="collapsed",
                    disabled=not (llm_ok and docs_ok),
                )
            with send_col:
                submitted = st.form_submit_button(
                    "Send",
                    type="primary",
                    use_container_width=True,
                    disabled=not (llm_ok and docs_ok),
                )
        if not llm_ok:
            st.caption("Connect an LLM using the sidebar controls.")
        elif not docs_ok:
            st.caption("Index at least one document using the sidebar controls.")

        if submitted and question.strip():
            guard = guard_question(question)
            safe_question = guard.sanitized
            st.session_state.display_history.append({"role": "user", "content": safe_question})
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            active_configuration = _configuration_snapshot({
                "provider": svc.provider,
                "model": svc.model,
            })
            try:
                if guard.blocked:
                    raise ValueError(
                        "This query was blocked by the heuristic prompt-attack screen. "
                        "Rephrase it as a biomedical question."
                    )
                stream, sources, details = svc.query_with_details(
                    question=safe_question,
                    chat_history=st.session_state.chat_history,
                    top_k=st.session_state.top_k,
                    mode=st.session_state.retrieval_mode,
                )
                answer = "".join(stream)
                answer_pii_count = len(find_pii(answer))
                answer = redact_pii(answer)[0]
                answer_audit = audit_answer(
                    answer, [str(source.get("text", "")) for source in sources]
                )
                latency_ms = (time.perf_counter() - started) * 1000
                st.session_state.total_queries += 1
                total = st.session_state.total_queries
                previous_average = st.session_state.avg_latency_ms
                st.session_state.avg_latency_ms = (
                    previous_average + (latency_ms - previous_average) / total
                )
                st.session_state.display_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "latency_ms": latency_ms,
                    "details": details,
                    "provider": svc.provider,
                    "model": svc.model,
                    "retrieval_mode": st.session_state.retrieval_mode,
                    "top_k": st.session_state.top_k,
                })
                st.session_state.selected_evidence_rank = 1
                st.session_state.chat_history = commit_turn(
                    st.session_state.chat_history, safe_question, answer
                )
                st.session_state.audit_events.append({
                    "event": "query",
                    "id": started_at,
                    "timestamp": started_at,
                    "success": True,
                    "query": safe_question,
                    "answer": answer,
                    "sources": [source.get("source", "Unknown") for source in sources],
                    "source_count": len(sources),
                    "is_grounded": details.get("is_grounded"),
                    "warnings": details.get("warnings", []),
                    "audit": answer_audit,
                    "security": {
                        "pii_redacted": len(guard.pii),
                        "attack_score": guard.attack.score,
                        "attack_patterns": guard.attack.patterns,
                        "attack_blocked": guard.blocked,
                        "pii_exposed_in_answer": answer_pii_count,
                    },
                    "latency_ms": round(latency_ms, 2),
                    "retrieval_mode": st.session_state.retrieval_mode,
                    "top_k": st.session_state.top_k,
                    "provider": svc.provider,
                    "model": svc.model,
                    "configuration": active_configuration,
                })
            except Exception as error:
                latency_ms = (time.perf_counter() - started) * 1000
                st.session_state.display_history.append({
                    "role": "error",
                    "content": str(error),
                })
                st.session_state.audit_events.append({
                    "event": "query",
                    "id": started_at,
                    "timestamp": started_at,
                    "success": False,
                    "query": safe_question,
                    "error": str(error),
                    "security": {
                        "pii_redacted": len(guard.pii),
                        "attack_score": guard.attack.score,
                        "attack_patterns": guard.attack.patterns,
                        "attack_blocked": guard.blocked,
                    },
                    "latency_ms": round(latency_ms, 2),
                    "retrieval_mode": st.session_state.retrieval_mode,
                    "top_k": st.session_state.top_k,
                    "provider": svc.provider,
                    "model": svc.model,
                    "configuration": active_configuration,
                })
            st.rerun()

    with evidence_col:
        st.markdown("#### Evidence · click a citation to inspect")
        evidence = latest_answer.get("sources", []) if latest_answer else []
        cited_indices = sorted({
            int(match)
            for match in re.findall(
                r"\[(?:Evidence\s+)?(\d+)\]",
                latest_answer["content"],
                flags=re.IGNORECASE,
            )
            if 1 <= int(match) <= len(evidence)
        }) if latest_answer else []
        if cited_indices:
            citation_cols = st.columns(min(4, len(cited_indices)))
            for rank in cited_indices:
                with citation_cols[(rank - 1) % len(citation_cols)]:
                    if st.button(
                        f"[{rank}]",
                        key=f"citation_jump_{rank}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_evidence_rank = rank
        if evidence:
            selected_rank = st.session_state.get("selected_evidence_rank", 1)
            for rank, source in enumerate(evidence, 1):
                score = source.get("rerank_score", source.get("score", 0)) or 0
                title = source.get("file", source.get("source", "Unknown source"))
                page = source.get("page") or source.get("page_number")
                if page:
                    title = f"{title} · p.{page}"
                excerpt = source.get("text", "") or "No excerpt was returned for this source."
                border_color = "#00C9A7" if rank == selected_rank else "#1E2D45"
                st.markdown(
                    f"<div class='src-card' style='border-color:{border_color}'>"
                    f"<strong>[{rank}] {html.escape(str(title))}</strong>"
                    f"<span style='float:right'>{score:.3f}</span>"
                    f"<div class='src-excerpt'>{html.escape(str(excerpt))}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Retrieved evidence will appear here after a query.")

        st.markdown("#### Quick review")
        if latest_answer:
            latest_event = next(
                (
                    event for event in reversed(st.session_state.audit_events)
                    if event.get("event") == "query"
                    and event.get("success")
                    and event.get("answer") == latest_answer["content"]
                ),
                None,
            )
            latest_event_id = str(
                latest_event.get("id", latest_event.get("timestamp"))
                if latest_event else ""
            )
            already_reviewed = latest_event_id in reviewed_event_ids
            if already_reviewed:
                st.success("Reviewed by a human")
            else:
                with st.form("playground_quick_review_form", clear_on_submit=True):
                    quick_ratings = {
                        "factual_correctness": st.select_slider(
                            "Factual correctness", options=[1, 2, 3, 4, 5], value=3,
                        ),
                        "hallucination_concern": st.select_slider(
                            "Hallucination concern (1 = none, 5 = high)",
                            options=[1, 2, 3, 4, 5],
                            value=1,
                        ),
                        "evidence_support": st.select_slider(
                            "Evidence support", options=[1, 2, 3, 4, 5], value=3,
                        ),
                        "faithfulness": st.select_slider(
                            "Faithfulness to evidence",
                            options=[1, 2, 3, 4, 5],
                            value=3,
                        ),
                        "explainability": st.select_slider(
                            "Explainability",
                            options=[1, 2, 3, 4, 5],
                            value=3,
                        ),
                        "completeness": st.select_slider(
                            "Completeness", options=[1, 2, 3, 4, 5], value=3,
                        ),
                        "safety": st.select_slider(
                            "Safety", options=[1, 2, 3, 4, 5], value=3,
                        ),
                    }
                    evaluator = st.text_input(
                        "Reviewer (optional)", key="quick_review_evaluator",
                    )
                    save_review = st.form_submit_button("Save human review")
                if save_review and latest_event:
                    review = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "query_event_id": latest_event_id,
                        "query": latest_event["query"],
                        "answer": latest_event["answer"],
                        "sources": latest_answer.get("sources", []),
                        "evaluator": evaluator.strip(),
                        "ratings": quick_ratings,
                        "notes": "",
                        "configuration": latest_event.get("configuration"),
                    }
                    st.session_state.answer_quality_reviews.append(review)
                    st.session_state.audit_events.append({
                        "event": "answer_quality_review",
                        "timestamp": review["timestamp"],
                        "query_event_id": latest_event_id,
                        "query": review["query"],
                        "ratings": quick_ratings,
                        "configuration": review["configuration"],
                    })
                    st.rerun()

with evaluation_tab:
    st.subheader("Evaluation Lab")
    st.markdown("#### RAG audit framework")
    audit_dimension_cols = st.columns(4)
    audit_dimension_cols[0].metric("Hallucinations", "Human review")
    audit_dimension_cols[1].metric("Faithfulness", "Human review")
    audit_dimension_cols[2].metric("Explainability", "Human review")
    audit_dimension_cols[3].metric("Retrieval accuracy", "Labeled metrics")
    st.warning(
        "Retrieval scores use manually labeled relevant sources. Answer-quality "
        "scores are human rubric judgments, not automated validation, clinical "
        "quality measurements, BLEU, or ROUGE."
    )
    from dashboard.audit_utils import (
        QUALITY_DIMENSIONS,
        summarize_answer_quality_reviews,
    )

    st.markdown("#### Human answer-quality review")
    if successful_queries:
        query_options = {
            f"{event['timestamp']} · {event['query'][:70]}": event
            for event in successful_queries
        }
        selected_query_label = st.selectbox(
            "Select a successful answer to review",
            list(query_options),
            key="quality_query_selection",
        )
        selected_query = query_options[selected_query_label]
        st.markdown(f"**Question:** {selected_query['query']}")
        st.markdown(f"**Answer:** {selected_query['answer']}")
        st.caption(
            "Rate each dimension from 1 (poor) to 5 (strong). Ratings are subjective "
            "research annotations and do not establish medical correctness."
        )
        with st.expander("Evidence returned for this answer"):
            st.write(", ".join(selected_query.get("sources", [])) or "No sources returned")

        with st.form("answer_quality_review_form", clear_on_submit=True):
            evaluator = st.text_input("Evaluator name or identifier (optional)")
            ratings = {
                "factual_correctness": st.select_slider(
                    "Factual correctness",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    help="1 = poor, 5 = strong; reviewer judgment only.",
                ),
                "hallucination_concern": st.select_slider(
                    "Hallucination concern",
                    options=[1, 2, 3, 4, 5],
                    value=1,
                    help="1 = no unsupported claims observed; 5 = many claims appear unsupported.",
                ),
                "evidence_support": st.select_slider(
                    "Support from retrieved evidence",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                ),
                "faithfulness": st.select_slider(
                    "Faithfulness to retrieved evidence",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                ),
                "explainability": st.select_slider(
                    "Explainability",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                ),
                "completeness": st.select_slider(
                    "Completeness for the question",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                ),
                "safety": st.select_slider(
                    "Safety and appropriate limitations",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                ),
            }
            quality_notes = st.text_area("Reviewer notes")
            submit_review = st.form_submit_button("Save human review")
        if submit_review:
            review = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_event_id": selected_query.get("id", selected_query["timestamp"]),
                "query": selected_query["query"],
                "answer": selected_query["answer"],
                "sources": selected_query.get("sources", []),
                "evaluator": evaluator.strip(),
                "ratings": ratings,
                "notes": quality_notes.strip(),
                "configuration": selected_query.get(
                    "configuration",
                    _configuration_snapshot(),
                ),
            }
            st.session_state.answer_quality_reviews.append(review)
            st.session_state.audit_events.append({
                "event": "answer_quality_review",
                "timestamp": review["timestamp"],
                "query_event_id": review["query_event_id"],
                "query": review["query"],
                "ratings": ratings,
                "configuration": review["configuration"],
            })
            st.success("Human quality review saved to this session.")

        quality_summary = summarize_answer_quality_reviews(
            st.session_state.answer_quality_reviews
        )
        if quality_summary["review_count"]:
            st.caption(
                f"{quality_summary['review_count']} human review(s); means are "
                "descriptive summaries, not model validation."
            )
            labels = {
                "factual_correctness": "Factual correctness",
                "hallucination_concern": "Hallucination concern",
                "evidence_support": "Evidence support",
                "faithfulness": "Faithfulness",
                "explainability": "Explainability",
                "completeness": "Completeness",
                "safety": "Safety",
            }
            for start in range(0, len(QUALITY_DIMENSIONS), 4):
                quality_cols = st.columns(4)
                for column, dimension in zip(
                    quality_cols,
                    QUALITY_DIMENSIONS[start:start + 4],
                ):
                    mean = quality_summary[dimension]
                    if mean is not None:
                        value = f"{mean:.2f}/5"
                        if dimension == "hallucination_concern":
                            value += " concern"
                    else:
                        value = "—"
                    column.metric(labels[dimension], value)
            st.caption(
                "Retrieval accuracy is assessed separately using manually labeled "
                "relevant sources and Recall@k, Precision@k, MRR, and NDCG@k."
            )
            st.dataframe(
                [
                    {
                        "timestamp": review["timestamp"],
                        "query": review["query"],
                        "evaluator": review["evaluator"] or "—",
                        **review["ratings"],
                        "notes": review["notes"],
                    }
                    for review in reversed(st.session_state.answer_quality_reviews)
                ],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Run a successful playground query before reviewing answer quality.")

    st.markdown("#### Labeled retrieval evaluation")
    st.caption(
        "Each run stores the query, relevance labels, ranked source excerpts, "
        "retrieval settings, and configuration fingerprint for reproducibility."
    )
    if not svc or not st.session_state.document_records:
        st.info("Connect the service and index documents before running retrieval evaluation.")
    else:
        source_names = sorted({
            record["source"] for record in st.session_state.document_records
        })
        with st.form("retrieval_evaluation_form"):
            eval_query = st.text_input("Evaluation query")
            relevant_sources = st.multiselect(
                "Known relevant source documents",
                source_names,
                help="Labels are used only for this manual evaluation run.",
            )
            eval_k = st.number_input(
                "Evaluate at k",
                min_value=1,
                max_value=20,
                value=min(st.session_state.top_k, 20),
            )
            eval_mode = st.selectbox(
                "Retrieval mode",
                ["hybrid", "dense", "sparse"],
                index=["hybrid", "dense", "sparse"].index(
                    st.session_state.retrieval_mode
                ),
            )
            run_evaluation = st.form_submit_button("Run retrieval evaluation")

        if run_evaluation:
            if not eval_query.strip():
                st.error("Enter an evaluation query.")
            elif not relevant_sources:
                st.error("Select at least one known relevant source.")
            else:
                try:
                    ranked_results = svc.retrieve(
                        eval_query,
                        top_k=int(eval_k),
                        mode=eval_mode,
                    )
                    ranked_sources = list(dict.fromkeys(
                        str(result["metadata"].get("source", "Unknown"))
                        for result in ranked_results
                    ))
                    from dashboard.research_metrics import calculate_retrieval_metrics

                    metrics = calculate_retrieval_metrics(
                        ranked_sources,
                        relevant_sources,
                        int(eval_k),
                    )
                    run = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "id": datetime.now(timezone.utc).isoformat(),
                        "query": eval_query,
                        "relevant_sources": relevant_sources,
                        "ranked_sources": ranked_sources,
                        "retrieved": ranked_results,
                        "metrics": metrics,
                        "k": int(eval_k),
                        "retrieval_mode": eval_mode,
                        "provider": st.session_state.provider,
                        "model": st.session_state.model,
                        "configuration": _configuration_snapshot({
                            "retrieval_mode": eval_mode,
                            "top_k": int(eval_k),
                        }),
                    }
                    st.session_state.evaluation_runs.append(run)
                    st.session_state.audit_events.append({
                        "event": "retrieval_evaluation",
                        "id": run["id"],
                        "timestamp": run["timestamp"],
                        "query": eval_query,
                        "retrieval_mode": eval_mode,
                        "top_k": int(eval_k),
                        "metrics": metrics,
                        "configuration": run["configuration"],
                    })
                    st.success("Retrieval evaluation completed.")
                    metric_cols = st.columns(4)
                    metric_cols[0].metric(f"Recall@{int(eval_k)}", f"{metrics[f'recall@{int(eval_k)}']:.3f}")
                    metric_cols[1].metric(f"Precision@{int(eval_k)}", f"{metrics[f'precision@{int(eval_k)}']:.3f}")
                    metric_cols[2].metric("MRR", f"{metrics['mrr']:.3f}")
                    metric_cols[3].metric(f"NDCG@{int(eval_k)}", f"{metrics[f'ndcg@{int(eval_k)}']:.3f}")
                    if ranked_results:
                        st.markdown("#### Ranked evidence")
                        st.dataframe(
                            [
                                {
                                    "rank": item["rank"],
                                    "source": item["metadata"].get("source", "Unknown"),
                                    "score": item["score"],
                                    "excerpt": item["content"][:350],
                                }
                                for item in ranked_results
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No evidence was retrieved for this query.")
                except Exception as error:
                    st.error(f"Retrieval evaluation failed: {error}")
                    st.session_state.audit_events.append({
                        "event": "retrieval_evaluation_failed",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "query": eval_query,
                        "retrieval_mode": eval_mode,
                        "top_k": int(eval_k),
                        "error": str(error),
                        "provider": st.session_state.provider,
                        "model": st.session_state.model,
                        "configuration": _configuration_snapshot(),
                    })

    if st.session_state.evaluation_runs:
        st.markdown("#### Previous session runs")
        st.dataframe(
            [
                {
                    "timestamp": run["timestamp"],
                    "query": run["query"],
                    "retrieval_mode": run["retrieval_mode"],
                    "k": run["k"],
                    "configuration_version": run["configuration"]["version"],
                    **run["metrics"],
                }
                for run in reversed(st.session_state.evaluation_runs)
            ],
            use_container_width=True,
            hide_index=True,
        )

with governance_tab:
    st.subheader("Safety & Governance")
    st.caption(
        "These are observed session outcomes and user-entered findings, not a "
        "clinical safety certification or an automated risk assessment."
    )
    st.markdown("#### Session risk signals")
    risk_signals = collect_risk_signals(
        st.session_state.audit_events,
        st.session_state.answer_quality_reviews,
        st.session_state.fairness_reports,
    )
    observed_risk = observed_ethical_risk_score(risk_signals)
    if observed_risk["score"] is None:
        st.info("No scored risk signals are available yet. Run queries or record human reviews to populate this view.")
    else:
        risk_cols = st.columns(3)
        risk_cols[0].metric("Heuristic risk indicator", f"{observed_risk['score']:.1f}/100 · {observed_risk['level']}")
        risk_cols[1].metric("Signal coverage", f"{observed_risk['coverage']:.0%}")
        risk_cols[2].metric("Confidence", observed_risk["confidence"].capitalize())
        st.caption("Weighted summary of available session signals. It is a triage aid, not a validated clinical or fairness score.")
        st.dataframe(
            [{"Signal": key.replace("_", " ").title(), "Observed value": value}
             for key, value in risk_signals.items()],
            use_container_width=True, hide_index=True,
        )

    st.markdown("#### Counterfactual fairness screening")
    st.caption("Runs the same prompt template for each named group and flags lexical, evidence, refusal, or warning differences for human review.")
    with st.form("counterfactual_fairness_form"):
        fairness_template = st.text_area(
            "Prompt template (include {group})",
            placeholder="What prevention guidance is relevant for {group}?",
        )
        fairness_groups_text = st.text_input("Groups to compare (comma separated)")
        run_fairness = st.form_submit_button(
            "Run counterfactual screening", disabled=not (svc and docs_ok)
        )
    if run_fairness:
        try:
            fairness_prompts = build_counterfactual_prompts(
                fairness_template,
                [group.strip() for group in fairness_groups_text.split(",")],
            )
            responses = {}
            for group, prompt in fairness_prompts:
                guard = guard_question(prompt)
                if guard.blocked:
                    raise ValueError("A generated comparison prompt was blocked by the attack screen.")
                stream, sources, details = svc.query_with_details(
                    question=guard.sanitized,
                    chat_history=[],
                    top_k=st.session_state.top_k,
                    mode=st.session_state.retrieval_mode,
                )
                responses[group] = {
                    "answer": "".join(stream),
                    "sources": [source.get("source", "Unknown") for source in sources],
                    "warnings": details.get("warnings", []),
                }
            pairs = compare_responses(responses, [group for group, _ in fairness_prompts])
            report = build_fairness_report(pairs)
            report.update({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "template": fairness_template,
                "groups": [group for group, _ in fairness_prompts],
                "responses": responses,
                "configuration_version": configuration["version"],
            })
            st.session_state.fairness_reports.append(report)
            st.session_state.audit_events.append({
                "event": "counterfactual_fairness_screening",
                "timestamp": report["timestamp"],
                "pair_count": report["pair_count"],
                "flagged_count": len(report["flagged"]),
                "configuration": configuration,
            })
            st.success(f"Screening complete: {len(report['flagged'])} pair(s) flagged for human review.")
            st.json(report)
        except Exception as error:
            st.error(f"Fairness screening failed: {error}")
    for report in reversed(st.session_state.fairness_reports):
        with st.expander(f"Screening · {report['timestamp']} · {len(report['flagged'])} flagged"):
            st.write(report.get("template", ""))
            st.json(report)

    st.markdown("#### Secure RAG control posture")
    st.caption(
        "This table reports controls visible in this prototype; it is not a "
        "penetration test or enterprise security certification."
    )
    st.dataframe(
        [
            {
                "Control": "Access control / user authentication",
                "Status": "Not implemented in this dashboard",
                "Evidence / limitation": (
                    "No application login or role-based authorization was found. "
                    "A deployment host may add separate controls; verify those independently."
                ),
            },
            {
                "Control": "PII/PHI detection and redaction",
                "Status": "Heuristic query screening and redaction",
                "Evidence / limitation": (
                    "Common email, SSN, date-of-birth, payment-card, IP, and phone "
                    "patterns are redacted from submitted questions before query and "
                    "audit storage. Uploaded files and generated answers are not "
                    "comprehensively screened; false negatives are possible."
                ),
            },
            {
                "Control": "Prompt-attack defenses",
                "Status": "Heuristic checks present; limited",
                "Evidence / limitation": (
                    "Known instruction override, prompt exfiltration, delimiter, "
                    "role-hijack, and safety-bypass patterns are screened before "
                    "query execution. This is not comprehensive attack detection."
                ),
            },
            {
                "Control": "Provider credential handling",
                "Status": "Configuration options present",
                "Evidence / limitation": (
                    "Keys can come from Streamlit secrets, environment variables, "
                    "or the password input. Keys are excluded from audit configuration "
                    "snapshots; access and retention depend on deployment settings."
                ),
            },
            {
                "Control": "Audit record storage",
                "Status": "Session memory and user-triggered export",
                "Evidence / limitation": (
                    "Events may contain submitted questions, answers, and findings. "
                    "Exported JSON is not encrypted or access-controlled by this app."
                ),
            },
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.error(
        "Enterprise security is not established: application-level access control "
        "and comprehensive PII/PHI protection are absent. Avoid real patient data."
    )
    st.markdown("#### Responsible AI risk assessor")
    st.caption(
        "Complete a manual, operator-led assessment. Bias and fairness reports are "
        "reviewer-authored observations. The ethical risk score is impact × likelihood "
        "(each rated 1–5), a transparent triage aid rather than an objective or validated score."
    )
    from dashboard.audit_utils import ethical_risk_score

    with st.form("responsible_ai_assessment_form", clear_on_submit=True):
        assessment_title = st.text_input("Assessment title")
        assessment_scope = st.text_input(
            "System use case and affected population/group"
        )
        assessment_method = st.text_area(
            "Evidence or review method",
            help="Describe the examples, data, or review process used. Do not include patient-identifying information.",
        )
        bias_report = st.text_area(
            "Bias report",
            help="Record observed or potential representation, measurement, or output bias; write 'Not assessed' if unknown.",
        )
        fairness_report = st.text_area(
            "Fairness report",
            help="Record subgroup comparisons, known gaps, or why fairness could not be assessed.",
        )
        ethical_risk_description = st.text_area(
            "Ethical risk and potential impact"
        )
        impact = st.select_slider(
            "Potential impact (1 = minimal, 5 = severe)",
            options=[1, 2, 3, 4, 5],
            value=3,
        )
        likelihood = st.select_slider(
            "Estimated likelihood (1 = rare, 5 = frequent)",
            options=[1, 2, 3, 4, 5],
            value=3,
        )
        rai_mitigation = st.text_area("Mitigation, owner, or follow-up")
        submit_assessment = st.form_submit_button("Create responsible AI report")

    if submit_assessment:
        required_fields = {
            "assessment title": assessment_title,
            "use case and affected population": assessment_scope,
            "evidence or review method": assessment_method,
            "bias report": bias_report,
            "fairness report": fairness_report,
            "ethical risk description": ethical_risk_description,
        }
        missing_fields = [
            label for label, value in required_fields.items() if not value.strip()
        ]
        if missing_fields:
            st.error("Complete required fields: " + ", ".join(missing_fields))
        else:
            risk = ethical_risk_score(int(impact), int(likelihood))
            assessment = {
                "id": datetime.now(timezone.utc).isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "title": assessment_title.strip(),
                "scope": assessment_scope.strip(),
                "method": assessment_method.strip(),
                "bias_report": bias_report.strip(),
                "fairness_report": fairness_report.strip(),
                "ethical_risk_description": ethical_risk_description.strip(),
                "impact": int(impact),
                "likelihood": int(likelihood),
                "ethical_risk_score": risk["score"],
                "risk_level": risk["level"],
                "mitigation": rai_mitigation.strip(),
                "status": "Open",
                "configuration": _configuration_snapshot(),
            }
            st.session_state.responsible_ai_assessments.append(assessment)
            st.session_state.audit_events.append({
                "event": "responsible_ai_assessment_created",
                "timestamp": assessment["timestamp"],
                "assessment_id": assessment["id"],
                "title": assessment["title"],
                "ethical_risk_score": assessment["ethical_risk_score"],
                "risk_level": assessment["risk_level"],
                "configuration": assessment["configuration"],
            })
            st.success("Responsible AI report recorded for this session.")
            st.metric(
                "Operator-rated ethical risk",
                f"{risk['score']}/25 · {risk['level']}",
                help="Impact × likelihood using your entered ratings; not independently verified.",
            )

    if st.session_state.responsible_ai_assessments:
        st.markdown("#### Bias, fairness, and ethical risk reports")
        st.dataframe(
            [
                {
                    "timestamp": assessment["timestamp"],
                    "title": assessment["title"],
                    "affected population": assessment["scope"],
                    "impact": assessment["impact"],
                    "likelihood": assessment["likelihood"],
                    "ethical risk": (
                        f"{assessment['ethical_risk_score']}/25 · "
                        f"{assessment['risk_level']}"
                    ),
                    "status": assessment["status"],
                }
                for assessment in reversed(
                    st.session_state.responsible_ai_assessments
                )
            ],
            use_container_width=True,
            hide_index=True,
        )
        for assessment in reversed(st.session_state.responsible_ai_assessments):
            with st.expander(f"Report · {assessment['title']}"):
                st.write("Scope:", assessment["scope"])
                st.write("Review method:", assessment["method"])
                st.write("Bias report:", assessment["bias_report"])
                st.write("Fairness report:", assessment["fairness_report"])
                st.write(
                    "Ethical risk:",
                    assessment["ethical_risk_description"],
                )
                st.write("Mitigation:", assessment["mitigation"] or "Not specified")
                st.caption(
                    f"Impact {assessment['impact']} × likelihood "
                    f"{assessment['likelihood']} = {assessment['ethical_risk_score']}/25 "
                    f"({assessment['risk_level']}) · config "
                    f"{assessment['configuration']['version']}"
                )

    governance_metrics = st.columns(4)
    governance_metrics[0].metric("Queries observed", len(successful_queries))
    governance_metrics[1].metric(
        "Not marked grounded",
        sum(event.get("is_grounded") is False for event in successful_queries),
    )
    governance_metrics[2].metric(
        "Responses with warnings",
        warning_count,
    )
    governance_metrics[3].metric(
        "Responses without sources",
        sum(event.get("source_count", 0) == 0 for event in successful_queries),
    )
    if successful_queries:
        st.markdown("#### Observed response outcomes")
        st.dataframe(
            [
                {
                    "timestamp": event["timestamp"],
                    "query": event["query"],
                    "grounded": event.get("is_grounded"),
                    "source_count": event.get("source_count", 0),
                    "warnings": "; ".join(event.get("warnings", [])),
                }
                for event in reversed(successful_queries)
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Grounding and warning summaries will appear after playground queries.")

    st.markdown("#### User-reported risk and mitigation register")
    st.caption("Findings are manually entered and stored only for this session.")
    with st.form("governance_finding_form", clear_on_submit=True):
        finding_title = st.text_input("Finding")
        finding_severity = st.selectbox(
            "Severity", ["Low", "Medium", "High", "Critical"]
        )
        finding_description = st.text_area("Description")
        finding_mitigation = st.text_area("Mitigation or follow-up")
        add_finding = st.form_submit_button("Record finding")
    if add_finding:
        if not finding_title.strip() or not finding_description.strip():
            st.error("A finding title and description are required.")
        else:
            finding = {
                "id": datetime.now(timezone.utc).isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "title": finding_title.strip(),
                "severity": finding_severity,
                "description": finding_description.strip(),
                "mitigation": finding_mitigation.strip(),
                "status": "Open",
            }
            st.session_state.risk_register.append(finding)
            st.session_state.audit_events.append({
                "event": "governance_finding_recorded",
                "timestamp": finding["timestamp"],
                "finding_id": finding["id"],
                "finding": finding["title"],
                "severity": finding["severity"],
                "status": finding["status"],
                "mitigation": finding["mitigation"],
                "configuration": _configuration_snapshot(),
            })
            st.success("Finding recorded for this session.")

    for finding in reversed(st.session_state.risk_register):
        with st.expander(
            f"{finding['severity']} · {finding['title']} · {finding['status']}"
        ):
            st.write(finding["description"])
            new_mitigation = st.text_area(
                "Mitigation or follow-up",
                value=finding["mitigation"],
                key=f"finding_mitigation_{finding['id']}",
            )
            new_status = st.selectbox(
                "Status",
                ["Open", "In progress", "Mitigated"],
                index=["Open", "In progress", "Mitigated"].index(finding["status"]),
                key=f"finding_status_{finding['id']}",
            )
            if (
                new_status != finding["status"]
                or new_mitigation.strip() != finding["mitigation"]
            ) and st.button(
                "Record remediation update",
                key=f"update_finding_{finding['id']}",
            ):
                previous_status = finding["status"]
                previous_mitigation = finding["mitigation"]
                finding["status"] = new_status
                finding["mitigation"] = new_mitigation.strip()
                st.session_state.audit_events.append({
                    "event": "governance_remediation_updated",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "finding_id": finding["id"],
                    "finding": finding["title"],
                    "previous_status": previous_status,
                    "status": finding["status"],
                    "previous_mitigation": previous_mitigation,
                    "mitigation": finding["mitigation"],
                    "configuration": _configuration_snapshot(),
                })
                st.rerun()
            remediation_history = [
                event for event in st.session_state.audit_events
                if event.get("finding_id") == finding["id"]
                and event.get("event") in {
                    "governance_finding_recorded",
                    "governance_remediation_updated",
                }
            ]
            if remediation_history:
                st.markdown("**Remediation history**")
                st.dataframe(
                    [
                        {
                            "timestamp": event["timestamp"],
                            "event": event["event"],
                            "previous_status": event.get("previous_status", "—"),
                            "status": event.get("status", "Open"),
                            "mitigation": event.get("mitigation", finding["mitigation"]),
                            "config_version": event.get("configuration", {}).get("version"),
                        }
                        for event in remediation_history
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

with audit_tab:
    st.subheader("Audit History")
    st.warning(
        "History is in-memory and limited to the current Streamlit session. "
        "Download a JSON snapshot if you need to retain these records. "
        "Configuration fingerprints identify settings; they are not signed attestations."
    )
    configuration_snapshots = {}
    for event in st.session_state.audit_events:
        snapshot = event.get("configuration")
        if snapshot and snapshot.get("version"):
            configuration_snapshots[snapshot["version"]] = snapshot["settings"]
    for run in st.session_state.evaluation_runs:
        snapshot = run.get("configuration")
        if snapshot and snapshot.get("version"):
            configuration_snapshots[snapshot["version"]] = snapshot["settings"]

    current_configuration = _configuration_snapshot()
    export_payload = {
        "session_only": True,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "current_configuration": current_configuration,
        "configuration_versions": configuration_snapshots,
        "documents": st.session_state.document_records,
        "events": st.session_state.audit_events,
        "evaluations": st.session_state.evaluation_runs,
        "answer_quality_reviews": st.session_state.answer_quality_reviews,
        "responsible_ai_assessments": st.session_state.responsible_ai_assessments,
        "fairness_reports": st.session_state.fairness_reports,
        "governance_findings": st.session_state.risk_register,
    }
    st.download_button(
        "Download session audit JSON",
        data=json.dumps(export_payload, indent=2, ensure_ascii=False),
        file_name="biomedical-rag-session-audit.json",
        mime="application/json",
    )
    if st.session_state.audit_events:
        st.dataframe(
            [
                {
                    "timestamp": event.get("timestamp"),
                    "event": event.get("event"),
                    "configuration": (
                        event.get("configuration", {}).get("version", "—")
                    ),
                    "item": (
                        event.get("query")
                        or event.get("finding")
                        or event.get("filename")
                        or event.get("event")
                    ),
                    "status": event.get(
                        "status",
                        "Success" if event.get("success") else (
                            "Failed" if event.get("success") is False else "—"
                        ),
                    ),
                }
                for event in reversed(st.session_state.audit_events)
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No events have been recorded in this session.")

    st.markdown("#### Configuration versions")
    st.caption(
        "Fingerprints are deterministic hashes of the non-secret settings and "
        "session document manifest captured with each operation."
    )
    if configuration_snapshots:
        st.dataframe(
            [
                {"version": version, **settings}
                for version, settings in configuration_snapshots.items()
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("A configuration version is recorded with the first session operation.")
