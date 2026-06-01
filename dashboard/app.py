"""
dashboard/app.py  — Biomedical RAG System
Streamlit MVP for supervisor review
Entry point: streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import time
from typing import List, Optional

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

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
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
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "rag_service":      None,
        "chat_history":     [],   # clean LLM format
        "display_history":  [],   # {role, content, sources, latency_ms}
        "docs_indexed":     [],
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


# ── Lazy import of RAG service ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising biomedical embedding model…")
def _load_service(provider: str, model: str, api_key: str):
    try:
        from app.services import RAGService
        svc = RAGService(provider=provider, model=model, api_key=api_key)
        return svc, None
    except Exception as e:
        return None, str(e)


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
    secret_key = f"{st.session_state.provider}_api_key"
    api_key_val = ""
    if hasattr(st, "secrets"):
        api_key_val = st.secrets.get(secret_key, "")

    if api_key_val:
        st.markdown('<span class="pill pill-green">✓ API key loaded from secrets</span>', unsafe_allow_html=True)
    else:
        api_key_val = st.text_input(
            "API Key",
            type="password",
            placeholder=f"Enter your {sel_prov} API key",
        )

    if st.button("⚡ Initialise LLM", use_container_width=True):
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
        new_files = [f for f in uploaded if f.name not in st.session_state.docs_indexed]
        if new_files:
            if st.button(f"📥 Index {len(new_files)} file(s)", use_container_width=True):
                svc = st.session_state.rag_service
                if not svc:
                    st.warning("Initialise the LLM first.")
                else:
                    bar = st.progress(0)
                    total_chunks = 0
                    for i, f in enumerate(new_files):
                        with st.spinner(f"Processing {f.name}…"):
                            try:
                                n = svc.ingest(f.read(), f.name)
                                total_chunks += n
                                st.session_state.docs_indexed.append(f.name)
                            except Exception as e:
                                st.error(f"{f.name}: {e}")
                        bar.progress((i + 1) / len(new_files))
                    bar.empty()
                    st.success(f"Indexed {total_chunks:,} chunks from {len(new_files)} file(s)")
        else:
            st.info("All files already indexed.")

    if st.session_state.docs_indexed:
        for fn in st.session_state.docs_indexed:
            st.markdown(f"<div style='font-size:0.8em;color:#64748b;padding:2px 0'>📄 {fn}</div>", unsafe_allow_html=True)

    st.divider()

    # ── Retrieval Settings toggle ──────────────────────────────────────────
    chevron = "▲" if st.session_state.settings_open else "▼"
    if st.button(f"⚙️ Retrieval Settings  {chevron}", use_container_width=True, key="settings_btn"):
        st.session_state.settings_open = not st.session_state.settings_open
        st.rerun()

    if st.session_state.settings_open:
        st.session_state.top_k = st.slider(
            "Top-K chunks", min_value=3, max_value=15, value=st.session_state.top_k,
            help="Number of chunks retrieved before reranking"
        )
        st.session_state.retrieval_mode = st.radio(
            "Retrieval mode",
            ["hybrid", "dense", "sparse"],
            index=["hybrid", "dense", "sparse"].index(st.session_state.retrieval_mode),
            help="hybrid = FAISS + BM25 + RRF fusion"
        )
        st.caption(
            f"Hybrid uses FAISS dense + BM25 sparse + Reciprocal Rank Fusion. "
            f"Current: **{st.session_state.retrieval_mode}**, top-{st.session_state.top_k}."
        )

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
        st.cache_resource.clear()
        st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────
llm_ok  = st.session_state.llm_ready
docs_ok = bool(st.session_state.docs_indexed)
svc     = st.session_state.rag_service

# Header
st.markdown(f"""
<div class="hdr-band">
    <div class="hdr-title">
        <span class="accent-dot">●</span> Biomedical RAG
        <span style="font-size:0.55em; color:var(--muted); margin-left:12px;">
            Medical Question Answering System
        </span>
    </div>
    <div class="hdr-sub">
        Grounded answers from biomedical literature using hybrid retrieval (FAISS + BM25) and Cohere Command-R. 
        Corpus: MedQuAD · 32,814 QA pairs · SentenceTransformers embeddings.
    </div>
    <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <span class="pill {'pill-green' if llm_ok else 'pill-red'}">
            {'✓ LLM Ready' if llm_ok else '✗ LLM Not Connected'}
        </span>
        <span class="pill {'pill-green' if docs_ok else 'pill-red'}">
            {'✓ ' + str(len(st.session_state.docs_indexed)) + ' Doc(s) Indexed' if docs_ok else '✗ No Documents'}
        </span>
        <span class="pill pill-blue">
            {st.session_state.retrieval_mode.upper()} Retrieval
        </span>
        <span class="phase-badge">📋 MSc MVP · Phase 3</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Three-column layout ───────────────────────────────────────────────────────
chat_col, insight_col = st.columns([2.2, 1])

# ── Chat column ───────────────────────────────────────────────────────────────
with chat_col:
    st.markdown("### 💬 Medical Q&A")

    if not st.session_state.display_history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔬</div>
            <div class="empty-title">READY FOR BIOMEDICAL QUERIES</div>
            <div class="empty-hint">
                Upload documents and connect your LLM to begin.<br><br>
                <strong>Example questions:</strong><br>
                "What are the symptoms of type 2 diabetes?"<br>
                "How does Parkinson's disease affect dopamine?"<br>
                "What is the relationship between obesity and hypertension?"<br>
                "What are the treatment options for acute leukemia?"
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Render history
    for turn in st.session_state.display_history:
        if turn["role"] == "user":
            st.markdown(f'<div class="user-msg">{turn["content"]}</div>', unsafe_allow_html=True)
        elif turn["role"] == "error":
            st.markdown(f'<div class="error-msg">⚠ {turn["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">{turn["content"]}</div>', unsafe_allow_html=True)
            if turn.get("sources"):
                with st.expander(
                    f"📚 {len(turn['sources'])} source(s)  ·  ⚡ {turn.get('latency_ms', 0):.0f}ms",
                    expanded=False,
                ):
                    for i, src in enumerate(turn["sources"], 1):
                        score  = src.get("rerank_score") or src.get("score", 0)
                        conf   = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
                        c_cls  = {"High": "pill-green", "Medium": "pill-amber", "Low": "pill-red"}[conf]
                        st.markdown(f"""
                        <div class="src-card">
                            <strong>#{i}  {src.get('source','Unknown')}</strong>
                            &nbsp;<span class="pill {c_cls}">{conf}</span>
                            &nbsp;<span style="font-family:var(--mono);font-size:0.85em;color:var(--muted)">
                                score: {score:.3f}
                            </span>
                            <div class="src-excerpt">{src.get('text','')[:240]}…</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Input
    question = st.chat_input(
        "Ask a biomedical question…",
        disabled=not (llm_ok and docs_ok),
    )

    if not llm_ok and not docs_ok:
        st.caption("⬅ Connect LLM and upload documents in the sidebar to begin.")
    elif not llm_ok:
        st.caption("⬅ Connect your LLM in the sidebar.")
    elif not docs_ok:
        st.caption("⬅ Upload and index documents in the sidebar.")

    if question:
        st.session_state.display_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "user", "content": question})

        placeholder = st.empty()
        full_answer = ""
        sources     = []
        latency_ms  = 0.0

        try:
            t0 = time.perf_counter()
            stream, sources = svc.query_stream(
                question=question,
                chat_history=st.session_state.chat_history[:-1],
                top_k=st.session_state.top_k,
                mode=st.session_state.retrieval_mode,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            for token in stream:
                full_answer += token
                placeholder.markdown(
                    f'<div class="bot-msg">{full_answer}▌</div>',
                    unsafe_allow_html=True,
                )
            placeholder.empty()

            # Update running stats
            st.session_state.total_queries += 1
            n = st.session_state.total_queries
            prev_avg = st.session_state.avg_latency_ms
            st.session_state.avg_latency_ms = prev_avg + (latency_ms - prev_avg) / n

        except Exception as e:
            full_answer = str(e)
            placeholder.empty()
            st.session_state.display_history.append({"role": "error", "content": full_answer})
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {full_answer}"})
            st.rerun()

        st.session_state.display_history.append({
            "role":       "assistant",
            "content":    full_answer,
            "sources":    sources,
            "latency_ms": latency_ms,
        })
        st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
        st.rerun()


# ── Insights column ───────────────────────────────────────────────────────────
with insight_col:
    st.markdown("### 📊 System Status")

    # Live metrics
    n_chunks = 0
    try:
        if svc:
            n_chunks = getattr(svc, "total_chunks", 0)
    except Exception:
        pass

    st.markdown(f"""
    <div class="m-grid">
        <div class="m-card">
            <div class="m-val">{n_chunks:,}</div>
            <div class="m-lbl">Chunks</div>
        </div>
        <div class="m-card">
            <div class="m-val">{len(st.session_state.docs_indexed)}</div>
            <div class="m-lbl">Docs</div>
        </div>
        <div class="m-card">
            <div class="m-val">{st.session_state.total_queries}</div>
            <div class="m-lbl">Queries</div>
        </div>
        <div class="m-card">
            <div class="m-val">{st.session_state.avg_latency_ms:.0f}ms</div>
            <div class="m-lbl">Avg Latency</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown(f"""
    <div style="font-size:0.78em; color:var(--muted); line-height:2; font-family:var(--mono)">
        Embed  · all-MiniLM-L6-v2<br>
        Index  · FAISS (IndexFlatIP)<br>
        Sparse · BM25 (rank-bm25)<br>
        Fuse   · Reciprocal Rank Fusion<br>
        LLM    · {st.session_state.model.split('/')[-1]}<br>
        Mode   · {st.session_state.retrieval_mode.upper()}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Project progress
    st.markdown("**Project Progress**")
    phases = [
        ("Phase 1 · Data Pipeline",   95, "pill-green"),
        ("Phase 2 · RAG System",       85, "pill-green"),
        ("Phase 3 · Evaluation",       30, "pill-amber"),
        ("Phase 4 · Safety & Ethics",   5, "pill-red"),
    ]
    for label, pct, cls in phases:
        color = {"pill-green": "#10B981", "pill-amber": "#F59E0B", "pill-red": "#EF4444"}[cls]
        st.markdown(f"""
        <div class="prog-wrap">
            <div class="prog-label"><span>{label}</span><span>{pct}%</span></div>
            <div class="prog-track">
                <div class="prog-fill" style="width:{pct}%;background:{color}"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Last query sources
    st.markdown("**Last Query**")
    last = next(
        (t for t in reversed(st.session_state.display_history) if t["role"] == "assistant"),
        None
    )
    if last and last.get("sources"):
        st.caption(f"⚡ {last.get('latency_ms', 0):.0f}ms retrieval")
        for i, src in enumerate(last["sources"][:3], 1):
            score = src.get("rerank_score") or src.get("score", 0)
            conf  = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
            st.markdown(f"""
            <div class="src-card">
                <strong>#{i} {src.get('source','?')[:28]}</strong>
                {conf} {score:.3f}<br>
                <span class="src-excerpt">{src.get('text','')[:90]}…</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No query results yet.")

    st.divider()
    st.markdown("**Session**")
    n_q = len([t for t in st.session_state.display_history if t["role"] == "user"])
    n_a = len([t for t in st.session_state.display_history if t["role"] == "assistant"])
    st.markdown(f"""
    <div style="font-size:0.78em;color:var(--muted);line-height:2;font-family:var(--mono)">
        Questions  · {n_q}<br>
        Answers    · {n_a}<br>
        Errors     · {len([t for t in st.session_state.display_history if t['role']=='error'])}
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.display_history = []
        st.session_state.chat_history    = []
        st.session_state.total_queries   = 0
        st.session_state.avg_latency_ms  = 0.0
        st.rerun()
