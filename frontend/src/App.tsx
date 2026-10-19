import { useEffect, useMemo, useState } from "react";
import {
  get,
  post,
  queryPath,
  upload,
  type AuditEvent,
  type DashboardState,
  type SessionCreated,
} from "./api";

type Tab = "Overview" | "Knowledge" | "Playground" | "Evaluation" | "Governance" | "Audit";
type EvidenceSource = { source?: string; text?: string; score?: number; rank?: number };
type QueryResult = {
  event_id: string;
  answer: string;
  sources: EvidenceSource[];
  details: { is_grounded?: boolean; warnings?: string[]; validation_passed?: boolean };
  latency_ms: number;
};

const TABS: Tab[] = ["Overview", "Knowledge", "Playground", "Evaluation", "Governance", "Audit"];
const QUALITY_DIMENSIONS = [
  "factual_correctness",
  "hallucination_concern",
  "evidence_support",
  "faithfulness",
  "explainability",
  "completeness",
  "safety",
];

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("Overview");
  const [sessionId, setSessionId] = useState("");
  const [dashboard, setDashboard] = useState<DashboardState | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState("cohere");
  const [model, setModel] = useState("command-a-plus-05-2026");
  const [topK, setTopK] = useState(5);
  const [mode, setMode] = useState("hybrid");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<QueryResult | null>(null);
  const [retrieved, setRetrieved] = useState<EvidenceSource[]>([]);
  const [relevantSources, setRelevantSources] = useState("");
  const [auditAnswerText, setAuditAnswerText] = useState("");
  const [auditEvidence, setAuditEvidence] = useState("");
  const [answerAudit, setAnswerAudit] = useState<Record<string, unknown> | null>(null);
  const [securityText, setSecurityText] = useState("");
  const [securityReport, setSecurityReport] = useState<Record<string, unknown> | null>(null);
  const [fairnessTemplate, setFairnessTemplate] = useState("What treatment options are available for {group} patients?");
  const [groupsText, setGroupsText] = useState("adult, older adult");
  const [riskTitle, setRiskTitle] = useState("");
  const [riskRationale, setRiskRationale] = useState("");
  const [impact, setImpact] = useState(3);
  const [likelihood, setLikelihood] = useState(3);
  const [findingTitle, setFindingTitle] = useState("");
  const [findingSeverity, setFindingSeverity] = useState("moderate");
  const [findingStatus, setFindingStatus] = useState("open");
  const [findingRationale, setFindingRationale] = useState("");
  const [governanceSummary, setGovernanceSummary] = useState<Record<string, unknown> | null>(null);
  const [reviewNotes, setReviewNotes] = useState("");
  const [reviewRatings, setReviewRatings] = useState<Record<string, number>>(
    Object.fromEntries(QUALITY_DIMENSIONS.map((dimension) => [dimension, 3])),
  );

  const sessionPath = useMemo(
    () => sessionId ? `/api/sessions/${encodeURIComponent(sessionId)}` : "",
    [sessionId],
  );

  async function refreshState(id = sessionId) {
    if (!id) return;
    const state = await get<DashboardState>(`/api/sessions/${encodeURIComponent(id)}`);
    setDashboard(state);
    const settings = state.configuration.settings;
    setProvider(String(settings.provider ?? "cohere"));
    setModel(String(settings.model ?? "command-a-plus-05-2026"));
    setMode(String(settings.retrieval_mode ?? "hybrid"));
    setTopK(Number(settings.top_k ?? 5));
  }

  useEffect(() => {
    let active = true;
    async function start() {
      try {
        let id = window.localStorage.getItem("biomed-rag-session") ?? "";
        if (id) {
          try {
            await get<DashboardState>(`/api/sessions/${encodeURIComponent(id)}`);
          } catch {
            id = "";
          }
        }
        if (!id) {
          const created = await post<SessionCreated>("/api/sessions", {});
          id = created.session_id;
          window.localStorage.setItem("biomed-rag-session", id);
        }
        if (!active) return;
        setSessionId(id);
        await refreshState(id);
      } catch (cause) {
        if (active) setError(errorMessage(cause));
      }
    }
    void start();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (tab !== "Governance" || !sessionId) return;
    let active = true;
    void get<Record<string, unknown>>(`/api/governance/summary${queryPath(sessionId)}`)
      .then((summary) => { if (active) setGovernanceSummary(summary); })
      .catch((cause: unknown) => { if (active) setError(errorMessage(cause)); });
    return () => { active = false; };
  }, [tab, sessionId, dashboard?.events.length, dashboard?.fairness_reports.length]);

  async function perform<T>(action: () => Promise<T>, onSuccess?: (value: T) => void) {
    setBusy(true);
    setError("");
    try {
      const result = await action();
      onSuccess?.(result);
      await refreshState();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  function configure() {
    void perform(
      () => post<DashboardState>(`${sessionPath}/configure`, { provider, model }),
      (state) => setDashboard(state),
    );
  }

  function handleUpload(file?: File) {
    if (!file || !sessionId) return;
    const body = new FormData();
    body.append("session_id", sessionId);
    body.append("file", file);
    void perform(() => upload("/api/documents", body));
  }

  function submitQuery() {
    if (!sessionId || !question.trim()) return;
    void perform(
      () => post<QueryResult>("/api/query", {
        session_id: sessionId,
        question,
        top_k: topK,
        retrieval_mode: mode,
      }),
      (result) => {
        setAnswer(result);
        setQuestion("");
      },
    );
  }

  function retrieveEvidence() {
    if (!sessionId || !question.trim()) return;
    void perform(
      () => post<{ sources: EvidenceSource[] }>("/api/retrieve", {
        session_id: sessionId,
        question,
        top_k: topK,
        retrieval_mode: mode,
      }),
      (result) => setRetrieved(result.sources),
    );
  }

  function submitRetrievalEvaluation() {
    if (!sessionId) return;
    const sources = retrieved.map((source) => String(source.source ?? "")).filter(Boolean);
    void perform(() => post(
      `/api/evaluation/retrieval${queryPath(sessionId)}`,
      {
        retrieved_sources: sources,
        relevant_sources: relevantSources.split(/\r?\n/).map((value) => value.trim()).filter(Boolean),
        k: topK,
      },
    ));
  }

  function submitAnswerAudit() {
    void perform(
      () => post<Record<string, unknown>>(`/api/evaluation/answer${queryPath(sessionId)}`, {
        answer: auditAnswerText,
        evidence_texts: auditEvidence.split(/\r?\n/).map((value) => value.trim()).filter(Boolean),
      }),
      (report) => setAnswerAudit(report),
    );
  }

  function submitSecurityScan() {
    const body = new FormData();
    body.append("text", securityText);
    void perform(
      () => upload<Record<string, unknown>>("/api/governance/security-scan", body),
      (report) => setSecurityReport(report),
    );
  }

  function submitFairness() {
    if (!sessionId) return;
    void perform(
      () => post("/api/governance/fairness", {
        session_id: sessionId,
        template: fairnessTemplate,
        groups: groupsText.split(",").map((value) => value.trim()).filter(Boolean),
        top_k: topK,
      }),
    );
  }

  function submitRisk() {
    if (!sessionId || !riskTitle.trim()) return;
    void perform(() => post(
      `/api/governance/risk${queryPath(sessionId)}`,
      { title: riskTitle, rationale: riskRationale, impact, likelihood },
    ), () => {
      setRiskTitle("");
      setRiskRationale("");
    });
  }

  function submitFinding() {
    if (!sessionId || !findingTitle.trim()) return;
    void perform(() => post(
      `/api/governance/findings${queryPath(sessionId)}`,
      {
        title: findingTitle,
        severity: findingSeverity,
        status: findingStatus,
        rationale: findingRationale,
      },
    ), () => {
      setFindingTitle("");
      setFindingRationale("");
    });
  }

  function submitReview() {
    const target = [...(dashboard?.events ?? [])]
      .reverse()
      .find((event) => event.event === "query" && event.success);
    if (!sessionId || !target) {
      setError("Run a successful query before submitting an answer review.");
      return;
    }
    void perform(() => post(
      `/api/reviews${queryPath(sessionId)}`,
      { query_event_id: target.id, ratings: reviewRatings, notes: reviewNotes },
    ), () => setReviewNotes(""));
  }

  function exportAudit() {
    if (!sessionId) return;
    void perform(async () => {
      const report = await get<Record<string, unknown>>(
        `/api/audit/export${queryPath(sessionId)}`,
      );
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "biomedical-rag-session-audit.json";
      link.click();
      URL.revokeObjectURL(link.href);
    });
  }

  const successfulEvents = (dashboard?.events ?? []).filter(
    (event) => event.event === "query" && event.success,
  );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">B</span><div><strong>BioMed RAG</strong><small>RESEARCH WORKSPACE</small></div></div>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Dashboard sections">
          {TABS.map((item) => (
            <button className={`nav-item ${tab === item ? "active" : ""}`} key={item} onClick={() => setTab(item)}>
              <span className="nav-dot" />{item}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className={`status-dot ${dashboard?.configured ? "ready" : ""}`} />
          <span>{dashboard?.configured ? "LLM connected" : "LLM not configured"}</span>
          <small>Research use only</small>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div><p className="eyebrow">BIOMEDICAL QUESTION ANSWERING</p><h1>{tab}</h1></div>
          <div className="top-actions">
            <span className="config-pill">{dashboard?.configuration.version ?? "Connecting…"}</span>
            <button className="button quiet" onClick={exportAudit} disabled={!dashboard || busy}>Export audit</button>
          </div>
        </header>

        {error && <div className="alert error" role="alert"><strong>Request failed</strong><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError("")}>×</button></div>}
        {!dashboard && <div className="alert"><span>Connecting to the Python API…</span></div>}

        <section className="config-card">
          <div className="config-heading"><div><h2>Model connection</h2><p>Provider credentials stay on the API server.</p></div>
            <span className={`badge ${dashboard?.configured ? "success" : "neutral"}`}>{dashboard?.configured ? "READY" : "NOT CONNECTED"}</span>
          </div>
          <div className="config-fields">
            <label>Provider<select value={provider} onChange={(event) => setProvider(event.target.value)}>
              <option value="cohere">Cohere</option><option value="openai">OpenAI</option><option value="huggingface">Hugging Face</option>
            </select></label>
            <label>Model<input value={model} onChange={(event) => setModel(event.target.value)} /></label>
            <button className="button primary connect-button" onClick={configure} disabled={!sessionId || busy}>{busy ? "Working…" : "Connect model"}</button>
          </div>
        </section>

        {tab === "Overview" && <section className="content-section">
          <div className="section-heading"><div><h2>System overview</h2><p>Session-scoped research and audit indicators.</p></div><span className="badge warning">NOT CLINICALLY VALIDATED</span></div>
          <div className="metric-grid">
            <Metric label="Indexed documents" value={dashboard?.document_count ?? 0} />
            <Metric label="Indexed chunks" value={dashboard?.chunk_count ?? 0} />
            <Metric label="Successful queries" value={dashboard?.summary.successful_queries ?? 0} />
            <Metric label="Grounded answers" value={dashboard?.summary.grounded_queries ?? 0} />
            <Metric label="Mean latency" value={`${Math.round(dashboard?.summary.mean_latency_ms ?? 0)} ms`} />
          </div>
          <div className="panel"><div className="panel-heading"><h3>Recent activity</h3><span>{dashboard?.events.length ?? 0} events</span></div><EventTable events={dashboard?.events.slice(-8).reverse() ?? []} /></div>
          <p className="disclaimer">For research and education only. Not for diagnosis or treatment decisions; consult a qualified healthcare professional.</p>
        </section>}

        {tab === "Knowledge" && <section className="content-section">
          <div className="section-heading"><div><h2>Knowledge base</h2><p>Prepared biomedical corpus and session uploads.</p></div></div>
          <div className="upload-panel"><label className="upload-control">Choose PDF, DOCX, or TXT<input aria-label="Upload a document" type="file" accept=".pdf,.docx,.txt" onChange={(event) => handleUpload(event.target.files?.[0])} /></label>
            <span>Up to 20 MB. Uploaded files and indexes remain in server memory.</span></div>
          <div className="metric-grid compact"><Metric label="Documents" value={dashboard?.document_count ?? 0} /><Metric label="Chunks" value={dashboard?.chunk_count ?? 0} /><Metric label="Uploaded sources" value={dashboard?.documents.length ?? 0} /></div>
          <div className="panel"><div className="panel-heading"><h3>Document metadata & lineage</h3></div>
            {dashboard?.documents.length ? <div className="table-wrap"><table><thead><tr><th>Source</th><th>Type</th><th>Size</th><th>Chunks</th><th>SHA-256</th></tr></thead><tbody>
              {dashboard.documents.map((item, index) => <tr key={`${String(item.sha256)}-${index}`}><td>{String(item.filename ?? item.source)}</td><td>{String(item.file_type ?? "").toUpperCase()}</td><td>{Number(item.size_bytes ?? 0).toLocaleString()} B</td><td>{String(item.chunks ?? "—")}</td><td className="mono">{String(item.sha256 ?? "").slice(0, 14)}…</td></tr>)}
            </tbody></table></div> : <p className="empty">No uploaded documents in this session.</p>}
          </div>
          {!!dashboard?.source_names.length && <div className="source-chips">{dashboard.source_names.map((source) => <span key={source}>{source}</span>)}</div>}
        </section>}

        {tab === "Playground" && <section className="content-section">
          <div className="section-heading"><div><h2>RAG playground</h2><p>Ask a question, inspect evidence, and review the checked response.</p></div></div>
          <div className="query-settings">
            <label>Retrieval mode<select value={mode} onChange={(event) => setMode(event.target.value)}><option value="hybrid">Hybrid</option><option value="dense">Dense</option><option value="sparse">Sparse</option></select></label>
            <label>Top-K chunks<select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>{[3, 5, 8, 10, 15].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
          </div>
          <div className="playground-grid">
            <div className="panel question-panel">
              <label htmlFor="question">Biomedical question</label>
              <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="How does Parkinson's disease affect dopamine?" rows={4} />
              <div className="button-row"><button className="button primary" onClick={submitQuery} disabled={busy || !dashboard?.configured || !question.trim()}>Ask with RAG</button><button className="button" onClick={retrieveEvidence} disabled={busy || !dashboard?.configured || !question.trim()}>Retrieve evidence</button></div>
              {!dashboard?.configured && <p className="help-text">Connect a model above before querying.</p>}
              {answer && <div className="answer-card"><div className="answer-status">{answer.details.is_grounded ? "EVIDENCE GROUNDED" : "GROUNDING NOT CONFIRMED"}<span>{(answer.latency_ms / 1000).toFixed(1)} s</span></div><p>{answer.answer}</p>
                {!!answer.details.warnings?.length && <div className="alert warning">{answer.details.warnings.join("; ")}</div>}
                <button className="text-button" onClick={() => { setAuditAnswerText(answer.answer); setAuditEvidence(answer.sources.map((source) => (source.text ?? "").replace(/\s+/g, " ").trim()).join("\n")); setTab("Evaluation"); }}>Audit this answer →</button>
              </div>}
            </div>
            <div className="panel"><div className="panel-heading"><h3>Retrieved evidence</h3><span>{answer?.sources.length ?? retrieved.length} sources</span></div>
              <EvidenceList sources={answer?.sources ?? retrieved} />
            </div>
          </div>
          <div className="panel review-panel"><div className="panel-heading"><h3>Human answer review</h3><span>1 = low · 5 = high</span></div>
            {!successfulEvents.length ? <p className="empty">A successful answer is required before a review can be recorded.</p> :
              <><div className="rating-grid">{QUALITY_DIMENSIONS.map((dimension) => <label key={dimension}>{dimension.replaceAll("_", " ")}<select value={reviewRatings[dimension]} onChange={(event) => setReviewRatings({ ...reviewRatings, [dimension]: Number(event.target.value) })}>{[1, 2, 3, 4, 5].map((score) => <option key={score}>{score}</option>)}</select></label>)}</div>
                <label className="stacked-label">Reviewer notes<textarea rows={2} value={reviewNotes} onChange={(event) => setReviewNotes(event.target.value)} /></label>
                <button className="button" onClick={submitReview} disabled={busy}>Save human review</button></>}
          </div>
        </section>}

        {tab === "Evaluation" && <section className="content-section">
          <div className="section-heading"><div><h2>Evaluation lab</h2><p>Transparent source-level metrics and lexical answer audit proxies for human review.</p></div></div>
          <div className="evaluation-grid">
            <div className="panel"><h3>Retrieval evaluation</h3><p className="muted">Uses the latest retrieved evidence from the Playground.</p>
              <label>Relevant source names (one per line)<textarea rows={5} value={relevantSources} onChange={(event) => setRelevantSources(event.target.value)} /></label>
              <button className="button primary" onClick={submitRetrievalEvaluation} disabled={busy || !retrieved.length || !relevantSources.trim()}>Calculate Recall@K, Precision@K, MRR, NDCG</button>
              {dashboard?.evaluation_runs.length ? <ResultList items={dashboard.evaluation_runs} /> : <p className="empty">No labeled retrieval runs yet.</p>}
            </div>
            <div className="panel"><h3>Answer audit</h3><p className="muted">Lexical overlap is an audit proxy, not truth verification. Evidence citations such as [Evidence 1] are checked against the passage list.</p>
              <label>Answer<textarea rows={5} value={auditAnswerText} onChange={(event) => setAuditAnswerText(event.target.value)} /></label>
              <label>Evidence passages (one per line)<textarea rows={5} value={auditEvidence} onChange={(event) => setAuditEvidence(event.target.value)} /></label>
              <button className="button primary" onClick={submitAnswerAudit} disabled={busy || !auditAnswerText.trim()}>Audit answer against evidence</button>
              {answerAudit && <pre className="json-result">{JSON.stringify(answerAudit, null, 2)}</pre>}
            </div>
          </div>
          <div className="panel"><h3>Research metrics summary</h3><div className="metric-grid compact">
            {Object.entries(dashboard?.summary.retrieval_metrics ?? {}).map(([label, value]) => <Metric key={label} label={label} value={value == null ? "—" : Number(value).toFixed(3)} />)}
            {Object.entries(dashboard?.summary.answer_audits ?? {}).map(([label, value]) => <Metric key={label} label={label.replaceAll("_", " ")} value={value == null ? "—" : Number(value).toFixed(3)} />)}
            {Object.entries(dashboard?.summary.answer_quality_reviews ?? {}).map(([label, value]) => <Metric key={label} label={label.replaceAll("_", " ")} value={value == null ? "—" : Number(value).toFixed(2)} />)}
          </div></div>
        </section>}

        {tab === "Governance" && <section className="content-section">
          <div className="section-heading"><div><h2>Safety & governance</h2><p>Heuristic security screens, responsible-AI review, and transparent risk tracking.</p></div><span className="badge warning">SCREENING ONLY</span></div>
          <div className="panel"><div className="panel-heading"><h3>Observed governance signals</h3><span>Research indicators · not certification</span></div>
            {governanceSummary ? <pre className="json-result">{JSON.stringify({
              observed_risk: governanceSummary.observed_risk,
              risk_signals: governanceSummary.risk_signals,
              limitations: governanceSummary.limitations,
            }, null, 2)}</pre> : <p className="empty">Loading governance indicators…</p>}
          </div>
          <div className="evaluation-grid">
            <div className="panel"><h3>PII & prompt-attack screen</h3><label>Text to scan<textarea rows={5} value={securityText} onChange={(event) => setSecurityText(event.target.value)} placeholder="Paste text to scan…" /></label>
              <button className="button" onClick={submitSecurityScan} disabled={busy || !securityText.trim()}>Scan and redact</button>
              {securityReport && <pre className="json-result">{JSON.stringify(securityReport, null, 2)}</pre>}
            </div>
            <div className="panel"><h3>Fairness comparison</h3><p className="muted">Counterfactual wording comparison; flagged differences need qualified human review.</p>
              <label>Question template (include {"{group}"})<textarea rows={3} value={fairnessTemplate} onChange={(event) => setFairnessTemplate(event.target.value)} /></label>
              <label>Comparison groups, comma-separated<input value={groupsText} onChange={(event) => setGroupsText(event.target.value)} /></label>
              <button className="button" onClick={submitFairness} disabled={busy || !dashboard?.configured}>Run fairness screen</button>
              {!!dashboard?.fairness_reports.length && <ResultList items={dashboard.fairness_reports} />}
            </div>
          </div>
          <div className="evaluation-grid">
            <div className="panel"><h3>Operator-rated ethical risk</h3><div className="rating-grid two">
              <label>Impact (1–5)<input type="number" min={1} max={5} value={impact} onChange={(event) => setImpact(Number(event.target.value))} /></label>
              <label>Likelihood (1–5)<input type="number" min={1} max={5} value={likelihood} onChange={(event) => setLikelihood(Number(event.target.value))} /></label>
            </div>
              <label>Risk title<input value={riskTitle} onChange={(event) => setRiskTitle(event.target.value)} /></label>
              <label>Rationale<textarea rows={3} value={riskRationale} onChange={(event) => setRiskRationale(event.target.value)} /></label>
              <button className="button" onClick={submitRisk} disabled={busy || !riskTitle.trim()}>Record risk assessment</button>
            </div>
            <div className="panel"><h3>Risk register</h3>{dashboard?.risk_register.length ? <ResultList items={dashboard.risk_register} /> : <p className="empty">No operator findings recorded.</p>}</div>
          </div>
          <div className="panel"><h3>Record a governance finding</h3><div className="config-fields">
            <label>Finding<input value={findingTitle} onChange={(event) => setFindingTitle(event.target.value)} /></label>
            <label>Severity<select value={findingSeverity} onChange={(event) => setFindingSeverity(event.target.value)}><option value="low">Low</option><option value="moderate">Moderate</option><option value="high">High</option><option value="critical">Critical</option></select></label>
            <label>Status<select value={findingStatus} onChange={(event) => setFindingStatus(event.target.value)}><option value="open">Open</option><option value="mitigated">Mitigated</option><option value="accepted">Accepted</option></select></label>
          </div><label className="stacked-label">Rationale<textarea rows={2} value={findingRationale} onChange={(event) => setFindingRationale(event.target.value)} /></label>
            <button className="button" onClick={submitFinding} disabled={busy || !findingTitle.trim()}>Save governance finding</button>
          </div>
          <p className="disclaimer">Heuristic security and fairness signals are not comprehensive protection, proof of fairness, enterprise certification, or clinical validation.</p>
        </section>}

        {tab === "Audit" && <section className="content-section">
          <div className="section-heading"><div><h2>Audit history</h2><p>Events and reviews from this in-memory API session.</p></div><button className="button" onClick={exportAudit} disabled={busy}>Download JSON</button></div>
          <div className="metric-grid compact">
            <Metric label="All events" value={dashboard?.events.length ?? 0} />
            <Metric label="Successful queries" value={dashboard?.summary.successful_queries ?? 0} />
            <Metric label="Failed queries" value={dashboard?.summary.failed_queries ?? 0} />
            <Metric label="Answer reviews" value={dashboard?.answer_quality_reviews.length ?? 0} />
          </div>
          <div className="panel"><EventTable events={[...(dashboard?.events ?? [])].reverse()} /></div>
          <div className="panel"><div className="panel-heading"><h3>Human quality reviews</h3><span>{dashboard?.answer_quality_reviews.length ?? 0}</span></div><ResultList items={dashboard?.answer_quality_reviews ?? []} /></div>
          <p className="disclaimer">This session is held in server memory. Export data you need to retain; it is not durable storage.</p>
        </section>}

        <footer className="page-footer">Biomedical RAG · Research and educational use only · Session-scoped audit data</footer>
      </main>
    </div>
  );
}

function EvidenceList({ sources }: { sources: EvidenceSource[] }) {
  if (!sources.length) return <p className="empty">Evidence will appear after retrieval.</p>;
  return <div className="evidence-list">{sources.map((source, index) => <article className="evidence-item" key={`${source.source ?? "source"}-${index}`}>
    <div><span className="source-rank">#{source.rank ?? index + 1}</span><strong>{source.source ?? "Biomedical source"}</strong>{source.score != null && <span className="source-score">{source.score.toFixed(3)}</span>}</div>
    <p>{source.text ?? "No source excerpt provided."}</p>
  </article>)}</div>;
}

function EventTable({ events }: { events: AuditEvent[] }) {
  if (!events.length) return <p className="empty">No activity has been recorded in this session.</p>;
  return <div className="table-wrap"><table><thead><tr><th>Time</th><th>Event</th><th>Status</th><th>Question / source</th><th>Evidence</th><th>Latency</th></tr></thead><tbody>
    {events.map((event) => <tr key={event.id}><td>{new Date(event.timestamp).toLocaleString()}</td><td>{event.event}</td><td><span className={`event-status ${event.success ? "success" : event.success === false ? "failed" : ""}`}>{event.success === undefined ? "Recorded" : event.success ? "Success" : "Failed"}</span></td><td>{String(event.query ?? event.source ?? event.error ?? "—")}</td><td>{String(event.source_count ?? "—")}</td><td>{event.latency_ms == null ? "—" : `${Number(event.latency_ms).toFixed(0)} ms`}</td></tr>)}
  </tbody></table></div>;
}

function ResultList({ items }: { items: Array<Record<string, unknown>> }) {
  if (!items.length) return <p className="empty">No results recorded.</p>;
  return <div className="result-list">{[...items].reverse().map((item, index) => <details key={String(item.id ?? index)}>
    <summary>{String(item.title ?? item.timestamp ?? "Evaluation result")} <span>{String(item.level ?? item.severity ?? item.status ?? "")}</span></summary>
    <pre className="json-result">{JSON.stringify(item, null, 2)}</pre>
  </details>)}</div>;
}
