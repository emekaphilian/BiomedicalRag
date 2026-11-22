import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import {
  getApiToken,
  get,
  post,
  queryPath,
  setApiToken,
  upload,
  type AuditEvent,
  type DashboardState,
  type SessionCreated,
} from "./api";

type Tab = "Overview" | "Knowledge" | "Playground" | "Evaluation" | "Governance" | "Audit";
type EvidenceSource = { evidence_id?: string; source?: string; text?: string; prompt_excerpt?: string; score?: number; rerank_score?: number; rank?: number; url?: string; source_provider?: string; source_origin?: string; source_quality?: string };
type QueryResult = {
  event_id: string;
  answer: string;
  sources: EvidenceSource[];
  details: { is_grounded?: boolean; generation_success?: boolean; answer_abstained?: boolean; decision?: string; rejection_reason?: string | null; risk_tier?: string; question_category?: string; evidence_confidence?: string | null; confidence_basis?: string | null; claim_support?: Array<Record<string, unknown>>; validation_diagnostics?: { factual_sentence_count?: number; citation_coverage?: number | null; citation_support_coverage?: number | null; lexical_support?: number | null; method?: string } | null; warnings?: string[]; validation_passed?: boolean; validation_states?: Record<string, boolean | string | null> };
  latency_ms: number;
};

const TABS: Tab[] = ["Playground", "Knowledge", "Evaluation", "Governance", "Audit", "Overview"];
const QUALITY_DIMENSIONS = [
  "factual_correctness",
  "hallucination_concern",
  "evidence_support",
  "faithfulness",
  "explainability",
  "completeness",
  "safety",
];

const VALIDATION_LABELS: Record<string, string> = {
  RETRIEVAL_RELEVANCE_VALIDATED: "Evidence relevance",
  CITATION_REFERENCES_VALIDATED: "Citation references",
  CLAIM_SUPPORT_VALIDATED: "Claim support",
  CLAIM_SUPPORT_SCREEN_PASSED: "Lexical support screen",
  ANSWER_QUALITY_EVALUATED: "Answer quality",
  MEDICAL_ACCURACY_NOT_ESTABLISHED: "Medical accuracy",
};
const REJECTION_LABELS: Record<string, string> = {
  query_safety_blocked: "The question was blocked by the safety screen",
  insufficient_topic_matched_evidence: "No topic matched the retrieved evidence",
  source_prompt_injection_blocked: "A retrieved source was blocked by the security screen",
  generated_answer_off_topic: "The generated draft did not address the question",
  generation_failed: "The language model did not return a usable response",
  answer_validation_failed: "The draft did not pass citation and evidence-support checks",
  extractive_evidence_fallback: "A cited passage was used after the model abstained",
  model_abstained: "The model declined to answer even though relevant evidence was retrieved",
  lexical_support_uncertain: "Citation checks passed, but lexical support was inconclusive",
};

function rejectionLabel(reason?: string | null) {
  if (!reason) return "The response did not pass validation";
  return REJECTION_LABELS[reason] ?? reason.replaceAll("_", " ");
}

type ResultState = "approved" | "informational" | "validation_review_required" | "insufficient" | "no_relevant_evidence" | "evidence_extraction_failed" | "safety_override" | "out_of_scope" | "generation_failed";

function resultState(decision?: string, answerAbstained?: boolean, reason?: string | null, success?: boolean, answer?: string): ResultState {
  if (decision === "safety_override" || reason?.startsWith("medical_")) return "safety_override";
  if (decision === "out_of_scope" || reason === "non_medical_question") return "out_of_scope";
  if (decision === "validation_review_required") return "validation_review_required";
  if (decision === "no_relevant_evidence" || reason === "no_relevant_evidence") return "no_relevant_evidence";
  if (decision === "evidence_extraction_failed" || reason === "evidence_extraction_failed") return "evidence_extraction_failed";
  if (decision === "generation_failed") return "generation_failed";
  if (/could not produce an answer that passed|couldn.t provide a reliable answer|couldn.t find enough (?:supporting|relevant) evidence|could not find enough (?:supporting|relevant) evidence|generation failed:/i.test(answer ?? "")) return "insufficient";
  if (answerAbstained || success === false || decision === "insufficient_evidence") return "insufficient";
  if (decision === "informational") return "informational";
  if (decision === "answer_approved" || (decision == null && success)) return "approved";
  return "insufficient";
}

function outcomeLabel(state: ResultState) {
  return ({ approved: "Verified answer", informational: "Informational answer", validation_review_required: "Validation review required", insufficient: "Insufficient Supporting Evidence", no_relevant_evidence: "No Relevant Evidence Found", evidence_extraction_failed: "Evidence Extraction Failed", safety_override: "Medical safety override", out_of_scope: "Medical questions only", generation_failed: "Answer generation failed" })[state];
}

function outcomeTone(state: ResultState) {
  return state === "approved" ? "approved" : state === "informational" ? "informational" : state === "validation_review_required" ? "review" : state === "out_of_scope" ? "scope" : "unavailable";
}

function stateAnswer(state: ResultState, answer: string, sourceCount = 0) {
  if (state === "validation_review_required") {
    if (/relevant evidence was retrieved/i.test(answer)) return answer;
    return "Source records were retrieved, but this earlier response did not retain passage text needed to check whether they support an answer. Review the source records in Audit or ask again to generate a fresh response.";
  }
  if (state === "evidence_extraction_failed") return "A source record was retrieved, but no usable evidence excerpt was available for answer generation. Review the source record or ask again to retrieve its text.";
  if (state === "no_relevant_evidence") return "No sufficiently relevant passages were found in the available knowledge base or linked biomedical sources.";
  if (state === "out_of_scope") return "BioMed RAG answers medical and biomedical questions only. Ask about a health condition, symptom, treatment evidence, anatomy, or biology topic.";
  if (state === "generation_failed") return "Relevant evidence may be available, but answer generation failed. Please retry or review the retrieved sources.";
  if (state === "insufficient") return sourceCount > 0
    ? "Relevant sources were retrieved, but the available excerpts did not contain enough validated evidence to support a reliable answer."
    : "No relevant evidence passages were found to support an answer.";
  if (state === "safety_override" && /could not produce an answer that passed|couldn['’]t provide a reliable answer|generation failed:/i.test(answer)) {
    return "This question may involve urgent or individualized medical risk. Seek qualified medical help when appropriate.";
  }
  return answer;
}

function whySummary(decision?: string, reason?: string | null) {
  if (decision === "informational" && reason === "extractive_evidence_fallback") return "The model did not produce a usable answer, so this informational response quotes a relevant retrieved passage with its evidence citation. Review the source before relying on it; BioMed RAG does not establish clinical accuracy.";
  if (decision === "informational" && reason === "external_source_quality_unverified") return "The answer passed the evidence and citation checks, but it uses external research sources whose quality BioMed RAG has not independently verified. Review the linked source records.";
  if (decision === "informational" && reason === "retrieval_relevance_limited") return "Topic-matched evidence was retrieved, but the retrieval score was below the usual threshold. The answer uses the available passages and should be checked against the linked sources.";
  if (decision === "informational" && reason === "validation_support_inconclusive") return "The answer is shown from relevant retrieved passages, but citation or claim-support checks were incomplete. Review the listed sources before relying on it.";
  if (decision === "validation_review_required") return "Relevant evidence was retrieved, but the validator could not confirm the generated answer. The answer is withheld; review the specific findings and retrieved passages.";
  if (decision === "out_of_scope" || reason === "non_medical_question") return "This workspace is limited to medical and biomedical questions.";
  if (reason === "no_relevant_evidence") return "No relevant passages were found in the indexed collection or available external biomedical sources. Refine the question, add a source, or try again later.";
  if (reason === "evidence_extraction_failed") return "A source was located, but the system did not receive usable passage text to send to answer generation.";
  if (reason === "evidence_too_weak") return "Some passages were retrieved, but they were too weak or indirect to support an answer. Review them in Audit or add a more directly relevant source.";
  if (decision === "informational") return "The response has citations, but a lexical support check was inconclusive. That screen can miss valid paraphrases. Review the cited passages; BioMed RAG does not independently verify source quality or clinical accuracy.";
  if (decision === "safety_override" || reason?.startsWith("medical_")) return "This question may involve urgent or individualized medical risk, so the system did not generate a general answer. Seek qualified medical help when appropriate.";
  if (reason === "source_prompt_injection_blocked") return "Retrieved text triggered a security screen, so it was withheld from answer generation. Review the source in Audit.";
  if (decision === "insufficient_evidence") return "The retrieved passages did not provide enough directly relevant support for a reliable answer. Try a more specific question or review the available sources.";
  return rejectionLabel(reason);
}

function outcomeWhySummary(state: ResultState, decision?: string, reason?: string | null) {
  if (state === "evidence_extraction_failed") return "A source record was found, but no usable passage text was retained for answer generation. The source title alone is not enough to ground a medical answer.";
  if (state === "insufficient" && reason === "evidence_too_weak") return "Relevant source excerpts were retrieved, but they did not contain enough validated support for an answer. Review them in Sources or Audit.";
  return whySummary(decision, reason);
}

function percentage(value?: number | null) {
  return value == null ? "Not available" : `${Math.round(value * 100)}%`;
}

function answerParts(value: string) {
  const marker = "MEDICAL DISCLAIMER:";
  const index = value.indexOf(marker);
  if (index < 0) return { answer: value.trim(), disclaimer: "" };
  const start = value.lastIndexOf("\n\n", index);
  return {
    answer: value.slice(0, start >= 0 ? start : index).trim(),
    disclaimer: value.slice(index + marker.length).trim(),
  };
}

function validationValue(value: boolean | string | null) {
  if (value === true) return { label: "Passed", state: "passed" };
  if (value === false) return { label: "Failed", state: "failed" };
  if (value === null) return { label: "Not evaluated", state: "pending" };
  if (value === "NOT ESTABLISHED") return { label: "Not established", state: "pending" };
  return { label: String(value), state: "pending" };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}

function isBearerAuthError(error: unknown): boolean {
  return error instanceof Error && error.message.toLowerCase().includes("bearer token");
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("Playground");
  const [apiToken, setApiTokenState] = useState(getApiToken);
  const [tokenInput, setTokenInput] = useState("");
  const [showTokenPrompt, setShowTokenPrompt] = useState(false);
  const [apiAuthRequired, setApiAuthRequired] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [dashboard, setDashboard] = useState<DashboardState | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState("cohere");
  const [model, setModel] = useState("command-a-plus-05-2026");
  const [topK, setTopK] = useState(5);
  const [mode, setMode] = useState("hybrid");
  const [question, setQuestion] = useState("");
  const [answeredQuestion, setAnsweredQuestion] = useState("");
  const [answer, setAnswer] = useState<QueryResult | null>(null);
  const [retrieved, setRetrieved] = useState<EvidenceSource[]>([]);
  const [retrievalAttempted, setRetrievalAttempted] = useState(false);
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
  const chatEndRef = useRef<HTMLDivElement>(null);

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
        if (active) {
          setError(errorMessage(cause));
          if (isBearerAuthError(cause)) {
            setApiAuthRequired(true);
            setShowTokenPrompt(true);
          }
        }
      }
    }
    void start();
    return () => { active = false; };
  }, [apiToken]);

  useEffect(() => {
    if (tab !== "Governance" || !sessionId) return;
    let active = true;
    void get<Record<string, unknown>>(`/api/governance/summary${queryPath(sessionId)}`)
      .then((summary) => { if (active) setGovernanceSummary(summary); })
      .catch((cause: unknown) => { if (active) setError(errorMessage(cause)); });
    return () => { active = false; };
  }, [tab, sessionId, dashboard?.events.length, dashboard?.fairness_reports.length]);

  useEffect(() => {
    if (tab === "Playground" && answer) {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [tab, answer?.event_id]);

  async function perform<T>(action: () => Promise<T>, onSuccess?: (value: T) => void) {
    setBusy(true);
    setError("");
    try {
      const result = await action();
      onSuccess?.(result);
      await refreshState();
    } catch (cause) {
      setError(errorMessage(cause));
      if (isBearerAuthError(cause)) {
        setApiAuthRequired(true);
        setShowTokenPrompt(true);
      }
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
    const submittedQuestion = question.trim();
    if (!sessionId || !submittedQuestion) return;
    void perform(
      () => post<QueryResult>("/api/query", {
        session_id: sessionId,
        question: submittedQuestion,
        top_k: topK,
        retrieval_mode: mode,
      }),
      (result) => {
        setAnswer(result);
        setAnsweredQuestion(submittedQuestion);
        setQuestion("");
        setRetrievalAttempted(true);
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
      (result) => {
        setRetrieved(result.sources);
        setAnswer(null);
        setRetrievalAttempted(true);
      },
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
      .find((event) => event.event === "query" && event.answer);
    if (!sessionId || !target) {
      setError("Run a query before reviewing its answer or abstention.");
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

  function submitApiToken(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const token = tokenInput.trim();
    if (!token) return;
    setApiToken(token);
    setApiTokenState(token);
    setTokenInput("");
    setError("");
    setShowTokenPrompt(false);
    setApiAuthRequired(false);
  }

  function signOut() {
    setApiToken("");
    setApiTokenState("");
    setSessionId("");
    setDashboard(null);
    setError("");
    window.localStorage.removeItem("biomed-rag-session");
  }

  async function startNewChat() {
    if (!sessionId || busy) return;
    setBusy(true);
    setError("");
    try {
      let created: SessionCreated;
      try {
        created = await post<SessionCreated>(
          `/api/sessions/${encodeURIComponent(sessionId)}/new-chat`,
          {},
        );
      } catch (cause) {
        // The persisted browser session can outlive an API process or auth
        // principal. Recover from a stale session id by creating a clean one.
        if (!/session not found|request failed \(404\)/i.test(errorMessage(cause))) throw cause;
        created = await post<SessionCreated>("/api/sessions", {});
      }
      window.localStorage.setItem("biomed-rag-session", created.session_id);
      setSessionId(created.session_id);
      await refreshState(created.session_id);
      setQuestion("");
      setAnsweredQuestion("");
      setAnswer(null);
      setRetrieved([]);
      setRetrievalAttempted(false);
      setRelevantSources("");
      setAuditAnswerText("");
      setAuditEvidence("");
      setAnswerAudit(null);
      setSecurityText("");
      setSecurityReport(null);
      setReviewNotes("");
      setTab("Playground");
    } catch (cause) {
      setError(errorMessage(cause));
      if (isBearerAuthError(cause)) {
        setApiAuthRequired(true);
        setShowTokenPrompt(true);
      }
    } finally {
      setBusy(false);
    }
  }

  const reviewableEvents = (dashboard?.events ?? []).filter(
    (event) => event.event === "query" && event.answer,
  );
  const currentAnswerReviewed = Boolean(answer && dashboard?.answer_quality_reviews.some(
    (review) => String(review.query_event_id ?? "") === answer.event_id,
  ));
  const previousTurns = (dashboard?.events ?? [])
    .filter((event) => event.event === "query" && event.id !== answer?.event_id && (event.query || event.answer))
    .slice(-12)
    .reverse();
  const recentQuestions = (dashboard?.events ?? [])
    .filter((event) => event.event === "query" && event.query)
    .slice(-10)
    .reverse();

  const currentResultState = answer
    ? (() => {
        const state = resultState(answer.details.decision, answer.details.answer_abstained, answer.details.rejection_reason, answer.details.generation_success, answer.answer);
        const hasSourceRecords = answer.sources.length > 0;
        const hasUsableExcerpt = answer.sources.some((source) => Boolean(source.text?.trim()));
        return state === "insufficient" && hasSourceRecords && !hasUsableExcerpt ? "evidence_extraction_failed" : state;
      })()
    : "insufficient";
  const currentAnswerBody = answer
    ? stateAnswer(currentResultState, answer.answer, answer.sources.length)
    : "";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">B</span><div><strong>BioMed RAG</strong><small>RESEARCH WORKSPACE</small></div></div>
        {tab === "Playground" && <button className="new-chat-button" onClick={startNewChat} disabled={busy || !sessionId}><span>＋</span>New chat</button>}
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Dashboard sections">
          {TABS.map((item) => (
            <button className={`nav-item ${tab === item ? "active" : ""}`} key={item} onClick={() => setTab(item)}>
              <span className="nav-dot" />{item}
            </button>
          ))}
        </nav>
        {tab === "Playground" && <div className="recent-list"><div className="nav-label">RECENT QUESTIONS</div>{recentQuestions.length ? recentQuestions.map((event) => <button key={event.id} onClick={() => setQuestion(String(event.query ?? ""))} title={String(event.query ?? "")}>{String(event.query ?? "")}</button>) : <p>Questions from this session will appear here.</p>}</div>}
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
            <span className="config-pill">{dashboard?.configuration.version ?? (error ? "API unavailable" : "Connecting…")}</span>
            <button className="button quiet" onClick={exportAudit} disabled={!dashboard || busy}>Export audit</button>
            {apiToken
              ? <button className="button quiet" onClick={signOut}>Sign out</button>
              : apiAuthRequired && !showTokenPrompt
                ? <button className="button primary" onClick={() => setShowTokenPrompt(true)}>Connect API</button>
                : null}
          </div>
        </header>

        {showTokenPrompt && <section className="panel auth-inline" aria-label="API access">
          <div className="panel-heading"><h2>Connect to the API</h2><button className="button quiet" type="button" onClick={() => setShowTokenPrompt(false)}>Close</button></div>
          <p className="muted">Enter your BiomedicalRag bearer token. It stays in page memory and is not included in the frontend build.</p>
          <form onSubmit={submitApiToken}>
            <label htmlFor="api-token">Bearer token<input
              id="api-token"
              type="password"
              autoComplete="current-password"
              value={tokenInput}
              onChange={(event) => setTokenInput(event.target.value)}
              required
            /></label>
            <button className="button primary" type="submit">Connect</button>
          </form>
        </section>}
        {error && <div className="alert error" role="alert"><strong>Request failed</strong><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError("")}>×</button></div>}
        {!dashboard && !error && <div className="alert"><span>Connecting to the Python API…</span></div>}

        {tab !== "Playground" && <section className="config-card">
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
        </section>}

        {tab === "Overview" && <section className="content-section">
          <div className="section-heading"><div><h2>System overview</h2><p>Session-scoped research and audit indicators.</p></div><span className="badge warning">NOT CLINICALLY VALIDATED</span></div>
          <div className="metric-grid">
            <Metric label="Answer approval rate" value={percentage(dashboard?.summary.approval_rate)} />
            <Metric label="Warning rate" value={percentage(dashboard?.summary.warning_rate)} />
            <Metric label="Validation reviews" value={dashboard?.summary.validation_review_queries ?? 0} />
            <Metric label="Abstention rate" value={percentage(dashboard?.summary.abstention_rate)} />
            <Metric label="Evidence coverage" value={percentage(dashboard?.summary.answer_audits.evidence_trace_coverage)} />
            <Metric label="Citation coverage" value={percentage(dashboard?.summary.answer_audits.citation_support_coverage)} />
            <Metric label="Median latency" value={dashboard?.summary.median_latency_ms == null ? "—" : `${Math.round(dashboard.summary.median_latency_ms)} ms`} />
            <Metric label="Corpus size" value={`${dashboard?.document_count ?? 0} docs · ${dashboard?.chunk_count ?? 0} chunks`} />
          </div>
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

        {tab === "Playground" && <section className="chat-workspace">
          <div className="chat-toolbar">
            <div className="chat-model"><span className={`status-dot ${dashboard?.configured ? "ready" : ""}`} />{dashboard?.configured ? `${provider} · ${model}` : "Model not connected"}</div>
            <details className="chat-settings"><summary>Settings</summary><div className="chat-settings-popover">
              <div className="config-heading"><div><h2>Model connection</h2><p>Provider credentials stay on the API server.</p></div><span className={`badge ${dashboard?.configured ? "success" : "neutral"}`}>{dashboard?.configured ? "READY" : "NOT CONNECTED"}</span></div>
              <div className="config-fields"><label>Provider<select value={provider} onChange={(event) => setProvider(event.target.value)}><option value="cohere">Cohere</option><option value="openai">OpenAI</option><option value="huggingface">Hugging Face</option></select></label><label>Model<input value={model} onChange={(event) => setModel(event.target.value)} /></label><button className="button primary" onClick={configure} disabled={!sessionId || busy}>{busy ? "Working…" : "Connect model"}</button></div>
              <div className="chat-retrieval-settings"><label>Retrieval mode<select value={mode} onChange={(event) => setMode(event.target.value)}><option value="hybrid">Hybrid</option><option value="dense">Dense</option><option value="sparse">Sparse</option></select></label><label>Top-K passages<select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>{[3, 5, 8, 10, 15].map((value) => <option key={value} value={value}>{value}</option>)}</select></label></div>
              <button className="button quiet retrieve-settings-button" onClick={retrieveEvidence} disabled={busy || !question.trim()}>Retrieve evidence for draft question</button>
            </div></details>
          </div>
          <div className="chat-thread">
            {!previousTurns.length && !answer && <div className="chat-welcome"><span className="welcome-mark">B</span><h2>What would you like to research?</h2><p>Ask a biomedical question. Responses are grounded in your indexed evidence.</p><div className="suggestion-row">{["How does Parkinson’s disease affect dopamine?", "What are common symptoms of diabetes?"].map((suggestion) => <button key={suggestion} onClick={() => setQuestion(suggestion)}>{suggestion}<span>↗</span></button>)}</div></div>}
            {[...previousTurns].reverse().map((turn) => { const decision = String(turn.decision ?? (turn.success ? "answer_approved" : "insufficient_evidence")); const warnings = Array.isArray(turn.warnings) ? turn.warnings : []; const hasEvidenceRecord = Array.isArray(turn.retrieved_evidence); const savedEvidence = hasEvidenceRecord ? turn.retrieved_evidence as EvidenceSource[] : []; const savedNames = Array.isArray(turn.sources) ? turn.sources.map((source) => ({ source: String(source) })) : []; const oldSources = savedEvidence.length ? savedEvidence : savedNames; const body = String(turn.answer ?? turn.error ?? ""); const state = historyResultState(turn); const hasAnswer = state === "approved" || state === "informational"; const missingEvidenceMessage = hasEvidenceRecord ? "The retrieved source record contains no passage text." : "This response predates passage retention. Re-run the question to capture its evidence excerpts."; return <article className="chat-turn" key={turn.id}><div className="user-message"><span className="message-avatar">You</span><p>{String(turn.query ?? "")}</p></div><div className="assistant-message"><span className="assistant-avatar">B</span><div className="assistant-body"><div className="answer-status"><span className={`outcome-pill ${outcomeTone(state)}`}>{outcomeLabel(state)}</span></div><p><AnswerText text={answerParts(stateAnswer(state, hasAnswer || state === "safety_override" ? body : "", Number(turn.source_count ?? oldSources.length))).answer} /></p><EvidenceConfidence level={String(turn.evidence_confidence ?? "")} basis={String(turn.confidence_basis ?? "")} /><SourceSummary sources={oldSources} />{state !== "approved" && <details className="why-answer"><summary>{state === "validation_review_required" ? "Validation findings" : "Why this result?"}</summary><p>{outcomeWhySummary(state, decision, String(turn.rejection_reason ?? ""))}</p><p>{state === "validation_review_required" ? "Evidence found" : "Retrieved"}: {String(turn.source_count ?? 0)} passages.</p>{warnings.length > 0 && <ul>{warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}</ul>}</details>}<details className="inline-evidence" open><summary>Evidence and sources ({oldSources.length || Number(turn.source_count ?? 0)})</summary><EvidenceList sources={oldSources} emptyMessage={missingEvidenceMessage} missingTextMessage={missingEvidenceMessage} /></details><button className="text-button view-audit-link" onClick={() => setTab("Audit")}>View audit →</button></div></div></article>; })}
            {answer && <article className="chat-turn current-turn"><div className="user-message"><span className="message-avatar">You</span><p>{answeredQuestion}</p></div><div className="assistant-message"><span className="assistant-avatar">B</span><div className={`assistant-body ${(currentResultState === "insufficient" || currentResultState === "evidence_extraction_failed") ? "abstained" : ""}`}>
              <div className="answer-status"><span className={`outcome-pill ${outcomeTone(currentResultState)}`}>{outcomeLabel(currentResultState)}</span></div>
              {currentResultState === "informational" && <p className="outcome-subtitle">Evidence was found, but source verification or one or more validation checks are incomplete. Review the sources before relying on this answer.</p>}
              {currentResultState === "approved" && <p className="outcome-subtitle">Citation and support checks passed. This does not establish clinical accuracy.</p>}
              {currentResultState === "insufficient" || currentResultState === "evidence_extraction_failed" ? <p className="chat-answer-text abstained-message">{currentAnswerBody}</p> : (() => { const parts = answerParts(currentAnswerBody); return <><p className="chat-answer-text"><AnswerText text={parts.answer} /></p>{parts.disclaimer && <p className="medical-disclaimer"><span aria-hidden="true">⚠</span><span><strong>Medical disclaimer</strong>{parts.disclaimer}</span></p>}</>; })()}
              <EvidenceConfidence level={answer.details.evidence_confidence} basis={answer.details.confidence_basis} />
              <SourceSummary sources={answer.sources} />
              <div className="validation-states"><strong>Validation checks</strong><div className="validation-grid">{Object.entries(answer.details.validation_states ?? {}).filter(([key]) => key !== "claim_support_method").map(([key, value]) => { const result = key === "ANSWER_QUALITY_EVALUATED" && currentAnswerReviewed ? { label: "Human reviewed", state: "passed" } : validationValue(value); return <div className={`validation-item ${result.state}`} key={key}><span>{VALIDATION_LABELS[key] ?? key.replaceAll("_", " ")}</span><b>{result.label}</b></div>; })}</div><small>Answer quality is not scored automatically; use Human answer review for human ratings. Claim support is a lexical overlap screen, not semantic verification. Medical accuracy requires qualified human review and is not established by this system.</small>
                {answer.details.validation_diagnostics && <div className="audit-metrics"><strong>Lexical audit diagnostics</strong><span>Citation coverage <b>{percentage(answer.details.validation_diagnostics.citation_coverage)}</b></span><span>Citation-linked support <b>{percentage(answer.details.validation_diagnostics.citation_support_coverage)}</b></span><span>Lexical overlap <b>{percentage(answer.details.validation_diagnostics.lexical_support)}</b></span><small>Heuristic diagnostics, not calibrated confidence.</small></div>}
              </div>
              {(currentResultState !== "approved") && <details className="why-answer"><summary>{currentResultState === "validation_review_required" ? "Validation findings" : "Why this result?"}</summary><p>{outcomeWhySummary(currentResultState, answer.details.decision, answer.details.rejection_reason)}</p>{answer.details.risk_tier === "R4" && <p>If symptoms may be life-threatening, contact local emergency services now.</p>}<p>{currentResultState === "validation_review_required" ? "Evidence found" : "Retrieved"}: {answer.sources.length} passage{answer.sources.length === 1 ? "" : "s"}.</p>{!!answer.details.warnings?.length && <ul>{answer.details.warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}</ul>}{currentResultState !== "safety_override" && <p>Source quality and clinical accuracy are not independently verified by BioMed RAG.</p>}</details>}
              {!!answer.details.warnings?.length && <details className="validation-details"><summary>Why checks failed ({answer.details.warnings.length})</summary><ul>{answer.details.warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}</ul></details>}
              {answer.details.generation_success !== false && !answer.details.answer_abstained && <button className="text-button" onClick={() => { setAuditAnswerText(answer.answer); setAuditEvidence(answer.sources.map((source) => (source.text ?? "").replace(/\s+/g, " ").trim()).join("\n")); setTab("Evaluation"); }}>Audit this answer →</button>}
              <button className="text-button view-audit-link" onClick={() => setTab("Audit")}>View audit →</button>
              <details className="inline-evidence" open><summary>Evidence and sources ({answer.sources.length})</summary><EvidenceList sources={answer.sources} emptyMessage="No passage text was returned by retrieval for this response." /></details>
            </div></div></article>}
            {!answer && retrievalAttempted && <div className="assistant-message retrieval-result"><span className="assistant-avatar">B</span><div><strong>Retrieved evidence</strong><EvidenceList sources={retrieved} emptyMessage="No topic-matched passages were found. Try a more specific query or add a directly relevant source." /></div></div>}
            <div ref={chatEndRef} />
          </div>
          <form className="chat-composer" onSubmit={(event) => { event.preventDefault(); submitQuery(); }}>
            <label className="sr-only" htmlFor="chat-question">Ask a biomedical question</label><textarea id="chat-question" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submitQuery(); } }} placeholder="Ask a biomedical question" rows={2} />
            <div className="composer-bottom"><span>Evidence-grounded · Research use only</span><div><button type="button" className="button quiet" onClick={retrieveEvidence} disabled={busy || !dashboard?.configured || !question.trim()}>Retrieve evidence</button><button type="submit" className="send-button" aria-label="Send question" disabled={busy || !dashboard?.configured || !question.trim()}>{busy ? "…" : "↑"}</button></div></div>
          </form>
          {!dashboard?.configured && <p className="help-text chat-connection-hint">{apiToken ? "Connect a model in Settings before sending a question." : "Connect to the API to load a session and configure a model."}</p>}
          <details className="review-details" hidden><summary>Human answer review</summary><div className="review-panel">{!reviewableEvents.length && <p className="empty">Run a query before reviewing its answer or abstention.</p>}<div className="rating-grid">{QUALITY_DIMENSIONS.map((dimension) => <label key={dimension}>{dimension.replaceAll("_", " ")}<select value={reviewRatings[dimension]} onChange={(event) => setReviewRatings({ ...reviewRatings, [dimension]: Number(event.target.value) })}>{[1, 2, 3, 4, 5].map((score) => <option key={score}>{score}</option>)}</select></label>)}</div><label className="stacked-label">Reviewer notes<textarea rows={2} value={reviewNotes} onChange={(event) => setReviewNotes(event.target.value)} /></label><button className="button" onClick={submitReview} disabled={busy || !reviewableEvents.length}>Save human review</button></div></details>
        </section>}

        {tab === "Playground" && <section className="content-section legacy-playground" hidden>
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
              {!dashboard?.configured && <p className="help-text">{apiToken ? "Connect a model above before querying." : "Connect to the API to load a session and configure a model."}</p>}
              {!!previousTurns.length && <div className="chat-history" aria-label="Previous conversation turns"><div className="history-heading">Previous questions in this session</div>{previousTurns.map((turn) => <article className="history-turn" key={turn.id}><p><strong>You</strong>{turn.query}</p><p><strong>{turn.success ? "BioMed RAG" : "Generation failed"}</strong>{String(turn.answer ?? turn.error ?? "No answer was recorded.")}</p></article>)}</div>}
              {answer && <div className="answer-card"><div className="answer-status">{answer.details.generation_success === false ? "GENERATION FAILED" : answer.details.answer_abstained ? "ABSTAINED" : "CHECKS REPORTED BELOW"}<span>{(answer.latency_ms / 1000).toFixed(1)} s</span></div>{(() => { const parts = answerParts(answer.answer); return <><p>{parts.answer}</p>{parts.disclaimer && <p className="medical-disclaimer"><span aria-hidden="true">⚠</span><span><strong>Medical disclaimer</strong>{parts.disclaimer}</span></p>}</>; })()}
                <div className="validation-states"><strong>Validation checks</strong><div className="validation-grid">{Object.entries(answer.details.validation_states ?? {}).filter(([key]) => key !== "claim_support_method").map(([key, value]) => { const result = validationValue(value); return <div className={`validation-item ${result.state}`} key={key}><span>{VALIDATION_LABELS[key] ?? key.replaceAll("_", " ")}</span><b>{result.label}</b></div>; })}</div><small>Claim support is a lexical overlap screen. Medical accuracy is not established by this system.</small></div>
                {!!answer.details.warnings?.length && <details className="validation-details"><summary>Why checks failed ({answer.details.warnings.length})</summary><ul>{answer.details.warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}</ul></details>}
                {answer.details.generation_success !== false && !answer.details.answer_abstained && <button className="text-button" onClick={() => { setAuditAnswerText(answer.answer); setAuditEvidence(answer.sources.map((source) => (source.text ?? "").replace(/\s+/g, " ").trim()).join("\n")); setTab("Evaluation"); }}>Audit this answer →</button>}
              </div>}
            </div>
            <div className="panel"><div className="panel-heading"><h3>Retrieved evidence</h3><span>{answer?.sources.length ?? retrieved.length} sources</span></div>
              <EvidenceList sources={answer?.sources ?? retrieved} emptyMessage={retrievalAttempted ? "No topic-matched passages were found. Try a more specific query or add a directly relevant source." : undefined} />
            </div>
          </div>
          <div className="panel review-panel"><div className="panel-heading"><h3>Human answer review</h3><span>1 = low · 5 = high</span></div>
            {!reviewableEvents.length ? <p className="empty">Run a query before reviewing its answer or abstention.</p> :
              <><div className="rating-grid">{QUALITY_DIMENSIONS.map((dimension) => <label key={dimension}>{dimension.replaceAll("_", " ")}<select value={reviewRatings[dimension]} onChange={(event) => setReviewRatings({ ...reviewRatings, [dimension]: Number(event.target.value) })}>{[1, 2, 3, 4, 5].map((score) => <option key={score}>{score}</option>)}</select></label>)}</div>
                <label className="stacked-label">Reviewer notes<textarea rows={2} value={reviewNotes} onChange={(event) => setReviewNotes(event.target.value)} /></label>
                <button className="button" onClick={submitReview} disabled={busy}>Save human review</button></>}
          </div>
        </section>}

        {tab === "Evaluation" && <section className="content-section">
          <div className="section-heading"><div><h2>Evaluation lab</h2><p>Transparent source-level metrics and lexical answer audit proxies for human review.</p></div></div>
          <div className="metric-grid compact evaluation-outcomes">
            <Metric label="Answer rate" value={dashboard?.summary.answer_rate == null ? "—" : `${Math.round(dashboard.summary.answer_rate * 100)}%`} />
            <Metric label="Faithfulness proxy" value={percentage(dashboard?.summary.answer_audits.faithfulness)} />
            <Metric label="Citation-linked coverage" value={percentage(dashboard?.summary.answer_audits.citation_support_coverage)} />
            <Metric label={`Precision@${topK}`} value={dashboard?.summary.retrieval_metrics[`precision@${topK}`] == null ? "—" : Number(dashboard.summary.retrieval_metrics[`precision@${topK}`]).toFixed(3)} />
            <Metric label={`Recall@${topK}`} value={dashboard?.summary.retrieval_metrics[`recall@${topK}`] == null ? "—" : Number(dashboard.summary.retrieval_metrics[`recall@${topK}`]).toFixed(3)} />
            <Metric label="Grounding rate" value={dashboard?.summary.successful_queries ? percentage(dashboard.summary.grounded_queries / dashboard.summary.successful_queries) : "—"} />
          </div>
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
          <div className="panel review-panel"><div className="panel-heading"><h3>Human answer review</h3><span>1 = low · 5 = high</span></div>
            {!reviewableEvents.length ? <p className="empty">Run a query before reviewing its answer or abstention.</p> :
              <><div className="rating-grid">{QUALITY_DIMENSIONS.map((dimension) => <label key={dimension}>{dimension.replaceAll("_", " ")}<select value={reviewRatings[dimension]} onChange={(event) => setReviewRatings({ ...reviewRatings, [dimension]: Number(event.target.value) })}>{[1, 2, 3, 4, 5].map((score) => <option key={score}>{score}</option>)}</select></label>)}</div>
                <label className="stacked-label">Reviewer notes<textarea rows={2} value={reviewNotes} onChange={(event) => setReviewNotes(event.target.value)} /></label>
                <button className="button" onClick={submitReview} disabled={busy}>Save human review</button></>}
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
          <div className="section-heading"><div><h2>Audit history</h2><p>Inspect model prompts, generated drafts, retrieved passages, and validation results.</p></div><button className="button" onClick={exportAudit} disabled={busy}>Download JSON</button></div>
          <div className="metric-grid compact">
            <Metric label="All events" value={dashboard?.events.length ?? 0} />
            <Metric label="Approved answers" value={dashboard?.summary.successful_queries ?? 0} />
            <Metric label="Abstentions" value={dashboard?.summary.abstained_queries ?? 0} />
            <Metric label="Generation failures" value={dashboard?.summary.failed_queries ?? 0} />
            <Metric label="Answer reviews" value={dashboard?.answer_quality_reviews.length ?? 0} />
          </div>
          <div className="panel"><EventTable events={[...(dashboard?.events ?? [])].reverse()} showTrace /></div>
          <div className="panel"><div className="panel-heading"><h3>Human quality reviews</h3><span>{dashboard?.answer_quality_reviews.length ?? 0}</span></div><ResultList items={dashboard?.answer_quality_reviews ?? []} /></div>
          <p className="disclaimer">This session is held in server memory. Export data you need to retain; it is not durable storage.</p>
        </section>}

        <footer className="page-footer">Biomedical RAG · Research and educational use only · Session-scoped audit data</footer>
      </main>
    </div>
  );
}

function EvidenceList({ sources, emptyMessage, missingTextMessage = "No source excerpt was retained with this result.", showScores = false }: { sources: EvidenceSource[]; emptyMessage?: string; missingTextMessage?: string; showScores?: boolean }) {
  if (!sources.length) return <p className="empty">{emptyMessage ?? "Evidence will appear after retrieval."}</p>;
  return <div className="evidence-list">{sources.map((source, index) => <article className="evidence-item" key={`${source.source ?? "source"}-${index}`}>
    <div id={source.evidence_id ?? `evidence-${index + 1}`} className="evidence-heading"><span className="source-rank">{source.evidence_id ?? `Evidence ${index + 1}`}</span><strong>{source.source ?? "Biomedical source"}</strong>{source.source_origin === "external" && <span className="source-origin">External research · not independently verified</span>}{source.url?.startsWith("https://") && <a className="source-link" href={source.url} target="_blank" rel="noreferrer">Open source ↗</a>}{showScores && (source.rerank_score ?? source.score) != null && <span className="source-score">rank score {Number(source.rerank_score ?? source.score).toFixed(3)}</span>}</div>
    <p>{source.text?.trim() || missingTextMessage}</p>
    {!!source.prompt_excerpt?.trim() && source.prompt_excerpt.trim() !== source.text?.trim() && <details className="prompt-excerpt"><summary>Passage excerpt used for the answer</summary><p>{source.prompt_excerpt.trim()}</p></details>}
  </article>)}</div>;
}

function AnswerText({ text }: { text: string }) {
  const parts: Array<string | { citation: string; label: string }> = [];
  const citationPattern = /\[Evidence\s+(\d+)\]/gi;
  let cursor = 0;
  for (const match of text.matchAll(citationPattern)) {
    const index = match.index ?? 0;
    if (index > cursor) parts.push(text.slice(cursor, index));
    parts.push({ citation: `evidence-${match[1]}`, label: match[0] });
    cursor = index + match[0].length;
  }
  if (cursor < text.length) parts.push(text.slice(cursor));
  return <>{parts.map((part, index) => typeof part === "string"
    ? <span key={`answer-text-${index}`}>{part}</span>
    : <a className="inline-citation" href={`#${part.citation}`} key={`answer-citation-${index}`}>{part.label}</a>)}</>;
}

function EvidenceConfidence({ level, basis }: { level?: string | null; basis?: string | null }) {
  if (!level) return null;
  return <p className="evidence-confidence"><strong>Evidence confidence:</strong> {level[0].toUpperCase() + level.slice(1)}{basis && <small>{basis}</small>}</p>;
}

function sourceProvider(source: EvidenceSource) {
  const identity = `${source.source_provider ?? ""} ${source.source ?? ""}`.toLowerCase();
  if (identity.includes("medlineplus")) return "MedlinePlus";
  if (identity.includes("pubmed") || identity.includes("pmid") || identity.includes("ncbi")) return "PubMed";
  if (source.source_origin === "external") return "Internet";
  return source.source_provider || source.source || "Biomedical source";
}

function SourceSummary({ sources }: { sources: EvidenceSource[] }) {
  if (!sources.length) return null;
  const providers = [...new Set(sources.map(sourceProvider))];
  return <p className="source-summary"><strong>Sources:</strong> {providers.join(" · ")}</p>;
}

function eventStatus(event: AuditEvent) {
  if (event.decision === "answer_approved") return "Verified";
  if (event.decision === "informational") return "Informational";
  if (event.decision === "validation_review_required") return "Validation review";
  if (event.decision === "safety_override") return "Safety override";
  if (event.decision === "out_of_scope") return "Out of scope";
  if (event.decision === "insufficient_evidence" && Number(event.source_count ?? 0) > 0 && Array.isArray(event.retrieved_evidence) && (event.retrieved_evidence as EvidenceSource[]).length > 0 && !(event.retrieved_evidence as EvidenceSource[]).some((source) => Boolean(source.text?.trim()))) return "Evidence extraction failed";
  if (event.decision === "insufficient_evidence" || event.answer_abstained) return "Insufficient supporting evidence";
  if (event.success === undefined) return "Recorded";
  return event.success ? "Success" : "Failed";
}

function historyResultState(event: AuditEvent) {
  const state = resultState(
    event.decision,
    event.answer_abstained,
    event.rejection_reason,
    event.success,
    event.answer,
  );
  const hasSourceRecords = Number(event.source_count ?? 0) > 0;
  const passages = Array.isArray(event.retrieved_evidence) ? event.retrieved_evidence as EvidenceSource[] : [];
  const hasUsableExcerpt = passages.some((source) => Boolean(source.text?.trim()));
  if (state === "insufficient" && hasSourceRecords && passages.length > 0 && !hasUsableExcerpt) {
    return "evidence_extraction_failed" as const;
  }
  return state;
}

function EventTable({ events, showTrace = false }: { events: AuditEvent[]; showTrace?: boolean }) {
  if (!events.length) return <p className="empty">No activity has been recorded in this session.</p>;
  return <div className="table-wrap"><table><thead><tr><th>Time</th><th>Event</th><th>Status</th><th>Question / source</th><th>Evidence</th><th>Latency</th>{showTrace && <th>Trace</th>}</tr></thead><tbody>
    {events.map((event) => {
      const trace = event.audit_trace && typeof event.audit_trace === "object" ? event.audit_trace as Record<string, unknown> : {};
      const evidence = Array.isArray(event.retrieved_evidence) ? event.retrieved_evidence as EvidenceSource[] : [];
      const warnings = Array.isArray(event.warnings) ? event.warnings : [];
      return <tr key={event.id}><td>{new Date(event.timestamp).toLocaleString()}</td><td>{event.event}</td><td><span className={`event-status ${event.decision === "answer_approved" ? "success" : event.decision === "informational" || event.decision === "validation_review_required" ? "warning" : event.success === false ? "failed" : ""}`}>{eventStatus(event)}</span></td><td>{String(event.query ?? event.source ?? event.error ?? "—")}</td><td>{String(event.source_count ?? "—")}</td><td>{event.latency_ms == null ? "—" : `${Number(event.latency_ms).toFixed(0)} ms`}</td>{showTrace && <td><details className="event-trace"><summary>Inspect</summary><div><strong>Decision</strong><p>{String(event.decision ?? "Recorded event")}{event.rejection_reason ? ` · ${rejectionLabel(String(event.rejection_reason))}` : ""}</p><strong>Model prompt</strong><pre className="json-result">{JSON.stringify(trace.model_prompt ?? "Not captured for this event.", null, 2)}</pre><strong>Response sent to user</strong><pre className="json-result">{String(event.answer ?? "No response was recorded.")}</pre><strong>Validation results</strong><pre className="json-result">{JSON.stringify({ states: event.validation_states, diagnostics: event.validation_diagnostics, warnings: warnings, security: event.security }, null, 2)}</pre><strong>Retrieved passages</strong><EvidenceList sources={evidence} emptyMessage="No passage text was recorded for this event." showScores /></div></details></td>}</tr>;
    })}
  </tbody></table></div>;
}

function ResultList({ items }: { items: Array<Record<string, unknown>> }) {
  if (!items.length) return <p className="empty">No results recorded.</p>;
  return <div className="result-list">{[...items].reverse().map((item, index) => <details key={String(item.id ?? index)}>
    <summary>{String(item.title ?? item.timestamp ?? "Evaluation result")} <span>{String(item.level ?? item.severity ?? item.status ?? "")}</span></summary>
    <pre className="json-result">{JSON.stringify(item, null, 2)}</pre>
  </details>)}</div>;
}
