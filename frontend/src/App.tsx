import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
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
type EvidenceSource = { evidence_id?: string; citation_index?: number; id?: number; source?: string; title?: string; document_title?: string; text?: string; prompt_excerpt?: string; score?: number; rerank_score?: number; rank?: number; url?: string; source_provider?: string; source_origin?: string; source_quality?: string; evidence_classification?: { category?: string; reason?: string } };
type QueryResult = {
  event_id: string;
  answer: string;
  sources: EvidenceSource[];
  details: { is_grounded?: boolean; generation_success?: boolean; answer_abstained?: boolean; decision?: string; rejection_reason?: string | null; risk_tier?: string; question_category?: string; evidence_confidence?: string | null; confidence_basis?: string | null; learning_answer?: string | null; claim_support?: Array<Record<string, unknown>>; audit_trace?: { retrieval_configuration?: { mode?: string; expanded_query?: string; expanded_queries?: string[]; executed_queries?: string[]; top_ranked_candidates?: Array<Record<string, unknown>> }; evidence_audit?: { relevance_counts?: Record<string, number>; reason?: string; coverage_assessment?: { status?: string; reason?: string; coverage_score?: number | null; question_components?: Array<Record<string, unknown>> } }; validation_issues?: string[] }; validation_diagnostics?: { generated_claims?: number; supported_claims?: number; unsupported_claims?: number; coverage_score?: number | null; comparison_coverage_score?: number | null; method?: string; factual_sentence_count?: number; citation_coverage?: number | null; citation_support_coverage?: number | null; lexical_support?: number | null } | null; warnings?: string[]; validation_passed?: boolean; validation_states?: Record<string, boolean | string | null> };
  latency_ms: number;
};
type SavedItem = { id: string; title: string; body: string; created_at: string };
type DisplayMode = "research" | "learning" | "compare";

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

type ResultState = "approved" | "informational" | "validation_review_required" | "conflicting_evidence" | "insufficient" | "no_relevant_evidence" | "evidence_extraction_failed" | "safety_override" | "out_of_scope" | "generation_failed";

function resultState(decision?: string, answerAbstained?: boolean, reason?: string | null, success?: boolean, answer?: string): ResultState {
  if (decision === "safety_override" || reason?.startsWith("medical_")) return "safety_override";
  if (decision === "out_of_scope" || reason === "non_medical_question") return "out_of_scope";
  if (decision === "validation_review_required") return "validation_review_required";
  if (decision === "conflicting_evidence") return "conflicting_evidence";
  if (decision === "no_relevant_evidence" || reason === "no_relevant_evidence") return "no_relevant_evidence";
  if (decision === "evidence_extraction_failed" || reason === "evidence_extraction_failed") return "evidence_extraction_failed";
  if (decision === "generation_failed") return "generation_failed";
  if (/could not produce an answer that passed|couldn.t provide a reliable answer|couldn.t find enough (?:supporting|relevant) evidence|could not find enough (?:supporting|relevant) evidence|generation failed:/i.test(answer ?? "")) return "insufficient";
  if (answerAbstained || success === false || decision === "insufficient_evidence") return "insufficient";
  if (decision === "informational") return "informational";
  if (decision === "answer_approved" || (decision == null && success)) return "approved";
  return "insufficient";
}

function stateAnswer(state: ResultState, answer: string, sourceCount = 0) {
  if (state === "validation_review_required") {
    return "I found related biomedical information, but the retrieved sources do not directly answer your question. Review the passages below or refine your question.";
  }
  if (state === "evidence_extraction_failed") return "I found source records, but usable passage text was not available. Review the source records or add a source with accessible text.";
  if (state === "no_relevant_evidence") return "I could not find a close match in the available sources. Try a more specific question or add a relevant source.";
  if (state === "out_of_scope") return "BioMed RAG answers medical and biomedical questions only. Ask about a health condition, symptom, treatment evidence, anatomy, or biology topic.";
  if (state === "generation_failed") return "Relevant evidence may be available, but answer generation failed. Please retry or review the retrieved sources.";
  if (state === "insufficient") return sourceCount > 0
    ? "I found related biomedical information, but these sources do not contain enough direct evidence to answer reliably. Review the passages below or refine your question."
    : "I could not find a close source match for this question. Try a more specific question or add a relevant source.";
  if (state === "safety_override" && /could not produce an answer that passed|couldn['’]t provide a reliable answer|generation failed:/i.test(answer)) {
    return "This question may involve urgent or individualized medical risk. Seek qualified medical help when appropriate.";
  }
  if (!answer.trim()) return sourceCount
    ? "The model returned an empty answer. Review the retrieved passages or retry the question."
    : "No answer text was returned. Please retry the question.";
  return answer;
}

function validationIssue(value: string) {
  const message = value.toLowerCase();
  if (message.includes("failed the lexical claim-support screen")) return "A factual claim could not be matched by the lexical support screen. This screen can miss valid paraphrases; inspect the cited passage.";
  if (message.includes("lack a citation linked to overlapping evidence")) return "A citation was present, but the passage did not overlap enough with the claim to verify the citation link.";
  if (message.includes("lack a citation")) return "A factual statement is missing an inline evidence citation.";
  if (message.includes("does not directly establish") || message.includes("related condition")) return "The retrieved passages concern a related subject but do not directly establish the requested information.";
  if (message.includes("contradict")) return "The retrieved sources contain opposing information; review their source context before drawing a conclusion.";
  return value;
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

function answerReferences(sources: EvidenceSource[], answerText: string, claimSupport: Array<Record<string, unknown>> = []) {
  const citedNumbers = [...answerText.matchAll(/\[Evidence\s+(\d+)\]/gi)]
    .map((match) => Number(match[1])).filter((value) => Number.isInteger(value) && value > 0);
  if (citedNumbers.length) {
    return [...new Set(citedNumbers)].map((number) => sources[number - 1]).filter(Boolean);
  }
  const supportedIds = new Set(claimSupport
    .filter((claim) => claim.support_status === "supported")
    .flatMap((claim) => Array.isArray(claim.evidence_ids) ? claim.evidence_ids.map(String) : []));
  if (!supportedIds.size) return [];
  return sources.filter((source) => source.evidence_id && supportedIds.has(source.evidence_id));
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

function loadSavedItems(key: string): SavedItem[] {
  try {
    const value = JSON.parse(window.localStorage.getItem(key) ?? "[]");
    return Array.isArray(value) ? value as SavedItem[] : [];
  } catch {
    return [];
  }
}

function learningExcerptFromSources(sources: EvidenceSource[]) {
  for (const [index, source] of sources.entries()) {
    const category = source.evidence_classification?.category;
    if (["irrelevant", "contradictory", "uncertain"].includes(String(category ?? ""))) continue;
    const passage = String(source.prompt_excerpt?.trim() || source.text || "").replace(/\s+/g, " ").trim();
    const sentences = passage.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [];
    const excerpt = sentences.map((sentence) => sentence.trim()).find((sentence) => sentence.length >= 35 && !sentence.endsWith("?"));
    if (excerpt) return `The retrieved passage says: ${excerpt} [Evidence ${index + 1}]`;
  }
  return "";
}

function isBearerAuthError(error: unknown): boolean {
  return error instanceof Error && error.message.toLowerCase().includes("bearer token");
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("Playground");
  const [viewHistory, setViewHistory] = useState<Tab[]>(["Playground"]);
  const [viewHistoryIndex, setViewHistoryIndex] = useState(0);
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
  const [displayMode, setDisplayMode] = useState<DisplayMode>("research");
  const [answerModeUsed, setAnswerModeUsed] = useState<DisplayMode>("research");
  const [supersededAnswerIds, setSupersededAnswerIds] = useState<Set<string>>(() => new Set());
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarSection, setSidebarSection] = useState<"recent" | "notes" | "bookmarks">("recent");
  const [savedNotes, setSavedNotes] = useState<SavedItem[]>(() => loadSavedItems("biomed-rag-notes"));
  const [bookmarks, setBookmarks] = useState<SavedItem[]>(() => loadSavedItems("biomed-rag-bookmarks"));
  const [noteDraft, setNoteDraft] = useState("");
  const [noteEditorOpen, setNoteEditorOpen] = useState(false);
  const questionInputRef = useRef<HTMLTextAreaElement>(null);
  function navigateToTab(nextTab: Tab) {
    if (nextTab === tab) return;
    setViewHistory((history) => [...history.slice(0, viewHistoryIndex + 1), nextTab]);
    setViewHistoryIndex((index) => index + 1);
    setTab(nextTab);
  }
  function moveViewHistory(direction: -1 | 1) {
    const nextIndex = viewHistoryIndex + direction;
    if (nextIndex < 0 || nextIndex >= viewHistory.length) return;
    setViewHistoryIndex(nextIndex);
    setTab(viewHistory[nextIndex]);
    setSidebarOpen(false);
  }
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
      chatEndRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
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
        answer_mode: displayMode,
      }),
      (result) => {
        setAnswer(result);
        setAnswerModeUsed(displayMode);
        setAnsweredQuestion(submittedQuestion);
        setQuestion("");
        setRetrievalAttempted(true);
      },
    );
  }

  function changeDisplayMode(nextMode: DisplayMode) {
    if (busy || (nextMode === displayMode && (!answer || answerModeUsed === nextMode))) return;
    setDisplayMode(nextMode);
    if (!answer || !answeredQuestion || !sessionId) return;
    void perform(
      () => post<QueryResult>("/api/query", {
        session_id: sessionId,
        question: answeredQuestion,
        top_k: topK,
        retrieval_mode: mode,
        answer_mode: nextMode,
        supersedes_event_id: answer.event_id,
      }),
      (result) => {
        setSupersededAnswerIds((ids) => new Set(ids).add(answer.event_id));
        setAnswer(result);
        setAnswerModeUsed(nextMode);
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
        navigateToTab("Knowledge");
        setSidebarOpen(false);
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

  function exportCurrentAnswer() {
    if (!answer) {
      exportAudit();
      return;
    }
    const visibleAnswer = answerModeUsed === "learning" && answer.details.learning_answer?.trim()
      ? answerParts(answer.details.learning_answer).answer
      : currentAnswerBody;
    const sources = answerReferences(answer.sources, visibleAnswer, answer.details.claim_support ?? []);
    const markdown = `# ${answeredQuestion}\n\n${visibleAnswer}${referenceMarkdown(sources)}`;
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "biomedgpt-research-note.md";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  function copyAnswer(text: string, sources: EvidenceSource[]) {
    const content = `${text}${referenceMarkdown(sources)}`;
    void navigator.clipboard?.writeText(content);
  }

  function saveCurrentNote() {
    if (!answer || !noteDraft.trim()) return;
    const item: SavedItem = {
      id: `${Date.now()}`,
      title: answeredQuestion,
      body: `${noteDraft.trim()}\n\n${currentAnswerBody}${referenceMarkdown(answerReferences(answer.sources, currentAnswerBody, answer.details.claim_support ?? []))}`,
      created_at: new Date().toISOString(),
    };
    const updated = [item, ...savedNotes];
    setSavedNotes(updated);
    window.localStorage.setItem("biomed-rag-notes", JSON.stringify(updated));
    setNoteDraft("");
    setNoteEditorOpen(false);
    setSidebarSection("notes");
  }

  function bookmarkCurrentAnswer() {
    if (!answer) return;
    const item: SavedItem = {
      id: answer.event_id,
      title: answeredQuestion,
      body: `${currentAnswerBody}${referenceMarkdown(answerReferences(answer.sources, currentAnswerBody, answer.details.claim_support ?? []))}`,
      created_at: new Date().toISOString(),
    };
    const updated = [item, ...bookmarks.filter((bookmark) => bookmark.id !== item.id)];
    setBookmarks(updated);
    window.localStorage.setItem("biomed-rag-bookmarks", JSON.stringify(updated));
    setSidebarSection("bookmarks");
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
      setSupersededAnswerIds(new Set());
      setDisplayMode("research");
      setAnswerModeUsed("research");
      setRetrieved([]);
      setRetrievalAttempted(false);
      setRelevantSources("");
      setAuditAnswerText("");
      setAuditEvidence("");
      setAnswerAudit(null);
      setSecurityText("");
      setSecurityReport(null);
      setReviewNotes("");
      navigateToTab("Playground");
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
  const queryEvents = (dashboard?.events ?? []).filter((event) => event.event === "query");
  const supersededEventIds = new Set(supersededAnswerIds);
  for (const event of dashboard?.events ?? []) {
    if (typeof event.supersedes_event_id === "string") supersededEventIds.add(event.supersedes_event_id);
  }
  for (let index = 1; index < queryEvents.length; index += 1) {
    const previous = queryEvents[index - 1];
    const current = queryEvents[index];
    if (String(previous.query ?? "").trim() === String(current.query ?? "").trim()
      && previous.answer_mode && current.answer_mode && previous.answer_mode !== current.answer_mode) {
      supersededEventIds.add(previous.id);
    }
  }
  const previousTurns = (dashboard?.events ?? [])
    .filter((event) => event.event === "query" && event.id !== answer?.event_id && !supersededEventIds.has(event.id) && (event.query || event.answer))
    .slice(-12)
    .reverse();
  const recentQuestions = (dashboard?.events ?? [])
    .filter((event) => event.event === "query" && event.query && !supersededEventIds.has(event.id))
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
    ? answerParts(stateAnswer(currentResultState, answer.answer, answer.sources.length)).answer
    : "";
  const learningAnswerText = answerModeUsed === "learning" && answer
    ? answer.details.learning_answer?.trim()
      || (answer.details.answer_abstained ? learningExcerptFromSources(answer.sources).trim() : "")
      || currentAnswerBody
    : currentAnswerBody;
  const retrievalDebug = answer?.details.audit_trace;
  const displayAnswerText = learningAnswerText.trim()
    ? learningAnswerText
    : answer ? "No answer text was returned. Review the sources or retry the question." : "";

  return (
    <div className={`app-shell mode-${displayMode}`}>
      {sidebarOpen && <button className="sidebar-scrim" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand"><span className="brand-mark">B</span><div><strong>BioMedGPT</strong><small>BIOMEDICAL RESEARCH</small></div></div>
        <div className="sidebar-history" aria-label="Menu navigation">
          <button type="button" aria-label="Back" title="Back" onClick={() => moveViewHistory(-1)} disabled={viewHistoryIndex === 0}>←</button>
          <button type="button" aria-label="Forward" title="Forward" onClick={() => moveViewHistory(1)} disabled={viewHistoryIndex >= viewHistory.length - 1}>→</button>
        </div>
        {tab === "Playground" && <button className="new-chat-button" onClick={startNewChat} disabled={busy || !sessionId}><span>＋</span>New chat</button>}
        <div className="sidebar-section-heading">{sidebarSection === "recent" ? "Recent Research" : sidebarSection === "notes" ? "Saved Notes" : "Bookmarks"}</div>
        <div className="recent-list">
          {sidebarSection === "recent" && (recentQuestions.length ? recentQuestions.slice(0, 8).map((event) => <button key={event.id} onClick={() => { navigateToTab("Playground"); setSidebarOpen(false); window.setTimeout(() => document.getElementById(`turn-${event.id}`)?.scrollIntoView({ behavior: "smooth", block: "center" }), 80); }} title={String(event.query ?? "")}>{String(event.query ?? "")}</button>) : <p>Your research questions will appear here.</p>)}
          {sidebarSection === "notes" && (savedNotes.length ? savedNotes.map((note) => <article className="saved-sidebar-item" key={note.id}><strong>{note.title}</strong><small>{note.body.slice(0, 110)}</small></article>) : <p>Save a note from an answer to keep it here.</p>)}
          {sidebarSection === "bookmarks" && (bookmarks.length ? bookmarks.map((item) => <article className="saved-sidebar-item" key={item.id}><strong>{item.title}</strong><small>{item.body.slice(0, 110)}</small></article>) : <p>Bookmark useful answers to find them here.</p>)}
        </div>
        <nav className="product-nav" aria-label="Research tools">
          <button className={tab === "Playground" ? "selected" : ""} onClick={() => { navigateToTab("Playground"); setSidebarOpen(false); }}>Chat</button>
          <button className={tab === "Knowledge" ? "selected" : ""} onClick={() => { navigateToTab("Knowledge"); setSidebarOpen(false); }}>Research Library</button>
          <button className={sidebarSection === "notes" ? "selected" : ""} onClick={() => setSidebarSection("notes")}>Saved Notes <span>{savedNotes.length || ""}</span></button>
          <button className={sidebarSection === "bookmarks" ? "selected" : ""} onClick={() => setSidebarSection("bookmarks")}>Bookmarks <span>{bookmarks.length || ""}</span></button>
          <button onClick={exportCurrentAnswer}>Export {answer ? "Answer" : "Research"}</button>
        </nav>
        <details className="workspace-tools"><summary>Workspace tools</summary><nav aria-label="Workspace tools">
          {(["Overview", "Evaluation", "Governance", "Audit"] as Tab[]).map((item) => <button key={item} onClick={() => { navigateToTab(item); setSidebarOpen(false); }}>{item}</button>)}
        </nav></details>
        <div className="sidebar-bottom"><span className={`status-dot ${dashboard?.configured ? "ready" : ""}`} /><span>{displayMode[0].toUpperCase() + displayMode.slice(1)} mode · {dashboard?.configured ? "Model connected" : "Connect a model in Settings"}</span><small>Research and education only</small></div>
      </aside>

      <main className="main-content">
        <header className={`topbar ${tab === "Playground" ? "product-topbar" : ""}`}>
          <div className="topbar-title">{tab === "Playground" && <button className="mobile-menu-button" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}>☰</button>}<div><p className="eyebrow">{tab === "Playground" ? "BIOMEDGPT · EVIDENCE-BASED RESEARCH" : "BIOMEDICAL RESEARCH WORKSPACE"}</p><h1>{tab === "Playground" ? "BioMedGPT" : tab}</h1></div></div>
          <div className="top-actions">
            {tab === "Playground" && <div className="mode-toggle" role="group" aria-label="Answer mode"><span>Mode</span>{(["research", "learning", "compare"] as DisplayMode[]).map((modeOption) => <button key={modeOption} disabled={busy} aria-pressed={displayMode === modeOption} className={displayMode === modeOption ? "selected" : ""} onClick={() => changeDisplayMode(modeOption)}>{modeOption[0].toUpperCase() + modeOption.slice(1)}</button>)}</div>}
            <span className="config-pill">{dashboard?.configuration.version ?? (error ? "API unavailable" : "Connecting…")}</span>
            {tab !== "Playground" && <button className="button quiet" onClick={exportAudit} disabled={!dashboard || busy}>Export audit</button>}
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
          {(retrievalAttempted || answer?.sources.length) && <div className="panel research-evidence-panel"><div className="panel-heading"><h3>Retrieved evidence</h3><span>{answer?.sources.length ?? retrieved.length} passages</span></div><EvidenceList sources={answer?.sources ?? retrieved} emptyMessage="No matching evidence was retrieved for the latest search." showScores /></div>}
        </section>}

        {tab === "Playground" && <section className="chat-workspace">
          <div className="chat-toolbar">
            <details className="chat-settings"><summary>Settings</summary><div className="chat-settings-popover">
              <div className="config-heading"><div><h2>Model connection</h2><p>Provider credentials stay on the API server.</p></div><span className={`badge ${dashboard?.configured ? "success" : "neutral"}`}>{dashboard?.configured ? "READY" : "NOT CONNECTED"}</span></div>
              <div className="config-fields"><label>Provider<select value={provider} onChange={(event) => setProvider(event.target.value)}><option value="cohere">Cohere</option><option value="openai">OpenAI</option><option value="huggingface">Hugging Face</option></select></label><label>Model<input value={model} onChange={(event) => setModel(event.target.value)} /></label><button className="button primary" onClick={configure} disabled={!sessionId || busy}>{busy ? "Working…" : "Connect model"}</button></div>
              <div className="chat-retrieval-settings"><label>Retrieval mode<select value={mode} onChange={(event) => setMode(event.target.value)}><option value="hybrid">Hybrid</option><option value="dense">Dense</option><option value="sparse">Sparse</option></select></label><label>Top-K passages<select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>{[3, 5, 8, 10, 15].map((value) => <option key={value} value={value}>{value}</option>)}</select></label></div>
              <button className="button quiet retrieve-settings-button" onClick={retrieveEvidence} disabled={busy || !question.trim()}>Retrieve evidence for draft question</button>
            </div></details>
          </div>
          <div className="chat-thread">
            {!previousTurns.length && !answer && <div className="chat-welcome"><span className="welcome-mark">B</span><p className="hero-kicker">EVIDENCE-BASED BIOMEDICAL RESEARCH</p><h2>Research medicine.<br />Understand the evidence.</h2><p>Explore biomedical topics with answers grounded in cited sources.</p><button className="hero-start" onClick={() => questionInputRef.current?.focus()}>Start Research <span>→</span></button><h3>Popular queries</h3><div className="suggestion-row">{[["🧬", "Explain CRISPR gene editing"], ["❤️", "What causes hypertension?"], ["🩺", "Compare type 1 vs type 2 diabetes"], ["🔬", "Summarize Alzheimer's therapies"]].map(([icon, suggestion]) => <button key={suggestion} onClick={() => { setQuestion(suggestion); questionInputRef.current?.focus(); }}><span className="suggestion-icon">{icon}</span>{suggestion}<span>↗</span></button>)}</div></div>}
            {[...previousTurns].reverse().map((turn) => {
              const body = String(turn.answer ?? turn.error ?? "");
              const state = historyResultState(turn);
              const answerBody = answerParts(stateAnswer(state, body, Number(turn.source_count ?? 0))).answer;
              const historySources = Array.isArray(turn.retrieved_evidence)
                ? turn.retrieved_evidence as EvidenceSource[]
                : Array.isArray(turn.sources)
                  ? turn.sources.map((source) => ({ source: String(source) }))
                  : [];
              const citedRefs = answerReferences(historySources, answerBody, Array.isArray(turn.claim_support) ? turn.claim_support as Array<Record<string, unknown>> : []);
              const refs = citedRefs.length ? citedRefs : historySources;
              const showRetrievedPassages = !["approved", "informational"].includes(state);
              return <article className="chat-turn" id={`turn-${turn.id}`} key={turn.id}>
                <div className="user-message"><span className="message-avatar">You</span><p>{String(turn.query ?? "")}</p></div>
                <div className="assistant-message"><span className="assistant-avatar">B</span><div className="assistant-body">
                  <FormattedAnswer text={answerBody} />
                  <ReferenceSources sources={refs} showPassages={showRetrievedPassages} />
                </div></div>
              </article>;
            })}
            {answer && (() => {
              const visibleAnswer = answerParts(displayAnswerText);
              const citedRefs = answerReferences(answer.sources, displayAnswerText, answer.details.claim_support ?? []);
              const refs = citedRefs.length ? citedRefs : answer.sources;
              const showRetrievedPassages = !["approved", "informational"].includes(currentResultState);
              return <article className="chat-turn current-turn" id={`turn-${answer.event_id}`}>
                <div className="user-message"><span className="message-avatar">You</span><p>{answeredQuestion}</p></div>
                <div className="assistant-message"><span className="assistant-avatar">B</span><div className="assistant-body">
                  <FormattedAnswer text={visibleAnswer.answer} />
                  {visibleAnswer.disclaimer && <p className="medical-disclaimer"><span aria-hidden="true">⚠</span><span><strong>Medical disclaimer</strong>{visibleAnswer.disclaimer}</span></p>}
                  <ReferenceSources sources={refs} showPassages={showRetrievedPassages} />
                  <div className="answer-actions">
                    {!answer.details.answer_abstained && <><button onClick={() => setNoteEditorOpen((open) => !open)}>＋ Save note</button><button onClick={bookmarkCurrentAnswer}>☆ Bookmark</button><button onClick={exportCurrentAnswer}>↓ Export Markdown</button></>}
                    {currentAnswerReviewed && <span>Human reviewed</span>}
                    <button className="answer-copy-button" onClick={() => void copyAnswer(displayAnswerText, refs)}>Copy answer</button>
                  </div>
                  {noteEditorOpen && <form className="note-editor" onSubmit={(event) => { event.preventDefault(); saveCurrentNote(); }}>
                    <label htmlFor="research-note">Research note<textarea id="research-note" value={noteDraft} onChange={(event) => setNoteDraft(event.target.value)} rows={3} /></label>
                    <button className="button primary" disabled={!noteDraft.trim()}>Save note</button>
                  </form>}
                </div></div>
              </article>;
            })()}
            <div ref={chatEndRef} />
          </div>
          {busy && <div className="thinking-indicator" role="status"><span className="thinking-pulse" />BioMedGPT is thinking…</div>}
          <form className="chat-composer" onSubmit={(event) => { event.preventDefault(); submitQuery(); }}>
            <label className="sr-only" htmlFor="chat-question">Ask a biomedical question</label><textarea ref={questionInputRef} id="chat-question" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submitQuery(); } }} placeholder="Ask a biomedical question…" rows={2} />
            <div className="composer-bottom"><span aria-hidden="true" /><div><button type="submit" className="send-button" aria-label="Send question" disabled={busy || !dashboard?.configured || !question.trim()}>{busy ? "…" : "↑"}</button></div></div>
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
                {!!answer.details.warnings?.length && <details className="validation-details"><summary>Why checks failed ({answer.details.warnings.length})</summary><ul>{answer.details.warnings.map((warning, index) => <li key={`${warning}-${index}`}>{validationIssue(String(warning))}</li>)}</ul></details>}
                {answer.details.generation_success !== false && !answer.details.answer_abstained && <button className="text-button" onClick={() => { setAuditAnswerText(answer.answer); setAuditEvidence(answer.sources.map((source) => (source.text ?? "").replace(/\s+/g, " ").trim()).join("\n")); navigateToTab("Evaluation"); }}>Audit this answer →</button>}
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
          {answer && <details className="panel research-diagnostics">
            <summary>Research diagnostics for the latest answer</summary>
            <div className="research-diagnostics-content">
              <strong>Question</strong><p>{answeredQuestion}</p>
              {retrievalDebug?.retrieval_configuration && <>
                <strong>Executed queries</strong>
                <ul>{(retrievalDebug.retrieval_configuration.executed_queries ?? retrievalDebug.retrieval_configuration.expanded_queries ?? []).map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul>
                <strong>Retrieved ranking and classifications</strong>
                <ul>{(retrievalDebug.retrieval_configuration.top_ranked_candidates ?? []).map((candidate, index) => <li key={`${String(candidate.source ?? "source")}-${index}`}>{String(candidate.source ?? "Biomedical source")} · {String(candidate.evidence_category ?? "category not recorded")} · score {Number(candidate.retrieval_score ?? candidate.score ?? 0).toFixed(3)}</li>)}</ul>
              </>}
              <strong>Validation findings</strong>
              <pre className="json-result">{JSON.stringify({ decision: answer.details.decision, rejection_reason: answer.details.rejection_reason, diagnostics: answer.details.validation_diagnostics, claims: answer.details.claim_support, findings: answer.details.audit_trace?.validation_issues }, null, 2)}</pre>
              <strong>Retrieved passages</strong><EvidenceList sources={answer.sources} emptyMessage="No passage text was recorded for this answer." showScores />
            </div>
          </details>}
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

        <footer className="page-footer">Biomedical RAG · Session-scoped audit data</footer>
      </main>
    </div>
  );
}

function EvidenceList({ sources, emptyMessage, missingTextMessage = "No source excerpt was retained with this result.", showScores = false, answerText = "" }: { sources: EvidenceSource[]; emptyMessage?: string; missingTextMessage?: string; showScores?: boolean; answerText?: string }) {
  if (!sources.length) return <p className="empty">{emptyMessage ?? "Evidence will appear after retrieval."}</p>;
  return <div className="evidence-list">{sources.map((source, index) => <article className="evidence-item" key={`${source.source ?? "source"}-${index}`}>
    <div id={source.evidence_id ?? `evidence-${index + 1}`} className="evidence-heading"><div className="source-card-main"><strong>{source.source ?? "Biomedical source"}</strong><span>{source.evidence_classification?.category?.replaceAll("_", " ") ?? "Retrieved source"}</span></div>{answerText && <span className={`source-use ${answerText.includes(`[Evidence ${index + 1}]`) ? "used" : ""}`}>{answerText.includes(`[Evidence ${index + 1}]`) ? "✓ Used in answer" : "Source"}</span>}{source.source_origin === "external" && <span className="source-origin">External · quality not independently verified</span>}{source.url?.startsWith("https://") && <a className="source-link" href={source.url} target="_blank" rel="noreferrer">Open source ↗</a>}{showScores && <span className="source-rank">Source {index + 1}</span>}</div>
    <p>{source.text?.trim() || missingTextMessage}</p>
    {!!source.prompt_excerpt?.trim() && source.prompt_excerpt.trim() !== source.text?.trim() && <details className="prompt-excerpt"><summary>Passage excerpt used for the answer</summary><p>{source.prompt_excerpt.trim()}</p></details>}
  </article>)}</div>;
}

function AnswerText({ text }: { text: string }) {
  // Citation indices remain in the underlying answer for source mapping and
  // audits, while the conversational answer stays readable like plain prose.
  return <>{text.replace(/\s*\[Evidence\s+\d+\]/gi, "")}</>;
}

function FormattedAnswer({ text }: { text: string }) {
  const lines = text.split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) { index += 1; continue; }
    const validationNote = line.match(/^(?:validation note:\s*|)(some wording could not be fully confirmed by automated support checks\..*|the cited evidence is relevant, but this answer did not meet all validation requirements\..*)$/i)
      ?? line.match(/^validation note:\s*(.*)$/i);
    if (validationNote) {
      const message = validationNote[1] || validationNote[2] || line.replace(/^validation note:\s*/i, "");
      blocks.push(<aside className="validation-note" role="note" key={`validation-note-${index}`}><strong>Validation note</strong><p><AnswerText text={message} /></p></aside>);
      index += 1;
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const Tag = `h${heading[1].length + 1}` as "h2" | "h3" | "h4";
      blocks.push(<Tag key={`heading-${index}`}><AnswerText text={heading[2]} /></Tag>);
      index += 1;
      continue;
    }
    if (line.includes("|") && index + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[index + 1])) {
      const cells = (value: string) => value.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
      const header = cells(line);
      index += 2;
      const rows: string[][] = [];
      while (index < lines.length && lines[index].includes("|")) { rows.push(cells(lines[index])); index += 1; }
      blocks.push(<div className="answer-table-wrap" key={`table-${index}`}><table className="answer-table"><thead><tr>{header.map((cell, cellIndex) => <th key={cellIndex}><AnswerText text={cell} /></th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{header.map((_, cellIndex) => <td key={cellIndex}><AnswerText text={row[cellIndex] ?? ""} /></td>)}</tr>)}</tbody></table></div>);
      continue;
    }
    const listMatch = line.match(/^([-*•])\s+(.+)$/) ?? line.match(/^\d+[.)]\s+(.+)$/);
    if (listMatch) {
      const ordered = /^\d/.test(line);
      const items: string[] = [];
      while (index < lines.length) {
        const item = lines[index].trim().match(ordered ? /^\d+[.)]\s+(.+)$/ : /^[-*•]\s+(.+)$/);
        if (!item) break;
        items.push(item[1]);
        index += 1;
      }
      const List = ordered ? "ol" : "ul";
      blocks.push(<List key={`list-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}><AnswerText text={item} /></li>)}</List>);
      continue;
    }
    const paragraph = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() && !/^(#{1,3})\s+/.test(lines[index].trim()) && !/^([-*•])\s+/.test(lines[index].trim()) && !/^\d+[.)]\s+/.test(lines[index].trim())) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    blocks.push(<p key={`paragraph-${index}`}><AnswerText text={paragraph.join(" ")} /></p>);
  }
  return <div className="formatted-answer">{blocks}</div>;
}

function ReferenceSources({ sources, showPassages = false }: { sources: EvidenceSource[]; showPassages?: boolean }) {
  if (!sources.length) return null;
  return <details className="answer-sources" open={showPassages}>
    <summary>{showPassages ? `Retrieved passages (${sources.length})` : `Sources (${sources.length})`}</summary>
    <div className="answer-source-list">{sources.map((source, index) => {
      const citationIndex = source.citation_index ?? index + 1;
      const title = source.title ?? source.document_title;
      const passage = source.prompt_excerpt?.trim() || source.text?.trim();
      return <article className="answer-source" id={`evidence-${citationIndex}`} key={`${source.evidence_id ?? source.url ?? source.source ?? "reference"}-${index}`}>
        <strong>{sourceProvider(source)}</strong>
        {title && <p>{title}</p>}
        {source.url?.startsWith("https://") && <a href={source.url} target="_blank" rel="noreferrer">Open source ↗</a>}
        {passage && (showPassages ? <p className="retrieved-passage">{passage}</p> : <details><summary>Relevant passage</summary><p>{passage}</p></details>)}
      </article>;
    })}</div>
  </details>;
}

function referenceMarkdown(sources: EvidenceSource[]) {
  if (!sources.length) return "";
  const rows = sources.map((source) => {
    const name = source.title ?? source.document_title ?? sourceProvider(source);
    return source.url?.startsWith("https://") ? `- [${name}](${source.url})` : `- ${name}`;
  });
  return `\n\n## Sources\n${rows.join("\n")}`;
}

function sourceProvider(source: EvidenceSource) {
  const identity = `${source.source_provider ?? ""} ${source.source ?? ""}`.toLowerCase();
  if (identity.includes("medlineplus")) return "MedlinePlus";
  if (identity.includes("pubmed") || identity.includes("pmid") || identity.includes("ncbi")) return "PubMed";
  if (source.source_origin === "external") return "Internet";
  return source.source_provider || source.source || "Biomedical source";
}

function eventStatus(event: AuditEvent) {
  if (event.decision === "answer_approved") return "Verified";
  if (event.decision === "informational") return "Informational";
  if (event.decision === "conflicting_evidence") return "Conflicting evidence";
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
