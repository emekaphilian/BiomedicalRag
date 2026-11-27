import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { getApiToken, setApiToken } from "./api";
import type { AuditEvent, DashboardState } from "./api";
import App from "./App";

function stateResponse(): DashboardState {
  return {
    session_id: "session-test",
    configured: false,
    configuration: {
      version: "cfg-test",
      settings: {
        provider: "cohere",
        model: "command-a-plus-05-2026",
        retrieval_mode: "hybrid",
        top_k: 5,
      },
    },
    documents: [],
    document_count: 0,
    chunk_count: 0,
    source_names: [],
    events: [] as AuditEvent[],
    evaluation_runs: [],
    answer_quality_reviews: [],
    fairness_reports: [],
    risk_register: [],
    summary: {
      successful_queries: 0,
      failed_queries: 0,
      grounded_queries: 0,
      mean_latency_ms: 0,
      answer_audits: {},
      retrieval_metrics: {},
      answer_quality_reviews: {},
    },
  };
}

describe("research dashboard", () => {
  beforeEach(() => {
    window.localStorage.clear();
    setApiToken("researcher-test-token");
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/sessions")) {
        return new Response(JSON.stringify({
          session_id: "session-test",
          created_at: "2026-01-01T00:00:00Z",
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      if (url.includes("/api/sessions/session-test")) {
        return new Response(JSON.stringify(stateResponse()), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response(JSON.stringify({}), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }));
  });

  afterEach(() => {
    cleanup();
    setApiToken("");
    vi.unstubAllGlobals();
  });

  it("creates a session and displays dashboard metrics", async () => {
    render(<App />);
    expect(await screen.findByRole("heading", { name: "Research medicine. Understand the evidence." })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /New chat/i })).toBeInTheDocument();
    expect(window.localStorage.getItem("biomed-rag-session")).toBe("session-test");
  });

  it("shows the selected answer mode in the sidebar status", async () => {
    render(<App />);
    await screen.findByRole("heading", { name: "Research medicine. Understand the evidence." });
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    expect(screen.getByText("Compare mode · Connect a model in Settings")).toBeInTheDocument();
  });

  it("shows a visible message instead of a blank answer when the API returns empty text", async () => {
    const configuredState = { ...stateResponse(), configured: true };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/sessions")) {
        return new Response(JSON.stringify({ session_id: "session-test", created_at: "2026-01-01T00:00:00Z" }), { status: 200 });
      }
      if (url.endsWith("/api/query")) {
        return new Response(JSON.stringify({
          event_id: "empty-answer",
          answer: "",
          sources: [],
          details: { decision: "answer_approved", answer_abstained: false, generation_success: true, learning_answer: "" },
          latency_ms: 100,
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      return new Response(JSON.stringify(configuredState), { status: 200, headers: { "Content-Type": "application/json" } });
    }));

    render(<App />);
    await screen.findByRole("heading", { name: "Research medicine. Understand the evidence." });
    fireEvent.change(screen.getByLabelText("Ask a biomedical question"), { target: { value: "What is the function of dopamine?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));

    expect(await screen.findByText("No answer text was returned. Please retry the question.")).toBeInTheDocument();
  });

  it("sends the browser-session bearer token with API requests", async () => {
    const requestHeaders: Headers[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      requestHeaders.push(new Headers(init?.headers));
      const url = String(input);
      const body = url.endsWith("/api/sessions")
        ? { session_id: "session-test", created_at: "2026-01-01T00:00:00Z" }
        : stateResponse();
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }));

    render(<App />);
    await screen.findByRole("heading", { name: "Research medicine. Understand the evidence." });

    expect(requestHeaders.length).toBeGreaterThan(0);
    expect(requestHeaders[0].get("Authorization")).toBe("Bearer researcher-test-token");
  });

  it("loads the standalone workspace without requesting a browser token", async () => {
    setApiToken("");
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Research medicine. Understand the evidence." })).toBeInTheDocument();
    expect(screen.queryByLabelText("Bearer token")).not.toBeInTheDocument();
    expect(getApiToken()).toBe("");
  });

  it("navigates between the research and governance surfaces", async () => {
    render(<App />);
    await screen.findByRole("heading", { name: "Research medicine. Understand the evidence." });
    fireEvent.click(screen.getByRole("button", { name: "Governance" }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Safety & governance" })).toBeInTheDocument());
    expect(screen.getByRole("heading", { name: "Fairness comparison" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Risk register" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Chat" }));
    expect(await screen.findByRole("heading", { name: "Research medicine. Understand the evidence." })).toBeInTheDocument();
  });

  it("moves backward and forward through menu views", async () => {
    render(<App />);
    await screen.findByRole("heading", { name: "Research medicine. Understand the evidence." });
    fireEvent.click(screen.getByRole("button", { name: "Research Library" }));
    expect(await screen.findByRole("heading", { name: "Knowledge base" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(await screen.findByRole("heading", { name: "Research medicine. Understand the evidence." })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Forward" }));
    expect(await screen.findByRole("heading", { name: "Knowledge base" })).toBeInTheDocument();
  });

  it("restores previous chat turns from the active session", async () => {
    const restoredState = stateResponse();
    restoredState.events = [{
      id: "query-previous",
      event: "query",
      timestamp: "2026-10-20T00:00:00Z",
      success: true,
      query: "What is Parkinson's disease?",
      answer: "The indexed evidence describes it as a movement disorder.",
    }];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const body = url.endsWith("/api/sessions")
        ? { session_id: "session-test", created_at: "2026-10-20T00:00:00Z" }
        : restoredState;
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }));

    render(<App />);
    await screen.findByRole("heading", { name: "BioMedGPT" });

    expect(await screen.findAllByText("What is Parkinson's disease?")).not.toHaveLength(0);
    expect(screen.getAllByText("The indexed evidence describes it as a movement disorder.")).not.toHaveLength(0);
  });

  it("shows validation notes as a separate, labeled callout", async () => {
    const restoredState = stateResponse();
    restoredState.events = [{
      id: "query-validation-note",
      event: "query",
      timestamp: "2026-10-20T00:00:00Z",
      success: true,
      query: "What is cancer?",
      answer: "Cancer develops when cells grow abnormally [Evidence 1].\n\nValidation note: The cited evidence is relevant, but this answer did not meet all validation requirements. Review the cited passages for research use; this is not clinical validation.",
      source_count: 1,
      retrieved_evidence: [{ evidence_id: "evidence-1", citation_index: 1, source: "MedlinePlus", text: "Cancer develops when cells grow abnormally." }],
    }];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const body = url.endsWith("/api/sessions")
        ? { session_id: "session-test", created_at: "2026-10-20T00:00:00Z" }
        : restoredState;
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }));

    const { container } = render(<App />);
    const note = await screen.findByRole("note");

    expect(note).toHaveClass("validation-note");
    expect(note).toHaveTextContent("Validation note");
    expect(note).toHaveTextContent("did not meet all validation requirements");
    expect(note.previousElementSibling).toHaveTextContent("Cancer develops when cells grow abnormally");
    expect(container.querySelectorAll(".validation-note")).toHaveLength(1);
    expect(screen.queryByRole("link", { name: "[Evidence 1]" })).not.toBeInTheDocument();
  });

  it("opens retrieved passages when an answer asks the user to review them", async () => {
    const restoredState = stateResponse();
    restoredState.events = [{
      id: "query-insufficient-with-passage",
      event: "query",
      timestamp: "2026-10-20T00:00:00Z",
      success: false,
      query: "What is glaucoma?",
      answer: "I found related biomedical information, but the retrieved sources do not directly answer your question. Review the passages below or refine your question.",
      decision: "insufficient_evidence",
      answer_abstained: true,
      source_count: 1,
      retrieved_evidence: [{
        evidence_id: "evidence-1",
        citation_index: 1,
        source: "MedlinePlus",
        title: "Glaucoma",
        text: "Glaucoma is a group of eye diseases that can damage the optic nerve.",
      }],
    }];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const body = url.endsWith("/api/sessions")
        ? { session_id: "session-test", created_at: "2026-10-20T00:00:00Z" }
        : restoredState;
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    }));

    render(<App />);

    expect(await screen.findByText("Glaucoma is a group of eye diseases that can damage the optic nerve.")).toBeInTheDocument();
    expect(screen.getByText("Retrieved passages (1)").closest("details")).toHaveAttribute("open");
  });

  it("keeps evidence relevance diagnostics out of chat and available in audit", async () => {
    const restoredState = stateResponse();
    restoredState.events = [{
      id: "query-indirect",
      event: "query",
      timestamp: "2026-10-20T00:00:00Z",
      success: false,
      query: "What are common symptoms of diabetes?",
      answer: "Related biomedical sources were retrieved, but none directly addresses the requested symptoms information.",
      decision: "insufficient_evidence",
      answer_abstained: true,
      source_count: 2,
      rejection_reason: "evidence_too_weak",
      audit_trace: {
        evidence_audit: {
          reason: "No retrieved passage directly addresses the requested information type.",
          relevance_counts: { direct: 0, supporting_contextual: 0, related_indirect: 1, complication: 1, contradictory: 0, irrelevant: 0, uncertain: 0 },
        },
      },
      retrieved_evidence: [
        { evidence_id: "evidence-1", source: "MedlinePlus", text: "Prediabetes", evidence_classification: { category: "related_indirect" } },
        { evidence_id: "evidence-2", source: "MedlinePlus", text: "Diabetic eye problems", evidence_classification: { category: "complication" } },
      ],
    }];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const body = url.endsWith("/api/sessions")
        ? { session_id: "session-test", created_at: "2026-10-20T00:00:00Z" }
        : restoredState;
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    }));

    render(<App />);

    expect(await screen.findAllByText("What are common symptoms of diabetes?")).not.toHaveLength(0);
    expect(screen.queryByText("Evidence relevance: Related, but not sufficiently direct")).not.toBeInTheDocument();
    expect(screen.queryByText("Direct: 0")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Workspace tools"));
    fireEvent.click(screen.getByRole("button", { name: "Audit" }));
    const inspect = await screen.findByText("Inspect");
    fireEvent.click(inspect);
    expect(await screen.findByText("related indirect")).toBeInTheDocument();
    expect(screen.getByText("complication")).toBeInTheDocument();
  });

  it("explains withheld answers without claiming retained passages are missing", async () => {
    const restoredState = stateResponse();
    restoredState.events = [{
      id: "query-review-required",
      event: "query",
      timestamp: "2026-10-20T00:00:00Z",
      success: false,
      query: "What are common symptoms of diabetes?",
      decision: "validation_review_required",
      answer_abstained: true,
      source_count: 1,
      retrieved_evidence: [{
        evidence_id: "evidence-1",
        source: "MedlinePlus",
        prompt_excerpt: "Information about diabetes complications.",
        evidence_classification: { category: "related_indirect" },
      }],
    }];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const body = url.endsWith("/api/sessions")
        ? { session_id: "session-test", created_at: "2026-10-20T00:00:00Z" }
        : restoredState;
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    }));

    render(<App />);

    expect(await screen.findByText(/do not directly answer your question/i)).toBeInTheDocument();
    expect(screen.queryByText(/did not retain passage text/i)).not.toBeInTheDocument();
  });

  it("recovers from a stale session when starting a new chat", async () => {
    const freshState = { ...stateResponse(), session_id: "fresh-session" };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/sessions")) {
        return new Response(JSON.stringify({ session_id: "fresh-session", created_at: "2026-10-20T00:00:00Z" }), { status: 200 });
      }
      if (url.includes("/new-chat")) {
        return new Response(JSON.stringify({ detail: "Dashboard session not found." }), { status: 404 });
      }
      return new Response(JSON.stringify(freshState), { status: 200 });
    }));

    window.localStorage.setItem("biomed-rag-session", "session-test");
    render(<App />);
    await screen.findByRole("heading", { name: "Research medicine. Understand the evidence." });
    fireEvent.click(screen.getByRole("button", { name: /New chat/i }));

    await waitFor(() => expect(window.localStorage.getItem("biomed-rag-session")).toBe("fresh-session"));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows the API error rather than hiding a failed request", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      JSON.stringify({ detail: "API unavailable" }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    )));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("API unavailable");
  });
});
