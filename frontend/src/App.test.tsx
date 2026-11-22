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
    expect(await screen.findByText("System overview")).toBeInTheDocument();
    expect(screen.getByText("Successful queries")).toBeInTheDocument();
    expect(window.localStorage.getItem("biomed-rag-session")).toBe("session-test");
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
    await screen.findByText("System overview");

    expect(requestHeaders.length).toBeGreaterThan(0);
    expect(requestHeaders[0].get("Authorization")).toBe("Bearer researcher-test-token");
  });

  it("loads the standalone workspace without requesting a browser token", async () => {
    setApiToken("");
    render(<App />);

    expect(await screen.findByRole("heading", { name: "System overview" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Bearer token")).not.toBeInTheDocument();
    expect(getApiToken()).toBe("");
  });

  it("navigates between the research and governance surfaces", async () => {
    render(<App />);
    await screen.findByText("System overview");
    fireEvent.click(screen.getByRole("button", { name: "Governance" }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Safety & governance" })).toBeInTheDocument());
    expect(screen.getByRole("heading", { name: "Fairness comparison" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Risk register" })).toBeInTheDocument();
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
    await screen.findByText("System overview");
    fireEvent.click(screen.getByRole("button", { name: "Playground" }));

    expect(await screen.findByText("What is Parkinson's disease?")).toBeInTheDocument();
    expect(screen.getByText("The indexed evidence describes it as a movement disorder.")).toBeInTheDocument();
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
