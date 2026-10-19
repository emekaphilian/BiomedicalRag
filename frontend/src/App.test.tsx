import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";

function stateResponse() {
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
    events: [],
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
    vi.unstubAllGlobals();
  });

  it("creates a session and displays dashboard metrics", async () => {
    render(<App />);
    expect(await screen.findByText("System overview")).toBeInTheDocument();
    expect(screen.getByText("Successful queries")).toBeInTheDocument();
    expect(window.localStorage.getItem("biomed-rag-session")).toBe("session-test");
  });

  it("navigates between the research and governance surfaces", async () => {
    render(<App />);
    await screen.findByText("System overview");
    fireEvent.click(screen.getByRole("button", { name: "Governance" }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Safety & governance" })).toBeInTheDocument());
    expect(screen.getByRole("heading", { name: "Fairness comparison" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Risk register" })).toBeInTheDocument();
  });

  it("shows the API error rather than hiding a failed request", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      JSON.stringify({ detail: "API unavailable" }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    )));
    render(<App />);
    expect(await screen.findByText("API unavailable")).toBeInTheDocument();
  });
});
