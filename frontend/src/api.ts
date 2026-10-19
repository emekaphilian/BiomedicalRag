const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body: { detail?: string } = await response.json();
      if (body.detail) message = body.detail;
    } catch {
      // Preserve the status-based message if the server did not return JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function get<T>(path: string): Promise<T> {
  return request<T>(path);
}

export function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function upload<T>(path: string, body: FormData): Promise<T> {
  return request<T>(path, { method: "POST", body });
}

export function queryPath(sessionId: string): string {
  return `?session_id=${encodeURIComponent(sessionId)}`;
}

export interface SessionCreated {
  session_id: string;
  created_at: string;
}

export interface AuditEvent {
  id: string;
  event: string;
  timestamp: string;
  success?: boolean;
  query?: string;
  answer?: string;
  sources?: string[];
  source_count?: number;
  is_grounded?: boolean;
  latency_ms?: number;
  warnings?: string[];
  audit?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface DashboardState {
  session_id: string;
  configured: boolean;
  configuration: { version: string; settings: Record<string, unknown> };
  documents: Array<Record<string, unknown>>;
  document_count: number;
  chunk_count: number;
  source_names: string[];
  events: AuditEvent[];
  evaluation_runs: Array<Record<string, unknown>>;
  answer_quality_reviews: Array<Record<string, unknown>>;
  fairness_reports: Array<Record<string, unknown>>;
  risk_register: Array<Record<string, unknown>>;
  summary: {
    successful_queries: number;
    failed_queries: number;
    grounded_queries: number;
    mean_latency_ms: number;
    answer_audits: Record<string, number | null>;
    retrieval_metrics: Record<string, number | null>;
    answer_quality_reviews: Record<string, number | null>;
  };
}
