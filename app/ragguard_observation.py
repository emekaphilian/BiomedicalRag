"""Best-effort RAGGuard observation delivery for BiomedicalRag.

Configuration is read from server-side environment variables. Telemetry is
minimal by default: the query is redacted, and answer/source text is omitted.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import httpx


@dataclass(frozen=True, slots=True)
class ObservationDelivery:
    delivered: bool
    status_code: int | None = None
    error: str | None = None


class RAGGuardObserver:
    def __init__(
        self,
        *,
        base_url: str,
        application_id: str,
        environment: str,
        token: str,
        timeout_seconds: float = 1.0,
        include_query: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.observation_url = (
            f"{self.base_url}/observations"
            if self.base_url.endswith("/api/v1")
            else f"{self.base_url}/api/v1/observations"
        )
        self.application_id = application_id
        self.environment = environment
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.include_query = include_query

    @classmethod
    def from_environment(cls) -> "RAGGuardObserver | None":
        """Create an observer only when all required server-side settings exist."""
        base_url = os.getenv("RAGGUARD_URL", "").strip()
        application_id = os.getenv("RAGGUARD_APPLICATION_ID", "biomedicalrag").strip()
        environment = os.getenv("RAGGUARD_ENVIRONMENT", "development").strip()
        token = os.getenv("RAGGUARD_OBSERVATION_TOKEN", "").strip()
        if not (base_url and application_id and environment and token):
            return None
        return cls(
            base_url=base_url,
            application_id=application_id,
            environment=environment,
            token=token,
        )

    def observe(
        self,
        *,
        query: str,
        retrieved_chunks: Iterable[tuple[str, float]] = (),
        retrieval_method: str = "unknown",
        retrieval_latency_ms: float = 0,
        generation_status: str = "succeeded",
    ) -> ObservationDelivery:
        payload = {
            "contract_version": "v1",
            "source": {
                "application_id": self.application_id,
                "environment": self.environment,
            },
            "query": query if self.include_query else "[redacted]",
            "retrieved_chunks": [
                {"id": chunk_id, "score": score}
                for chunk_id, score in retrieved_chunks
            ],
            "retrieval_method": retrieval_method,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_status": generation_status,
        }
        try:
            response = httpx.post(
                self.observation_url,
                json=payload,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError:
            return ObservationDelivery(delivered=False, error="RAGGuard is unavailable")
        if not response.is_success:
            return ObservationDelivery(
                delivered=False,
                status_code=response.status_code,
                error="RAGGuard rejected the observation",
            )
        return ObservationDelivery(delivered=True, status_code=response.status_code)
