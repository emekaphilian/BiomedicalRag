"""Best-effort RAGGuard observation delivery for BiomedicalRag.

Configuration is read from server-side environment variables. Telemetry is
minimal by default: the query is redacted, and answer/source text is omitted.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from queue import Full, Queue
from threading import Lock, Thread
from time import monotonic, sleep
from typing import Iterable

import httpx


logger = logging.getLogger("biomedicalrag.ragguard_observation")
_OBSERVATION_LATENCY_BUDGET_SECONDS = 4.0
_OBSERVATION_RETRY_LIMIT = 3
_RETRYABLE_HTTP_STATUS_CODES = {408, 429}
_OBSERVATION_QUEUE: Queue = Queue(maxsize=256)
_worker_lock = Lock()
_worker_started = False


def get_server_setting(name: str, default: str = "") -> str:
    """Read server-side env or Streamlit secrets without exposing secret values."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        value = str(st.secrets.get(name, default)).strip()
    except Exception:
        # Streamlit is optional for the FastAPI service and secrets may be absent.
        value = default
    return value


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
        """Create an observer only when all required server-side settings exist.

        Environment variables take precedence over Streamlit server secrets.
        """
        base_url = get_server_setting("RAGGUARD_URL")
        application_id = get_server_setting("RAGGUARD_APPLICATION_ID", "biomedicalrag")
        environment = get_server_setting("RAGGUARD_ENVIRONMENT", "development")
        token = get_server_setting("RAGGUARD_OBSERVATION_TOKEN")
        if not (base_url and application_id and environment and token):
            return None
        return cls(
            base_url=base_url,
            application_id=application_id,
            environment=environment,
            token=token,
            # Observation delivery is telemetry; repair authorization is a
            # separate channel and must never extend this request's budget.
            timeout_seconds=0.75,
        )

    def observe(
        self,
        *,
        query: str,
        request_id: str | None = None,
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
            "request_id": request_id,
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
                timeout=httpx.Timeout(
                    self.timeout_seconds,
                    connect=min(self.timeout_seconds, 1.0),
                    write=min(self.timeout_seconds, 1.0),
                    pool=min(self.timeout_seconds, 1.0),
                ),
            )
        except httpx.HTTPError:
            return ObservationDelivery(delivered=False, error="RAGGuard is unavailable")
        if not response.is_success:
            return ObservationDelivery(
                delivered=False,
                status_code=response.status_code,
                error="RAGGuard rejected the observation",
            )
        return ObservationDelivery(
            delivered=True,
            status_code=response.status_code,
        )


def enqueue_observation(observer: RAGGuardObserver, **kwargs) -> bool:
    """Queue best-effort telemetry without waiting on RAGGuard or repair."""
    global _worker_started
    deadline = monotonic() + _OBSERVATION_LATENCY_BUDGET_SECONDS
    try:
        _OBSERVATION_QUEUE.put_nowait((observer, kwargs, deadline))
    except Full:
        logger.warning("RAGGuard observation queue is full; dropping observation")
        return False

    with _worker_lock:
        if not _worker_started:
            Thread(
                target=_deliver_observations,
                name="ragguard-observations",
                daemon=True,
            ).start()
            _worker_started = True
    return True


def _deliver_observations() -> None:
    while True:
        observer, kwargs, deadline = _OBSERVATION_QUEUE.get()
        try:
            result = ObservationDelivery(delivered=False, error="delivery budget expired")
            for attempt in range(_OBSERVATION_RETRY_LIMIT):
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                # The observer's HTTP timeout is capped by the remaining
                # end-to-end budget, including time spent waiting in queue.
                original_timeout = getattr(observer, "timeout_seconds", None)
                if original_timeout is not None:
                    observer.timeout_seconds = min(original_timeout, remaining)
                try:
                    result = observer.observe(**kwargs)
                except Exception:
                    result = ObservationDelivery(delivered=False, error="delivery failed")
                finally:
                    if original_timeout is not None:
                        observer.timeout_seconds = original_timeout
                if result.delivered or (
                    result.status_code is not None
                    and result.status_code < 500
                    and result.status_code not in _RETRYABLE_HTTP_STATUS_CODES
                ):
                    break
                if attempt + 1 < _OBSERVATION_RETRY_LIMIT:
                    sleep(min(0.2 * (2 ** attempt), max(0.0, deadline - monotonic())))
            else:
                result = ObservationDelivery(delivered=False, error="delivery failed")
            logger.debug(
                "RAGGuard observation completed request_id=%s delivered=%s status=%s",
                kwargs.get("request_id"),
                result.delivered,
                result.status_code,
            )
        except Exception:
            logger.debug("RAGGuard observation worker failed safely", exc_info=True)
        finally:
            _OBSERVATION_QUEUE.task_done()
