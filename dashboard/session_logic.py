"""Streamlit-free helpers for dashboard session and configuration logic."""

from __future__ import annotations

import hashlib
import os
from statistics import median
from typing import Callable, Mapping, MutableMapping, Optional, Sequence


def get_saved_api_key(provider: str, secrets, environ: Optional[Mapping[str, str]] = None) -> str:
    """Read a provider key from Streamlit secrets, falling back to the environment."""
    environ = os.environ if environ is None else environ
    name = f"{provider}_api_key"
    try:
        value = secrets.get(name, "")
    except Exception:  # Streamlit's missing-secrets exception varies by version.
        value = ""
    return str(value or environ.get(name.upper(), "") or "")


def configuration_drift(active: Optional[Mapping], selected: Mapping) -> list:
    """Return provider/model fields that differ from the initialized service."""
    if not active:
        return []
    return [key for key in ("provider", "model") if active.get(key) != selected.get(key)]


def classify_upload(name: str, sha256: str, records: Sequence[Mapping]) -> str:
    """Classify a source by content hash and name: duplicate, changed, or new."""
    if any(record.get("sha256") == sha256 for record in records):
        return "duplicate"
    if any(name in (record.get("source"), record.get("filename")) for record in records):
        return "changed"
    return "new"


def unique_source_name(name: str, sha256: str, records: Sequence[Mapping]) -> str:
    """Give changed content a distinct provenance key while preserving its extension."""
    if classify_upload(name, sha256, records) != "changed":
        return name
    stem, ext = os.path.splitext(name)
    return f"{stem} [{sha256[:8]}]{ext}"


def commit_turn(chat_history: Sequence[Mapping], question: str, answer: str) -> list:
    """Return a copied history with one successful question and answer appended."""
    return [
        *chat_history,
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def median_latency(latencies: Sequence[float]) -> Optional[float]:
    values = [value for value in latencies if isinstance(value, (int, float))]
    return float(median(values)) if values else None


def review_coverage(successful_events: Sequence[Mapping], reviews: Sequence[Mapping]) -> tuple:
    """Return (successful answers with a review, successful answer count)."""
    reviewed_ids = {review.get("query_event_id") for review in reviews}
    reviewed = sum(
        1 for event in successful_events
        if event.get("id", event.get("timestamp")) in reviewed_ids
    )
    return reviewed, len(successful_events)


class ServiceLoader:
    """Cache initialized services only; failures remain retryable."""

    def __init__(self, factory: Callable[[str, str, str], object]):
        self._factory = factory
        self._cache: MutableMapping[tuple, object] = {}

    @staticmethod
    def _key(provider: str, model: str, api_key: str) -> tuple:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        return provider, model, digest

    def load(self, provider: str, model: str, api_key: str) -> tuple:
        key = self._key(provider, model, api_key)
        if key in self._cache:
            return self._cache[key], None
        try:
            service = self._factory(provider, model, api_key)
        except Exception as error:
            return None, str(error)
        self._cache[key] = service
        return service, None

    def clear(self) -> None:
        self._cache.clear()


def _default_factory(provider: str, model: str, api_key: str):
    from app.services import RAGService

    return RAGService(provider=provider, model=model, api_key=api_key)


SERVICE_LOADER = ServiceLoader(_default_factory)
