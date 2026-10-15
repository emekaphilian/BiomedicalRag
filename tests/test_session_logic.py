"""Regression tests for dashboard.session_logic."""

from dashboard.session_logic import (
    ServiceLoader, classify_upload, commit_turn, configuration_drift,
    get_saved_api_key, median_latency, review_coverage, unique_source_name,
)


class MissingSecrets:
    def get(self, *_):
        raise RuntimeError("no secrets file")


def test_saved_key_and_environment_fallback():
    assert get_saved_api_key("cohere", {"cohere_api_key": "secret"}, {"COHERE_API_KEY": "env"}) == "secret"
    assert get_saved_api_key("cohere", {"openai_api_key": "x"}, {"COHERE_API_KEY": "env"}) == "env"
    assert get_saved_api_key("cohere", MissingSecrets(), {"COHERE_API_KEY": "env"}) == "env"
    assert get_saved_api_key("cohere", {}, {}) == ""


def test_configuration_drift_reports_only_changed_fields():
    active = {"provider": "cohere", "model": "command-r-plus"}
    assert configuration_drift(active, dict(active)) == []
    assert configuration_drift(active, {**active, "model": "command-r"}) == ["model"]
    assert configuration_drift(None, active) == []


def test_uploads_deduplicate_by_hash_and_version_changed_sources():
    records = [{"source": "a.txt", "filename": "a.txt", "sha256": "aaaa1111bbbb"}]
    assert classify_upload("copy.txt", "aaaa1111bbbb", records) == "duplicate"
    assert classify_upload("a.txt", "cccc2222dddd", records) == "changed"
    assert classify_upload("b.txt", "eeee3333ffff", records) == "new"
    assert unique_source_name("a.txt", "cccc2222dddd", records) == "a [cccc2222].txt"
    assert unique_source_name("b.txt", "eeee3333ffff", records) == "b.txt"


def test_commit_turn_copies_history_and_adds_successful_exchange():
    history = [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"}]
    updated = commit_turn(history, "q1", "a1")
    assert len(history) == 2
    assert updated[-2:] == [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]


def test_median_latency_and_review_coverage():
    assert median_latency([1000, 1200, 30000]) == 1200.0
    assert median_latency([]) is None
    events = [{"id": "t1", "timestamp": "t1"}, {"id": "t2", "timestamp": "t2"}, {"timestamp": "t3"}]
    reviews = [{"query_event_id": "t1"}, {"query_event_id": "t1"}, {"query_event_id": "t3"}]
    assert review_coverage(events, reviews) == (2, 3)


def test_service_loader_retries_failures_caches_success_and_clears():
    calls = {"n": 0}

    def flaky(*_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient")
        return object()

    loader = ServiceLoader(flaky)
    service, error = loader.load("cohere", "m", "secret")
    assert service is None and "transient" in error
    service, error = loader.load("cohere", "m", "secret")
    assert service is not None and error is None
    assert loader.load("cohere", "m", "secret")[0] is service
    assert calls["n"] == 2
    loader.clear()
    assert loader.load("cohere", "m", "secret")[0] is not service


def test_service_loader_keys_include_provider_model_and_secret_digest():
    loader = ServiceLoader(lambda *_: object())
    first, _ = loader.load("cohere", "command-r", "secret")
    assert loader.load("cohere", "command-r", "secret")[0] is first
    assert loader.load("cohere", "command-r-plus", "secret")[0] is not first
    assert loader.load("cohere", "command-r", "other-secret")[0] is not first
