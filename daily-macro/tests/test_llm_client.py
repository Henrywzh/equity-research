"""Tests for low-level LLM HTTP dispatch helpers."""

from __future__ import annotations

import time

import pytest
import requests

from daily_macro import llm_client


class _FakeSession:
    """Minimal stand-in for requests.Session with controllable post timing."""

    def __init__(self, *, sleep_seconds: float = 0.0, raise_exc: Exception | None = None):
        self.sleep_seconds = sleep_seconds
        self.raise_exc = raise_exc
        self.closed = False
        self.post_calls = 0

    def post(self, url, json=None, timeout=None):  # noqa: A002 - match requests API
        self.post_calls += 1
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        if self.raise_exc is not None:
            raise self.raise_exc
        return f"response-for:{url}"

    def close(self):
        self.closed = True


def test_post_with_deadline_returns_fast_response():
    session = _FakeSession()
    result = llm_client._post_with_deadline(
        session,
        "https://example.test/chat",
        json_body={"model": "x"},
        connect_timeout=1.0,
        read_timeout=1.0,
        total_deadline=2.0,
    )
    assert result == "response-for:https://example.test/chat"
    assert not session.closed


def test_post_with_deadline_aborts_on_hang():
    # Worker would block for 5s; the 0.5s hard deadline must fire first and
    # close the session so the socket is abandoned.
    session = _FakeSession(sleep_seconds=5.0)
    started = time.monotonic()
    with pytest.raises(llm_client.LLMRequestDeadlineError):
        llm_client._post_with_deadline(
            session,
            "https://example.test/chat",
            json_body={"model": "x"},
            connect_timeout=1.0,
            read_timeout=10.0,
            total_deadline=0.5,
        )
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"deadline did not fire promptly (took {elapsed:.2f}s)"
    assert session.closed, "session must be closed to release the stuck socket"


def test_post_with_deadline_propagates_request_error():
    boom = requests.exceptions.ConnectionError("refused")
    session = _FakeSession(raise_exc=boom)
    with pytest.raises(requests.exceptions.ConnectionError):
        llm_client._post_with_deadline(
            session,
            "https://example.test/chat",
            json_body={"model": "x"},
            connect_timeout=1.0,
            read_timeout=1.0,
            total_deadline=2.0,
        )


def test_deadline_error_is_a_request_exception():
    # The retry loop catches requests.exceptions.RequestException; the deadline
    # error must be caught by that handler.
    assert issubclass(llm_client.LLMRequestDeadlineError, requests.exceptions.RequestException)


def test_request_timeouts_respect_env(monkeypatch):
    monkeypatch.setenv("DAILY_MACRO_LLM_CONNECT_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("DAILY_MACRO_LLM_READ_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("DAILY_MACRO_LLM_DEADLINE_SECONDS", "90")
    assert llm_client._llm_request_timeouts() == (5.0, 30.0, 90.0)


def test_request_timeouts_floor_deadline_to_connect_plus_read(monkeypatch):
    monkeypatch.setenv("DAILY_MACRO_LLM_CONNECT_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("DAILY_MACRO_LLM_READ_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("DAILY_MACRO_LLM_DEADLINE_SECONDS", "5")  # too small
    _, _, total = llm_client._llm_request_timeouts()
    assert total == 70.0
