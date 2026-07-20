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


def test_estimate_tokens_counts_cjk_per_char():
    # CJK characters count ~1 token each; latin text ~4 chars/token.
    assert llm_client._estimate_tokens("中文字符測試") == 6  # 6 CJK chars
    assert llm_client._estimate_tokens("abcd") == 1  # 4 latin chars / 4
    assert llm_client._estimate_tokens("") == 0
    # Mixed: 4 CJK + 8 latin -> 4 + ceil(8/4) = 6
    assert llm_client._estimate_tokens("關稅政策ABCDEFGH") == 6
    # CJK estimate must exceed the old flat len/4 for Chinese text.
    zh = "中美貿易談判最新進展，市場關注關稅政策變化。" * 5
    assert llm_client._estimate_tokens(zh) > len(zh) // 4 * 2


def test_cloudflare_neuron_estimate_uses_asymmetric_token_rates():
    capability = llm_client.get_capability("@cf/zai-org/glm-4.7-flash", provider="cloudflare")
    assert llm_client._compute_units_for(capability, 10_000, 1_000) == 92


def test_zai_uses_max_tokens_request_field():
    messages = [{"role": "user", "content": "summarise"}]
    model = llm_client.ModelConfig("glm", provider="zai", max_completion_tokens=777)
    body = llm_client._chat_request_body(model, messages)
    assert body["max_tokens"] == 777
    assert "max_completion_tokens" not in body


def test_cloudflare_disables_thinking_for_bulk_qwen_requests():
    model = llm_client.ModelConfig("@cf/qwen/qwen3-30b-a3b-fp8", provider="cloudflare", max_completion_tokens=777)
    body = llm_client._chat_request_body(model, [{"role": "user", "content": "x"}])
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_cloudflare_uses_max_tokens_request_field_for_gemma_compatibility():
    model = llm_client.ModelConfig("@cf/google/gemma-4-26b-a4b-it", provider="cloudflare", max_completion_tokens=777)
    body = llm_client._chat_request_body(model, [{"role": "user", "content": "x"}])
    assert body["max_tokens"] == 777
    assert "max_completion_tokens" not in body


def test_response_content_falls_back_to_cloudflare_qwen_reasoning_field():
    assert llm_client._response_message_content({"content": None, "reasoning_content": "{\"ok\":true}"}) == (
        '{"ok":true}'
    )


def test_response_content_prefers_normal_content():
    assert llm_client._response_message_content({"content": "answer", "reasoning_content": "internal"}) == "answer"


def test_groq_uses_max_completion_tokens_request_field():
    provider = "groq"
    model = llm_client.ModelConfig("model", provider=provider, max_completion_tokens=888)
    body = llm_client._chat_request_body(model, [{"role": "user", "content": "x"}])
    assert body["max_completion_tokens"] == 888
    assert "max_tokens" not in body


def test_groq_gpt_oss_bounds_and_hides_reasoning():
    model = llm_client.ModelConfig("openai/gpt-oss-120b", provider="groq", max_completion_tokens=888)
    body = llm_client._chat_request_body(model, [{"role": "user", "content": "x"}])
    assert body["reasoning_effort"] == "low"
    assert body["reasoning_format"] == "hidden"


def test_request_timeouts_floor_deadline_to_connect_plus_read(monkeypatch):
    monkeypatch.setenv("DAILY_MACRO_LLM_CONNECT_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("DAILY_MACRO_LLM_READ_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("DAILY_MACRO_LLM_DEADLINE_SECONDS", "5")  # too small
    _, _, total = llm_client._llm_request_timeouts()
    assert total == 70.0


def test_endpoint_cooldown_is_tracked_separately_from_quota_state():
    now = [10.0]
    governor = llm_client.RateLimitGovernor(time_fn=lambda: now[0])

    governor.mark_endpoint_cooldown("cerebras:cerebras_1:gpt-oss-120b", 30.0)

    assert governor.endpoint_cooldown_seconds("cerebras:cerebras_1:gpt-oss-120b") == 30.0
    assert governor.endpoint_cooldown_seconds("cerebras:cerebras_2:gpt-oss-120b") == 0.0
    now[0] = 40.0
    assert governor.endpoint_cooldown_seconds("cerebras:cerebras_1:gpt-oss-120b") == 0.0


def test_runtime_failover_advances_after_exact_endpoint_with_duplicate_model_ids():
    first = llm_client.ModelConfig("gpt-oss-120b", provider="cerebras", account_id="cerebras_1")
    second = llm_client.ModelConfig("gpt-oss-120b", provider="cerebras", account_id="cerebras_2")
    third = llm_client.ModelConfig("gpt-oss-120b", provider="cerebras", account_id="cerebras_3")
    runtime = llm_client.AnalysisRuntime(
        governor=llm_client.RateLimitGovernor(),
        model_chain=[first, second, third],
    )

    assert runtime.next_model_after(second) == third
    assert runtime.switch_to_next_model("timeout", failed_model=second)
    assert runtime.current_model.endpoint_id == third.endpoint_id
    assert runtime.model_switches[-1]["from_endpoint"] == second.endpoint_id
