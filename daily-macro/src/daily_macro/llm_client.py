"""LLM client infrastructure for the daily-macro analysis pipeline.

Contains:
- Parsing utilities for HTTP headers and durations
- RateLimitGovernor — proactive per-(model, key) rate limit tracking
- Session management and API key loading
- AnalysisRuntime — execution state (model chain, sessions, diagnostics, budgets)
- _chat_completion / _invoke_json_with_retry — HTTP dispatch loop
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import requests

from .config import get_project_root
from .model_catalog import ModelCapability, get_capability
from .types import (
    BatchContext,
    CategoryBudgetState,
    CategoryDiagnostics,
    DEFAULT_CHAT_RETRIES,
    DEFAULT_OUTPUT_TOKENS,
    DEFAULT_PROVIDER,
    GROQ_CHAT_COMPLETIONS_URL,
    MAX_CATEGORY_SYNTHESIS_RETRIES,
    MAX_CATEGORY_SYNTHESIS_WAIT_SECONDS,
    MIN_INPUT_BUDGET_TOKENS,
    MIN_REQUEST_BYTE_BUDGET,
    MIN_SYNTHESIS_INPUT_BUDGET_TOKENS,
    MIN_SYNTHESIS_REQUEST_BYTE_BUDGET,
    ModelConfig,
    ModelRateLimitState,
    RATE_LIMIT_REQUEST_FLOOR,
    RATE_LIMIT_TOKEN_FLOOR,
    RuntimeDiagnostics,
    SynthesisBudgetExceeded,
    _section_profile,
)

LOGGER = logging.getLogger(__name__)


class LLMTask(StrEnum):
    ROUTING = "routing"
    ARTICLE_ANALYSIS = "article_analysis"
    CATEGORY_SYNTHESIS = "category_synthesis"
    TOP_ALERTS = "top_alerts"
    JSON_REPAIR = "json_repair"
    CRITIC = "critic"


@dataclass(frozen=True)
class ModelSelection:
    model: ModelConfig
    rejections: list[dict[str, str]] = field(default_factory=list)
    avoided_wait_seconds: float = 0.0


DEFAULT_MAX_LLM_WAIT_SECONDS = 45.0

# Tasks whose output quality most benefits from the strongest models; these may
# select premium models. All other (high-volume) tasks reserve premium capacity
# so bulk work cannot starve synthesis/alerts of the scarce, low-throughput
# high-quality models.
HIGH_VALUE_TASKS = frozenset({"category_synthesis", "top_alerts", "critic"})
# A model counts as "premium" (reserved) when it is strong at synthesis. Derived
# from scores so newly released large models are reserved automatically.
PREMIUM_SYNTHESIS_THRESHOLD = 0.85
# Score penalty applied to a premium model on a bulk task: large enough that any
# non-premium model with headroom outranks it, small enough that a premium model
# is still chosen over sleeping on a rate-limited one.
PREMIUM_BULK_PENALTY = 0.6


def _resolve_max_wait_seconds(explicit: float | None = None) -> float:
    """Resolve the rate-limit wait cap shared by the resolver and governor.

    Precedence: explicit argument, then ``DAILY_MACRO_MAX_LLM_WAIT_SECONDS``,
    then :data:`DEFAULT_MAX_LLM_WAIT_SECONDS`.
    """
    if explicit is not None:
        return max(0.0, explicit)
    raw_wait = os.environ.get("DAILY_MACRO_MAX_LLM_WAIT_SECONDS")
    if raw_wait:
        try:
            return max(0.0, float(raw_wait))
        except ValueError:
            return DEFAULT_MAX_LLM_WAIT_SECONDS
    return DEFAULT_MAX_LLM_WAIT_SECONDS


class ModelResolver:
    def __init__(
        self,
        *,
        active_model_ids: set[str] | None = None,
        model_policy: str | None = None,
        max_wait_seconds: float | None = None,
        capabilities: dict[str, ModelCapability] | None = None,
    ) -> None:
        self.active_model_ids = active_model_ids
        self.model_policy = (model_policy or os.environ.get("DAILY_MACRO_MODEL_POLICY") or "production_only").strip().lower()
        self.max_wait_seconds = _resolve_max_wait_seconds(max_wait_seconds)
        self.capabilities = capabilities or {}

    def task_preferences(self, task: str) -> list[str]:
        env_name = f"DAILY_MACRO_MODEL_{task.upper()}_PREFERENCES"
        raw = os.environ.get(env_name) or os.environ.get("DAILY_MACRO_MODEL_PREFERENCES", "")
        return [item.strip() for item in raw.split(",") if item.strip()]

    def capability_for(self, model: ModelConfig) -> ModelCapability:
        return self.capabilities.get(model.model_id) or get_capability(model.model_id, provider=model.provider)

    def resolve(
        self,
        task: LLMTask | str,
        model_chain: list[ModelConfig],
        *,
        estimated_input_tokens: int,
        requested_output_tokens: int,
        rate_limit_waits: dict[str, float] | None = None,
        preferred_model_id: str | None = None,
    ) -> ModelSelection:
        task_value = task.value if isinstance(task, LLMTask) else str(task)
        task_preferences = self.task_preferences(task_value)
        is_high_value = task_value in HIGH_VALUE_TASKS
        rejections: list[dict[str, str]] = []
        scored: list[tuple[float, int, ModelConfig, float]] = []
        wait_eligible: list[tuple[float, int, ModelConfig]] = []
        rate_limit_waits = rate_limit_waits or {}

        for index, model in enumerate(model_chain):
            capability = self.capability_for(model)
            if self.model_policy == "production_only" and capability.lifecycle != "production":
                rejections.append({"model_id": model.model_id, "reason": "preview_model_disallowed"})
                continue
            if self.active_model_ids is not None and model.model_id not in self.active_model_ids:
                rejections.append({"model_id": model.model_id, "reason": "model_not_active"})
                continue
            if not capability.supports_json:
                rejections.append({"model_id": model.model_id, "reason": "json_not_supported"})
                continue
            if estimated_input_tokens + requested_output_tokens > capability.context_window:
                rejections.append({"model_id": model.model_id, "reason": "context_window_exceeded"})
                continue
            output_cap = min(capability.max_output_tokens, model.max_completion_tokens)
            if requested_output_tokens > output_cap:
                rejections.append({"model_id": model.model_id, "reason": "output_limit_exceeded"})
                continue
            wait_seconds = rate_limit_waits.get(model.model_id, 0.0)
            # Passed every hard constraint (policy, active, json, context,
            # output). Remember it as a fallback that respects those constraints
            # even if its wait exceeds the cap.
            wait_eligible.append((wait_seconds, index, model))
            if wait_seconds > self.max_wait_seconds:
                rejections.append({"model_id": model.model_id, "reason": "rate_limit_wait_too_long"})
                continue

            task_score = capability.task_scores.get(task_value, capability.task_scores.get("article_analysis", 0.5))
            preferred_bonus = 0.5 if model.model_id == preferred_model_id else 0.0
            preference_bonus = 0.0
            if model.model_id in task_preferences:
                preference_bonus = max(0.0, 0.35 - task_preferences.index(model.model_id) * 0.03)
            wait_penalty = min(wait_seconds / max(self.max_wait_seconds, 1.0), 1.0) * 0.4
            order_penalty = index * 0.01
            # Reserve premium models for high-value tasks: penalize them on bulk
            # tasks so any non-premium model with headroom outranks them, keeping
            # bulk work off the scarce high-quality models. The penalty is soft —
            # a premium model is still chosen over sleeping on a rate-limited one,
            # and an explicit env preference is exempt.
            reservation_penalty = 0.0
            is_premium = capability.task_scores.get("category_synthesis", 0.0) >= PREMIUM_SYNTHESIS_THRESHOLD
            if is_premium and not is_high_value and model.model_id not in task_preferences:
                reservation_penalty = PREMIUM_BULK_PENALTY
            scored.append(
                (
                    task_score + preferred_bonus + preference_bonus - wait_penalty - order_penalty - reservation_penalty,
                    index,
                    model,
                    wait_seconds,
                )
            )

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            _score, _index, model, wait_seconds = scored[0]
            return ModelSelection(model=model, rejections=rejections, avoided_wait_seconds=max(rate_limit_waits.get(preferred_model_id or "", 0.0) - wait_seconds, 0.0))

        # Nothing scored within the wait cap. Prefer a model that satisfies every
        # hard constraint (policy/active/context/output) and just has a long
        # wait — the governor now caps the actual sleep — over silently
        # returning model_chain[0], which may be a preview model the policy
        # forbids or one whose context window can't fit the request.
        if wait_eligible:
            wait_eligible.sort(key=lambda item: (item[0], item[1]))
            wait_seconds, _index, model = wait_eligible[0]
            return ModelSelection(model=model, rejections=rejections, avoided_wait_seconds=0.0)

        # Truly no eligible model (e.g. request too large for every context
        # window, or no active production model). Fall back to the chain head as
        # a last resort so the caller can surface a meaningful error.
        fallback = model_chain[0]
        LOGGER.warning(
            "Model resolver found no eligible model for task %s; falling back to %s. Rejections: %s",
            task_value,
            fallback.model_id,
            rejections,
        )
        return ModelSelection(model=fallback, rejections=rejections, avoided_wait_seconds=0.0)


# ---------------------------------------------------------------------------
# Config-file helpers
# ---------------------------------------------------------------------------


def _candidate_config_paths() -> list[Path]:
    project_root = get_project_root()
    return [project_root / ".config", project_root.parent / ".config"]


def _parse_simple_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


# ---------------------------------------------------------------------------
# Duration / retry-after parsing
# ---------------------------------------------------------------------------


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_duration_seconds(value: str | None) -> float | None:
    if not value:
        return None
    stripped = value.strip().lower()
    try:
        return float(stripped)
    except ValueError:
        pass

    total = 0.0
    number = ""
    unit_found = False
    index = 0
    while index < len(stripped):
        char = stripped[index]
        if char.isdigit() or char == ".":
            number += char
            index += 1
            continue
        if not number:
            index += 1
            continue

        unit_found = True
        if stripped.startswith("ms", index):
            total += float(number) / 1000.0
            index += 2
        elif char == "s":
            total += float(number)
            index += 1
        elif char == "m":
            total += float(number) * 60.0
            index += 1
        elif char == "h":
            total += float(number) * 3600.0
            index += 1
        else:
            index += 1
        number = ""

    if number:
        total += float(number)
        unit_found = True
    return total if unit_found else None


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return _parse_duration_seconds(value)


def _retry_delay_seconds(response: requests.Response, attempt: int) -> float:
    retry_after = _parse_retry_after_seconds(response.headers.get("Retry-After"))
    if retry_after is not None:
        return max(1.0, retry_after)
    base_delay = min(15.0, 2.0 * (2**attempt))
    return base_delay + random.uniform(0.0, 0.5)


# ---------------------------------------------------------------------------
# Exception classification
# ---------------------------------------------------------------------------


def _classify_exception(exc: Exception) -> str:
    if isinstance(exc, SynthesisBudgetExceeded):
        return "synthesis_budget_exhausted"
    if isinstance(exc, ValueError):
        return "invalid_json"
    if isinstance(exc, requests.HTTPError):
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 413:
            return "payload_too_large"
        if status_code == 429:
            return "rate_limited"
        return "http_error"
    return "unexpected_error"


# ---------------------------------------------------------------------------
# Token / byte estimation
# ---------------------------------------------------------------------------


# CJK ideographs, kana, and full-width punctuation tokenize at roughly one
# token per character for the llama/Qwen tokenizers we target, unlike latin
# text which averages ~4 chars/token. HKEJ content is predominantly Chinese, so
# a flat len/4 heuristic undercounts by ~4x and lets oversized requests through.
_CJK_RE = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk_chars = len(_CJK_RE.findall(text))
    other_chars = len(text) - cjk_chars
    return math.ceil(cjk_chars + other_chars / 4)


def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return _estimate_tokens(json.dumps(messages, ensure_ascii=False))


def _estimate_request_payload_bytes(model_id: str, messages: list[dict[str, Any]], max_completion_tokens: int) -> int:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.1,
        "max_completion_tokens": max_completion_tokens,
        "response_format": {"type": "json_object"},
    }
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


# ---------------------------------------------------------------------------
# JSON parsing (used by _invoke_json_with_retry)
# ---------------------------------------------------------------------------


def _parse_json_content(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Response did not contain valid JSON.")
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Response did not contain valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object response.")
    return payload


# ---------------------------------------------------------------------------
# Session / API key management
# ---------------------------------------------------------------------------


def load_groq_api_keys() -> list[str]:
    """Load all Groq API keys from GROQ_API_KEY (comma-separated for multiple keys)."""
    raw: str | None = os.environ.get("GROQ_API_KEY")
    if not raw:
        for config_path in _candidate_config_paths():
            if not config_path.exists():
                continue
            parsed = _parse_simple_env_file(config_path)
            raw = parsed.get("GROQ_API_KEY")
            if raw:
                break
    if not raw:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export GROQ_API_KEY or add it to the repo-root .config file."
        )
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("GROQ_API_KEY contains no valid keys.")
    return keys


def load_groq_api_key() -> str:
    """Return the first Groq API key (backward-compat shim)."""
    return load_groq_api_keys()[0]


def _build_groq_session(api_key: str) -> requests.Session:
    return _build_provider_session(api_key)


def load_active_groq_model_ids(api_key: str, *, timeout: int = 10) -> set[str] | None:
    if str(os.environ.get("DAILY_MACRO_REFRESH_MODEL_CATALOG") or "").strip().lower() not in {"1", "true", "yes"}:
        return None
    session = _build_groq_session(api_key)
    try:
        response = session.get("https://api.groq.com/openai/v1/models", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        LOGGER.warning("Could not refresh Groq model catalog: %s", exc)
        return None
    finally:
        session.close()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    model_ids = {str(item.get("id") or "").strip() for item in data if isinstance(item, dict)}
    return {model_id for model_id in model_ids if model_id}


def _build_provider_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    )
    return session


class LLMRequestDeadlineError(requests.exceptions.Timeout):
    """Raised when an LLM HTTP request exceeds its hard wall-clock deadline."""


def _llm_request_timeouts() -> tuple[float, float, float]:
    """Return (connect_timeout, read_timeout, total_deadline) in seconds.

    ``requests``' ``timeout`` only bounds the gap between received bytes, so the
    total deadline is enforced separately by :func:`_post_with_deadline`.
    """

    def _read(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if not raw:
            return default
        try:
            return max(1.0, float(raw))
        except ValueError:
            return default

    connect_timeout = _read("DAILY_MACRO_LLM_CONNECT_TIMEOUT_SECONDS", 10.0)
    read_timeout = _read("DAILY_MACRO_LLM_READ_TIMEOUT_SECONDS", 60.0)
    total_deadline = _read("DAILY_MACRO_LLM_DEADLINE_SECONDS", 120.0)
    # The total deadline must cover at least one connect + read cycle.
    total_deadline = max(total_deadline, connect_timeout + read_timeout)
    return connect_timeout, read_timeout, total_deadline


def _post_with_deadline(
    session: requests.Session,
    url: str,
    *,
    json_body: dict[str, Any],
    connect_timeout: float,
    read_timeout: float,
    total_deadline: float,
) -> requests.Response:
    """POST with a hard wall-clock deadline.

    ``requests``' ``timeout`` only limits the gap between received bytes, so a
    server that trickles data or stalls mid-stream can hang far longer than
    intended. We run the request on a daemon worker thread and enforce an
    absolute ceiling; on expiry we close the session to abandon the stuck socket
    and raise :class:`LLMRequestDeadlineError` so the caller's retry/rotation
    logic can recover instead of blocking forever.
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["response"] = session.post(
                url, json=json_body, timeout=(connect_timeout, read_timeout)
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised to the caller below
            box["error"] = exc

    worker = threading.Thread(target=_run, name="llm-post", daemon=True)
    worker.start()
    worker.join(total_deadline)
    if worker.is_alive():
        # Close the session to break the blocked recv on the worker thread; the
        # caller evicts and rebuilds it before the next attempt.
        try:
            session.close()
        except Exception:  # noqa: BLE001 - best-effort socket teardown
            pass
        raise LLMRequestDeadlineError(
            f"LLM request exceeded hard deadline of {total_deadline:.0f}s"
        )
    if "error" in box:
        raise box["error"]
    return box["response"]


def _load_model_api_key(model: ModelConfig) -> str:
    if model.provider == DEFAULT_PROVIDER:
        return load_groq_api_key()

    env_name = model.api_key_env or "OPENAI_API_KEY"
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value

    for config_path in _candidate_config_paths():
        if not config_path.exists():
            continue
        parsed = _parse_simple_env_file(config_path)
        value = parsed.get(env_name)
        if value:
            return value

    raise RuntimeError(f"{env_name} is not set for provider {model.provider}.")


# ---------------------------------------------------------------------------
# RateLimitGovernor
# ---------------------------------------------------------------------------


class RateLimitGovernor:
    def __init__(self, *, time_fn=time.monotonic, sleep_fn=time.sleep, max_wait_seconds: float | None = None):
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self.max_wait_seconds = _resolve_max_wait_seconds(max_wait_seconds)
        # Keyed by (model_id, key_index) to track rate limits per API key independently.
        self._states: dict[tuple[str, int], ModelRateLimitState] = {}

    def _state(self, model_id: str, key_index: int) -> ModelRateLimitState:
        return self._states.setdefault((model_id, key_index), ModelRateLimitState())

    def select_key(
        self,
        model_id: str,
        current_key: int,
        num_keys: int,
        estimated_input_tokens: int = 0,
    ) -> tuple[int, float]:
        """Return (best_key_index, seconds_slept).

        Picks the key with the most remaining token capacity. If a better key
        exists, rotates silently (0 sleep). Only sleeps when ALL keys are
        exhausted, waiting until the soonest-resetting key unlocks.
        """
        now = self._time_fn()
        best_key: int | None = None
        best_remaining_tokens = -1
        earliest_reset = float("inf")

        for ki in range(num_keys):
            state = self._state(model_id, ki)
            is_observed_key = (
                ki == current_key
                or state.remaining_requests is not None
                or state.remaining_tokens is not None
            )
            req_ok = (
                state.remaining_requests is None
                or state.remaining_requests > RATE_LIMIT_REQUEST_FLOOR
            )
            tok_ok = (
                state.remaining_tokens is None
                or state.remaining_tokens > estimated_input_tokens + RATE_LIMIT_TOKEN_FLOOR
            )
            if req_ok and tok_ok and is_observed_key:
                remaining = state.remaining_tokens if state.remaining_tokens is not None else 9_999_999
                if best_key is None or remaining > best_remaining_tokens:
                    best_key = ki
                    best_remaining_tokens = remaining
            else:
                reset_at = max(
                    state.reset_requests_at or 0.0,
                    state.reset_tokens_at or 0.0,
                )
                if reset_at > now:
                    earliest_reset = min(earliest_reset, reset_at)

        if best_key is not None:
            return best_key, 0.0

        # All keys exhausted — sleep until the soonest one resets, but never
        # longer than the configured cap. A far-future reset (e.g. a daily token
        # limit) would otherwise block the whole run for hours; capping lets the
        # caller's retry/model-fallback logic degrade gracefully instead.
        sleep_secs = max(0.0, earliest_reset - now) if earliest_reset != float("inf") else 0.0
        sleep_secs = min(sleep_secs, self.max_wait_seconds)
        if sleep_secs > 0:
            self._sleep_fn(sleep_secs)
        return current_key, sleep_secs

    def peek_key(
        self,
        model_id: str,
        current_key: int,
        num_keys: int,
        estimated_input_tokens: int = 0,
    ) -> tuple[int, float]:
        now = self._time_fn()
        best_key: int | None = None
        best_remaining_tokens = -1
        earliest_reset = float("inf")

        for ki in range(num_keys):
            state = self._state(model_id, ki)
            is_observed_key = (
                ki == current_key
                or state.remaining_requests is not None
                or state.remaining_tokens is not None
            )
            req_ok = (
                state.remaining_requests is None
                or state.remaining_requests > RATE_LIMIT_REQUEST_FLOOR
            )
            tok_ok = (
                state.remaining_tokens is None
                or state.remaining_tokens > estimated_input_tokens + RATE_LIMIT_TOKEN_FLOOR
            )
            if req_ok and tok_ok and is_observed_key:
                remaining = state.remaining_tokens if state.remaining_tokens is not None else 9_999_999
                if best_key is None or remaining > best_remaining_tokens:
                    best_key = ki
                    best_remaining_tokens = remaining
            else:
                reset_at = max(
                    state.reset_requests_at or 0.0,
                    state.reset_tokens_at or 0.0,
                )
                if reset_at > now:
                    earliest_reset = min(earliest_reset, reset_at)

        if best_key is not None:
            return best_key, 0.0
        sleep_secs = max(0.0, earliest_reset - now) if earliest_reset != float("inf") else 0.0
        return current_key, sleep_secs

    def before_request(self, model_id: str, estimated_input_tokens: int = 0) -> float:
        """Backward-compatible single-key check. Uses key_index=0."""
        _key, waited = self.select_key(model_id, 0, 1, estimated_input_tokens)
        return waited

    def record_response(self, model_id: str, response: requests.Response, key_index: int = 0) -> None:
        state = self._state(model_id, key_index)
        now = self._time_fn()
        headers = {key.lower(): value for key, value in response.headers.items()}

        remaining_requests = _parse_int(headers.get("x-ratelimit-remaining-requests"))
        if remaining_requests is not None:
            state.remaining_requests = remaining_requests

        remaining_tokens = _parse_int(headers.get("x-ratelimit-remaining-tokens"))
        if remaining_tokens is not None:
            state.remaining_tokens = remaining_tokens

        reset_requests = _parse_duration_seconds(headers.get("x-ratelimit-reset-requests"))
        if reset_requests is not None:
            state.reset_requests_at = now + reset_requests

        reset_tokens = _parse_duration_seconds(headers.get("x-ratelimit-reset-tokens"))
        if reset_tokens is not None:
            state.reset_tokens_at = now + reset_tokens

        if response.status_code == 429:
            retry_after = _parse_retry_after_seconds(headers.get("retry-after"))
            if retry_after is not None:
                state.remaining_requests = 0
                state.remaining_tokens = 0
                state.reset_requests_at = now + retry_after
                state.reset_tokens_at = now + retry_after

    def apply_backoff(self, model_id: str, response: requests.Response, attempt: int) -> float:
        delay_seconds = _retry_delay_seconds(response, attempt)
        self._sleep_fn(delay_seconds)
        return delay_seconds


# ---------------------------------------------------------------------------
# AnalysisRuntime
# ---------------------------------------------------------------------------


@dataclass
class AnalysisRuntime:
    governor: RateLimitGovernor
    model_chain: list[ModelConfig]
    groq_api_keys: list[str] = field(default_factory=list)
    session: requests.Session | None = None
    delayed_retry_final_model: ModelConfig | None = None
    resolver: ModelResolver | None = None
    current_model_index: int = 0
    current_key_index: int = 0
    groq_sessions: dict[int, requests.Session] = field(default_factory=dict)
    model_switches: list[dict[str, Any]] = field(default_factory=list)
    last_attempted_model: str | None = None
    diagnostics: RuntimeDiagnostics = field(default_factory=RuntimeDiagnostics)
    category_diagnostics: dict[str, CategoryDiagnostics] = field(default_factory=dict)
    category_budgets: dict[str, CategoryBudgetState] = field(default_factory=dict)
    provider_sessions: dict[str, requests.Session] = field(default_factory=dict)
    market_context_string: str = ""
    macro_release_digest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.resolver is None:
            self.resolver = ModelResolver()
        if self.session is not None:
            if not self.groq_api_keys:
                self.groq_api_keys = ["session-backed-groq-key"]
            self.groq_sessions.setdefault(0, self.session)

    @property
    def primary_model(self) -> ModelConfig:
        return self.model_chain[0]

    @property
    def fallback_models(self) -> list[ModelConfig]:
        return self.model_chain[1:]

    @property
    def current_model(self) -> ModelConfig:
        return self.model_chain[self.current_model_index]

    def get_groq_session(self, key_index: int | None = None) -> requests.Session:
        ki = key_index if key_index is not None else self.current_key_index
        if ki not in self.groq_sessions:
            self.groq_sessions[ki] = _build_groq_session(self.groq_api_keys[ki])
        return self.groq_sessions[ki]

    def get_session_for_model(self, model: ModelConfig) -> requests.Session:
        if model.provider == DEFAULT_PROVIDER:
            return self.get_groq_session(self.current_key_index)
        session = self.provider_sessions.get(model.provider)
        if session is not None:
            return session
        session = _build_provider_session(_load_model_api_key(model))
        self.provider_sessions[model.provider] = session
        return session

    def rotate_key(self, reason: str) -> bool:
        """Rotate to the next API key. Returns False if only one key available."""
        if len(self.groq_api_keys) <= 1:
            return False
        next_idx = (self.current_key_index + 1) % len(self.groq_api_keys)
        LOGGER.info("Rotating Groq API key %d → %d: %s", self.current_key_index, next_idx, reason)
        self.current_key_index = next_idx
        self.diagnostics.key_rotation_count += 1
        # Clear old sessions to force new authentication
        for ki in list(self.groq_sessions.keys()):
            self.groq_sessions[ki].close()
            del self.groq_sessions[ki]
        return True

    def close_sessions(self) -> None:
        for session in self.groq_sessions.values():
            session.close()
        self.groq_sessions.clear()

    def reset_session_for_model(self, model: ModelConfig) -> None:
        """Drop the cached session for a model so a fresh one is built next use.

        Used after a request error (e.g. a hard-deadline abort closed the
        session) to avoid reusing a torn-down connection pool.
        """
        if model.provider == DEFAULT_PROVIDER:
            for ki in list(self.groq_sessions.keys()):
                try:
                    self.groq_sessions[ki].close()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
                del self.groq_sessions[ki]
            return
        session = self.provider_sessions.pop(model.provider, None)
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        for session in self.provider_sessions.values():
            session.close()
        self.provider_sessions.clear()

    def get_model_config(self, model_id: str) -> ModelConfig:
        for model in self.model_chain:
            if model.model_id == model_id:
                return model
        raise KeyError(model_id)

    def next_model_after(self, model_id: str) -> ModelConfig | None:
        for index, model in enumerate(self.model_chain):
            if model.model_id == model_id:
                next_index = index + 1
                if next_index < len(self.model_chain):
                    return self.model_chain[next_index]
                return None
        return None

    def switch_to_next_model(self, reason: str) -> bool:
        next_model = self.next_model_after(self.current_model.model_id)
        if next_model is None:
            return False
        self.diagnostics.fallback_switch_count += 1
        switch = {
            "switched_at": datetime.now().astimezone().isoformat(),
            "from_model": self.current_model.model_id,
            "to_model": next_model.model_id,
            "reason": reason,
        }
        self.model_switches.append(switch)
        LOGGER.info(
            "Switching Groq model from %s to %s: %s",
            self.current_model.model_id,
            next_model.model_id,
            reason,
        )
        self.current_model_index += 1
        return True

    def reset_model_for_category(self) -> None:
        self.current_model_index = 0

    def evict_model(self, model_id: str) -> bool:
        """Drop a decommissioned/unavailable model from the pool for this run.

        Returns False (and keeps the model) if it is the only one left, so the
        pool is never emptied. The resolver re-resolves over the reduced chain on
        the next attempt.
        """
        if len(self.model_chain) <= 1:
            return False
        index = next((i for i, m in enumerate(self.model_chain) if m.model_id == model_id), None)
        if index is None:
            return False
        self.model_chain.pop(index)
        if self.current_model_index >= len(self.model_chain):
            self.current_model_index = len(self.model_chain) - 1
        if self.resolver is not None and self.resolver.active_model_ids is not None:
            self.resolver.active_model_ids.discard(model_id)
        self.diagnostics.fallback_switch_count += 1
        LOGGER.info("Evicted model %s from the pool; %d remain.", model_id, len(self.model_chain))
        return True

    def get_category_diagnostics(self, category_name: str) -> CategoryDiagnostics:
        return self.category_diagnostics.setdefault(category_name, CategoryDiagnostics())

    def get_category_budget(self, category_name: str) -> CategoryBudgetState:
        if category_name not in self.category_budgets:
            profile = _section_profile(category_name)
            self.category_budgets[category_name] = CategoryBudgetState(
                article_input_budget_tokens=profile.article_input_budget_tokens,
                article_request_byte_budget=profile.article_request_byte_budget,
                synthesis_input_budget_tokens=profile.synthesis_input_budget_tokens,
                synthesis_request_byte_budget=profile.synthesis_request_byte_budget,
            )
        return self.category_budgets[category_name]

    def record_wait(self, category_name: str, delay_seconds: float, *, batch_kind: str = "") -> None:
        if delay_seconds <= 0:
            return
        self.diagnostics.rate_limit_wait_count += 1
        self.diagnostics.rate_limit_wait_seconds_total += delay_seconds
        diagnostics = self.get_category_diagnostics(category_name)
        diagnostics.rate_limit_waits += 1
        if batch_kind.startswith("synthesis"):
            diagnostics.synthesis_wait_seconds_total += delay_seconds

    def record_retry(self, category_name: str, *, batch_kind: str = "") -> None:
        if batch_kind.startswith("synthesis"):
            self.get_category_diagnostics(category_name).synthesis_retry_count += 1

    def record_model_switch(self, category_name: str, switch: dict[str, Any]) -> None:
        self.get_category_diagnostics(category_name).model_switches.append(dict(switch))

    def note_synthesis_merge_depth(self, category_name: str, depth: int) -> None:
        diagnostics = self.get_category_diagnostics(category_name)
        diagnostics.synthesis_merge_depth_max = max(diagnostics.synthesis_merge_depth_max, depth)

    def mark_degraded_merge(self, category_name: str, reason: str) -> None:
        diagnostics = self.get_category_diagnostics(category_name)
        diagnostics.degraded_merge_used = True
        diagnostics.degraded_merge_reason = reason
        self.diagnostics.degraded_merge_count += 1

    def ensure_synthesis_budget(self, category_name: str) -> None:
        diagnostics = self.get_category_diagnostics(category_name)
        if diagnostics.synthesis_wait_seconds_total >= MAX_CATEGORY_SYNTHESIS_WAIT_SECONDS:
            if not diagnostics.synthesis_budget_exhausted:
                self.diagnostics.synthesis_budget_exhausted_count += 1
            diagnostics.synthesis_budget_exhausted = True
            diagnostics.synthesis_retry_skipped_count += 1
            raise SynthesisBudgetExceeded(
                f"Category {category_name} exhausted synthesis wait budget after "
                f"{diagnostics.synthesis_wait_seconds_total:.1f} seconds."
            )
        if diagnostics.synthesis_retry_count >= MAX_CATEGORY_SYNTHESIS_RETRIES:
            if not diagnostics.synthesis_budget_exhausted:
                self.diagnostics.synthesis_budget_exhausted_count += 1
            diagnostics.synthesis_budget_exhausted = True
            diagnostics.synthesis_retry_skipped_count += 1
            raise SynthesisBudgetExceeded(
                f"Category {category_name} exhausted synthesis retry budget after "
                f"{diagnostics.synthesis_retry_count} retries."
            )

    def record_split(self, category_name: str, reason: str) -> None:
        diagnostics = self.get_category_diagnostics(category_name)
        if reason not in diagnostics.split_reasons:
            diagnostics.split_reasons.append(reason)
        if reason == "pre_send_budget":
            self.diagnostics.pre_send_split_count += 1
        elif reason == "response_413":
            self.diagnostics.response_413_split_count += 1

    def record_batch_attempt(self, context: BatchContext, model_id: str) -> None:
        if context.llm_task != LLMTask.TOP_ALERTS.value:
            self.diagnostics.batch_count += 1
        task_counts = self.diagnostics.model_task_counts.setdefault(context.llm_task, {})
        task_counts[model_id] = task_counts.get(model_id, 0) + 1
        diagnostics = self.get_category_diagnostics(context.category_name)
        if model_id not in diagnostics.models_attempted:
            diagnostics.models_attempted.append(model_id)
        diagnostics.estimated_input_tokens_max = max(
            diagnostics.estimated_input_tokens_max,
            context.estimated_input_tokens,
        )
        diagnostics.serialized_request_bytes_max = max(
            diagnostics.serialized_request_bytes_max,
            context.serialized_request_bytes,
        )

    def record_json_repair_retry(self) -> None:
        self.diagnostics.json_repair_retry_count += 1

    def record_resolver_selection(
        self,
        *,
        task: str,
        preferred_model: str,
        selected_model: str,
        rejections: list[dict[str, str]],
        avoided_wait_seconds: float,
    ) -> None:
        for rejection in rejections:
            item = {"task": task, **rejection}
            if item not in self.diagnostics.resolver_rejections:
                self.diagnostics.resolver_rejections.append(item)
        if selected_model != preferred_model:
            substitution = {
                "task": task,
                "from_model": preferred_model,
                "to_model": selected_model,
            }
            if avoided_wait_seconds > 0:
                substitution["avoided_wait_seconds"] = round(avoided_wait_seconds, 3)
                self.diagnostics.avoided_rate_limit_wait_count += 1
                self.diagnostics.avoided_rate_limit_wait_seconds_total += avoided_wait_seconds
            self.diagnostics.model_substitutions.append(substitution)

    def record_failed_batch(self) -> None:
        self.diagnostics.failed_batch_count += 1

    def tighten_category_budget(self, category_name: str, classification: str, *, batch_kind: str = "article_batch") -> None:
        state = self.get_category_budget(category_name)
        if batch_kind == "synthesis":
            token_attr = "synthesis_input_budget_tokens"
            byte_attr = "synthesis_request_byte_budget"
            min_tokens = MIN_SYNTHESIS_INPUT_BUDGET_TOKENS
            min_bytes = MIN_SYNTHESIS_REQUEST_BYTE_BUDGET
        else:
            token_attr = "article_input_budget_tokens"
            byte_attr = "article_request_byte_budget"
            min_tokens = MIN_INPUT_BUDGET_TOKENS
            min_bytes = MIN_REQUEST_BYTE_BUDGET

        current_tokens = getattr(state, token_attr)
        current_bytes = getattr(state, byte_attr)
        if classification == "payload_too_large":
            setattr(state, byte_attr, max(min_bytes, current_bytes // 2))
        elif classification == "rate_limited":
            setattr(state, token_attr, max(min_tokens, current_tokens // 2))
            setattr(state, byte_attr, max(min_bytes, int(current_bytes * 0.75)))
        else:
            setattr(state, token_attr, max(min_tokens, int(current_tokens * 0.8)))
            setattr(state, byte_attr, max(min_bytes, int(current_bytes * 0.8)))


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------


def _chat_completion(
    runtime: AnalysisRuntime,
    messages: list[dict[str, Any]],
    estimated_input_tokens: int,
    context: BatchContext,
    *,
    model_override: ModelConfig | None = None,
) -> tuple[str, str]:
    response: requests.Response | None = None
    for attempt in range(DEFAULT_CHAT_RETRIES):
        preferred_model = model_override or runtime.current_model
        task = context.llm_task
        candidate_chain = [preferred_model] if model_override is not None else runtime.model_chain
        rate_limit_waits = {
            candidate.model_id: runtime.governor.peek_key(
                candidate.model_id,
                runtime.current_key_index,
                len(runtime.groq_api_keys),
                estimated_input_tokens,
            )[1]
            for candidate in candidate_chain
        }
        selection = (runtime.resolver or ModelResolver()).resolve(
            task,
            candidate_chain,
            estimated_input_tokens=estimated_input_tokens,
            requested_output_tokens=preferred_model.max_completion_tokens,
            rate_limit_waits=rate_limit_waits,
            preferred_model_id=preferred_model.model_id,
        )
        model = selection.model
        runtime.record_resolver_selection(
            task=task,
            preferred_model=preferred_model.model_id,
            selected_model=model.model_id,
            rejections=selection.rejections,
            avoided_wait_seconds=selection.avoided_wait_seconds,
        )
        if context.batch_kind.startswith("synthesis"):
            runtime.ensure_synthesis_budget(context.category_name)
        api_url = model.api_url or GROQ_CHAT_COMPLETIONS_URL
        session = runtime.get_session_for_model(model)
        runtime.last_attempted_model = model.model_id
        attempt_context = BatchContext(
            category_name=context.category_name,
            batch_kind=context.batch_kind,
            batch_label=context.batch_label,
            article_count=context.article_count,
            estimated_input_tokens=estimated_input_tokens,
            serialized_request_bytes=_estimate_request_payload_bytes(model.model_id, messages, model.max_completion_tokens),
            content_shrunk=context.content_shrunk,
            llm_task=task,
        )
        selected_wait = rate_limit_waits.get(model.model_id, 0.0)
        if task == LLMTask.TOP_ALERTS.value and selected_wait > 0:
            raise RuntimeError(f"Top-alert generation degraded instead of waiting {selected_wait:.1f}s for rate limit reset.")
        runtime.record_batch_attempt(attempt_context, model.model_id)
        LOGGER.debug(
            "Sending %s for %s batch=%s model=%s articles=%s estimated_tokens=%s serialized_bytes=%s shrunk=%s attempt=%s.",
            attempt_context.batch_kind,
            attempt_context.category_name,
            attempt_context.batch_label,
            model.model_id,
            attempt_context.article_count,
            attempt_context.estimated_input_tokens,
            attempt_context.serialized_request_bytes,
            attempt_context.content_shrunk,
            attempt + 1,
        )
        # Select the API key with the most remaining capacity; rotate silently if a
        # better key exists, sleep only when all keys are exhausted for this model.
        best_key, wait_seconds = runtime.governor.select_key(
            model.model_id,
            runtime.current_key_index,
            len(runtime.groq_api_keys),
            estimated_input_tokens,
        )
        if best_key != runtime.current_key_index and model.provider == DEFAULT_PROVIDER:
            LOGGER.info(
                "Governor rotating key %d → %d for model %s (more capacity available).",
                runtime.current_key_index,
                best_key,
                model.model_id,
            )
            runtime.current_key_index = best_key
            runtime.diagnostics.key_rotation_count += 1
            session = runtime.get_session_for_model(model)
        runtime.record_wait(context.category_name, wait_seconds, batch_kind=context.batch_kind)
        if wait_seconds > 0:
            LOGGER.info(
                "Waiting %.1f seconds before %s for category %s batch=%s on %s (key=%d).",
                wait_seconds,
                attempt_context.batch_kind,
                attempt_context.category_name,
                attempt_context.batch_label,
                model.model_id,
                runtime.current_key_index,
            )
        connect_timeout, read_timeout, total_deadline = _llm_request_timeouts()
        try:
            response = _post_with_deadline(
                session,
                api_url,
                json_body={
                    "model": model.model_id,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_completion_tokens": model.max_completion_tokens,
                    "response_format": {"type": "json_object"},
                },
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                total_deadline=total_deadline,
            )
        except requests.exceptions.RequestException as exc:
            # Network failure or hard-deadline abort: never sit on a stuck
            # socket. Treat like a transient server error and let key rotation /
            # model fallback recover instead of failing the whole category.
            runtime.diagnostics.request_timeout_count += 1
            LOGGER.warning(
                "LLM request error for category %s batch=%s on %s (key=%d): %s",
                attempt_context.category_name,
                attempt_context.batch_label,
                model.model_id,
                runtime.current_key_index,
                exc,
            )
            runtime.reset_session_for_model(model)
            runtime.record_retry(attempt_context.category_name, batch_kind=attempt_context.batch_kind)
            if attempt == DEFAULT_CHAT_RETRIES - 1:
                raise
            if model.provider == DEFAULT_PROVIDER and runtime.rotate_key(
                f"request error on key {runtime.current_key_index} / model {model.model_id}"
            ):
                session = runtime.get_session_for_model(model)
                continue
            if model_override is None and runtime.switch_to_next_model(
                f"Request error on {model.model_id}: {exc}"
            ):
                runtime.current_key_index = 0
                session = runtime.get_session_for_model(runtime.current_model)
                runtime.record_model_switch(context.category_name, runtime.model_switches[-1])
                continue
            delay_seconds = min(5.0 * (attempt + 1), 20.0)
            runtime.record_wait(attempt_context.category_name, delay_seconds, batch_kind=attempt_context.batch_kind)
            time.sleep(delay_seconds)
            session = runtime.get_session_for_model(model)
            continue
        runtime.governor.record_response(model.model_id, response, key_index=runtime.current_key_index)
        LOGGER.debug(
            "Received HTTP %s for %s category=%s batch=%s model=%s.",
            response.status_code,
            attempt_context.batch_kind,
            attempt_context.category_name,
            attempt_context.batch_label,
            model.model_id,
        )

        if response.status_code == 413:
            LOGGER.info(
                "Groq returned HTTP 413 for category %s batch=%s on %s.",
                attempt_context.category_name,
                attempt_context.batch_label,
                model.model_id,
            )
            response.raise_for_status()

        if response.status_code == 429 and model_override is None:
            LOGGER.info(
                "Groq returned HTTP 429 for category %s batch=%s on %s (key=%d).",
                attempt_context.category_name,
                attempt_context.batch_label,
                model.model_id,
                runtime.current_key_index,
            )
            runtime.tighten_category_budget(
                attempt_context.category_name,
                "rate_limited",
                batch_kind="synthesis" if attempt_context.batch_kind.startswith("synthesis") else "article_batch",
            )
            # Try rotating to another key before switching models.
            if runtime.rotate_key(f"429 on key {runtime.current_key_index} / model {model.model_id}"):
                session = runtime.get_session_for_model(model)
                continue
            # All keys exhausted for this model — switch model and reset key index.
            if runtime.switch_to_next_model(f"All keys returned 429 on {model.model_id}."):
                runtime.current_key_index = 0
                session = runtime.get_session_for_model(runtime.current_model)
                runtime.record_model_switch(context.category_name, runtime.model_switches[-1])
                continue

        if response.status_code in {429, 500, 502, 503, 504}:
            if response.status_code == 429:
                LOGGER.info(
                    "Groq returned HTTP 429 for category %s batch=%s on %s.",
                    attempt_context.category_name,
                    attempt_context.batch_label,
                    model.model_id,
                )
                runtime.tighten_category_budget(
                    attempt_context.category_name,
                    "rate_limited",
                    batch_kind="synthesis" if attempt_context.batch_kind.startswith("synthesis") else "article_batch",
                )
            runtime.record_retry(attempt_context.category_name, batch_kind=attempt_context.batch_kind)
            if attempt == DEFAULT_CHAT_RETRIES - 1:
                response.raise_for_status()
            delay_seconds = runtime.governor.apply_backoff(model.model_id, response, attempt)
            runtime.record_wait(attempt_context.category_name, delay_seconds, batch_kind=attempt_context.batch_kind)
            LOGGER.info(
                "Retrying category %s batch=%s on %s after HTTP %s in %.1f seconds.",
                attempt_context.category_name,
                attempt_context.batch_label,
                model.model_id,
                response.status_code,
                delay_seconds,
            )
            continue

        if response.status_code in {400, 404}:
            # A 404 (model gone) or a 400 on an already-minimal request (≤1 item,
            # so it cannot be split smaller) means the model is rejecting our
            # request shape, not that the request is too big. Evict it for this
            # run and re-resolve onto a working model instead of failing. A 400
            # on a multi-item batch is treated as oversized and falls through to
            # the split/retry path below.
            minimal_request = attempt_context.article_count <= 1
            if (response.status_code == 404 or minimal_request) and model_override is None:
                LOGGER.warning(
                    "Model %s rejected request (HTTP %s) for category %s batch=%s; evicting and re-resolving.",
                    model.model_id,
                    response.status_code,
                    attempt_context.category_name,
                    attempt_context.batch_label,
                )
                if runtime.evict_model(model.model_id):
                    runtime.diagnostics.model_decommissioned_count += 1
                    runtime.current_key_index = 0
                    session = runtime.get_session_for_model(runtime.current_model)
                    continue

        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Groq response content was not a string.")
        return content, model.model_id

    if response is not None:
        response.raise_for_status()
    raise RuntimeError("Groq request retries exhausted.")


def _invoke_json_with_retry(
    runtime: AnalysisRuntime,
    messages: list[dict[str, Any]],
    estimated_input_tokens: int,
    context: BatchContext,
    *,
    model_override: ModelConfig | None = None,
) -> tuple[dict[str, Any], str]:
    raw_text, model_used = _chat_completion(
        runtime,
        messages,
        estimated_input_tokens,
        context,
        model_override=model_override,
    )
    try:
        return _parse_json_content(raw_text), model_used
    except ValueError:
        runtime.record_json_repair_retry()
        LOGGER.info(
            "Repairing invalid JSON for category %s %s %s.",
            context.category_name,
            context.batch_kind,
            context.batch_label,
        )
        repair_messages = list(messages) + [
            {"role": "assistant", "content": raw_text},
            {
                "role": "user",
                "content": "Repair your previous reply into one valid JSON object only. Do not add markdown fences or commentary.",
            },
        ]
        repaired_tokens = _estimate_messages_tokens(repair_messages)
        repair_context = BatchContext(
            category_name=context.category_name,
            batch_kind=f"{context.batch_kind}_repair",
            batch_label=context.batch_label,
            article_count=context.article_count,
            estimated_input_tokens=repaired_tokens,
            serialized_request_bytes=_estimate_request_payload_bytes(
                (model_override or runtime.current_model).model_id,
                repair_messages,
                (model_override or runtime.current_model).max_completion_tokens,
            ),
            content_shrunk=context.content_shrunk,
            llm_task=LLMTask.JSON_REPAIR.value,
        )
        repaired_text, repaired_model = _chat_completion(
            runtime,
            repair_messages,
            repaired_tokens,
            repair_context,
            model_override=model_override,
        )
        return _parse_json_content(repaired_text), repaired_model
