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
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from .config import get_project_root
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


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


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


def _build_provider_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    )
    return session


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
    def __init__(self, *, time_fn=time.monotonic, sleep_fn=time.sleep):
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
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

        # All keys exhausted — sleep until the soonest one resets.
        sleep_secs = max(0.0, earliest_reset - now) if earliest_reset != float("inf") else 0.0
        if sleep_secs > 0:
            self._sleep_fn(sleep_secs)
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


@dataclass(slots=True)
class AnalysisRuntime:
    governor: RateLimitGovernor
    model_chain: list[ModelConfig]
    groq_api_keys: list[str] = field(default_factory=list)
    session: requests.Session | None = None
    delayed_retry_final_model: ModelConfig | None = None
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
        self.diagnostics.batch_count += 1
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
        model = model_override or runtime.current_model
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
        )
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
        response = session.post(
            api_url,
            json={
                "model": model.model_id,
                "messages": messages,
                "temperature": 0.1,
                "max_completion_tokens": model.max_completion_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
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
        )
        repaired_text, repaired_model = _chat_completion(
            runtime,
            repair_messages,
            repaired_tokens,
            repair_context,
            model_override=model_override,
        )
        return _parse_json_content(repaired_text), repaired_model
