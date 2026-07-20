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

from .budget import UNLIMITED, DailyBudgetLedger
from .config import get_project_root
from .model_catalog import ModelCapability, get_capability
from .types import (
    BatchContext,
    CategoryBudgetState,
    CategoryDiagnostics,
    DEFAULT_CHAT_RETRIES,
    DEFAULT_OUTPUT_TOKENS,
    DEFAULT_PROVIDER,
    CEREBRAS_CHAT_COMPLETIONS_URL,
    GOOGLE_AI_STUDIO_CHAT_COMPLETIONS_URL,
    GROQ_CHAT_COMPLETIONS_URL,
    MAX_CATEGORY_SYNTHESIS_RETRIES,
    MAX_CATEGORY_SYNTHESIS_WAIT_SECONDS,
    MIN_INPUT_BUDGET_TOKENS,
    MIN_REQUEST_BYTE_BUDGET,
    MIN_SYNTHESIS_INPUT_BUDGET_TOKENS,
    MIN_SYNTHESIS_REQUEST_BYTE_BUDGET,
    ModelConfig,
    ModelRateLimitState,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    RATE_LIMIT_REQUEST_FLOOR,
    RATE_LIMIT_TOKEN_FLOOR,
    RuntimeDiagnostics,
    SynthesisBudgetExceeded,
    ZAI_CHAT_COMPLETIONS_URL,
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


class NoEligibleEndpoint(RuntimeError):
    """Raised when every endpoint exceeds the task's allowed quota wait."""


@dataclass(frozen=True)
class ModelSelection:
    model: ModelConfig
    rejections: list[dict[str, str]] = field(default_factory=list)
    avoided_wait_seconds: float = 0.0
    wait_seconds: float = 0.0
    wait_exceeded: bool = False


DEFAULT_MAX_LLM_WAIT_SECONDS = 45.0
DEFAULT_LLM_TIMEOUT_COOLDOWN_SECONDS = 30.0
DEFAULT_MODEL_POLICY = "production_with_qwen"
DEFAULT_PREVIEW_MODEL_ALLOWLIST = frozenset({"qwen/qwen3.6-27b"})
TASK_WAIT_BUDGETS_SECONDS = {
    LLMTask.ROUTING.value: 10.0,
    LLMTask.ARTICLE_ANALYSIS.value: 15.0,
    LLMTask.CATEGORY_SYNTHESIS.value: 75.0,
    LLMTask.TOP_ALERTS.value: 90.0,
    LLMTask.JSON_REPAIR.value: 10.0,
    LLMTask.CRITIC.value: 90.0,
}

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
# Fraction of a premium model's declared daily token budget held in reserve for
# high-value tasks: while remaining daily budget is above this floor, bulk work
# may use the premium model; below it, the bulk-reservation penalty kicks in so
# the remainder is saved for synthesis/top-alerts/critic.
RESERVE_FRACTION = 0.25


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


def _resolve_timeout_cooldown_seconds() -> float:
    raw = os.environ.get("DAILY_MACRO_LLM_TIMEOUT_COOLDOWN_SECONDS")
    if not raw:
        return DEFAULT_LLM_TIMEOUT_COOLDOWN_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_COOLDOWN_SECONDS


def _resolve_task_wait_seconds(task: str) -> float:
    """Return the maximum useful quota wait for one LLM task.

    Per-task environment variables take precedence. The legacy global setting
    remains an override for operators who want one uniform cap.
    """
    task_name = str(task).strip().lower()
    task_env = f"DAILY_MACRO_MAX_LLM_WAIT_SECONDS_{task_name.upper()}"
    raw = os.environ.get(task_env)
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    if os.environ.get("DAILY_MACRO_MAX_LLM_WAIT_SECONDS"):
        return _resolve_max_wait_seconds()
    return TASK_WAIT_BUDGETS_SECONDS.get(task_name, DEFAULT_MAX_LLM_WAIT_SECONDS)


def _resolve_category_synthesis_wait_seconds() -> float:
    raw = os.environ.get("DAILY_MACRO_MAX_CATEGORY_SYNTHESIS_WAIT_SECONDS")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return MAX_CATEGORY_SYNTHESIS_WAIT_SECONDS


class ModelResolver:
    def __init__(
        self,
        *,
        active_model_ids: set[str] | None = None,
        model_policy: str | None = None,
        max_wait_seconds: float | None = None,
        capabilities: dict[str, ModelCapability] | None = None,
        preview_model_allowlist: set[str] | None = None,
    ) -> None:
        self.active_model_ids = active_model_ids
        self.model_policy = (
            model_policy
            or os.environ.get("DAILY_MACRO_MODEL_POLICY")
            or DEFAULT_MODEL_POLICY
        ).strip().lower()
        self.max_wait_seconds = _resolve_max_wait_seconds(max_wait_seconds)
        self.capabilities = capabilities or {}
        configured_preview_models = os.environ.get("DAILY_MACRO_PREVIEW_MODEL_ALLOWLIST")
        if preview_model_allowlist is not None:
            self.preview_model_allowlist = set(preview_model_allowlist)
        elif configured_preview_models:
            self.preview_model_allowlist = {
                item.strip() for item in configured_preview_models.split(",") if item.strip()
            }
        else:
            self.preview_model_allowlist = set(DEFAULT_PREVIEW_MODEL_ALLOWLIST)

    def _preview_model_allowed(self, model: ModelConfig, capability: ModelCapability) -> bool:
        if capability.lifecycle == "production":
            return True
        if self.model_policy == "allow_preview":
            return True
        if self.model_policy == "production_with_qwen":
            return model.model_id in self.preview_model_allowlist
        return False

    def wait_budget_seconds(self, task: LLMTask | str) -> float:
        task_value = task.value if isinstance(task, LLMTask) else str(task)
        return _resolve_task_wait_seconds(task_value)

    def task_preferences(self, task: str) -> list[str]:
        env_name = f"DAILY_MACRO_MODEL_{task.upper()}_PREFERENCES"
        raw = os.environ.get(env_name) or os.environ.get("DAILY_MACRO_MODEL_PREFERENCES", "")
        return [item.strip() for item in raw.split(",") if item.strip()]

    def capability_for(self, model: ModelConfig) -> ModelCapability:
        return (
            self.capabilities.get(model.endpoint_id)
            or self.capabilities.get(f"{model.provider}:{model.model_id}")
            or self.capabilities.get(model.model_id)
            or get_capability(model.model_id, provider=model.provider)
        )

    @staticmethod
    def _mapping_value(mapping: dict[str, float] | None, model: ModelConfig, default: float = 0.0) -> float:
        if not mapping:
            return default
        return float(mapping.get(model.endpoint_id, mapping.get(model.model_id, default)))

    def resolve(
        self,
        task: LLMTask | str,
        model_chain: list[ModelConfig],
        *,
        estimated_input_tokens: int,
        requested_output_tokens: int,
        rate_limit_waits: dict[str, float] | None = None,
        preferred_model_id: str | None = None,
        budget_remaining: dict[str, float] | None = None,
        budget_required: dict[str, float] | None = None,
    ) -> ModelSelection:
        task_value = task.value if isinstance(task, LLMTask) else str(task)
        task_preferences = self.task_preferences(task_value)
        task_wait_budget = self.wait_budget_seconds(task_value)
        is_high_value = task_value in HIGH_VALUE_TASKS
        rejections: list[dict[str, str]] = []
        scored: list[tuple[float, int, ModelConfig, float]] = []
        wait_eligible: list[tuple[float, int, ModelConfig]] = []
        rate_limit_waits = rate_limit_waits or {}
        needed_tokens = estimated_input_tokens + requested_output_tokens

        for index, model in enumerate(model_chain):
            capability = self.capability_for(model)
            if not self._preview_model_allowed(model, capability):
                rejections.append({"model_id": model.model_id, "reason": "preview_model_disallowed"})
                continue
            if self.active_model_ids is not None and not (
                model.model_id in self.active_model_ids or model.endpoint_id in self.active_model_ids
            ):
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
            # Hard daily-budget gate: skip a model whose remaining TPD can't cover
            # this call rather than stalling on it. The floor model has a huge
            # daily budget and is always in the pool, so gating never strands the
            # run. budget_remaining is per the current key; UNLIMITED when the
            # model's TPD is unknown (no gating).
            if budget_remaining is not None:
                remaining_budget = self._mapping_value(budget_remaining, model, UNLIMITED)
                required_budget = self._mapping_value(budget_required, model, needed_tokens)
                if remaining_budget < required_budget:
                    rejections.append({"model_id": model.model_id, "reason": "daily_budget_exhausted"})
                    continue
            wait_seconds = self._mapping_value(rate_limit_waits, model)
            # Passed every hard constraint (policy, active, json, context,
            # output). Remember it as a fallback that respects those constraints
            # even if its wait exceeds the cap.
            wait_eligible.append((wait_seconds, index, model))
            if wait_seconds > task_wait_budget:
                rejections.append({"model_id": model.model_id, "reason": "rate_limit_wait_too_long"})
                continue

            task_score = capability.task_scores.get(task_value, capability.task_scores.get("article_analysis", 0.5))
            preferred_bonus = 0.5 if model.model_id == preferred_model_id or model.endpoint_id == preferred_model_id else 0.0
            preference_bonus = 0.0
            preference_key = model.endpoint_id if model.endpoint_id in task_preferences else model.model_id
            if preference_key in task_preferences:
                preference_bonus = max(0.0, 0.35 - task_preferences.index(preference_key) * 0.03)
            wait_penalty = min(wait_seconds / max(task_wait_budget, 1.0), 1.0) * 0.4
            order_penalty = index * 0.01
            # Reserve premium models for high-value tasks: penalize them on bulk
            # tasks so any non-premium model with headroom outranks them, keeping
            # bulk work off the scarce high-quality models. The penalty is soft —
            # a premium model is still chosen over sleeping on a rate-limited one,
            # and an explicit env preference is exempt.
            reservation_penalty = 0.0
            is_premium = capability.task_scores.get("category_synthesis", 0.0) >= PREMIUM_SYNTHESIS_THRESHOLD
            if is_premium and not is_high_value and model.model_id not in task_preferences:
                # Budget-aware reservation: while the premium model still has
                # ample daily budget, bulk work may use it; once remaining drops
                # below the reserve floor (or with no budget signal at all), apply
                # the penalty so the remainder is held for high-value tasks.
                declared_tpd = capability.limits.get("tpd")
                remaining_budget = self._mapping_value(budget_remaining, model, UNLIMITED) if budget_remaining else None
                reserve_floor = RESERVE_FRACTION * declared_tpd if declared_tpd else None
                if remaining_budget is None or reserve_floor is None or remaining_budget < reserve_floor:
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
            preferred_wait = 0.0
            if preferred_model_id:
                preferred_wait = float(rate_limit_waits.get(preferred_model_id, 0.0))
            return ModelSelection(
                model=model,
                rejections=rejections,
                avoided_wait_seconds=max(preferred_wait - wait_seconds, 0.0),
                wait_seconds=wait_seconds,
            )

        # Nothing scored within the wait cap. Prefer a model that satisfies every
        # hard constraint (policy/active/context/output) and just has a long
        # wait — the governor now caps the actual sleep — over silently
        # returning model_chain[0], which may be a preview model the policy
        # forbids or one whose context window can't fit the request.
        if wait_eligible:
            wait_eligible.sort(key=lambda item: (item[0], item[1]))
            wait_seconds, _index, model = wait_eligible[0]
            return ModelSelection(
                model=model,
                rejections=rejections,
                avoided_wait_seconds=0.0,
                wait_seconds=wait_seconds,
                wait_exceeded=wait_seconds > task_wait_budget,
            )

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
    if isinstance(exc, NoEligibleEndpoint):
        return "no_eligible_endpoint"
    if isinstance(exc, (requests.Timeout, LLMRequestDeadlineError)):
        return "provider_timeout"
    if isinstance(exc, requests.ConnectionError):
        return "provider_unavailable"
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


def _usage_token_counts(
    usage: dict[str, Any] | None,
    *,
    estimated_input_tokens: int,
    fallback_output_tokens: int,
) -> tuple[int, int, int]:
    """Normalize OpenAI, Gemini, and provider-specific usage fields."""
    usage = usage or {}
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("prompt_token_count")
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_token_count")
    total_tokens = usage.get("total_tokens") or usage.get("total_token_count")
    input_tokens = int(input_tokens) if isinstance(input_tokens, (int, float)) else 0
    output_tokens = int(output_tokens) if isinstance(output_tokens, (int, float)) else 0
    total_tokens = int(total_tokens) if isinstance(total_tokens, (int, float)) else 0
    if input_tokens <= 0:
        input_tokens = max(0, int(estimated_input_tokens))
    if total_tokens <= 0:
        total_tokens = input_tokens + (output_tokens if output_tokens > 0 else max(0, int(fallback_output_tokens)))
    if output_tokens <= 0:
        output_tokens = max(0, total_tokens - input_tokens)
    return input_tokens, output_tokens, max(total_tokens, input_tokens + output_tokens)


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
    # A stalled provider must not consume a full minute while other healthy
    # accounts sit idle. Operators can raise these values for unusually slow
    # models, but the default is deliberately short enough for daily-digest
    # failover to remain responsive.
    read_timeout = _read("DAILY_MACRO_LLM_READ_TIMEOUT_SECONDS", 30.0)
    total_deadline = _read("DAILY_MACRO_LLM_DEADLINE_SECONDS", 45.0)
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
    if model.api_key:
        return model.api_key
    if model.provider == DEFAULT_PROVIDER:
        return load_groq_api_key()

    env_name = model.api_key_env or "OPENAI_API_KEY"
    env_names = [env_name]
    if model.provider == "google_ai_studio":
        env_names.extend(["GOOGLE_AI_STUDIO_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"])
    for candidate_env in dict.fromkeys(env_names):
        env_value = os.environ.get(candidate_env)
        if env_value:
            return env_value

    for config_path in _candidate_config_paths():
        if not config_path.exists():
            continue
        parsed = _parse_simple_env_file(config_path)
        for candidate_env in dict.fromkeys(env_names):
            value = parsed.get(candidate_env)
            if value:
                return value

    raise RuntimeError(f"{env_name} is not set for provider {model.provider}.")


def _default_provider_url(provider: str) -> str:
    return {
        DEFAULT_PROVIDER: GROQ_CHAT_COMPLETIONS_URL,
        "cerebras": CEREBRAS_CHAT_COMPLETIONS_URL,
        "google_ai_studio": GOOGLE_AI_STUDIO_CHAT_COMPLETIONS_URL,
        "openrouter": OPENROUTER_CHAT_COMPLETIONS_URL,
        "zai": ZAI_CHAT_COMPLETIONS_URL,
    }.get(provider, GROQ_CHAT_COMPLETIONS_URL)


def _compute_units_for(capability: ModelCapability, input_tokens: int, output_tokens: int) -> int:
    """Convert token usage to a provider's metered compute units (Cloudflare neurons)."""
    input_rate = capability.limits.get("input_neurons_per_million")
    output_rate = capability.limits.get("output_neurons_per_million")
    if not input_rate and not output_rate:
        return 0
    units = (max(0, input_tokens) * (input_rate or 0) + max(0, output_tokens) * (output_rate or 0)) / 1_000_000
    return math.ceil(units)


def _chat_request_body(model: ModelConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the common OpenAI-compatible body with provider token-field quirks."""
    body: dict[str, Any] = {
        "model": model.model_id,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    # Cloudflare's OpenAI-compatible endpoint expects max_tokens. Using
    # max_completion_tokens can hang or truncate long structured Gemma calls.
    token_field = "max_tokens" if model.provider in {"zai", "cloudflare"} else "max_completion_tokens"
    body[token_field] = model.max_completion_tokens
    if model.provider == "cloudflare":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif model.provider == DEFAULT_PROVIDER and model.model_id in {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }:
        # Groq's GPT-OSS models spend completion budget on reasoning unless the
        # effort is bounded. Hidden reasoning keeps the response compatible
        # with the JSON parser used by the analysis pipeline.
        body["reasoning_effort"] = "low"
        body["reasoning_format"] = "hidden"
    return body


def _response_message_content(message: dict[str, Any]) -> str:
    """Extract a usable answer from normal or Cloudflare Qwen message fields."""
    content = message.get("content")
    if content is None:
        content = message.get("reasoning_content")
    if not isinstance(content, str):
        raise ValueError("LLM response content was not a string.")
    return content


# ---------------------------------------------------------------------------
# RateLimitGovernor
# ---------------------------------------------------------------------------


class RateLimitGovernor:
    def __init__(
        self,
        *,
        time_fn=time.monotonic,
        sleep_fn=time.sleep,
        max_wait_seconds: float | None = None,
        model_limits: dict[str, dict[str, int]] | None = None,
    ):
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self.max_wait_seconds = _resolve_max_wait_seconds(max_wait_seconds)
        # Declared per-model limits (rpm/rpd/tpm/tpd) from the catalog, used to
        # pre-seed per-minute headroom before any response header is observed.
        self.model_limits = model_limits or {}
        # When quota_scope is present, key credentials share one state bucket.
        # Without it, the legacy per-(model, key) behavior is retained.
        self._states: dict[tuple[str, int], ModelRateLimitState] = {}
        # Endpoint health is intentionally separate from quota state. Two
        # accounts may share a provider quota scope while only one account's
        # socket is unhealthy.
        self._endpoint_cooldowns: dict[str, float] = {}
        self._endpoint_attempts: dict[str, int] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _bucket_id(model_id: str, quota_scope: str | None) -> str:
        return f"{quota_scope}:{model_id}" if quota_scope else model_id

    def _state(self, model_id: str, key_index: int, quota_scope: str | None = None) -> ModelRateLimitState:
        bucket_id = self._bucket_id(model_id, quota_scope)
        key = (bucket_id, 0 if quota_scope else key_index)
        state = self._states.get(key)
        if state is None:
            state = ModelRateLimitState()
            # Pre-seed per-minute token headroom from the declared TPM so the
            # very first burst on a fresh key can't overshoot before a header is
            # seen. Real headers overwrite this on the next response.
            limits = self.model_limits.get(bucket_id, self.model_limits.get(model_id, {}))
            rpm = limits.get("rpm")
            if rpm:
                state.limit_requests = int(rpm)
                state.remaining_requests = int(rpm)
                state.reset_requests_at = self._time_fn() + 60.0
            tpm = limits.get("tpm")
            if tpm:
                state.limit_tokens = int(tpm)
                state.remaining_tokens = int(tpm)
                state.reset_tokens_at = self._time_fn() + 60.0
            self._states[key] = state
        return state

    @staticmethod
    def _refresh_expired_window(state: ModelRateLimitState, now: float) -> None:
        """Refill declared minute capacity after a known reset.

        Providers use moving token buckets, so this is deliberately a
        conservative local approximation. The next response header remains
        authoritative and overwrites the estimate.
        """
        if state.limit_requests is not None and state.reset_requests_at is not None and now >= state.reset_requests_at:
            state.remaining_requests = state.limit_requests
            state.reset_requests_at = now + 60.0
        if state.limit_tokens is not None and state.reset_tokens_at is not None and now >= state.reset_tokens_at:
            state.remaining_tokens = state.limit_tokens
            state.reset_tokens_at = now + 60.0

    def select_key(
        self,
        model_id: str,
        current_key: int,
        num_keys: int,
        estimated_input_tokens: int = 0,
        quota_scope: str | None = None,
        max_wait_seconds: float | None = None,
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
            state = self._state(model_id, ki, quota_scope)
            self._refresh_expired_window(state, now)
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
        wait_cap = self.max_wait_seconds if max_wait_seconds is None else max(0.0, float(max_wait_seconds))
        sleep_secs = min(sleep_secs, wait_cap)
        if sleep_secs > 0:
            self._sleep_fn(sleep_secs)
            # The normal sleep advances monotonic time. Test doubles and
            # injected schedulers may not advance their clock, so explicitly
            # release the window we just waited through; otherwise the next
            # task would count the same reset twice.
            for ki in range(num_keys):
                state = self._states.get((self._bucket_id(model_id, quota_scope), 0 if quota_scope else ki))
                if state is None:
                    continue
                if state.reset_requests_at is not None and state.reset_requests_at <= now + sleep_secs:
                    state.remaining_requests = state.limit_requests if state.limit_requests is not None else 1
                    state.reset_requests_at = now
                if state.reset_tokens_at is not None and state.reset_tokens_at <= now + sleep_secs:
                    state.remaining_tokens = state.limit_tokens if state.limit_tokens is not None else max(1, estimated_input_tokens + 1)
                    state.reset_tokens_at = now
        return current_key, sleep_secs

    def reserve_request(
        self,
        model_id: str,
        estimated_tokens: int = 0,
        *,
        key_index: int = 0,
        quota_scope: str | None = None,
    ) -> None:
        """Reserve declared request/token headroom before the HTTP call.

        Live headers replace these estimates after the response. Reserving the
        local catalog values prevents a burst of calls from overshooting a
        provider's published RPM/TPM before the first header arrives.
        """
        with self._lock:
            state = self._state(model_id, key_index, quota_scope)
            if state.remaining_requests is not None:
                state.remaining_requests = max(0, state.remaining_requests - 1)
            if state.remaining_tokens is not None:
                state.remaining_tokens = max(0, state.remaining_tokens - max(0, int(estimated_tokens)))

    def select_and_reserve(
        self,
        model_id: str,
        current_key: int,
        num_keys: int,
        estimated_input_tokens: int = 0,
        quota_scope: str | None = None,
        max_wait_seconds: float | None = None,
    ) -> tuple[int, float]:
        """Select and reserve one key atomically for concurrent workers."""
        with self._lock:
            selected_key, waited = self.select_key(
                model_id,
                current_key,
                num_keys,
                estimated_input_tokens,
                quota_scope=quota_scope,
                max_wait_seconds=max_wait_seconds,
            )
            self.reserve_request(
                model_id,
                estimated_input_tokens,
                key_index=selected_key,
                quota_scope=quota_scope,
            )
            return selected_key, waited

    def peek_key(
        self,
        model_id: str,
        current_key: int,
        num_keys: int,
        estimated_input_tokens: int = 0,
        quota_scope: str | None = None,
    ) -> tuple[int, float]:
        with self._lock:
            now = self._time_fn()
            best_key: int | None = None
            best_remaining_tokens = -1
            earliest_reset = float("inf")

            for ki in range(num_keys):
                state = self._state(model_id, ki, quota_scope)
                self._refresh_expired_window(state, now)
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

    def endpoint_cooldown_seconds(self, endpoint_id: str) -> float:
        with self._lock:
            return max(0.0, self._endpoint_cooldowns.get(endpoint_id, 0.0) - self._time_fn())

    def mark_endpoint_cooldown(self, endpoint_id: str, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            until = self._time_fn() + seconds
            self._endpoint_cooldowns[endpoint_id] = max(
                until,
                self._endpoint_cooldowns.get(endpoint_id, 0.0),
            )

    def clear_endpoint_cooldown(self, endpoint_id: str) -> None:
        with self._lock:
            self._endpoint_cooldowns.pop(endpoint_id, None)

    def record_endpoint_attempt(self, endpoint_id: str) -> None:
        with self._lock:
            self._endpoint_attempts[endpoint_id] = self._endpoint_attempts.get(endpoint_id, 0) + 1

    def endpoint_attempt_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._endpoint_attempts)

    def before_request(self, model_id: str, estimated_input_tokens: int = 0) -> float:
        """Backward-compatible single-key check. Uses key_index=0."""
        _key, waited = self.select_key(model_id, 0, 1, estimated_input_tokens)
        return waited

    def record_response(
        self,
        model_id: str,
        response: requests.Response,
        key_index: int = 0,
        quota_scope: str | None = None,
    ) -> None:
        with self._lock:
            state = self._state(model_id, key_index, quota_scope)
            now = self._time_fn()
            headers = {key.lower(): value for key, value in response.headers.items()}

            header_limit_requests = _parse_int(headers.get("x-ratelimit-limit-requests-minute"))
            if header_limit_requests is not None:
                state.limit_requests = header_limit_requests
            header_limit_tokens = _parse_int(
                headers.get("x-ratelimit-limit-tokens-minute")
                or headers.get("x-ratelimit-limit-tokens")
            )
            if header_limit_tokens is not None:
                state.limit_tokens = header_limit_tokens

            remaining_requests = _parse_int(
                headers.get("x-ratelimit-remaining-requests")
                or headers.get("x-ratelimit-remaining-requests-minute")
                or headers.get("x-ratelimit-remaining-requests-day")
            )
            if remaining_requests is not None:
                state.remaining_requests = remaining_requests

            remaining_tokens = _parse_int(
                headers.get("x-ratelimit-remaining-tokens")
                or headers.get("x-ratelimit-remaining-tokens-minute")
                or headers.get("x-ratelimit-remaining-tokens-day")
            )
            if remaining_tokens is not None:
                state.remaining_tokens = remaining_tokens

            reset_requests = _parse_duration_seconds(
                headers.get("x-ratelimit-reset-requests")
                or headers.get("x-ratelimit-reset-requests-minute")
                or headers.get("x-ratelimit-reset-requests-day")
            )
            if reset_requests is not None:
                state.reset_requests_at = now + reset_requests

            reset_tokens = _parse_duration_seconds(
                headers.get("x-ratelimit-reset-tokens")
                or headers.get("x-ratelimit-reset-tokens-minute")
                or headers.get("x-ratelimit-reset-tokens-day")
            )
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
    budget: DailyBudgetLedger = field(default_factory=DailyBudgetLedger)

    def __post_init__(self) -> None:
        if self.resolver is None:
            self.resolver = ModelResolver()
        if self.session is not None:
            if not self.groq_api_keys:
                self.groq_api_keys = ["session-backed-groq-key"]
            self.groq_sessions.setdefault(0, self.session)

    def fork_for_worker(self) -> "AnalysisRuntime":
        """Create isolated mutable execution state for one parallel batch."""
        worker_resolver = None
        if self.resolver is not None:
            worker_resolver = ModelResolver(
                active_model_ids=(
                    set(self.resolver.active_model_ids)
                    if self.resolver.active_model_ids is not None
                    else None
                ),
                model_policy=self.resolver.model_policy,
                max_wait_seconds=self.resolver.max_wait_seconds,
                capabilities=self.resolver.capabilities,
                preview_model_allowlist=set(self.resolver.preview_model_allowlist),
            )
        worker = AnalysisRuntime(
            governor=self.governor,
            model_chain=list(self.model_chain),
            groq_api_keys=list(self.groq_api_keys),
            delayed_retry_final_model=self.delayed_retry_final_model,
            resolver=worker_resolver,
            budget=self.budget,
        )
        worker.market_context_string = self.market_context_string
        worker.macro_release_digest = dict(self.macro_release_digest)
        return worker

    def merge_worker_diagnostics(self, worker: "AnalysisRuntime") -> None:
        """Merge one worker's counters into the parent runtime."""
        parent = self.diagnostics
        child = worker.diagnostics
        additive = (
            "rate_limit_wait_count", "rate_limit_wait_seconds_total", "fallback_switch_count",
            "pre_send_split_count", "response_413_split_count", "json_repair_retry_count",
            "batch_count", "failed_batch_count", "delayed_retry_candidate_count",
            "delayed_retry_attempted_count", "delayed_retry_recovered_count",
            "delayed_retry_failed_count", "delayed_retry_skipped_final_model_count",
            "synthesis_budget_exhausted_count", "degraded_merge_count", "key_rotation_count",
            "request_timeout_count", "endpoint_cooldown_count", "endpoint_cooldown_seconds_total",
            "model_decommissioned_count", "daily_budget_skip_count",
            "daily_budget_tokens_used", "cloudflare_neurons_used", "llm_request_count", "input_tokens_used",
            "output_tokens_used", "total_tokens_used", "avoided_rate_limit_wait_count",
            "avoided_rate_limit_wait_seconds_total", "degraded_mode_count", "parallel_batch_count",
            "llm_request_seconds_total", "timeout_seconds_total",
            "quality_review_count", "quality_review_failed_count", "quality_review_skipped_count",
        )
        for name in additive:
            setattr(parent, name, getattr(parent, name) + getattr(child, name))
        parent.high_medium_unresolved_count = max(parent.high_medium_unresolved_count, child.high_medium_unresolved_count)
        parent.light_unresolved_count = max(parent.light_unresolved_count, child.light_unresolved_count)
        parent.resolver_rejections.extend(child.resolver_rejections)
        parent.model_substitutions.extend(child.model_substitutions)
        parent.rate_limit_events.extend(child.rate_limit_events)
        self.model_switches.extend(worker.model_switches)
        for key, value in child.failure_classifications.items():
            parent.failure_classifications[key] = parent.failure_classifications.get(key, 0) + value
        for key, value in child.split_counts_by_kind.items():
            parent.split_counts_by_kind[key] = parent.split_counts_by_kind.get(key, 0) + value
        for endpoint, values in child.rate_limit_waits_by_endpoint.items():
            target = parent.rate_limit_waits_by_endpoint.setdefault(endpoint, {"count": 0, "seconds": 0.0})
            target["count"] = int(target.get("count", 0)) + int(values.get("count", 0))
            target["seconds"] = float(target.get("seconds", 0.0)) + float(values.get("seconds", 0.0))
        for field_name in ("model_task_counts", "endpoint_task_counts", "endpoint_usage"):
            target = getattr(parent, field_name)
            for outer_key, counts in getattr(child, field_name).items():
                target_counts = target.setdefault(outer_key, {})
                for key, value in counts.items():
                    target_counts[key] = target_counts.get(key, 0) + value
        for field_name in ("llm_request_seconds_by_task", "llm_request_seconds_by_endpoint"):
            target = getattr(parent, field_name)
            for key, value in getattr(child, field_name).items():
                target[key] = target.get(key, 0.0) + value
        for category_name, child_category in worker.category_diagnostics.items():
            target_category = self.category_diagnostics.setdefault(category_name, CategoryDiagnostics())
            for name in (
                "rate_limit_waits", "partial_article_count", "sub_batch_count",
                "synthesis_wait_seconds_total", "synthesis_retry_count",
                "synthesis_retry_skipped_count",
            ):
                setattr(target_category, name, getattr(target_category, name) + getattr(child_category, name))
            target_category.estimated_input_tokens_max = max(
                target_category.estimated_input_tokens_max,
                child_category.estimated_input_tokens_max,
            )
            target_category.serialized_request_bytes_max = max(
                target_category.serialized_request_bytes_max,
                child_category.serialized_request_bytes_max,
            )
            target_category.synthesis_merge_depth_max = max(
                target_category.synthesis_merge_depth_max,
                child_category.synthesis_merge_depth_max,
            )
            target_category.synthesis_budget_exhausted |= child_category.synthesis_budget_exhausted
            target_category.degraded_merge_used |= child_category.degraded_merge_used
            if child_category.degraded_merge_reason:
                target_category.degraded_merge_reason = child_category.degraded_merge_reason
            for name in ("split_reasons", "models_attempted"):
                target_list = getattr(target_category, name)
                for value in getattr(child_category, name):
                    if value not in target_list:
                        target_list.append(value)
            target_category.model_switches.extend(child_category.model_switches)

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
        session = self.provider_sessions.get(model.session_key)
        if session is not None:
            return session
        session = _build_provider_session(_load_model_api_key(model))
        self.provider_sessions[model.session_key] = session
        return session

    def key_index_for_model(self, model: ModelConfig) -> int:
        return self.current_key_index if model.provider == DEFAULT_PROVIDER else 0

    def key_count_for_model(self, model: ModelConfig) -> int:
        return max(1, len(self.groq_api_keys)) if model.provider == DEFAULT_PROVIDER else 1

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
        # Persist daily token/request usage so budget survives across same-day
        # runs; best-effort, never blocks teardown.
        self.budget.flush()
        for session in self.groq_sessions.values():
            session.close()
        self.groq_sessions.clear()
        for session in self.provider_sessions.values():
            session.close()
        self.provider_sessions.clear()

    def close_worker_sessions(self) -> None:
        """Close worker-local HTTP sessions without flushing shared state."""
        for session in self.groq_sessions.values():
            session.close()
        self.groq_sessions.clear()
        for session in self.provider_sessions.values():
            session.close()
        self.provider_sessions.clear()

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
        session = self.provider_sessions.pop(model.session_key, None)
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        # Other provider/account sessions remain valid and are kept cached.

    def get_model_config(self, model_id: str) -> ModelConfig:
        for model in self.model_chain:
            if model.model_id == model_id:
                return model
        raise KeyError(model_id)

    def _model_index(self, model: ModelConfig | str | None) -> int | None:
        if model is None:
            return self.current_model_index
        if isinstance(model, ModelConfig):
            for index, candidate in enumerate(self.model_chain):
                if candidate.endpoint_id == model.endpoint_id:
                    return index
            return None
        for index, candidate in enumerate(self.model_chain):
            if candidate.endpoint_id == model:
                return index
        matching = [index for index, candidate in enumerate(self.model_chain) if candidate.model_id == model]
        if not matching:
            return None
        return next((index for index in matching if index >= self.current_model_index), matching[0])

    def next_model_after(self, model: ModelConfig | str | None = None) -> ModelConfig | None:
        index = self._model_index(model)
        if index is None:
            return None
        next_index = index + 1
        if next_index < len(self.model_chain):
            return self.model_chain[next_index]
        return None

    def switch_to_next_model(self, reason: str, *, failed_model: ModelConfig | None = None) -> bool:
        failed = failed_model or self.current_model
        failed_index = self._model_index(failed)
        if failed_index is None:
            return False
        next_index = failed_index + 1
        if next_index >= len(self.model_chain):
            return False
        next_model = self.model_chain[next_index]
        if next_model is None:
            return False
        self.diagnostics.fallback_switch_count += 1
        switch = {
            "switched_at": datetime.now().astimezone().isoformat(),
            "from_model": failed.model_id,
            "to_model": next_model.model_id,
            "from_endpoint": failed.endpoint_id,
            "to_endpoint": next_model.endpoint_id,
            "reason": reason,
        }
        self.model_switches.append(switch)
        LOGGER.info(
            "Switching LLM endpoint from %s to %s (model=%s): %s",
            failed.endpoint_id,
            next_model.endpoint_id,
            next_model.model_id,
            reason,
        )
        self.current_model_index = next_index
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
        index = next(
            (i for i, m in enumerate(self.model_chain) if m.model_id == model_id or m.endpoint_id == model_id),
            None,
        )
        if index is None:
            return False
        removed_model = self.model_chain.pop(index)
        if self.current_model_index >= len(self.model_chain):
            self.current_model_index = len(self.model_chain) - 1
        if self.resolver is not None and self.resolver.active_model_ids is not None:
            self.resolver.active_model_ids.discard(model_id)
            self.resolver.active_model_ids.discard(removed_model.model_id)
            self.resolver.active_model_ids.discard(removed_model.endpoint_id)
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

    def record_wait(
        self,
        category_name: str,
        delay_seconds: float,
        *,
        batch_kind: str = "",
        model: ModelConfig | None = None,
        task: str | None = None,
        reason: str = "quota",
    ) -> None:
        if delay_seconds <= 0:
            return
        self.diagnostics.rate_limit_wait_count += 1
        self.diagnostics.rate_limit_wait_seconds_total += delay_seconds
        diagnostics = self.get_category_diagnostics(category_name)
        diagnostics.rate_limit_waits += 1
        if batch_kind.startswith("synthesis"):
            diagnostics.synthesis_wait_seconds_total += delay_seconds
        if model is not None:
            endpoint = model.endpoint_id
            endpoint_stats = self.diagnostics.rate_limit_waits_by_endpoint.setdefault(
                endpoint, {"count": 0, "seconds": 0.0}
            )
            endpoint_stats["count"] = int(endpoint_stats.get("count", 0)) + 1
            endpoint_stats["seconds"] = float(endpoint_stats.get("seconds", 0.0)) + delay_seconds
            self.diagnostics.rate_limit_events.append(
                {
                    "category": category_name,
                    "task": task or "",
                    "batch_kind": batch_kind,
                    "endpoint": endpoint,
                    "provider": model.provider,
                    "model": model.model_id,
                    "seconds": round(delay_seconds, 3),
                    "reason": reason,
                }
            )

    def record_phase(self, phase_name: str, elapsed_seconds: float) -> None:
        self.diagnostics.phase_seconds[phase_name] = max(0.0, float(elapsed_seconds))

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
        max_wait_seconds = _resolve_category_synthesis_wait_seconds()
        if diagnostics.synthesis_wait_seconds_total >= max_wait_seconds:
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

    def record_split(self, category_name: str, reason: str, *, batch_kind: str = "") -> None:
        diagnostics = self.get_category_diagnostics(category_name)
        if reason not in diagnostics.split_reasons:
            diagnostics.split_reasons.append(reason)
        if reason == "pre_send_budget":
            self.diagnostics.pre_send_split_count += 1
        elif reason == "response_413":
            self.diagnostics.response_413_split_count += 1
        split_key = batch_kind or "unknown"
        self.diagnostics.split_counts_by_kind[split_key] = (
            self.diagnostics.split_counts_by_kind.get(split_key, 0) + 1
        )

    def record_rate_limit_event(
        self,
        *,
        category_name: str,
        task: str,
        model: ModelConfig,
        wait_seconds: float,
        action: str,
        status_code: int = 429,
    ) -> None:
        self.diagnostics.rate_limit_events.append(
            {
                "category": category_name,
                "task": task,
                "endpoint": model.endpoint_id,
                "provider": model.provider,
                "model": model.model_id,
                "wait_seconds": round(max(0.0, wait_seconds), 3),
                "action": action,
                "status_code": status_code,
            }
        )

    def record_failure_classification(self, classification: str) -> None:
        self.diagnostics.failure_classifications[classification] = (
            self.diagnostics.failure_classifications.get(classification, 0) + 1
        )

    def record_batch_attempt(self, context: BatchContext, model_id: str) -> None:
        self.diagnostics.llm_request_count += 1
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

    def record_endpoint_attempt(self, context: BatchContext, model: ModelConfig) -> None:
        counts = self.diagnostics.endpoint_task_counts.setdefault(context.llm_task, {})
        counts[model.endpoint_id] = counts.get(model.endpoint_id, 0) + 1
        self.governor.record_endpoint_attempt(model.endpoint_id)

    def record_request_timing(
        self,
        context: BatchContext,
        model: ModelConfig,
        elapsed_seconds: float,
        *,
        timed_out: bool = False,
    ) -> None:
        elapsed_seconds = max(0.0, float(elapsed_seconds))
        self.diagnostics.llm_request_seconds_total += elapsed_seconds
        task_times = self.diagnostics.llm_request_seconds_by_task
        task_times[context.llm_task] = task_times.get(context.llm_task, 0.0) + elapsed_seconds
        endpoint_times = self.diagnostics.llm_request_seconds_by_endpoint
        endpoint_times[model.endpoint_id] = endpoint_times.get(model.endpoint_id, 0.0) + elapsed_seconds
        if timed_out:
            self.diagnostics.timeout_seconds_total += elapsed_seconds

    def record_usage(self, model: ModelConfig, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        input_tokens = max(0, int(input_tokens))
        output_tokens = max(0, int(output_tokens))
        total_tokens = max(0, int(total_tokens))
        self.diagnostics.input_tokens_used += input_tokens
        self.diagnostics.output_tokens_used += output_tokens
        self.diagnostics.total_tokens_used += total_tokens
        self.diagnostics.daily_budget_tokens_used += total_tokens
        usage = self.diagnostics.endpoint_usage.setdefault(
            model.endpoint_id,
            {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        usage["requests"] += 1
        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        usage["total_tokens"] += total_tokens

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
            if rejection.get("reason") == "daily_budget_exhausted":
                self.diagnostics.daily_budget_skip_count += 1
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
        if model_override is not None:
            candidate_chain = [preferred_model]
        else:
            candidate_chain = [
                candidate
                for candidate in runtime.model_chain
                if runtime.governor.endpoint_cooldown_seconds(candidate.endpoint_id) <= 0
            ]
            if not candidate_chain:
                raise RuntimeError("No healthy LLM endpoint is available after timeout cooldowns.")
        rate_limit_waits = {
            candidate.endpoint_id: runtime.governor.peek_key(
                candidate.model_id,
                runtime.key_index_for_model(candidate),
                runtime.key_count_for_model(candidate),
                estimated_input_tokens,
                quota_scope=candidate.quota_scope,
            )[1]
            for candidate in candidate_chain
        }
        resolver = runtime.resolver or ModelResolver()
        requested_output_tokens = min(preferred_model.max_completion_tokens, DEFAULT_OUTPUT_TOKENS)
        budget_remaining: dict[str, float] = {}
        budget_required: dict[str, float] = {}
        for candidate in candidate_chain:
            capability = resolver.capability_for(candidate)
            daily_neurons = capability.limits.get("daily_neurons")
            if daily_neurons and candidate.quota_scope:
                budget_remaining[candidate.endpoint_id] = runtime.budget.remaining_compute_units(
                    candidate.quota_scope, daily_neurons
                )
                budget_required[candidate.endpoint_id] = _compute_units_for(
                    capability, estimated_input_tokens, requested_output_tokens
                )
            else:
                budget_remaining[candidate.endpoint_id] = runtime.budget.remaining_tokens(
                    candidate.model_id,
                    runtime.key_index_for_model(candidate),
                    capability.limits.get("tpd"),
                    quota_scope=candidate.quota_scope,
                )

        # The resolver's budget check is advisory; this atomic reservation is
        # what prevents parallel calls from collectively overbooking a shared
        # account-wide compute allowance.
        reservation_units = 0
        accumulated_rejections: list[dict[str, str]] = []
        selectable_chain = list(candidate_chain)
        while selectable_chain:
            selection = resolver.resolve(
                task,
                selectable_chain,
                estimated_input_tokens=estimated_input_tokens,
                requested_output_tokens=requested_output_tokens,
                rate_limit_waits=rate_limit_waits,
                preferred_model_id=preferred_model.endpoint_id,
                budget_remaining=budget_remaining,
                budget_required=budget_required,
            )
            model = selection.model
            accumulated_rejections.extend(selection.rejections)
            capability = resolver.capability_for(model)
            daily_neurons = capability.limits.get("daily_neurons")
            reservation_units = int(budget_required.get(model.endpoint_id, 0))
            if not daily_neurons or not model.quota_scope or runtime.budget.try_reserve_compute_units(
                model.quota_scope, reservation_units, daily_neurons
            ):
                break
            accumulated_rejections.append(
                {"model_id": model.model_id, "reason": "daily_budget_exhausted"}
            )
            selectable_chain = [item for item in selectable_chain if item.endpoint_id != model.endpoint_id]
        else:
            raise NoEligibleEndpoint(f"No {task} endpoint has sufficient daily compute budget.")

        selection = ModelSelection(
            model=model,
            rejections=accumulated_rejections,
            avoided_wait_seconds=selection.avoided_wait_seconds,
            wait_seconds=selection.wait_seconds,
            wait_exceeded=selection.wait_exceeded,
        )
        runtime.record_resolver_selection(
            task=task,
            preferred_model=preferred_model.endpoint_id,
            selected_model=model.endpoint_id,
            rejections=selection.rejections,
            avoided_wait_seconds=selection.avoided_wait_seconds,
        )
        if selection.wait_exceeded:
            runtime.record_failure_classification("no_eligible_endpoint")
            runtime.record_rate_limit_event(
                category_name=context.category_name,
                task=task,
                model=model,
                wait_seconds=selection.wait_seconds,
                action="skipped_wait_budget",
            )
            raise NoEligibleEndpoint(
                f"No {task} endpoint is available within the "
                f"{runtime.resolver.wait_budget_seconds(task):.1f}s wait budget."
            )
        if context.batch_kind.startswith("synthesis"):
            runtime.ensure_synthesis_budget(context.category_name)
        api_url = model.api_url or _default_provider_url(model.provider)
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
        selected_wait = rate_limit_waits.get(model.endpoint_id, 0.0)
        runtime.record_batch_attempt(attempt_context, model.model_id)
        runtime.record_endpoint_attempt(attempt_context, model)
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
        best_key, wait_seconds = runtime.governor.select_and_reserve(
            model.model_id,
            runtime.key_index_for_model(model),
            runtime.key_count_for_model(model),
            estimated_input_tokens,
            quota_scope=model.quota_scope,
            max_wait_seconds=runtime.resolver.wait_budget_seconds(task),
        )
        if best_key != runtime.key_index_for_model(model) and model.provider == DEFAULT_PROVIDER and not model.quota_scope:
            LOGGER.info(
                "Governor rotating key %d → %d for model %s (more capacity available).",
                runtime.current_key_index,
                best_key,
                model.model_id,
            )
            runtime.current_key_index = best_key
            runtime.diagnostics.key_rotation_count += 1
            session = runtime.get_session_for_model(model)
        runtime.record_wait(
            context.category_name,
            wait_seconds,
            batch_kind=context.batch_kind,
            model=model,
            task=task,
            reason="quota_reservation",
        )
        if wait_seconds > 0:
            LOGGER.info(
                "Waiting %.1f seconds before %s for category %s batch=%s on %s/%s (key=%d).",
                wait_seconds,
                attempt_context.batch_kind,
                attempt_context.category_name,
                attempt_context.batch_label,
                model.provider,
                model.model_id,
                runtime.current_key_index,
            )
        connect_timeout, read_timeout, total_deadline = _llm_request_timeouts()
        request_started = time.monotonic()
        try:
            response = _post_with_deadline(
                session,
                api_url,
                json_body=_chat_request_body(model, messages),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                total_deadline=total_deadline,
            )
        except requests.exceptions.RequestException as exc:
            # Network failure or hard-deadline abort: never sit on a stuck
            # socket. Treat like a transient server error and let key rotation /
            # model fallback recover instead of failing the whole category.
            elapsed_seconds = time.monotonic() - request_started
            timed_out = isinstance(exc, requests.exceptions.Timeout)
            runtime.record_request_timing(
                attempt_context,
                model,
                elapsed_seconds,
                timed_out=timed_out,
            )
            if timed_out:
                runtime.diagnostics.request_timeout_count += 1
                cooldown_seconds = _resolve_timeout_cooldown_seconds()
                runtime.governor.mark_endpoint_cooldown(model.endpoint_id, cooldown_seconds)
                runtime.diagnostics.endpoint_cooldown_count += 1
                runtime.diagnostics.endpoint_cooldown_seconds_total += cooldown_seconds
            LOGGER.warning(
                "LLM request error for category %s batch=%s on %s/%s (endpoint=%s, key=%d): %s",
                attempt_context.category_name,
                attempt_context.batch_label,
                model.provider,
                model.model_id,
                model.endpoint_id,
                runtime.current_key_index,
                exc,
            )
            runtime.reset_session_for_model(model)
            runtime.record_retry(attempt_context.category_name, batch_kind=attempt_context.batch_kind)
            if attempt == DEFAULT_CHAT_RETRIES - 1:
                raise
            if model.provider == DEFAULT_PROVIDER and not model.quota_scope and runtime.rotate_key(
                f"request error on key {runtime.current_key_index} / model {model.model_id}"
            ):
                session = runtime.get_session_for_model(model)
                continue
            if model_override is None and runtime.switch_to_next_model(
                f"Request error on {model.model_id}: {exc}",
                failed_model=model,
            ):
                runtime.current_key_index = 0
                session = runtime.get_session_for_model(runtime.current_model)
                runtime.record_model_switch(context.category_name, runtime.model_switches[-1])
                continue
            if model_override is None and any(
                candidate.endpoint_id != model.endpoint_id
                and runtime.governor.endpoint_cooldown_seconds(candidate.endpoint_id) <= 0
                for candidate in runtime.model_chain
            ):
                continue
            delay_seconds = min(5.0 * (attempt + 1), 20.0)
            runtime.record_wait(attempt_context.category_name, delay_seconds, batch_kind=attempt_context.batch_kind)
            time.sleep(delay_seconds)
            session = runtime.get_session_for_model(model)
            continue
        runtime.record_request_timing(
            attempt_context,
            model,
            time.monotonic() - request_started,
        )
        runtime.governor.clear_endpoint_cooldown(model.endpoint_id)
        runtime.governor.record_response(
            model.model_id,
            response,
            key_index=runtime.key_index_for_model(model),
            quota_scope=model.quota_scope,
        )
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
                "LLM provider %s returned HTTP 413 for category %s batch=%s on %s.",
                model.provider,
                attempt_context.category_name,
                attempt_context.batch_label,
                model.model_id,
            )
            response.raise_for_status()

        if response.status_code == 429 and model_override is None:
            LOGGER.info(
                "LLM provider %s returned HTTP 429 for category %s batch=%s on %s (endpoint=%s, key=%d).",
                model.provider,
                attempt_context.category_name,
                attempt_context.batch_label,
                model.model_id,
                model.endpoint_id,
                runtime.current_key_index,
            )
            runtime.tighten_category_budget(
                attempt_context.category_name,
                "rate_limited",
                batch_kind="synthesis" if attempt_context.batch_kind.startswith("synthesis") else "article_batch",
            )
            retry_wait = runtime.governor.peek_key(
                model.model_id,
                runtime.key_index_for_model(model),
                runtime.key_count_for_model(model),
                estimated_input_tokens,
                quota_scope=model.quota_scope,
            )[1]
            task_wait_budget = runtime.resolver.wait_budget_seconds(task)
            if model.quota_scope and retry_wait <= task_wait_budget:
                runtime.record_rate_limit_event(
                    category_name=attempt_context.category_name,
                    task=task,
                    model=model,
                    wait_seconds=retry_wait,
                    action="wait_for_reset",
                )
                LOGGER.info(
                    "Keeping %s/%s for %.1f seconds after HTTP 429; within %s wait budget.",
                    model.provider,
                    model.model_id,
                    retry_wait,
                    task,
                )
                continue
            # Try rotating to another key before switching models.
            if not model.quota_scope and runtime.rotate_key(f"429 on key {runtime.current_key_index} / model {model.model_id}"):
                session = runtime.get_session_for_model(model)
                continue
            # All keys exhausted for this model — switch model and reset key index.
            if runtime.switch_to_next_model(
                f"All keys returned 429 on {model.model_id}.",
                failed_model=model,
            ):
                runtime.current_key_index = 0
                session = runtime.get_session_for_model(runtime.current_model)
                runtime.record_model_switch(context.category_name, runtime.model_switches[-1])
                runtime.record_rate_limit_event(
                    category_name=attempt_context.category_name,
                    task=task,
                    model=model,
                    wait_seconds=retry_wait,
                    action="switch_endpoint",
                )
                continue

        if response.status_code in {500, 502, 503, 504} and model_override is None and attempt < DEFAULT_CHAT_RETRIES - 1:
            cooldown_seconds = _resolve_timeout_cooldown_seconds()
            runtime.governor.mark_endpoint_cooldown(model.endpoint_id, cooldown_seconds)
            runtime.diagnostics.endpoint_cooldown_count += 1
            runtime.diagnostics.endpoint_cooldown_seconds_total += cooldown_seconds
            runtime.record_retry(attempt_context.category_name, batch_kind=attempt_context.batch_kind)
            if runtime.switch_to_next_model(
                f"Provider HTTP {response.status_code} on {model.model_id}; cooling down endpoint.",
                failed_model=model,
            ):
                runtime.current_key_index = 0
                session = runtime.get_session_for_model(runtime.current_model)
                runtime.record_model_switch(context.category_name, runtime.model_switches[-1])
                continue

        if response.status_code in {429, 500, 502, 503, 504}:
            if response.status_code == 429:
                LOGGER.info(
                    "LLM provider %s returned HTTP 429 for category %s batch=%s on %s.",
                    model.provider,
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
            runtime.record_wait(
                attempt_context.category_name,
                delay_seconds,
                batch_kind=attempt_context.batch_kind,
                model=model,
                task=task,
                reason=f"http_{response.status_code}_backoff",
            )
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
                    "Model %s/%s rejected request (HTTP %s) for category %s batch=%s; evicting and re-resolving.",
                    model.provider,
                    model.model_id,
                    response.status_code,
                    attempt_context.category_name,
                    attempt_context.batch_label,
                )
                if runtime.evict_model(model.endpoint_id):
                    runtime.diagnostics.model_decommissioned_count += 1
                    runtime.current_key_index = 0
                    session = runtime.get_session_for_model(runtime.current_model)
                    continue

        response.raise_for_status()
        payload = response.json()
        # Record actual daily token/request usage so budget survives across runs.
        usage = payload.get("usage") if isinstance(payload, dict) else None
        input_tokens, output_tokens, used_tokens = _usage_token_counts(
            usage if isinstance(usage, dict) else None,
            estimated_input_tokens=estimated_input_tokens,
            fallback_output_tokens=model.max_completion_tokens,
        )
        runtime.budget.record(
            model.model_id,
            runtime.key_index_for_model(model),
            used_tokens,
            quota_scope=model.quota_scope,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if reservation_units and model.quota_scope:
            actual_compute_units = _compute_units_for(
                resolver.capability_for(model), input_tokens, output_tokens
            )
            runtime.budget.settle_compute_units(model.quota_scope, reservation_units, actual_compute_units)
            if model.provider == "cloudflare":
                runtime.diagnostics.cloudflare_neurons_used += actual_compute_units
        runtime.record_usage(model, input_tokens, output_tokens, used_tokens)
        content = _response_message_content(payload["choices"][0]["message"])
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
