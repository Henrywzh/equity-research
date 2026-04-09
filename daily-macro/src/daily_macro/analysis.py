from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict
try:
    from typing import NotRequired
except ImportError:
    try:
        from typing_extensions import NotRequired
    except ImportError:
        # Fallback for Python < 3.11 without typing_extensions
        from typing import Optional as NotRequired

import requests
from langgraph.graph import END, START, StateGraph

from .config import DEFAULT_SOURCE_SITE, get_analysis_dir, get_data_dir, get_db_path, get_project_root
from .storage import Storage

LOGGER = logging.getLogger(__name__)

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
REPORT_FILE_NAME = "hkej-news-analysis.json"
REPORT_SCHEMA_VERSION = 6
DEFAULT_PROVIDER = "groq"
PRIMARY_MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_MODEL_IDS = ["qwen/qwen3-32b", "llama-3.1-8b-instant"]
DELAYED_RETRY_FINAL_MODEL_ID = "openai/gpt-oss-20b"
DELAYED_RETRY_WAIT_SECONDS = 60.0
MAX_CATEGORY_SYNTHESIS_WAIT_SECONDS = 1800.0
MAX_CATEGORY_SYNTHESIS_RETRIES = 24
MAX_SYNTHESIS_MERGE_DEPTH = 3
DEFAULT_OUTPUT_TOKENS = 1200
DEFAULT_INPUT_BUDGET_TOKENS = 4500
DEFAULT_SYNTHESIS_INPUT_BUDGET_TOKENS = 2400
DEFAULT_PROMPT_OVERHEAD_TOKENS = 900
SHORT_ARTICLE_FULL_TEXT_THRESHOLD = 1500
DEFAULT_CHAT_RETRIES = 4
RATE_LIMIT_REQUEST_FLOOR = 0
RATE_LIMIT_TOKEN_FLOOR = 800
DEFAULT_REQUEST_BYTE_BUDGET = 12000
DEFAULT_SYNTHESIS_REQUEST_BYTE_BUDGET = 8000
MIN_REQUEST_BYTE_BUDGET = 4000
MIN_SYNTHESIS_REQUEST_BYTE_BUDGET = 3000
MIN_INPUT_BUDGET_TOKENS = 1200
MIN_SYNTHESIS_INPUT_BUDGET_TOKENS = 800
CATEGORY_SHRINK_STEP_CHARS = 400
CATEGORY_MIN_CONTENT_CHARS = 0
CATEGORY_ORDER = [
    "國際財經",
    "時事脈搏",
    "香港財經",
    "港股直擊",
    "中國財經",
    "即巿股評",
    "重要通告",
    "港交所通告",
    "地產新聞",
]
ENTITY_TYPES = {"person", "company", "country", "institution", "index", "organization", "asset", "other"}
FAILURE_CLASSIFICATIONS = {
    "payload_too_large",
    "rate_limited",
    "invalid_json",
    "incomplete_model_output",
    "http_error",
    "unexpected_error",
    "synthesis_budget_exhausted",
}
LIGHT_ANALYSIS_SECTIONS = {"時事脈搏", "地產新聞"}
ATTENTION_TIERS = ("high", "medium", "light")
ATTENTION_TIER_RANK = {"high": 0, "medium": 1, "light": 2}
ROUTER_LLM_MIN_ARTICLES = 1
HIGH_ATTENTION_THEME_KEYWORDS = {
    "stocks": ("業績", "盈喜", "盈警", "回購", "配股", "供股", "新股", "上市", "股份", "股價", "股東", "盈利", "profit", "earnings", "guidance", "buyback", "placement", "ipo", "shares"),
    "macro": ("聯儲", "聯儲局", "人行", "央行", "利率", "通脹", "經濟", "衰退", "增長", "國債", "收益率", "匯率", "美元", "人民幣", "油價", "gdp", "inflation", "rates", "yield", "fx", "oil", "economy"),
    "geopolitics": ("戰爭", "制裁", "關稅", "貿易戰", "軍事", "衝突", "伊朗", "俄羅斯", "烏克蘭", "中東", "tariff", "sanction", "war", "trade", "conflict", "geopolit"),
    "property": ("樓市", "地產", "樓價", "按揭", "租金", "土地", "property", "housing", "mortgage", "real estate"),
}


@dataclass(frozen=True, slots=True)
class SectionProfile:
    name: str
    article_key_points_limit: int
    article_key_points_instruction: str
    entity_limit: int
    category_bullet_limit: int
    subgroup_bullet_limit: int
    article_input_budget_tokens: int
    article_request_byte_budget: int
    synthesis_input_budget_tokens: int
    synthesis_request_byte_budget: int
    subgroup_threshold: int
    subgroup_target_size: int
    salvage_max_depth: int


STANDARD_SECTION_PROFILE = SectionProfile(
    name="standard",
    article_key_points_limit=4,
    article_key_points_instruction="2 to 4 strings — each must include at least one specific detail such as a figure, percentage, named actor, date, cause, or consequence; do not restate the article title",
    entity_limit=5,
    category_bullet_limit=5,
    subgroup_bullet_limit=4,
    article_input_budget_tokens=DEFAULT_INPUT_BUDGET_TOKENS,
    article_request_byte_budget=DEFAULT_REQUEST_BYTE_BUDGET,
    synthesis_input_budget_tokens=DEFAULT_SYNTHESIS_INPUT_BUDGET_TOKENS,
    synthesis_request_byte_budget=DEFAULT_SYNTHESIS_REQUEST_BYTE_BUDGET,
    subgroup_threshold=5,
    subgroup_target_size=6,
    salvage_max_depth=3,
)

LIGHT_SECTION_PROFILE = SectionProfile(
    name="light",
    article_key_points_limit=2,
    article_key_points_instruction="1 to 2 strings — each must include a specific detail such as a figure, named actor, cause, or consequence; do not restate the article title",
    entity_limit=4,
    category_bullet_limit=3,
    subgroup_bullet_limit=3,
    article_input_budget_tokens=3000,
    article_request_byte_budget=9000,
    synthesis_input_budget_tokens=1600,
    synthesis_request_byte_budget=5600,
    subgroup_threshold=7,
    subgroup_target_size=8,
    salvage_max_depth=2,
)


class AnalysisGraphState(TypedDict):
    target_date: str
    source_site: str
    report_path: Path
    previous_report_path: Path
    articles: list[dict[str, Any]]
    previous_articles: list[dict[str, Any]]
    existing_report: dict[str, Any] | None
    previous_report: dict[str, Any] | None
    today_plan: dict[str, Any]
    previous_retry_plan: dict[str, Any]
    runtime: AnalysisRuntime | None
    category_reports: list[dict[str, Any]]
    previous_day_retry_successes: int
    updated_previous_report: NotRequired[dict[str, Any] | None]
    incremental: dict[str, int]
    market_context_string: str
    top_alerts: list[str]
    legacy_executive_summary: list[str]
    report: dict[str, Any]
    total_scraped_articles: int
    newly_analyzed_keys: set[str]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    provider: str = DEFAULT_PROVIDER
    max_completion_tokens: int = DEFAULT_OUTPUT_TOKENS
    api_url: str | None = None
    api_key_env: str | None = None


@dataclass(slots=True)
class ModelRateLimitState:
    remaining_requests: int | None = None
    reset_requests_at: float | None = None
    remaining_tokens: int | None = None
    reset_tokens_at: float | None = None


@dataclass(slots=True)
class CategoryBudgetState:
    article_input_budget_tokens: int = DEFAULT_INPUT_BUDGET_TOKENS
    article_request_byte_budget: int = DEFAULT_REQUEST_BYTE_BUDGET
    synthesis_input_budget_tokens: int = DEFAULT_SYNTHESIS_INPUT_BUDGET_TOKENS
    synthesis_request_byte_budget: int = DEFAULT_SYNTHESIS_REQUEST_BYTE_BUDGET


@dataclass(slots=True)
class RuntimeDiagnostics:
    rate_limit_wait_count: int = 0
    rate_limit_wait_seconds_total: float = 0.0
    fallback_switch_count: int = 0
    pre_send_split_count: int = 0
    response_413_split_count: int = 0
    json_repair_retry_count: int = 0
    batch_count: int = 0
    failed_batch_count: int = 0
    delayed_retry_candidate_count: int = 0
    delayed_retry_attempted_count: int = 0
    delayed_retry_recovered_count: int = 0
    delayed_retry_failed_count: int = 0
    high_medium_unresolved_count: int = 0
    light_unresolved_count: int = 0
    delayed_retry_skipped_final_model_count: int = 0
    synthesis_budget_exhausted_count: int = 0
    degraded_merge_count: int = 0
    key_rotation_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rate_limit_wait_count": self.rate_limit_wait_count,
            "rate_limit_wait_seconds_total": round(self.rate_limit_wait_seconds_total, 3),
            "fallback_switch_count": self.fallback_switch_count,
            "key_rotation_count": self.key_rotation_count,
            "pre_send_split_count": self.pre_send_split_count,
            "response_413_split_count": self.response_413_split_count,
            "json_repair_retry_count": self.json_repair_retry_count,
            "batch_count": self.batch_count,
            "failed_batch_count": self.failed_batch_count,
            "delayed_retry_candidate_count": self.delayed_retry_candidate_count,
            "delayed_retry_attempted_count": self.delayed_retry_attempted_count,
            "delayed_retry_recovered_count": self.delayed_retry_recovered_count,
            "delayed_retry_failed_count": self.delayed_retry_failed_count,
            "high_medium_unresolved_count": self.high_medium_unresolved_count,
            "light_unresolved_count": self.light_unresolved_count,
            "delayed_retry_skipped_final_model_count": self.delayed_retry_skipped_final_model_count,
            "synthesis_budget_exhausted_count": self.synthesis_budget_exhausted_count,
            "degraded_merge_count": self.degraded_merge_count,
        }


@dataclass(slots=True)
class CategoryDiagnostics:
    split_reasons: list[str] = field(default_factory=list)
    models_attempted: list[str] = field(default_factory=list)
    estimated_input_tokens_max: int = 0
    serialized_request_bytes_max: int = 0
    rate_limit_waits: int = 0
    partial_article_count: int = 0
    sub_batch_count: int = 0
    synthesis_wait_seconds_total: float = 0.0
    synthesis_retry_count: int = 0
    synthesis_retry_skipped_count: int = 0
    synthesis_budget_exhausted: bool = False
    degraded_merge_used: bool = False
    degraded_merge_reason: str = ""
    synthesis_merge_depth_max: int = 0
    model_switches: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sub_batch_count": self.sub_batch_count,
            "split_reasons": list(self.split_reasons),
            "models_attempted": list(self.models_attempted),
            "estimated_input_tokens_max": self.estimated_input_tokens_max,
            "serialized_request_bytes_max": self.serialized_request_bytes_max,
            "rate_limit_waits": self.rate_limit_waits,
            "partial_article_count": self.partial_article_count,
            "synthesis_wait_seconds_total": round(self.synthesis_wait_seconds_total, 3),
            "synthesis_retry_count": self.synthesis_retry_count,
            "synthesis_retry_skipped_count": self.synthesis_retry_skipped_count,
            "synthesis_budget_exhausted": self.synthesis_budget_exhausted,
            "degraded_merge_used": self.degraded_merge_used,
            "degraded_merge_reason": self.degraded_merge_reason,
            "synthesis_merge_depth_max": self.synthesis_merge_depth_max,
            "model_switches": list(self.model_switches),
        }


class SynthesisBudgetExceeded(RuntimeError):
    """Raised when a category has exhausted its allowed synthesis retry/wait budget."""


@dataclass(frozen=True, slots=True)
class BatchContext:
    category_name: str
    batch_kind: str
    batch_label: str
    article_count: int
    estimated_input_tokens: int
    serialized_request_bytes: int
    content_shrunk: bool = False


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


def run_analysis(
    *,
    date_string: str | None = None,
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    target_date = date_string or datetime.now().astimezone().date().isoformat()
    resolved_data_dir = get_data_dir(data_dir)
    resolved_db_path = get_db_path(db_path, resolved_data_dir)
    analysis_dir = get_analysis_dir(resolved_data_dir)
    report_path = analysis_dir / target_date / REPORT_FILE_NAME
    existing_report = _load_existing_report(report_path)

    if existing_report is not None and not force:
        cached = dict(existing_report)
        cached["output_path"] = str(report_path)
        cached["cached"] = True
        return cached

    LOGGER.info("Starting daily analysis for %s.", target_date)

    storage = Storage(resolved_db_path)
    try:
        articles = storage.fetch_published_articles_for_date(target_date, source_site=DEFAULT_SOURCE_SITE)
        total_scraped_articles = len(storage.fetch_articles_by_date(target_date))
        previous_date = (datetime.fromisoformat(target_date).date() - timedelta(days=1)).isoformat()
        previous_report_path = analysis_dir / previous_date / REPORT_FILE_NAME
        previous_report = _load_existing_report(previous_report_path)
        previous_articles = (
            storage.fetch_published_articles_for_date(previous_date, source_site=DEFAULT_SOURCE_SITE)
            if previous_report is not None
            else []
        )
    finally:
        storage.close()
    graph = _build_analysis_graph()
    state: AnalysisGraphState = {
        "target_date": target_date,
        "source_site": DEFAULT_SOURCE_SITE,
        "report_path": report_path,
        "previous_report_path": previous_report_path,
        "articles": articles,
        "previous_articles": previous_articles,
        "existing_report": existing_report,
        "previous_report": previous_report,
        "today_plan": {},
        "previous_retry_plan": {},
        "runtime": None,
        "category_reports": [],
        "previous_day_retry_successes": 0,
        "market_context_string": "",
        "market_snapshots": [],
        "top_alerts": [],
        "report": {},
        "total_scraped_articles": total_scraped_articles,
    }
    final_state = graph.invoke(state)
    report = final_state["report"]
    report["output_path"] = str(report_path)
    report["cached"] = False
    LOGGER.info(
        "Finished daily analysis for %s with status %s. Categories=%s, articles=%s.",
        target_date,
        report["status"],
        len(report.get("categories") or []),
        int((report.get("totals") or {}).get("article_count") or 0),
    )
    return report


def _build_analysis_graph():
    graph = StateGraph(AnalysisGraphState)
    graph.add_node("initialize", _graph_initialize)
    graph.add_node("fetch_market_data", _graph_fetch_market_data)
    graph.add_node("route_attention", _graph_route_attention)
    graph.add_node("analyze_today", _graph_analyze_today)
    graph.add_node("retry_previous_day", _graph_retry_previous_day)
    graph.add_node("summarize_top_alerts", _graph_summarize_top_alerts)
    graph.add_node("finalize", _graph_finalize)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "fetch_market_data")
    graph.add_edge("fetch_market_data", "route_attention")
    graph.add_edge("route_attention", "analyze_today")
    graph.add_edge("analyze_today", "retry_previous_day")
    graph.add_edge("retry_previous_day", "summarize_top_alerts")
    graph.add_edge("summarize_top_alerts", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def _graph_initialize(state: AnalysisGraphState) -> AnalysisGraphState:
    today_plan = _build_today_incremental_plan(state["articles"], state["existing_report"])
    previous_retry_plan = _build_previous_retry_plan(state["previous_articles"], state["previous_report"])
    runtime: AnalysisRuntime | None = None
    if today_plan["new_articles_analyzed"] > 0 or previous_retry_plan["retried_previous_day_articles"] > 0:
        groq_keys = load_groq_api_keys()
        LOGGER.info("Loaded %d Groq API key(s).", len(groq_keys))
        runtime = AnalysisRuntime(
            groq_api_keys=groq_keys,
            governor=RateLimitGovernor(),
            model_chain=[ModelConfig(PRIMARY_MODEL_ID), *(ModelConfig(model_id) for model_id in FALLBACK_MODEL_IDS)],
            delayed_retry_final_model=ModelConfig(
                DELAYED_RETRY_FINAL_MODEL_ID,
                provider="openai",
                api_url=OPENAI_CHAT_COMPLETIONS_URL,
                api_key_env="OPENAI_API_KEY",
            ),
        )
    state["today_plan"] = today_plan
    state["previous_retry_plan"] = previous_retry_plan
    state["runtime"] = runtime
    state["legacy_executive_summary"] = (state["existing_report"] or {}).get("executive_summary") or []
    state["newly_analyzed_keys"] = set()
    state["incremental"] = {
        "reused_successful_articles": today_plan["reused_successful_articles"],
        "new_articles_analyzed": today_plan["new_articles_analyzed"],
        "retried_previous_day_articles": previous_retry_plan["retried_previous_day_articles"],
        "previous_day_retry_successes": 0,
    }
    return state


def _graph_fetch_market_data(state: AnalysisGraphState) -> AnalysisGraphState:
    from .market import build_market_context_string, fetch_market_snapshot_for_date

    snapshots = fetch_market_snapshot_for_date(state["target_date"])
    state["market_context_string"] = build_market_context_string(snapshots)
    state["market_snapshots"] = snapshots
    if state["runtime"] is not None:
        state["runtime"].market_context_string = state["market_context_string"]
    LOGGER.info(
        "Loaded market snapshot with %d ticker(s) for %s.",
        len(snapshots),
        state["target_date"],
    )
    return state


def _graph_route_attention(state: AnalysisGraphState) -> AnalysisGraphState:
    runtime = state["runtime"]
    today_plan = dict(state["today_plan"])
    previous_retry_plan = dict(state["previous_retry_plan"])
    if today_plan.get("work_articles"):
        today_plan["work_articles"] = _route_articles(runtime, list(today_plan["work_articles"]))
    if previous_retry_plan.get("work_articles"):
        previous_retry_plan["work_articles"] = _route_articles(runtime, list(previous_retry_plan["work_articles"]))
    state["today_plan"] = today_plan
    state["previous_retry_plan"] = previous_retry_plan
    return state


def _graph_analyze_today(state: AnalysisGraphState) -> AnalysisGraphState:
    if not state["articles"]:
        state["category_reports"] = []
        return state
    state["category_reports"] = _build_incremental_report_categories(
        runtime=state["runtime"],
        db_articles=state["articles"],
        existing_report=state["existing_report"],
        plan=state["today_plan"],
        retry_only=False,
    )
    # Record which articles were actually newly analyzed in this run
    state["newly_analyzed_keys"].update(
        str(a.get("source_article_id") or a.get("canonical_url") or "")
        for a in state["today_plan"]["work_articles"]
    )
    return state


def _graph_retry_previous_day(state: AnalysisGraphState) -> AnalysisGraphState:
    runtime = state["runtime"]
    try:
        if state["previous_report"] is not None and state["previous_retry_plan"]["retried_previous_day_articles"] > 0:
            updated_previous_report, previous_day_retry_successes = _retry_previous_report(
                runtime=runtime,
                previous_date=(datetime.fromisoformat(state["target_date"]).date() - timedelta(days=1)).isoformat(),
                db_articles=state["previous_articles"],
                existing_report=state["previous_report"],
                retry_plan=state["previous_retry_plan"],
            )
            state["previous_day_retry_successes"] = previous_day_retry_successes
            state["incremental"]["previous_day_retry_successes"] = previous_day_retry_successes
            if previous_day_retry_successes > 0:
                _write_report(state["previous_report_path"], updated_previous_report)
                state["updated_previous_report"] = updated_previous_report
    finally:
        if runtime is not None:
            runtime.close_sessions()
    return state


def _graph_summarize_top_alerts(state: AnalysisGraphState) -> AnalysisGraphState:
    runtime = state["runtime"]
    if runtime is None or not state["category_reports"]:
        return state

    # Collect all key developments and their metadata mapping
    all_developments: list[dict[str, Any]] = []
    # Map article info by a temporary ID for the LLM to reference
    article_metadata: dict[str, dict[str, str]] = {}
    
    for category in state["category_reports"]:
        cat_name = category.get("category") or "Unknown"
        for subgroup in category.get("subgroups") or []:
            subgroup_title = subgroup.get("title") or "Overview"
            
            # Index articles in this subgroup for the LLM
            subgroup_articles = subgroup.get("articles") or []
            ref_ids = []
            for art in subgroup_articles:
                # Use source_article_id as the stabilizer
                aid = str(art.get("source_article_id") or art.get("canonical_url") or "")
                title = str(art.get("title") or "Untitled")
                pub_at = str(art.get("published_at") or "")
                # Default to report date if published_at is missing date part
                date_str = pub_at[:10] if len(pub_at) >= 10 else state["target_date"]
                url = str(art.get("canonical_url") or "").strip()
                
                if aid:
                    article_metadata[aid] = {
                        "title": title,
                        "date": date_str,
                        "url": url,
                    }
                    ref_ids.append(aid)
            
            for dev in subgroup.get("key_developments") or []:
                dev_text = str(dev)
                # If we are in an incremental run, we should only focus on developments 
                # that were derived from the newly analyzed articles.
                if state.get("legacy_executive_summary") and state.get("newly_analyzed_keys"):
                    subgroup_keys = {
                        _article_key(a.get("source_article_id"), a.get("canonical_url"))
                        for a in subgroup_articles
                    }
                    if not (subgroup_keys & state["newly_analyzed_keys"]):
                        continue

                all_developments.append(
                    {
                        "category": cat_name,
                        "subgroup": subgroup_title,
                        "text": dev_text,
                        "ref_ids": ref_ids,
                    }
                )
    
    if not all_developments:
        return state

    try:
        top_alerts = _generate_top_alerts(
            runtime=runtime,
            market_context=state["market_context_string"],
            developments=all_developments,
            article_metadata=article_metadata,
            is_incremental=bool(state.get("legacy_executive_summary")),
        )
        state["executive_summary"] = top_alerts
        # Legacy key support
        state["top_alerts"] = top_alerts
    except Exception as e:
        LOGGER.warning("Graph summarize_top_alerts failed: %s", e)
        state["executive_summary"] = []
        state["top_alerts"] = []

    if state.get("executive_summary"):
        LOGGER.info("Successfully generated %d top-level alerts.", len(state["executive_summary"]))
    return state


def _graph_finalize(state: AnalysisGraphState) -> AnalysisGraphState:
    if not state["articles"]:
        report = _build_empty_report(state["target_date"], incremental=state["incremental"])
        _write_report(state["report_path"], report)
        LOGGER.info("No published articles found for %s.", state["target_date"])
        state["report"] = report
        return state

    report = _finalize_report(
        target_date=state["target_date"],
        source_site=state["source_site"],
        input_article_count=len(state["articles"]),
        category_reports=state["category_reports"],
        top_alerts=state.get("top_alerts") or [],
        runtime=state["runtime"],
        incremental=state["incremental"],
        total_scraped_count=state.get("total_scraped_articles") or 0,
        market_snapshots=state.get("market_snapshots"),
        legacy_executive_summary=state.get("legacy_executive_summary"),
        newly_analyzed_keys=state.get("newly_analyzed_keys"),
    )
    _write_report(state["report_path"], report)
    state["report"] = report
    return state


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


def select_content_for_analysis(content_text: str) -> dict[str, Any]:
    normalized = _normalize_whitespace(content_text)
    original_length = len(normalized)
    original_token_estimate = _estimate_tokens(normalized)

    if original_length <= SHORT_ARTICLE_FULL_TEXT_THRESHOLD:
        return {
            "content_text": normalized,
            "content_truncated": False,
            "analysis_method": "full_text",
            "original_content_length_chars": original_length,
            "analyzed_content_length_chars": original_length,
            "original_content_token_estimate": original_token_estimate,
            "analyzed_content_token_estimate": original_token_estimate,
            "truncation_reason": None,
        }

    projected_input_tokens = original_token_estimate + DEFAULT_PROMPT_OVERHEAD_TOKENS
    if projected_input_tokens <= DEFAULT_INPUT_BUDGET_TOKENS:
        return {
            "content_text": normalized,
            "content_truncated": False,
            "analysis_method": "full_text",
            "original_content_length_chars": original_length,
            "analyzed_content_length_chars": original_length,
            "original_content_token_estimate": original_token_estimate,
            "analyzed_content_token_estimate": original_token_estimate,
            "truncation_reason": None,
        }

    return _build_truncated_selection(
        normalized,
        original_length,
        original_token_estimate,
        max(DEFAULT_INPUT_BUDGET_TOKENS - DEFAULT_PROMPT_OVERHEAD_TOKENS, 500) * 4,
        (
            "Full article exceeded the working request budget after prompt overhead; "
            "the leading content slice was analyzed instead."
        ),
    )


def _prepare_single_article(article: dict[str, Any]) -> dict[str, Any]:
    selected = select_content_for_analysis(article.get("content_text") or "")
    attention = _article_attention_defaults(article)
    return {
        "source_article_id": article.get("source_article_id"),
        "title": article.get("title"),
        "canonical_url": article.get("canonical_url"),
        "published_at": article.get("published_at"),
        "section": article.get("article_section"),
        "summary_snippet": article.get("summary_snippet"),
        "content_text": selected["content_text"],
        "content_truncated": selected["content_truncated"],
        "analysis_method": selected["analysis_method"],
        "original_content_length_chars": selected["original_content_length_chars"],
        "analyzed_content_length_chars": selected["analyzed_content_length_chars"],
        "original_content_token_estimate": selected["original_content_token_estimate"],
        "analyzed_content_token_estimate": selected["analyzed_content_token_estimate"],
        "truncation_reason": selected["truncation_reason"],
        "attention_tier": attention["attention_tier"],
        "theme": attention["theme"],
        "attention_reason": attention["attention_reason"],
        "must_keep": attention["must_keep"],
    }


def _article_attention_defaults(article: dict[str, Any]) -> dict[str, Any]:
    return _normalize_attention_metadata(
        {
            "attention_tier": article.get("attention_tier"),
            "theme": article.get("theme"),
            "attention_reason": article.get("attention_reason"),
            "must_keep": article.get("must_keep"),
        },
        article,
    )


def _heuristic_attention_metadata(article: dict[str, Any]) -> dict[str, Any]:
    title = str(article.get("title") or "").strip()
    snippet = str(article.get("summary_snippet") or "").strip()
    section = str(article.get("article_section") or article.get("section") or "").strip()
    haystack = f"{title} {snippet} {section}".lower()

    matched_theme = "general"
    high_signal = False
    for theme, keywords in HIGH_ATTENTION_THEME_KEYWORDS.items():
        if any(keyword.lower() in haystack for keyword in keywords):
            matched_theme = theme
            high_signal = theme in {"stocks", "macro", "geopolitics"}
            break

    if high_signal:
        tier = "high"
        must_keep = True
        reason = f"High-signal {matched_theme} headline based on title/summary keywords."
    elif section in LIGHT_ANALYSIS_SECTIONS:
        tier = "light"
        must_keep = False
        if matched_theme == "general" and section == "地產新聞":
            matched_theme = "property"
        reason = f"Lighter treatment because this story sits in the {section} section without strong market-moving signals."
    else:
        tier = "medium"
        must_keep = False
        reason = "Default medium attention because the story is relevant but not an obvious top-priority market mover."

    return {
        "attention_tier": tier,
        "theme": matched_theme,
        "attention_reason": reason,
        "must_keep": must_keep,
    }


def _normalize_attention_metadata(metadata: dict[str, Any], article: dict[str, Any]) -> dict[str, Any]:
    heuristic = _heuristic_attention_metadata(article)
    attention_tier = str(metadata.get("attention_tier") or heuristic["attention_tier"]).strip().lower()
    if attention_tier not in ATTENTION_TIERS:
        attention_tier = heuristic["attention_tier"]
    theme = str(metadata.get("theme") or heuristic["theme"]).strip().lower() or heuristic["theme"]
    attention_reason = str(metadata.get("attention_reason") or heuristic["attention_reason"]).strip() or heuristic["attention_reason"]
    must_keep_value = metadata.get("must_keep")
    if isinstance(must_keep_value, bool):
        must_keep = must_keep_value
    elif must_keep_value is None:
        must_keep = bool(heuristic["must_keep"])
    else:
        must_keep = str(must_keep_value).strip().lower() in {"1", "true", "yes"}
    if must_keep and attention_tier == "light":
        attention_tier = "medium"
    return {
        "attention_tier": attention_tier,
        "theme": theme,
        "attention_reason": attention_reason,
        "must_keep": must_keep,
    }


def _apply_attention_metadata(article: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    applied = dict(article)
    normalized = _normalize_attention_metadata(metadata, applied)
    applied.update(normalized)
    return applied


def _order_articles_like_input(
    input_articles: list[dict[str, Any]],
    routed_articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    routed_by_key = {
        _article_key(article.get("source_article_id"), article.get("canonical_url")): article
        for article in routed_articles
    }
    return [
        routed_by_key.get(_article_key(article.get("source_article_id"), article.get("canonical_url")), article)
        for article in input_articles
    ]


def _load_existing_report(report_path: Path) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Ignoring unreadable analysis report at %s: %s", report_path, exc)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("Ignoring unexpected analysis report payload at %s: expected JSON object.", report_path)
        return None
    if payload.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        return None
    return payload


def _build_today_incremental_plan(
    articles: list[dict[str, Any]],
    existing_report: dict[str, Any] | None,
) -> dict[str, Any]:
    successful_keys = _successful_report_article_keys(existing_report)
    work_articles: list[dict[str, Any]] = []
    reused_successful_articles = 0
    for article in articles:
        key = _article_key(article.get("source_article_id"), article.get("canonical_url"))
        if key in successful_keys:
            reused_successful_articles += 1
        else:
            work_articles.append(article)
    return {
        "work_articles": work_articles,
        "reused_successful_articles": reused_successful_articles,
        "new_articles_analyzed": len(work_articles),
    }


def _build_previous_retry_plan(
    previous_articles: list[dict[str, Any]],
    previous_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if previous_report is None:
        return {"work_articles": [], "retried_previous_day_articles": 0}

    unresolved_keys = _unresolved_report_article_keys(previous_report)
    work_articles = [
        article
        for article in previous_articles
        if _article_key(article.get("source_article_id"), article.get("canonical_url")) in unresolved_keys
    ]
    return {
        "work_articles": work_articles,
        "retried_previous_day_articles": len(work_articles),
    }


def _route_articles(runtime: AnalysisRuntime | None, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seeded = [_apply_attention_metadata(article, _heuristic_attention_metadata(article)) for article in articles]
    if runtime is None or not seeded:
        return seeded

    routed_by_key = {
        _article_key(article.get("source_article_id"), article.get("canonical_url")): article
        for article in seeded
    }
    for category_name, category_articles in _group_source_articles(seeded):
        if not _should_use_llm_router(category_name, category_articles):
            continue
        routed_batch = _route_articles_with_llm(runtime, category_name, category_articles)
        for article in routed_batch:
            routed_by_key[_article_key(article.get("source_article_id"), article.get("canonical_url"))] = article
    return [
        routed_by_key[_article_key(article.get("source_article_id"), article.get("canonical_url"))]
        for article in seeded
    ]


def _should_use_llm_router(category_name: str, category_articles: list[dict[str, Any]]) -> bool:
    if category_name in LIGHT_ANALYSIS_SECTIONS:
        return False
    return len(category_articles) >= ROUTER_LLM_MIN_ARTICLES


def _route_articles_with_llm(
    runtime: AnalysisRuntime,
    category_name: str,
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _route_batch_recursive(runtime, category_name, articles, batch_label="route")


def _route_batch_recursive(
    runtime: AnalysisRuntime,
    category_name: str,
    batch_articles: list[dict[str, Any]],
    *,
    batch_label: str,
    depth: int = 0,
) -> list[dict[str, Any]]:
    if not batch_articles:
        return []

    messages = _build_attention_routing_messages(category_name, batch_articles)
    budget_state = runtime.get_category_budget(category_name)
    estimated_input_tokens = _estimate_messages_tokens(messages)
    request_bytes = _estimate_request_payload_bytes(runtime.current_model.model_id, messages, runtime.current_model.max_completion_tokens)
    if (
        len(batch_articles) > 1
        and (
            estimated_input_tokens > budget_state.synthesis_input_budget_tokens
            or request_bytes > budget_state.synthesis_request_byte_budget
        )
    ):
        runtime.record_split(category_name, "pre_send_budget")
        left, right = _split_batch(batch_articles)
        return _route_batch_recursive(runtime, category_name, left, batch_label=f"{batch_label}a", depth=depth + 1) + _route_batch_recursive(
            runtime, category_name, right, batch_label=f"{batch_label}b", depth=depth + 1
        )

    context = BatchContext(
        category_name=category_name,
        batch_kind="routing",
        batch_label=batch_label,
        article_count=len(batch_articles),
        estimated_input_tokens=estimated_input_tokens,
        serialized_request_bytes=request_bytes,
        content_shrunk=False,
    )

    try:
        payload, _ = _invoke_json_with_retry(runtime, messages, estimated_input_tokens, context)
        merged_results, missing_articles = _merge_attention_results(batch_articles, payload)
        if not missing_articles:
            return merged_results
        if len(missing_articles) == 1 or depth >= 1:
            fallback_results = [
                _apply_attention_metadata(article, _heuristic_attention_metadata(article))
                for article in missing_articles
            ]
            return _order_articles_like_input(batch_articles, merged_results + fallback_results)
        left, right = _split_batch(missing_articles)
        salvage_results = _route_batch_recursive(runtime, category_name, left, batch_label=f"{batch_label}m1", depth=depth + 1) + _route_batch_recursive(
            runtime, category_name, right, batch_label=f"{batch_label}m2", depth=depth + 1
        )
        return _order_articles_like_input(batch_articles, merged_results + salvage_results)
    except Exception as exc:
        classification = _classify_exception(exc)
        runtime.tighten_category_budget(category_name, classification, batch_kind="synthesis")
        if classification == "payload_too_large":
            runtime.record_split(category_name, "response_413")
        if len(batch_articles) > 1 and depth < 1:
            left, right = _split_batch(batch_articles)
            return _route_batch_recursive(runtime, category_name, left, batch_label=f"{batch_label}a", depth=depth + 1) + _route_batch_recursive(
                runtime, category_name, right, batch_label=f"{batch_label}b", depth=depth + 1
            )
        return [_apply_attention_metadata(article, _heuristic_attention_metadata(article)) for article in batch_articles]


def _build_attention_routing_messages(category_name: str, batch_articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a financial news triage router. Return one valid JSON object only. "
                "Route every article into an attention tier and compact theme. Do not omit any article."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Assign compact routing metadata before deeper article analysis.",
                    "required_schema": {
                        "routes": [
                            {
                                "source_article_id": "string or null",
                                "canonical_url": "string",
                                "attention_tier": "high|medium|light",
                                "theme": "short theme such as stocks|macro|geopolitics|property|general",
                                "reason": "one short sentence",
                                "must_keep": "boolean",
                            }
                        ]
                    },
                    "routing_policy": {
                        "high": [
                            "stocks, earnings, guidance, placements, buybacks, capital markets",
                            "macro, central banks, inflation, rates, growth, FX, oil",
                            "geopolitics, sanctions, tariffs, war, trade restrictions",
                        ],
                        "light": [
                            "softer pulse-style or local-interest items",
                            "stories in 時事脈搏 or 地產新聞 without obvious market-moving signals",
                        ],
                        "default": "Use medium when ambiguous.",
                    },
                    "category": category_name,
                    "articles": [
                        {
                            "source_article_id": article.get("source_article_id"),
                            "canonical_url": article.get("canonical_url"),
                            "title": article.get("title"),
                            "summary_snippet": article.get("summary_snippet"),
                            "article_section": article.get("article_section") or article.get("section"),
                            "published_at": article.get("published_at"),
                        }
                        for article in batch_articles
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]


def _merge_attention_results(
    batch_articles: list[dict[str, Any]],
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    response_items = payload.get("routes")
    response_list = response_items if isinstance(response_items, list) else []
    response_by_key: dict[tuple[str | None, str], dict[str, Any]] = {}
    for item in response_list:
        if isinstance(item, dict):
            response_by_key[_article_key(item.get("source_article_id"), item.get("canonical_url"))] = item

    merged: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for article in batch_articles:
        key = _article_key(article.get("source_article_id"), article.get("canonical_url"))
        routed = response_by_key.get(key)
        if routed is None:
            missing.append(article)
            continue
        merged.append(
            _apply_attention_metadata(
                article,
                {
                    "attention_tier": routed.get("attention_tier"),
                    "theme": routed.get("theme"),
                    "attention_reason": routed.get("reason"),
                    "must_keep": routed.get("must_keep"),
                },
            )
        )
    return merged, missing


def _build_incremental_report_categories(
    *,
    runtime: AnalysisRuntime | None,
    db_articles: list[dict[str, Any]],
    existing_report: dict[str, Any] | None,
    plan: dict[str, Any],
    retry_only: bool,
) -> list[dict[str, Any]]:
    existing_categories = _existing_categories_by_name(existing_report)
    existing_articles = _existing_articles_by_key(existing_report)
    db_articles_by_key = {
        _article_key(article.get("source_article_id"), article.get("canonical_url")): article
        for article in db_articles
    }
    work_by_key = {
        _article_key(article.get("source_article_id"), article.get("canonical_url")): article
        for article in plan["work_articles"]
    }
    analyzed_categories: dict[str, dict[str, Any]] = {}

    if runtime is not None and plan["work_articles"]:
        for category_name, category_articles in _group_source_articles(plan["work_articles"]):
            prepared_articles = [_prepare_single_article(article) for article in category_articles]
            category_report, _ = _analyze_category(runtime, category_name, prepared_articles)
            analyzed_categories[category_name] = category_report

    if retry_only:
        category_order = [category["category"] for category in existing_report.get("categories", [])] if existing_report else []
        source_groups = [(name, []) for name in category_order]
    else:
        source_groups = _group_source_articles(db_articles)

    category_reports: list[dict[str, Any]] = []
    for category_name, category_articles in source_groups:
        db_keys = [
            _article_key(article.get("source_article_id"), article.get("canonical_url"))
            for article in category_articles
        ]
        existing_category = existing_categories.get(category_name)
        analyzed_category = analyzed_categories.get(category_name)
        changed = analyzed_category is not None
        final_articles: list[dict[str, Any]] = []
        used_existing_results = False

        if retry_only and existing_category is not None:
            existing_category_articles = existing_category.get("articles") or []
            for article_result in existing_category_articles:
                key = _article_key(article_result.get("source_article_id"), article_result.get("canonical_url"))
                if analyzed_category is not None:
                    replacement = _article_result_by_key(analyzed_category["articles"]).get(key)
                    if replacement is not None:
                        final_articles.append(replacement)
                        continue
                final_articles.append(_ensure_attention_fields(article_result, db_articles_by_key.get(key)))
                used_existing_results = True
        else:
            analyzed_results = _article_result_by_key(analyzed_category["articles"] if analyzed_category else [])
            for article in category_articles:
                key = _article_key(article.get("source_article_id"), article.get("canonical_url"))
                if key in analyzed_results:
                    final_articles.append(analyzed_results[key])
                else:
                    existing_result = existing_articles.get(key)
                    if existing_result is not None:
                        final_articles.append(_ensure_attention_fields(existing_result, article))
                        used_existing_results = True

        if not final_articles:
            continue

        category_reports.append(
            _merge_category_report(
                runtime=runtime,
                category_name=category_name,
                final_articles=final_articles,
                existing_category=existing_category,
                analyzed_category=analyzed_category,
                changed=changed,
                used_existing_results=used_existing_results,
            )
        )

    return category_reports


def _retry_previous_report(
    *,
    runtime: AnalysisRuntime | None,
    previous_date: str,
    db_articles: list[dict[str, Any]],
    existing_report: dict[str, Any],
    retry_plan: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    if runtime is None or not retry_plan["work_articles"]:
        return existing_report, 0

    category_reports = _build_incremental_report_categories(
        runtime=runtime,
        db_articles=db_articles,
        existing_report=existing_report,
        plan=retry_plan,
        retry_only=True,
    )
    updated_report = _finalize_report(
        target_date=previous_date,
        source_site=DEFAULT_SOURCE_SITE,
        input_article_count=int((existing_report.get("input") or {}).get("article_count") or len(db_articles)),
        category_reports=category_reports,
        top_alerts=list(existing_report.get("executive_summary") or []),
        runtime=runtime,
        incremental={
            "reused_successful_articles": 0,
            "new_articles_analyzed": 0,
            "retried_previous_day_articles": retry_plan["retried_previous_day_articles"],
            "previous_day_retry_successes": 0,
        },
        total_scraped_count=int((existing_report.get("daily_stats") or {}).get("total_scraped") or len(db_articles)),
    )

    existing_articles = _existing_articles_by_key(existing_report)
    updated_articles = _existing_articles_by_key(updated_report)
    previous_day_retry_successes = 0
    for article in retry_plan["work_articles"]:
        key = _article_key(article.get("source_article_id"), article.get("canonical_url"))
        before = existing_articles.get(key)
        after = updated_articles.get(key)
        if before is not None and after is not None and before.get("error") and not after.get("error"):
            previous_day_retry_successes += 1

    updated_report["incremental"]["previous_day_retry_successes"] = previous_day_retry_successes
    if previous_day_retry_successes == 0:
        return existing_report, 0
    return updated_report, previous_day_retry_successes


def _merge_category_report(
    *,
    runtime: AnalysisRuntime | None,
    category_name: str,
    final_articles: list[dict[str, Any]],
    existing_category: dict[str, Any] | None,
    analyzed_category: dict[str, Any] | None,
    changed: bool,
    used_existing_results: bool,
) -> dict[str, Any]:
    if changed and analyzed_category is not None and not used_existing_results:
        merged = dict(analyzed_category)
        merged["article_count"] = len(final_articles)
        merged["articles"] = final_articles
        return merged

    profile = _section_profile(category_name)
    successful_articles = [article for article in final_articles if not article.get("error")]
    diagnostics = dict((analyzed_category or existing_category or {}).get("diagnostics") or {})
    sub_batch_count = int((analyzed_category or existing_category or {}).get("sub_batch_count") or diagnostics.get("sub_batch_count") or 0)
    key_developments = list((existing_category or {}).get("key_developments") or [])
    named_entities = list((existing_category or {}).get("named_entities") or [])
    model_used = str((existing_category or {}).get("model_used") or PRIMARY_MODEL_ID)
    subgroups = list((existing_category or {}).get("subgroups") or [])
    category_error_message: str | None = None

    if changed:
        diagnostics = dict((analyzed_category or {}).get("diagnostics") or diagnostics)
        sub_batch_count = int((analyzed_category or {}).get("sub_batch_count") or sub_batch_count)
        if successful_articles and runtime is not None:
            runtime.reset_model_for_category()
            try:
                synthesis_payload, synthesis_model, synthesis_batch_count, subgroup_reports = _build_category_outputs(
                    runtime,
                    category_name,
                    successful_articles,
                )
                key_developments = _normalize_string_list(synthesis_payload.get("key_developments"), limit=profile.category_bullet_limit)
                named_entities = _normalize_entities(synthesis_payload.get("named_entities"))[: profile.entity_limit]
                subgroups = subgroup_reports
                model_used = synthesis_model
                sub_batch_count += synthesis_batch_count
            except Exception as exc:
                category_error_message = str(exc)
        elif not successful_articles:
            category_error_message = "No successful article analyses available for synthesis."
    diagnostics["sub_batch_count"] = sub_batch_count
    diagnostics["partial_article_count"] = sum(1 for article in final_articles if article.get("error"))

    if successful_articles:
        status = "success" if all(not article.get("error") for article in final_articles) and not category_error_message else "partial"
    else:
        status = "failed"
        if category_error_message is None:
            category_error_message = "No successful article analyses available for synthesis."

    return {
        "category": category_name,
        "article_count": len(final_articles),
        "status": status,
        "key_developments": key_developments,
        "named_entities": named_entities if successful_articles else [],
        "articles": final_articles,
        "subgroups": subgroups if successful_articles else [],
        "analysis_profile": profile.name,
        "model_used": model_used,
        "sub_batch_count": sub_batch_count,
        "diagnostics": diagnostics,
        "error": category_error_message,
    }


def _finalize_report(
    *,
    target_date: str,
    source_site: str,
    input_article_count: int,
    category_reports: list[dict[str, Any]],
    top_alerts: list[str],
    runtime: AnalysisRuntime | None,
    incremental: dict[str, int],
    total_scraped_count: int,
    market_snapshots: list[dict[str, Any]] | None = None,
    legacy_executive_summary: list[str] | None = None,
    newly_analyzed_keys: set[tuple[str | None, str]] | None = None,
) -> dict[str, Any]:
    # Flag new articles
    if newly_analyzed_keys:
        for category in category_reports:
            for article in category.get("articles", []):
                key = _article_key(article.get("source_article_id"), article.get("canonical_url"))
                if key in newly_analyzed_keys:
                    article["is_new"] = True

    all_articles = [article for category in category_reports for article in category["articles"]]
    unresolved_articles = _collect_unresolved_articles(category_reports)
    successful_article_analyses = sum(1 for article in all_articles if not article.get("error"))
    failed_article_analyses = len(all_articles) - successful_article_analyses
    full_text_count = sum(1 for article in all_articles if not article.get("content_truncated"))
    truncated_count = len(all_articles) - full_text_count
    successful_categories = sum(1 for category in category_reports if category["status"] == "success")
    partial_categories = sum(1 for category in category_reports if category["status"] == "partial")
    failed_categories = sum(1 for category in category_reports if category["status"] == "failed")
    errors = _collect_report_errors(category_reports)
    status = "success" if not errors else "partial"
    if runtime is not None:
        runtime.diagnostics.high_medium_unresolved_count = sum(
            1 for article in unresolved_articles if str(article.get("attention_tier") or "medium") in {"high", "medium"}
        )
        runtime.diagnostics.light_unresolved_count = sum(
            1 for article in unresolved_articles if str(article.get("attention_tier") or "") == "light"
        )

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_date": target_date,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_site": source_site,
        "status": status,
        "executive_summary": top_alerts,
        "legacy_executive_summary": legacy_executive_summary or [],
        "model": {
            "provider": DEFAULT_PROVIDER,
            "primary_model": PRIMARY_MODEL_ID,
            "fallback_models": list(FALLBACK_MODEL_IDS),
        },
        "model_switches": list(runtime.model_switches if runtime is not None else []),
        "input": {
            "article_count": input_article_count,
            "category_count": len(category_reports),
        },
        "diagnostics": runtime.diagnostics.as_dict() if runtime is not None else RuntimeDiagnostics().as_dict(),
        "incremental": incremental,
        "daily_stats": {
            "total_scraped": total_scraped_count,
            "analyzed": input_article_count,
            "success_rate": round((input_article_count / max(total_scraped_count, 1)) * 100, 1) if total_scraped_count > 0 else 0,
        },
        "unresolved_articles": unresolved_articles,
        "totals": {
            "article_count": len(all_articles),
            "successful_article_analyses": successful_article_analyses,
            "failed_article_analyses": failed_article_analyses,
            "full_text_article_count": full_text_count,
            "truncated_article_count": truncated_count,
            "successful_categories": successful_categories,
            "partial_categories": partial_categories,
            "failed_categories": failed_categories,
        },
        "market_context": _build_market_context_for_report(market_snapshots or []),
        "categories": category_reports,
        "errors": errors,
    }


def _collect_report_errors(category_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for category in category_reports:
        for article in category.get("articles") or []:
            if article.get("error"):
                errors.append(
                    {
                        "type": "article",
                        "target": article.get("canonical_url"),
                        "message": article.get("error"),
                        "classification": article.get("error_classification") or "unexpected_error",
                    }
                )
        if category.get("error"):
            errors.append(
                {
                    "type": "category",
                    "target": category.get("category"),
                    "message": category.get("error"),
                    "classification": _category_error_classification(category),
                }
            )
    return errors


def _collect_unresolved_articles(category_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unresolved: list[dict[str, Any]] = []
    for category in category_reports:
        category_name = category.get("category")
        for article in category.get("articles") or []:
            if not article.get("error"):
                continue
            unresolved.append(
                {
                    "category": category_name,
                    "title": article.get("title"),
                    "canonical_url": article.get("canonical_url"),
                    "source_article_id": article.get("source_article_id"),
                    "attention_tier": article.get("attention_tier"),
                    "theme": article.get("theme"),
                    "must_keep": article.get("must_keep"),
                    "error_classification": article.get("error_classification"),
                    "error": article.get("error"),
                    "model_used": article.get("model_used"),
                    "delayed_retry_attempted": bool(article.get("delayed_retry_attempted")),
                    "delayed_retry_model_chain": list(article.get("delayed_retry_model_chain") or []),
                    "delayed_retry_final_model": article.get("delayed_retry_final_model"),
                    "published_at": article.get("published_at"),
                }
            )
    unresolved.sort(
        key=lambda item: (
            ATTENTION_TIER_RANK.get(str(item.get("attention_tier") or "medium"), ATTENTION_TIER_RANK["medium"]),
            str(item.get("category") or ""),
            str(item.get("published_at") or ""),
        )
    )
    return unresolved


def _category_error_classification(category: dict[str, Any]) -> str:
    message = str(category.get("error") or "")
    if "413" in message:
        return "payload_too_large"
    if category.get("status") == "failed":
        return "incomplete_model_output"
    return "unexpected_error"


def _existing_articles_by_key(report: dict[str, Any] | None) -> dict[tuple[str | None, str], dict[str, Any]]:
    if report is None:
        return {}
    items: dict[tuple[str | None, str], dict[str, Any]] = {}
    for category in report.get("categories") or []:
        for article in category.get("articles") or []:
            items[_article_key(article.get("source_article_id"), article.get("canonical_url"))] = article
    return items


def _existing_categories_by_name(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if report is None:
        return {}
    return {category.get("category"): category for category in report.get("categories") or []}


def _successful_report_article_keys(report: dict[str, Any] | None) -> set[tuple[str | None, str]]:
    return {
        key
        for key, article in _existing_articles_by_key(report).items()
        if not article.get("error") and not article.get("error_classification")
    }


def _unresolved_report_article_keys(report: dict[str, Any] | None) -> set[tuple[str | None, str]]:
    return {
        key
        for key, article in _existing_articles_by_key(report).items()
        if article.get("error") or article.get("error_classification")
    }


def _article_result_by_key(items: list[dict[str, Any]]) -> dict[tuple[str | None, str], dict[str, Any]]:
    return {
        _article_key(item.get("source_article_id"), item.get("canonical_url")): item
        for item in items
    }


def _ensure_attention_fields(article_result: dict[str, Any], source_article: dict[str, Any] | None) -> dict[str, Any]:
    hydrated = dict(article_result)
    article_context = {
        "source_article_id": hydrated.get("source_article_id"),
        "canonical_url": hydrated.get("canonical_url"),
        "title": hydrated.get("title") or (source_article or {}).get("title"),
        "summary_snippet": hydrated.get("summary_snippet") or (source_article or {}).get("summary_snippet"),
        "article_section": hydrated.get("section") or (source_article or {}).get("article_section"),
        "section": hydrated.get("section") or (source_article or {}).get("article_section"),
        "published_at": hydrated.get("published_at") or (source_article or {}).get("published_at"),
    }
    hydrated.update(_article_attention_defaults({**article_context, **hydrated}))
    return hydrated


def _analyze_category(
    runtime: AnalysisRuntime,
    category_name: str,
    prepared_articles: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime.reset_model_for_category()
    LOGGER.info("Analyzing category %s with %s article(s).", category_name, len(prepared_articles))
    profile = _section_profile(category_name)
    planned_batches = _plan_category_batches(runtime, category_name, prepared_articles)
    article_results: list[dict[str, Any]] = []
    article_errors: list[dict[str, Any]] = []
    sub_batch_count = 0
    
    # Analyze non-light articles in batches
    for index, batch in enumerate(planned_batches, start=1):
        if not batch:
            continue
        batch_results, batch_errors, batch_count = _process_batch_recursive(runtime, category_name, batch, str(index))
        article_results.extend(batch_results)
        article_errors.extend(batch_errors)
        sub_batch_count += batch_count
        
    # Handle bypassed light articles
    for article in prepared_articles:
        if str(article.get("attention_tier") or "").lower() == "light":
            article_results.append({
                "source_article_id": article.get("source_article_id"),
                "canonical_url": article.get("canonical_url"),
                "title": article.get("title"),
                "section": article.get("section"),
                "published_at": article.get("published_at"),
                "attention_tier": "light",
                "novelty_score": 5,
                "relevance_score": 5,
                "urgency_score": 5,
                "named_entities": [],
                "key_points": [],
                "content_truncated": article.get("content_truncated"),
                "analysis_method": "bypassed",
                "model_used": "direct",
            })

    article_results = _order_results_like_input(prepared_articles, article_results)
    article_results, delayed_retry_count = _run_delayed_retry_pass(runtime, category_name, prepared_articles, article_results)
    sub_batch_count += delayed_retry_count
    article_errors = _article_errors_from_results(article_results)
    successful_articles = [article for article in article_results if not article.get("error")]
    category_errors = list(article_errors)
    key_developments: list[str] = []
    named_entities: list[dict[str, str]] = _collect_entities_from_articles(successful_articles)
    subgroup_reports: list[dict[str, Any]] = []
    synthesis_model = _category_model_used(article_results)
    category_status = "success"
    category_error_message: str | None = None
    diagnostics = runtime.get_category_diagnostics(category_name)
    diagnostics.sub_batch_count = sub_batch_count
    diagnostics.partial_article_count = sum(1 for article in article_results if article.get("error"))

    if successful_articles:
        try:
            synthesis_payload, synthesis_model, synthesis_batch_count, subgroup_reports = _build_category_outputs(
                runtime,
                category_name,
                successful_articles,
            )
            sub_batch_count += synthesis_batch_count
            key_developments = _normalize_string_list(synthesis_payload.get("key_developments"), limit=profile.category_bullet_limit)
            named_entities = _normalize_entities(synthesis_payload.get("named_entities"))[: profile.entity_limit]
        except Exception as exc:
            category_status = "partial"
            category_error_message = str(exc)
            category_errors.append(
                {
                    "type": "category",
                    "target": category_name,
                    "message": str(exc),
                    "classification": _classify_exception(exc),
                }
            )
    elif prepared_articles:
        category_status = "failed"
        category_error_message = "No successful article analyses available for synthesis."
        category_errors.append(
            {
                "type": "category",
                "target": category_name,
                "message": category_error_message,
                "classification": "incomplete_model_output",
            }
        )
    else:
        # Graceful handling for exactly 0 articles case
        category_status = "success"


    if category_status == "success" and article_errors:
        category_status = "partial"

    if category_status == "partial":
        LOGGER.info("Category %s completed partially.", category_name)
    elif category_status == "failed":
        LOGGER.info("Category %s failed.", category_name)
    else:
        LOGGER.info("Category %s completed successfully.", category_name)

    return (
        {
            "category": category_name,
            "article_count": len(prepared_articles),
            "status": category_status,
            "key_developments": key_developments,
            "named_entities": named_entities,
            "articles": article_results,
            "subgroups": subgroup_reports,
            "analysis_profile": profile.name,
            "model_used": synthesis_model,
            "sub_batch_count": sub_batch_count,
            "diagnostics": diagnostics.as_dict(),
            "error": category_error_message,
        },
        category_errors,
    )


def _run_delayed_retry_pass(
    runtime: AnalysisRuntime,
    category_name: str,
    prepared_articles: list[dict[str, Any]],
    article_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    result_by_key = _article_result_by_key(article_results)
    prepared_by_key = {
        _article_key(article.get("source_article_id"), article.get("canonical_url")): article
        for article in prepared_articles
    }
    candidates = [
        result
        for result in article_results
        if result.get("error") and str(result.get("attention_tier") or "medium") in {"high", "medium"}
    ]
    runtime.diagnostics.delayed_retry_candidate_count += len(candidates)
    if not candidates:
        return article_results, 0

    LOGGER.info(
        "Waiting %.1f seconds before delayed retry for %s unresolved high/medium article(s) in %s.",
        DELAYED_RETRY_WAIT_SECONDS,
        len(candidates),
        category_name,
    )
    time.sleep(DELAYED_RETRY_WAIT_SECONDS)

    batch_count = 0
    for candidate in candidates:
        key = _article_key(candidate.get("source_article_id"), candidate.get("canonical_url"))
        prepared_article = prepared_by_key.get(key)
        if prepared_article is None:
            continue
        runtime.diagnostics.delayed_retry_attempted_count += 1
        retry_result, retry_count = _retry_unresolved_article_with_delay(runtime, category_name, prepared_article, candidate)
        batch_count += retry_count
        result_by_key[key] = retry_result
        if retry_result.get("error"):
            runtime.diagnostics.delayed_retry_failed_count += 1
        else:
            runtime.diagnostics.delayed_retry_recovered_count += 1

    merged_results = [
        result_by_key[_article_key(article.get("source_article_id"), article.get("canonical_url"))]
        for article in article_results
    ]
    return merged_results, batch_count


def _retry_unresolved_article_with_delay(
    runtime: AnalysisRuntime,
    category_name: str,
    prepared_article: dict[str, Any],
    current_result: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    model_chain = _delayed_retry_model_sequence(runtime, str(current_result.get("model_used") or ""))
    attempted_chain: list[str] = []
    final_model = DELAYED_RETRY_FINAL_MODEL_ID

    for model in model_chain:
        attempted_chain.append(model.model_id)
        try:
            retry_result = _attempt_single_article_with_model(runtime, category_name, prepared_article, model, current_result)
            retry_result["delayed_retry_attempted"] = True
            retry_result["delayed_retry_model_chain"] = list(attempted_chain)
            retry_result["delayed_retry_final_model"] = final_model
            current_result = retry_result
            if not retry_result.get("error"):
                return retry_result, 1
        except Exception as exc:
            if model.provider == "openai" and "OPENAI_API_KEY" in str(exc):
                runtime.diagnostics.delayed_retry_skipped_final_model_count += 1
                attempted_chain[-1] = f"{model.model_id} (skipped missing OPENAI_API_KEY)"
                break
            failed = _build_failed_article_result(
                article=prepared_article,
                error_message=str(exc),
                model_used=model.model_id,
                error_classification=_classify_exception(exc),
            )
            failed["delayed_retry_attempted"] = True
            failed["delayed_retry_model_chain"] = list(attempted_chain)
            failed["delayed_retry_final_model"] = final_model
            current_result = failed

    current_result = dict(current_result)
    current_result["delayed_retry_attempted"] = True
    current_result["delayed_retry_model_chain"] = list(attempted_chain)
    current_result["delayed_retry_final_model"] = final_model
    return current_result, 0


def _delayed_retry_model_sequence(runtime: AnalysisRuntime, current_model_used: str) -> list[ModelConfig]:
    sequence: list[ModelConfig] = []
    seen: set[str] = set()
    found_current = False
    for model in runtime.model_chain:
        if model.model_id == current_model_used:
            found_current = True
            continue
        if not found_current:
            continue
        if model.model_id == DELAYED_RETRY_FINAL_MODEL_ID:
            continue
        if model.model_id not in seen:
            sequence.append(model)
            seen.add(model.model_id)
    final_model = runtime.delayed_retry_final_model
    if final_model is not None and final_model.model_id not in seen:
        sequence.append(final_model)
    return sequence


def _attempt_single_article_with_model(
    runtime: AnalysisRuntime,
    category_name: str,
    prepared_article: dict[str, Any],
    model: ModelConfig,
    current_result: dict[str, Any],
) -> dict[str, Any]:
    working_article = _clone_prepared_article(prepared_article)
    budget_state = runtime.get_category_budget(category_name)
    profile = _section_profile(category_name)
    effective_budget = _effective_batch_budget(profile, budget_state, [working_article])
    _shrink_batch_to_budget(
        category_name,
        [working_article],
        input_budget_tokens=effective_budget["input_budget_tokens"],
        request_byte_budget=effective_budget["request_byte_budget"],
        model_id=model.model_id,
    )
    estimated_input_tokens = _estimate_batch_request_tokens(category_name, [working_article])
    payload, model_used = _invoke_article_batch(
        runtime,
        category_name,
        [working_article],
        estimated_input_tokens,
        batch_label="delayed",
        content_shrunk=bool(working_article.get("content_truncated")),
        model_override=model,
    )
    merged_results, missing_articles = _merge_batch_article_results([working_article], payload, model_used)
    if missing_articles:
        result = _build_failed_article_result(
            article=working_article,
            error_message="Model response omitted this article from the delayed retry batch.",
            model_used=model_used,
            error_classification="incomplete_model_output",
        )
        result["delayed_retry_attempted"] = True
        result["delayed_retry_model_chain"] = [model.model_id]
        result["delayed_retry_final_model"] = DELAYED_RETRY_FINAL_MODEL_ID
        return result
    recovered = merged_results[0]
    recovered["delayed_retry_attempted"] = True
    recovered["delayed_retry_model_chain"] = [model.model_id]
    recovered["delayed_retry_final_model"] = DELAYED_RETRY_FINAL_MODEL_ID
    return recovered


def _article_errors_from_results(article_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "article",
            "target": article.get("canonical_url"),
            "message": article.get("error"),
            "classification": article.get("error_classification") or "unexpected_error",
        }
        for article in article_results
        if article.get("error")
    ]


def _process_batch_recursive(
    runtime: AnalysisRuntime,
    category_name: str,
    batch_articles: list[dict[str, Any]],
    batch_label: str = "1",
    *,
    allow_salvage: bool = True,
    model_override: ModelConfig | None = None,
    depth: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    profile = _section_profile(category_name)
    working_batch = [_clone_prepared_article(article) for article in batch_articles]
    budget_state = runtime.get_category_budget(category_name)
    effective_budget = _effective_batch_budget(profile, budget_state, working_batch)
    active_model = model_override or runtime.current_model
    content_shrunk = _shrink_batch_to_budget(
        category_name,
        working_batch,
        input_budget_tokens=effective_budget["input_budget_tokens"],
        request_byte_budget=effective_budget["request_byte_budget"],
        model_id=active_model.model_id,
    )
    estimated_input_tokens = _estimate_batch_request_tokens(category_name, working_batch)
    request_bytes = _estimate_batch_request_bytes(category_name, working_batch, active_model.model_id)

    if (
        (
            estimated_input_tokens > effective_budget["input_budget_tokens"]
            or request_bytes > effective_budget["request_byte_budget"]
        )
        and len(working_batch) > 1
    ):
        runtime.record_split(category_name, "pre_send_budget")
        LOGGER.info(
            "Splitting category %s batch %s before send: estimated_input_tokens=%s request_bytes=%s article_count=%s.",
            category_name,
            batch_label,
            estimated_input_tokens,
            request_bytes,
            len(working_batch),
        )
        left, right = _split_batch(working_batch)
        left_results, left_errors, left_count = _process_batch_recursive(
            runtime,
            category_name,
            left,
            f"{batch_label}a",
            model_override=model_override,
            depth=depth + 1,
        )
        right_results, right_errors, right_count = _process_batch_recursive(
            runtime,
            category_name,
            right,
            f"{batch_label}b",
            model_override=model_override,
            depth=depth + 1,
        )
        return left_results + right_results, left_errors + right_errors, left_count + right_count

    try:
        payload, model_used = _invoke_article_batch(
            runtime,
            category_name,
            working_batch,
            estimated_input_tokens,
            batch_label=batch_label,
            content_shrunk=content_shrunk,
            model_override=model_override,
        )
        merged_results, missing_articles = _merge_batch_article_results(working_batch, payload, model_used)
        if missing_articles:
            if not allow_salvage:
                message = "Model response omitted this article from the category batch."
                failed_results = [
                    _build_failed_article_result(
                        article=article,
                        error_message=message,
                        model_used=model_used,
                        error_classification="incomplete_model_output",
                    )
                    for article in missing_articles
                ]
                failed_errors = [
                    {
                        "type": "article",
                        "target": article.get("canonical_url"),
                        "message": message,
                        "classification": "incomplete_model_output",
                    }
                    for article in missing_articles
                ]
                combined = _order_results_like_input(working_batch, merged_results + failed_results)
                return combined, failed_errors, 1

            LOGGER.info(
                "Salvaging %s omitted article(s) from category %s batch %s.",
                len(missing_articles),
                category_name,
                batch_label,
            )
            runtime.tighten_category_budget(category_name, "incomplete_model_output", batch_kind="article_batch")
            if len(missing_articles) == 1:
                next_model = runtime.next_model_after(active_model.model_id)
                salvage_model = next_model if profile.name == "light" and next_model is not None else model_override
                salvage_results, salvage_errors, salvage_count = _process_batch_recursive(
                    runtime,
                    category_name,
                    missing_articles,
                    f"{batch_label}m",
                    allow_salvage=False,
                    model_override=salvage_model,
                    depth=depth + 1,
                )
            elif depth + 1 >= effective_budget["salvage_max_depth"]:
                message = "Model response omitted this article from the category batch."
                salvage_results = [
                    _build_failed_article_result(
                        article=article,
                        error_message=message,
                        model_used=model_used,
                        error_classification="incomplete_model_output",
                    )
                    for article in missing_articles
                ]
                salvage_errors = [
                    {
                        "type": "article",
                        "target": article.get("canonical_url"),
                        "message": message,
                        "classification": "incomplete_model_output",
                    }
                    for article in missing_articles
                ]
                salvage_count = 0
            else:
                left, right = _split_batch(missing_articles)
                left_results, left_errors, left_count = _process_batch_recursive(
                    runtime,
                    category_name,
                    left,
                    f"{batch_label}ma",
                    model_override=model_override,
                    depth=depth + 1,
                )
                right_results, right_errors, right_count = _process_batch_recursive(
                    runtime,
                    category_name,
                    right,
                    f"{batch_label}mb",
                    model_override=model_override,
                    depth=depth + 1,
                )
                salvage_results = left_results + right_results
                salvage_errors = left_errors + right_errors
                salvage_count = left_count + right_count

            unresolved_results = [result for result in salvage_results if result.get("error")]
            if unresolved_results:
                next_model = runtime.next_model_after(active_model.model_id)
                if next_model is not None:
                    unresolved_keys = {
                        _article_key(result.get("source_article_id"), result.get("canonical_url"))
                        for result in unresolved_results
                    }
                    unresolved_articles = [
                        article
                        for article in missing_articles
                        if _article_key(article.get("source_article_id"), article.get("canonical_url")) in unresolved_keys
                    ]
                    if unresolved_articles:
                        LOGGER.info(
                            "Escalating %s omitted article(s) in category %s batch %s from %s to %s.",
                            len(unresolved_articles),
                            category_name,
                            batch_label,
                            active_model.model_id,
                            next_model.model_id,
                        )
                        retry_results, retry_errors, retry_count = _process_batch_recursive(
                            runtime,
                            category_name,
                            unresolved_articles,
                            f"{batch_label}e",
                            allow_salvage=False,
                            model_override=next_model,
                            depth=depth + 1,
                        )
                        salvage_results = [
                            result
                            for result in salvage_results
                            if _article_key(result.get("source_article_id"), result.get("canonical_url")) not in unresolved_keys
                        ] + retry_results
                        salvage_errors = [
                            error
                            for error in salvage_errors
                            if error.get("target") not in {article.get("canonical_url") for article in unresolved_articles}
                        ] + retry_errors
                        salvage_count += retry_count

            combined = _order_results_like_input(working_batch, merged_results + salvage_results)
            return combined, salvage_errors, 1 + salvage_count
        return merged_results, [], 1
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        classification = _classify_exception(exc)
        if status_code == 413 and len(working_batch) > 1:
            runtime.record_split(category_name, "response_413")
            runtime.tighten_category_budget(category_name, classification, batch_kind="article_batch")
            LOGGER.info(
                "Splitting category %s batch %s after HTTP 413: article_count=%s.",
                category_name,
                batch_label,
                len(working_batch),
            )
            left, right = _split_batch(working_batch)
            left_results, left_errors, left_count = _process_batch_recursive(
                runtime,
                category_name,
                left,
                f"{batch_label}a",
                model_override=model_override,
                depth=depth + 1,
            )
            right_results, right_errors, right_count = _process_batch_recursive(
                runtime,
                category_name,
                right,
                f"{batch_label}b",
                model_override=model_override,
                depth=depth + 1,
            )
            return left_results + right_results, left_errors + right_errors, left_count + right_count

        model_used = runtime.last_attempted_model or active_model.model_id
        message = str(exc)
        if len(working_batch) > 1 and classification in {"rate_limited", "http_error", "unexpected_error"} and depth + 1 < effective_budget["salvage_max_depth"]:
            runtime.tighten_category_budget(category_name, classification, batch_kind="article_batch")
            LOGGER.info(
                "Retrying category %s batch %s as smaller sub-batches after %s.",
                category_name,
                batch_label,
                classification,
            )
            left, right = _split_batch(working_batch)
            left_results, left_errors, left_count = _process_batch_recursive(
                runtime,
                category_name,
                left,
                f"{batch_label}a",
                model_override=model_override,
                depth=depth + 1,
            )
            right_results, right_errors, right_count = _process_batch_recursive(
                runtime,
                category_name,
                right,
                f"{batch_label}b",
                model_override=model_override,
                depth=depth + 1,
            )
            return left_results + right_results, left_errors + right_errors, left_count + right_count
        runtime.record_failed_batch()
        failed_results = [
            _build_failed_article_result(
                article=article,
                error_message=message,
                model_used=model_used,
                error_classification=classification,
            )
            for article in working_batch
        ]
        failed_errors = [
            {
                "type": "article",
                "target": article.get("canonical_url"),
                "message": message,
                "classification": classification,
            }
            for article in working_batch
        ]
        return failed_results, failed_errors, 1
    except Exception as exc:
        model_used = runtime.last_attempted_model or active_model.model_id
        message = str(exc)
        classification = _classify_exception(exc)
        if len(working_batch) > 1 and classification in {"invalid_json", "unexpected_error"} and depth + 1 < effective_budget["salvage_max_depth"]:
            runtime.tighten_category_budget(category_name, classification, batch_kind="article_batch")
            LOGGER.info(
                "Retrying category %s batch %s as smaller sub-batches after %s.",
                category_name,
                batch_label,
                classification,
            )
            left, right = _split_batch(working_batch)
            left_results, left_errors, left_count = _process_batch_recursive(
                runtime,
                category_name,
                left,
                f"{batch_label}a",
                model_override=model_override,
                depth=depth + 1,
            )
            right_results, right_errors, right_count = _process_batch_recursive(
                runtime,
                category_name,
                right,
                f"{batch_label}b",
                model_override=model_override,
                depth=depth + 1,
            )
            return left_results + right_results, left_errors + right_errors, left_count + right_count
        if len(working_batch) == 1 and classification in {"invalid_json", "incomplete_model_output", "unexpected_error"} and depth + 1 < effective_budget["salvage_max_depth"]:
            next_model = runtime.next_model_after(active_model.model_id)
            if next_model is not None:
                LOGGER.info(
                    "Escalating category %s batch %s from %s to %s after %s.",
                    category_name,
                    batch_label,
                    active_model.model_id,
                    next_model.model_id,
                    classification,
                )
                return _process_batch_recursive(
                    runtime,
                    category_name,
                    working_batch,
                    f"{batch_label}n",
                    allow_salvage=False,
                    model_override=next_model,
                    depth=depth + 1,
                )
        runtime.record_failed_batch()
        failed_results = [
            _build_failed_article_result(
                article=article,
                error_message=message,
                model_used=model_used,
                error_classification=classification,
            )
            for article in working_batch
        ]
        failed_errors = [
            {
                "type": "article",
                "target": article.get("canonical_url"),
                "message": message,
                "classification": classification,
            }
            for article in working_batch
        ]
        return failed_results, failed_errors, 1


def _invoke_article_batch(
    runtime: AnalysisRuntime,
    category_name: str,
    batch_articles: list[dict[str, Any]],
    estimated_input_tokens: int,
    *,
    batch_label: str,
    content_shrunk: bool,
    model_override: ModelConfig | None = None,
) -> tuple[dict[str, Any], str]:
    messages = _build_article_batch_messages(category_name, batch_articles, market_context=runtime.market_context_string)
    active_model = model_override or runtime.current_model
    context = BatchContext(
        category_name=category_name,
        batch_kind="article_batch",
        batch_label=batch_label,
        article_count=len(batch_articles),
        estimated_input_tokens=estimated_input_tokens,
        serialized_request_bytes=_estimate_request_payload_bytes(active_model.model_id, messages, active_model.max_completion_tokens),
        content_shrunk=content_shrunk,
    )
    return _invoke_json_with_retry(runtime, messages, estimated_input_tokens, context, model_override=model_override)


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


def _build_article_batch_messages(category_name: str, batch_articles: list[dict[str, Any]], market_context: str = "") -> list[dict[str, Any]]:
    profile = _section_profile(category_name)
    batch_tier = _batch_attention_tier(batch_articles)
    return [
        {
            "role": "system",
            "content": (
                "You are a financial news analyst. Return one valid JSON object only. "
                "Analyze every article in the batch. Use integer scores from 1 to 10. "
                "Do not omit any article. "
                f"This batch is primarily {batch_tier}-attention; keep low-attention stories concise."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Analyze one batch of articles from a fixed financial news category.",
                    "required_schema": {
                        "articles": [
                            {
                                "source_article_id": "string or null",
                                "canonical_url": "string",
                                "novelty_score": "integer 1-10",
                                "relevance_score": "integer 1-10",
                                "urgency_score": "integer 1-10",
                                "named_entities": [
                                    {"name": "entity name", "type": "person|company|country|institution|index|organization|asset|other"}
                                ],
                                "key_points": [profile.article_key_points_instruction],
                            }
                        ]
                    },
                    "rules": [
                        "Return one article result for every input article.",
                        "Match article results by source_article_id when available and also include canonical_url.",
                        "Do not include category-level summary in this response.",
                        "If an article is tagged as light attention, keep key points especially concise.",
                        "If market_context data is provided, reference specific price movements and percentage changes in your key_points when relevant to the article.",
                    ],
                    "scoring_rubric": {
                        "novelty_score": "How new or non-repetitive this development is within the current news flow.",
                        "relevance_score": "How important this article is for daily finance, equity research, or macro monitoring.",
                        "urgency_score": "How quickly a human analyst should pay attention today.",
                    },
                    "category": category_name,
                    "articles": [
                        {
                            "source_article_id": article["source_article_id"],
                            "canonical_url": article["canonical_url"],
                            "title": article["title"],
                            "published_at": article["published_at"],
                            "article_section": article["section"],
                            "summary_snippet": article["summary_snippet"],
                            "content_text": article["content_text"],
                            "content_truncated": article["content_truncated"],
                            "analysis_method": article["analysis_method"],
                            "attention_tier": article.get("attention_tier"),
                            "theme": article.get("theme"),
                            "must_keep": article.get("must_keep"),
                        }
                        for article in batch_articles
                    ],
                    "market_context": market_context,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _build_category_outputs(
    runtime: AnalysisRuntime,
    category_name: str,
    successful_articles: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, int, list[dict[str, Any]]]:
    profile = _section_profile(category_name)
    if len(successful_articles) < profile.subgroup_threshold:
        summary_payload, summary_model, summary_batches = _synthesize_summary_items(
            runtime,
            category_name,
            [_article_to_synthesis_item(article) for article in successful_articles],
            batch_label_prefix="summary",
            bullet_limit=profile.category_bullet_limit,
            scope_kind="category",
            scope_title=category_name,
        )
        subgroup = {
            "title": f"{category_name} overview",
            "theme_rationale": "Single subgroup because the category size did not require thematic splitting.",
            "article_count": len(successful_articles),
            "key_developments": _normalize_string_list(summary_payload.get("key_developments"), limit=profile.subgroup_bullet_limit),
            "named_entities": _normalize_entities(summary_payload.get("named_entities"))[: profile.entity_limit],
            "articles": successful_articles,
            "model_used": summary_model,
        }
        return summary_payload, summary_model, summary_batches, [subgroup]

    subgroup_specs = _assign_article_subgroups(runtime, category_name, successful_articles)
    subgroup_specs = _consolidate_subgroups(subgroup_specs)
    subgroup_reports: list[dict[str, Any]] = []
    subgroup_summary_items: list[dict[str, Any]] = []
    total_batches = 0
    model_used = runtime.current_model.model_id

    for index, subgroup in enumerate(subgroup_specs, start=1):
        subgroup_payload, subgroup_model, subgroup_batches = _synthesize_summary_items(
            runtime,
            category_name,
            [_article_to_synthesis_item(article) for article in subgroup["articles"]],
            batch_label_prefix=f"subgroup{index}",
            bullet_limit=profile.subgroup_bullet_limit,
            scope_kind="subgroup",
            scope_title=str(subgroup["title"]),
        )
        subgroup_reports.append(
            {
                "title": subgroup["title"],
                "theme_rationale": subgroup["theme_rationale"],
                "article_count": len(subgroup["articles"]),
                "key_developments": _normalize_string_list(subgroup_payload.get("key_developments"), limit=profile.subgroup_bullet_limit),
                "named_entities": _normalize_entities(subgroup_payload.get("named_entities"))[: profile.entity_limit],
                "articles": subgroup["articles"],
                "model_used": subgroup_model,
            }
        )
        subgroup_summary_items.append(
            {
                "kind": "summary",
                "label": subgroup["title"],
                "article_count": len(subgroup["articles"]),
                "theme_rationale": subgroup["theme_rationale"],
                "key_developments": _normalize_string_list(subgroup_payload.get("key_developments"), limit=profile.subgroup_bullet_limit),
                "named_entities": _normalize_entities(subgroup_payload.get("named_entities"))[: profile.entity_limit],
            }
        )
        total_batches += subgroup_batches
        model_used = subgroup_model

    if len(subgroup_summary_items) == 1:
        only_group = subgroup_reports[0]
        return {
            "key_developments": only_group["key_developments"],
            "named_entities": only_group["named_entities"],
        }, model_used, total_batches, subgroup_reports

    seen_entity_names: set[str] = set()
    merged_entities: list[dict[str, str]] = []
    for sg in subgroup_reports:
        for entity in (sg.get("named_entities") or []):
            name = entity.get("name") or ""
            if name and name not in seen_entity_names:
                seen_entity_names.add(name)
                merged_entities.append(entity)
    category_payload = {
        "key_developments": [],
        "named_entities": merged_entities[: profile.entity_limit],
    }
    return category_payload, model_used, total_batches, subgroup_reports


def _assign_article_subgroups(
    runtime: AnalysisRuntime,
    category_name: str,
    successful_articles: list[dict[str, Any]],
    *,
    batch_label: str = "groups",
    model_override: ModelConfig | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    profile = _section_profile(category_name)
    items = [_article_to_grouping_item(article) for article in successful_articles]
    messages = _build_grouping_messages(category_name, items)
    active_model = model_override or runtime.current_model
    estimated_input_tokens = _estimate_messages_tokens(messages)
    serialized_bytes = _estimate_request_payload_bytes(active_model.model_id, messages, active_model.max_completion_tokens)
    budget_state = runtime.get_category_budget(category_name)
    grouping_input_budget = max(budget_state.synthesis_input_budget_tokens, 2000)
    grouping_request_byte_budget = max(budget_state.synthesis_request_byte_budget, 7000)

    if (
        len(successful_articles) > 1
        and (
            estimated_input_tokens > grouping_input_budget
            or serialized_bytes > grouping_request_byte_budget
        )
    ):
        runtime.record_split(category_name, "pre_send_budget")
        left, right = _split_batch(successful_articles)
        return _assign_article_subgroups(runtime, category_name, left, batch_label=f"{batch_label}a", model_override=model_override, depth=depth) + _assign_article_subgroups(
            runtime, category_name, right, batch_label=f"{batch_label}b", model_override=model_override, depth=depth
        )

    context = BatchContext(
        category_name=category_name,
        batch_kind="synthesis_grouping",
        batch_label=batch_label,
        article_count=len(successful_articles),
        estimated_input_tokens=estimated_input_tokens,
        serialized_request_bytes=serialized_bytes,
        content_shrunk=False,
    )

    try:
        payload, _ = _invoke_json_with_retry(
            runtime,
            messages,
            estimated_input_tokens,
            context,
            model_override=model_override,
        )
        normalized, missing_articles = _normalize_grouping_payload(payload, successful_articles, category_name)
        if not missing_articles:
            return normalized
        runtime.tighten_category_budget(category_name, "incomplete_model_output", batch_kind="synthesis")
        if len(missing_articles) == 1:
            next_model = runtime.next_model_after(active_model.model_id)
            if next_model is not None:
                return normalized + _assign_article_subgroups(
                    runtime,
                    category_name,
                    missing_articles,
                    batch_label=f"{batch_label}m",
                    model_override=next_model,
                    depth=depth + 1,
                )
            return normalized + _fallback_subgroups(category_name, missing_articles)

        if depth + 1 >= profile.salvage_max_depth:
            return normalized + _fallback_subgroups(category_name, missing_articles)

        left, right = _split_batch(missing_articles)
        return normalized + _assign_article_subgroups(
            runtime,
            category_name,
            left,
            batch_label=f"{batch_label}ma",
            model_override=model_override,
            depth=depth + 1,
        ) + _assign_article_subgroups(
            runtime,
            category_name,
            right,
            batch_label=f"{batch_label}mb",
            model_override=model_override,
            depth=depth + 1,
        )
    except Exception as exc:
        classification = _classify_exception(exc)
        runtime.tighten_category_budget(category_name, classification, batch_kind="synthesis")
        if len(successful_articles) > 1 and depth + 1 < profile.salvage_max_depth:
            if classification == "payload_too_large":
                runtime.record_split(category_name, "response_413")
            left, right = _split_batch(successful_articles)
            return _assign_article_subgroups(
                runtime,
                category_name,
                left,
                batch_label=f"{batch_label}a",
                model_override=model_override,
                depth=depth + 1,
            ) + _assign_article_subgroups(
                runtime,
                category_name,
                right,
                batch_label=f"{batch_label}b",
                model_override=model_override,
                depth=depth + 1,
            )
        return _fallback_subgroups(category_name, successful_articles)


def _synthesize_summary_items(
    runtime: AnalysisRuntime,
    category_name: str,
    synthesis_items: list[dict[str, Any]],
    *,
    batch_label_prefix: str,
    bullet_limit: int,
    scope_kind: str,
    scope_title: str,
    model_override: ModelConfig | None = None,
    merge_depth: int = 0,
) -> tuple[dict[str, Any], str, int]:
    active_model = model_override or runtime.current_model
    planned_batches = _plan_synthesis_batches(runtime, category_name, synthesis_items, model_id=active_model.model_id)
    partial_summaries: list[dict[str, Any]] = []
    total_batch_count = 0
    model_used = active_model.model_id

    for index, batch in enumerate(planned_batches, start=1):
        batch_label = batch_label_prefix if len(planned_batches) == 1 else f"{batch_label_prefix}{index}"
        try:
            payload, batch_model = _invoke_synthesis_batch(
                runtime,
                category_name,
                batch,
                batch_label=batch_label,
                bullet_limit=bullet_limit,
                scope_kind=scope_kind,
                scope_title=scope_title,
                model_override=model_override,
            )
            partial_summaries.append(_summary_to_synthesis_item(payload, len(batch), batch_label))
            model_used = batch_model
            total_batch_count += 1
        except Exception as exc:
            classification = _classify_exception(exc)
            if len(batch) > 1 and classification in {"payload_too_large", "invalid_json", "unexpected_error"}:
                if classification == "payload_too_large":
                    runtime.record_split(category_name, "response_413")
                runtime.tighten_category_budget(category_name, classification, batch_kind="synthesis")
                left, right = _split_batch(batch)
                left_payload, left_model, left_count = _synthesize_summary_items(
                    runtime,
                    category_name,
                    left,
                    batch_label_prefix=f"{batch_label}a",
                    bullet_limit=bullet_limit,
                    scope_kind=scope_kind,
                    scope_title=scope_title,
                    model_override=model_override,
                )
                right_payload, right_model, right_count = _synthesize_summary_items(
                    runtime,
                    category_name,
                    right,
                    batch_label_prefix=f"{batch_label}b",
                    bullet_limit=bullet_limit,
                    scope_kind=scope_kind,
                    scope_title=scope_title,
                    model_override=model_override,
                )
                partial_summaries.extend(
                    [
                        _summary_to_synthesis_item(left_payload, len(left), f"{batch_label}a"),
                        _summary_to_synthesis_item(right_payload, len(right), f"{batch_label}b"),
                    ]
                )
                model_used = right_model or left_model
                total_batch_count += left_count + right_count
                continue
            raise

    if len(partial_summaries) == 1:
        summary = partial_summaries[0]
        return {
            "key_developments": summary["key_developments"],
            "named_entities": summary["named_entities"],
        }, model_used, total_batch_count

    next_merge_depth = merge_depth + 1
    runtime.note_synthesis_merge_depth(category_name, next_merge_depth)
    if next_merge_depth > MAX_SYNTHESIS_MERGE_DEPTH:
        runtime.mark_degraded_merge(category_name, f"merge_depth_cap:{MAX_SYNTHESIS_MERGE_DEPTH}")
        LOGGER.info(
            "Using local degraded merge for category %s scope=%s after reaching synthesis merge depth cap %s.",
            category_name,
            scope_title,
            MAX_SYNTHESIS_MERGE_DEPTH,
        )
        return _merge_summary_items_locally(partial_summaries, bullet_limit), model_used, total_batch_count

    try:
        merged_payload, merged_model, merged_count = _synthesize_summary_items(
            runtime,
            category_name,
            partial_summaries,
            batch_label_prefix=f"{batch_label_prefix}-merge{next_merge_depth}",
            bullet_limit=bullet_limit,
            scope_kind=scope_kind,
            scope_title=scope_title,
            model_override=model_override,
            merge_depth=next_merge_depth,
        )
        return merged_payload, merged_model, total_batch_count + merged_count
    except SynthesisBudgetExceeded:
        runtime.mark_degraded_merge(category_name, "synthesis_budget_exhausted")
        LOGGER.info(
            "Using local degraded merge for category %s scope=%s after exhausting synthesis budget.",
            category_name,
            scope_title,
        )
        return _merge_summary_items_locally(partial_summaries, bullet_limit), model_used, total_batch_count


def _invoke_synthesis_batch(
    runtime: AnalysisRuntime,
    category_name: str,
    synthesis_items: list[dict[str, Any]],
    *,
    batch_label: str,
    bullet_limit: int,
    scope_kind: str,
    scope_title: str,
    model_override: ModelConfig | None = None,
) -> tuple[dict[str, Any], str]:
    active_model = model_override or runtime.current_model
    messages = _build_synthesis_messages(
        category_name,
        synthesis_items,
        bullet_limit=bullet_limit,
        scope_kind=scope_kind,
        scope_title=scope_title,
        market_context=runtime.market_context_string,
    )
    estimated_input_tokens = _estimate_messages_tokens(messages)
    context = BatchContext(
        category_name=category_name,
        batch_kind="synthesis",
        batch_label=batch_label,
        article_count=len(synthesis_items),
        estimated_input_tokens=estimated_input_tokens,
        serialized_request_bytes=_estimate_request_payload_bytes(active_model.model_id, messages, active_model.max_completion_tokens),
        content_shrunk=False,
    )
    return _invoke_json_with_retry(runtime, messages, estimated_input_tokens, context, model_override=model_override)


def _build_synthesis_messages(
    category_name: str,
    synthesis_items: list[dict[str, Any]],
    *,
    bullet_limit: int,
    scope_kind: str,
    scope_title: str,
    market_context: str = "",
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a financial news analyst. Return one valid JSON object only. "
                f"Summarize the {scope_kind} using only the provided article analyses or prior summary blocks."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": f"Synthesize one {scope_kind} from already-analyzed article results.",
                    "required_schema": {
                        "key_developments": [f"2 to {bullet_limit} developments — each must contain a specific detail such as a figure, causal link, named company, or concrete outcome; avoid abstract statements"],
                        "named_entities": [
                            {"name": "entity name", "type": "person|company|country|institution|index|organization|asset|other"}
                        ],
                    },
                    "rules": [
                        "If market_context data is provided, incorporate specific price movements into developments where relevant.",
                    ],
                    "category": category_name,
                    "scope_title": scope_title,
                    "inputs": synthesis_items,
                    "market_context": market_context,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _build_grouping_messages(category_name: str, grouping_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profile = _section_profile(category_name)
    return [
        {
            "role": "system",
            "content": (
                "You are a financial news analyst. Return one valid JSON object only. "
                "Assign every article to one thematic subgroup. Do not omit any article."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Create thematic subgroups inside one financial news category from compact article analyses.",
                    "required_schema": {
                        "subgroups": [
                            {
                                "title": "short subgroup title",
                                "theme_rationale": "one sentence theme explanation",
                                "article_keys": ["article key strings"],
                            }
                        ]
                    },
                    "rules": [
                        "Every input article must appear in exactly one subgroup.",
                        "Avoid many tiny subgroups; focus on 2-3 broad thematic clusters.",
                        f"Target around {profile.subgroup_target_size} article(s) per subgroup when possible.",
                        "Use the provided theme and attention tier as hints when forming subgroups.",
                    ],
                    "category": category_name,
                    "analysis_profile": profile.name,
                    "articles": grouping_items,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _plan_category_batches(
    runtime: AnalysisRuntime,
    category_name: str,
    prepared_articles: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    budget_state = runtime.get_category_budget(category_name)
    profile = _section_profile(category_name)
    planned: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for article in prepared_articles:
        # Bypassing LLM analysis for light news
        if str(article.get("attention_tier") or "").lower() == "light":
            continue

        if current and _batch_attention_tier(current) != _batch_attention_tier([article]):
            planned.append(current)
            current = [article]
            continue
        candidate = current + [article]
        effective_budget = _effective_batch_budget(profile, budget_state, candidate)
        if current and not _batch_within_budget(
            category_name,
            candidate,
            input_budget_tokens=effective_budget["input_budget_tokens"],
            request_byte_budget=effective_budget["request_byte_budget"],
            model_id=runtime.current_model.model_id,
        ):
            runtime.record_split(category_name, "pre_send_budget")
            LOGGER.info(
                "Planning smaller batch for category %s before send: current_articles=%s next_article=%s.",
                category_name,
                len(current),
                article.get("source_article_id") or article.get("canonical_url"),
            )
            planned.append(current)
            current = [article]
        else:
            current = candidate

    if current:
        planned.append(current)
    return planned or [prepared_articles]


def _order_articles_for_analysis(prepared_articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        prepared_articles,
        key=lambda article: (
            ATTENTION_TIER_RANK.get(str(article.get("attention_tier") or "medium"), ATTENTION_TIER_RANK["medium"]),
            -(int(article.get("must_keep") or False)),
            article.get("published_at") or "",
        ),
    )


def _batch_attention_tier(batch_articles: list[dict[str, Any]]) -> str:
    if not batch_articles:
        return "medium"
    return min(
        (
            str(article.get("attention_tier") or "medium")
            for article in batch_articles
        ),
        key=lambda tier: ATTENTION_TIER_RANK.get(tier, ATTENTION_TIER_RANK["medium"]),
    )


def _effective_batch_budget(
    profile: SectionProfile,
    budget_state: CategoryBudgetState,
    batch_articles: list[dict[str, Any]],
) -> dict[str, int]:
    tier = _batch_attention_tier(batch_articles)
    input_budget = budget_state.article_input_budget_tokens
    request_budget = budget_state.article_request_byte_budget
    salvage_depth = profile.salvage_max_depth
    if tier == "light":
        if profile.name != "light":
            input_budget = min(input_budget, max(MIN_INPUT_BUDGET_TOKENS, int(profile.article_input_budget_tokens * 0.7)))
            request_budget = min(request_budget, max(MIN_REQUEST_BYTE_BUDGET, int(profile.article_request_byte_budget * 0.7)))
        salvage_depth = max(2, profile.salvage_max_depth - (0 if profile.name == "light" else 1))
    elif tier == "high":
        salvage_depth = profile.salvage_max_depth + 1
    return {
        "input_budget_tokens": input_budget,
        "request_byte_budget": request_budget,
        "salvage_max_depth": salvage_depth,
    }


def _article_key_points_limit(category_name: str, article: dict[str, Any]) -> int:
    profile = _section_profile(category_name)
    if str(article.get("attention_tier") or "medium") == "light":
        return min(profile.article_key_points_limit, 2)
    return profile.article_key_points_limit


def _article_entity_limit(category_name: str, article: dict[str, Any]) -> int:
    profile = _section_profile(category_name)
    if str(article.get("attention_tier") or "medium") == "light":
        return min(profile.entity_limit, 4)
    return profile.entity_limit


def _plan_synthesis_batches(
    runtime: AnalysisRuntime,
    category_name: str,
    synthesis_items: list[dict[str, Any]],
    *,
    model_id: str,
) -> list[list[dict[str, Any]]]:
    budget_state = runtime.get_category_budget(category_name)
    planned: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for item in synthesis_items:
        candidate = current + [item]
        if current and not _synthesis_within_budget(
            category_name,
            candidate,
            input_budget_tokens=budget_state.synthesis_input_budget_tokens,
            request_byte_budget=budget_state.synthesis_request_byte_budget,
            model_id=model_id,
        ):
            runtime.record_split(category_name, "pre_send_budget")
            planned.append(current)
            current = [item]
        else:
            current = candidate

    if current:
        planned.append(current)
    return planned or [synthesis_items]


def _article_to_synthesis_item(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "article",
        "source_article_id": article["source_article_id"],
        "canonical_url": article["canonical_url"],
        "title": article["title"],
        "published_at": article["published_at"],
        "attention_tier": article.get("attention_tier"),
        "theme": article.get("theme"),
        "must_keep": article.get("must_keep"),
        "scores": {
            "novelty_score": article["novelty_score"],
            "relevance_score": article["relevance_score"],
            "urgency_score": article["urgency_score"],
        },
        "named_entities": article["named_entities"][: _article_entity_limit(str(article.get("section") or ""), article)],
        "key_points": article["key_points"][: _article_key_points_limit(str(article.get("section") or ""), article)],
    }


def _article_to_grouping_item(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "article_key": _article_group_key(article),
        "source_article_id": article["source_article_id"],
        "canonical_url": article["canonical_url"],
        "title": article["title"],
        "published_at": article["published_at"],
        "attention_tier": article.get("attention_tier"),
        "theme": article.get("theme"),
        "must_keep": article.get("must_keep"),
        "attention_reason": article.get("attention_reason"),
        "scores": {
            "novelty_score": article["novelty_score"],
            "relevance_score": article["relevance_score"],
            "urgency_score": article["urgency_score"],
        },
        "named_entities": article["named_entities"][:4],
        "key_points": article["key_points"][:2],
    }


def _summary_to_synthesis_item(payload: dict[str, Any], article_count: int, label: str) -> dict[str, Any]:
    return {
        "kind": "summary",
        "label": label,
        "article_count": article_count,
        "key_developments": _normalize_string_list(payload.get("key_developments"), limit=5),
        "named_entities": _normalize_entities(payload.get("named_entities"))[:6],
    }


def _merge_summary_items_locally(summary_items: list[dict[str, Any]], bullet_limit: int) -> dict[str, Any]:
    key_developments: list[str] = []
    seen_developments: set[str] = set()
    seen_entities: set[str] = set()
    named_entities: list[dict[str, str]] = []

    for item in summary_items:
        for development in _normalize_string_list(item.get("key_developments"), limit=bullet_limit):
            normalized = development.strip()
            if not normalized or normalized in seen_developments:
                continue
            seen_developments.add(normalized)
            key_developments.append(normalized)
            if len(key_developments) >= bullet_limit:
                break
        for entity in _normalize_entities(item.get("named_entities")):
            name = str(entity.get("name") or "").strip()
            if not name or name in seen_entities:
                continue
            seen_entities.add(name)
            named_entities.append(entity)
        if len(key_developments) >= bullet_limit:
            break

    return {
        "key_developments": key_developments[:bullet_limit],
        "named_entities": named_entities[:6],
    }


def _normalize_grouping_payload(
    payload: dict[str, Any],
    successful_articles: list[dict[str, Any]],
    category_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    article_lookup = {_article_group_key(article): article for article in successful_articles}
    assigned_keys: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, subgroup in enumerate(payload.get("subgroups") or [], start=1):
        if not isinstance(subgroup, dict):
            continue
        subgroup_keys = []
        subgroup_articles = []
        for raw_key in subgroup.get("article_keys") or []:
            key = str(raw_key or "").strip()
            if not key or key in assigned_keys:
                continue
            article = article_lookup.get(key)
            if article is None:
                continue
            assigned_keys.add(key)
            subgroup_keys.append(key)
            subgroup_articles.append(article)
        if not subgroup_articles:
            continue
        normalized.append(
            {
                "title": str(subgroup.get("title") or f"{category_name} subgroup {index}").strip(),
                "theme_rationale": str(subgroup.get("theme_rationale") or "Grouped by related theme.").strip(),
                "articles": subgroup_articles,
            }
        )

    missing_articles = [article for key, article in article_lookup.items() if key not in assigned_keys]
    return normalized, missing_articles


def _consolidate_subgroups(
    subgroups: list[dict[str, Any]],
    target_count: int = 3,
    min_articles: int = 3
) -> list[dict[str, Any]]:
    """Merges smaller subgroups into larger ones if fragmentations are too high."""
    if len(subgroups) <= target_count:
        return subgroups
        
    to_keep = []
    to_merge = []
    for sg in subgroups:
        if len(sg.get("articles") or []) < min_articles:
            to_merge.append(sg)
        else:
            to_keep.append(sg)
            
    if not to_merge:
        return subgroups
        
    if not to_keep:
        # All were small, just return them as one big group
        merged_articles = []
        for sg in to_merge:
            merged_articles.extend(sg["articles"])
        return [{
            "title": "Combined Market Overview",
            "theme_rationale": "Consolidated smaller themes for brevity.",
            "articles": merged_articles
        }]
        
    # Merge small ones into the largest existing subgroup or the first one
    to_keep.sort(key=lambda x: len(x["articles"]), reverse=True)
    target_sg = to_keep[0]
    for sg in to_merge:
        target_sg["articles"].extend(sg["articles"])
    
    target_sg["title"] += " & Other News"
    return to_keep


def _fallback_subgroups(category_name: str, successful_articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profile = _section_profile(category_name)
    if not successful_articles:
        return []
    subgroups: list[dict[str, Any]] = []
    chunk_size = max(1, profile.subgroup_target_size)
    for index in range(0, len(successful_articles), chunk_size):
        chunk = successful_articles[index : index + chunk_size]
        subgroups.append(
            {
                "title": f"{category_name} subgroup {len(subgroups) + 1}",
                "theme_rationale": "Fallback subgroup created from nearby articles after subgroup assignment needed to be localized.",
                "articles": chunk,
            }
        )
    return subgroups


def _shrink_batch_to_budget(
    category_name: str,
    batch_articles: list[dict[str, Any]],
    *,
    input_budget_tokens: int,
    request_byte_budget: int,
    model_id: str,
) -> bool:
    shrunk = False
    while True:
        estimated_tokens = _estimate_batch_request_tokens(category_name, batch_articles)
        estimated_bytes = _estimate_batch_request_bytes(category_name, batch_articles, model_id)
        if estimated_tokens <= input_budget_tokens and estimated_bytes <= request_byte_budget:
            break
        candidate = _largest_shrinkable_article(batch_articles)
        if candidate is None:
            break
        new_length = max(CATEGORY_MIN_CONTENT_CHARS, candidate["analyzed_content_length_chars"] - CATEGORY_SHRINK_STEP_CHARS)
        _apply_category_shrink(candidate, new_length)
        shrunk = True
    return shrunk


def _estimate_batch_request_tokens(category_name: str, batch_articles: list[dict[str, Any]]) -> int:
    return _estimate_messages_tokens(_build_article_batch_messages(category_name, batch_articles))


def _estimate_batch_request_bytes(category_name: str, batch_articles: list[dict[str, Any]], model_id: str) -> int:
    return _estimate_request_payload_bytes(
        model_id,
        _build_article_batch_messages(category_name, batch_articles),
        DEFAULT_OUTPUT_TOKENS,
    )


def _estimate_synthesis_request_tokens(category_name: str, synthesis_items: list[dict[str, Any]]) -> int:
    profile = _section_profile(category_name)
    return _estimate_messages_tokens(
        _build_synthesis_messages(
            category_name,
            synthesis_items,
            bullet_limit=profile.category_bullet_limit,
            scope_kind="category",
            scope_title=category_name,
        )
    )


def _estimate_synthesis_request_bytes(category_name: str, synthesis_items: list[dict[str, Any]], model_id: str) -> int:
    profile = _section_profile(category_name)
    return _estimate_request_payload_bytes(
        model_id,
        _build_synthesis_messages(
            category_name,
            synthesis_items,
            bullet_limit=profile.category_bullet_limit,
            scope_kind="category",
            scope_title=category_name,
        ),
        DEFAULT_OUTPUT_TOKENS,
    )


def _batch_within_budget(
    category_name: str,
    batch_articles: list[dict[str, Any]],
    *,
    input_budget_tokens: int,
    request_byte_budget: int,
    model_id: str,
) -> bool:
    return (
        _estimate_batch_request_tokens(category_name, batch_articles) <= input_budget_tokens
        and _estimate_batch_request_bytes(category_name, batch_articles, model_id) <= request_byte_budget
    )


def _synthesis_within_budget(
    category_name: str,
    synthesis_items: list[dict[str, Any]],
    *,
    input_budget_tokens: int,
    request_byte_budget: int,
    model_id: str,
) -> bool:
    return (
        _estimate_synthesis_request_tokens(category_name, synthesis_items) <= input_budget_tokens
        and _estimate_synthesis_request_bytes(category_name, synthesis_items, model_id) <= request_byte_budget
    )


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


def _largest_shrinkable_article(batch_articles: list[dict[str, Any]]) -> dict[str, Any] | None:
    shrinkable = [article for article in batch_articles if article["analyzed_content_length_chars"] > CATEGORY_MIN_CONTENT_CHARS]
    if not shrinkable:
        return None
    return max(shrinkable, key=lambda article: article["analyzed_content_length_chars"])


def _apply_category_shrink(article: dict[str, Any], new_length: int) -> None:
    original_text = article.get("content_text") or ""
    truncated_text = original_text[: max(new_length, 0)].rstrip()
    original_length = article.get("original_content_length_chars", len(original_text))
    analyzed_length = len(truncated_text)
    article["content_text"] = truncated_text
    article["content_truncated"] = analyzed_length < original_length
    article["analysis_method"] = "truncated_text" if analyzed_length < original_length else article.get("analysis_method", "full_text")
    article["analyzed_content_length_chars"] = analyzed_length
    article["analyzed_content_token_estimate"] = _estimate_tokens(truncated_text)
    if analyzed_length < original_length:
        article["truncation_reason"] = "Category batch request exceeded the working request budget; article content was shortened to fit."


def _split_batch(batch_articles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    midpoint = max(1, len(batch_articles) // 2)
    return batch_articles[:midpoint], batch_articles[midpoint:]


def _merge_batch_article_results(
    batch_articles: list[dict[str, Any]],
    payload: dict[str, Any],
    model_used: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    category_name = batch_articles[0].get("section") if batch_articles else ""
    profile = _section_profile(str(category_name or ""))
    response_articles = payload.get("articles")
    response_items = response_articles if isinstance(response_articles, list) else []
    response_by_key: dict[tuple[str | None, str], dict[str, Any]] = {}
    for item in response_items:
        if isinstance(item, dict):
            response_by_key[_article_key(item.get("source_article_id"), item.get("canonical_url"))] = item

    merged_results: list[dict[str, Any]] = []
    missing_articles: list[dict[str, Any]] = []
    for article in batch_articles:
        key = _article_key(article.get("source_article_id"), article.get("canonical_url"))
        match = response_by_key.get(key)
        
        # Fallback 1: Match by source_article_id only if URL was tweaked by LLM
        if match is None and article.get("source_article_id"):
            for r_key, r_item in response_by_key.items():
                if r_key[0] == article.get("source_article_id"):
                    match = r_item
                    break
        
        if match is None:
            missing_articles.append(article)
            continue

        merged_results.append(
            {
                "source_article_id": article.get("source_article_id"),
                "title": article.get("title"),
                "canonical_url": article.get("canonical_url"),
                "published_at": article.get("published_at"),
                "section": article.get("section"),
                "novelty_score": _coerce_score(match.get("novelty_score")),
                "relevance_score": _coerce_score(match.get("relevance_score")),
                "urgency_score": _coerce_score(match.get("urgency_score")),
                "named_entities": _normalize_entities(match.get("named_entities"))[: _article_entity_limit(str(category_name or ""), article)],
                "key_points": _normalize_string_list(match.get("key_points"), limit=_article_key_points_limit(str(category_name or ""), article)),
                "content_truncated": article.get("content_truncated"),
                "original_content_length_chars": article.get("original_content_length_chars"),
                "analyzed_content_length_chars": article.get("analyzed_content_length_chars"),
                "original_content_token_estimate": article.get("original_content_token_estimate"),
                "analyzed_content_token_estimate": article.get("analyzed_content_token_estimate"),
                "truncation_reason": article.get("truncation_reason"),
                "analysis_method": article.get("analysis_method"),
                "attention_tier": article.get("attention_tier"),
                "theme": article.get("theme"),
                "attention_reason": article.get("attention_reason"),
                "must_keep": article.get("must_keep"),
                "model_used": model_used,
                "delayed_retry_attempted": False,
                "delayed_retry_model_chain": [],
                "delayed_retry_final_model": None,
                "error_classification": None,
                "error": None,
            }
        )

    return merged_results, missing_articles


def _order_results_like_input(
    batch_articles: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results_by_key = {
        _article_key(result.get("source_article_id"), result.get("canonical_url")): result
        for result in results
    }
    ordered: list[dict[str, Any]] = []
    for article in batch_articles:
        key = _article_key(article.get("source_article_id"), article.get("canonical_url"))
        result = results_by_key.get(key)
        if result is not None:
            ordered.append(result)
    return ordered


def _build_failed_article_result(
    *,
    article: dict[str, Any],
    error_message: str,
    model_used: str,
    error_classification: str,
) -> dict[str, Any]:
    return {
        "source_article_id": article.get("source_article_id"),
        "title": article.get("title"),
        "canonical_url": article.get("canonical_url"),
        "published_at": article.get("published_at"),
        "section": article.get("section"),
        "novelty_score": None,
        "relevance_score": None,
        "urgency_score": None,
        "named_entities": [],
        "key_points": [],
        "content_truncated": article.get("content_truncated"),
        "original_content_length_chars": article.get("original_content_length_chars"),
        "analyzed_content_length_chars": article.get("analyzed_content_length_chars"),
        "original_content_token_estimate": article.get("original_content_token_estimate"),
        "analyzed_content_token_estimate": article.get("analyzed_content_token_estimate"),
        "truncation_reason": article.get("truncation_reason"),
        "analysis_method": article.get("analysis_method"),
        "attention_tier": article.get("attention_tier"),
        "theme": article.get("theme"),
        "attention_reason": article.get("attention_reason"),
        "must_keep": article.get("must_keep"),
        "model_used": model_used,
        "delayed_retry_attempted": False,
        "delayed_retry_model_chain": [],
        "delayed_retry_final_model": None,
        "error_classification": error_classification,
        "error": error_message,
    }


def _group_source_articles(articles: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Ensure ordered categories exist even if empty
    for category in CATEGORY_ORDER:
        grouped[category] = []
        
    for article in articles:
        section_name = article.get("article_section") or "未分類"
        if section_name in {"二手市場", "新盤情報"}:
            section_name = "地產新聞"
        grouped[section_name].append(article)

    fixed_order = {name: index for index, name in enumerate(CATEGORY_ORDER)}
    ordered_categories = sorted(grouped, key=lambda category: (fixed_order.get(category, len(CATEGORY_ORDER)), category))
    for category in ordered_categories:
        grouped[category].sort(key=lambda item: (item.get("published_at") or "", item.get("id") or 0), reverse=True)
    return [(category, grouped[category]) for category in ordered_categories]


def _collect_entities_from_articles(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    combined: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        for entity in item.get("named_entities") or []:
            key = (entity["name"], entity["type"])
            if key in seen:
                continue
            seen.add(key)
            combined.append(entity)
    return combined


def _category_model_used(article_results: list[dict[str, Any]]) -> str:
    model_ids = {article.get("model_used") for article in article_results if article.get("model_used")}
    if not model_ids:
        return PRIMARY_MODEL_ID
    if len(model_ids) == 1:
        return next(iter(model_ids))
    return ",".join(sorted(model_ids))


def _build_empty_report(target_date: str, *, incremental: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_date": target_date,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_site": DEFAULT_SOURCE_SITE,
        "status": "empty",
        "model": {
            "provider": DEFAULT_PROVIDER,
            "primary_model": PRIMARY_MODEL_ID,
            "fallback_models": list(FALLBACK_MODEL_IDS),
        },
        "model_switches": [],
        "input": {
            "article_count": 0,
            "category_count": 0,
        },
        "diagnostics": RuntimeDiagnostics().as_dict(),
        "incremental": incremental
        or {
            "reused_successful_articles": 0,
            "new_articles_analyzed": 0,
            "retried_previous_day_articles": 0,
            "previous_day_retry_successes": 0,
        },
        "totals": {
            "article_count": 0,
            "successful_article_analyses": 0,
            "failed_article_analyses": 0,
            "full_text_article_count": 0,
            "truncated_article_count": 0,
            "successful_categories": 0,
            "partial_categories": 0,
            "failed_categories": 0,
        },
        "market_context": [],
        "categories": [],
        "unresolved_articles": [],
        "errors": [],
    }


def _build_truncated_selection(
    normalized_text: str,
    original_length: int,
    original_token_estimate: int,
    target_length: int,
    reason: str,
) -> dict[str, Any]:
    truncated_text = normalized_text[: max(target_length, 0)].rstrip()
    analyzed_length = len(truncated_text)
    return {
        "content_text": truncated_text,
        "content_truncated": analyzed_length < original_length,
        "analysis_method": "truncated_text" if analyzed_length < original_length else "full_text",
        "original_content_length_chars": original_length,
        "analyzed_content_length_chars": analyzed_length,
        "original_content_token_estimate": original_token_estimate,
        "analyzed_content_token_estimate": _estimate_tokens(truncated_text),
        "truncation_reason": reason if analyzed_length < original_length else None,
    }


def _clone_prepared_article(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_article_id": article.get("source_article_id"),
        "title": article.get("title"),
        "canonical_url": article.get("canonical_url"),
        "published_at": article.get("published_at"),
        "section": article.get("section"),
        "summary_snippet": article.get("summary_snippet"),
        "content_text": article.get("content_text"),
        "content_truncated": article.get("content_truncated"),
        "analysis_method": article.get("analysis_method"),
        "original_content_length_chars": article.get("original_content_length_chars"),
        "analyzed_content_length_chars": article.get("analyzed_content_length_chars"),
        "original_content_token_estimate": article.get("original_content_token_estimate"),
        "analyzed_content_token_estimate": article.get("analyzed_content_token_estimate"),
        "truncation_reason": article.get("truncation_reason"),
        "attention_tier": article.get("attention_tier"),
        "theme": article.get("theme"),
        "attention_reason": article.get("attention_reason"),
        "must_keep": article.get("must_keep"),
    }


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


def _normalize_entities(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if isinstance(item, str):
            name = item.strip()
            entity_type = "other"
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            entity_type = str(item.get("type") or "other").strip().lower() or "other"
        else:
            continue

        if not name:
            continue
        if entity_type not in ENTITY_TYPES:
            entity_type = "other"

        key = (name, entity_type)
        if key in seen:
            continue
        seen.add(key)
        entities.append({"name": name, "type": entity_type})
    return entities


def _normalize_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _coerce_score(value: Any) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 5
    return max(1, min(10, numeric))


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


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


def _retry_delay_seconds(response: requests.Response, attempt: int) -> float:
    retry_after = _parse_retry_after_seconds(response.headers.get("Retry-After"))
    if retry_after is not None:
        return max(1.0, retry_after)
    base_delay = min(15.0, 2.0 * (2**attempt))
    return base_delay + random.uniform(0.0, 0.5)


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return _parse_duration_seconds(value)


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


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


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


def _article_key(source_article_id: Any, canonical_url: Any) -> tuple[str | None, str]:
    normalized_id = str(source_article_id) if source_article_id not in {None, ""} else None
    normalized_url = str(canonical_url or "")
    return normalized_id, normalized_url


def _article_group_key(article: dict[str, Any]) -> str:
    source_article_id = str(article.get("source_article_id") or "").strip()
    if source_article_id:
        return source_article_id
    return str(article.get("canonical_url") or "").strip()


def _build_market_context_for_report(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from .market import build_market_context_for_report

    return build_market_context_for_report(snapshots)


def _section_profile(category_name: str) -> SectionProfile:
    return LIGHT_SECTION_PROFILE if category_name in LIGHT_ANALYSIS_SECTIONS else STANDARD_SECTION_PROFILE
def _generate_top_alerts(
    runtime: AnalysisRuntime,
    market_context: str,
    developments: list[dict[str, Any]],
    article_metadata: dict[str, dict[str, str]],
    is_incremental: bool = False,
) -> list[str]:
    """Uses LLM to pick exactly 3 most critical macro-moving developments."""
    import json

    morning_briefing = "morning briefing" if not is_incremental else "INTRA-DAY UPDATE session"
    system_prompt = (
        f"You are a Chief Investment Officer. Your task is to pick the 3 most critical news developments "
        f"from the following list for a {morning_briefing}. Your goal is high 'Alpha' — prioritize events "
        "that have the greatest macro impact, structural significance, or immediate market urgency.\n\n"
        "Guidelines:\n"
        "1. Pick EXACTLY 3 developments. If fewer than 3 are significant, pick the best available.\n"
        "2. Ensure each alert includes specific data (prices, % changes, specific figures) referenced in the news.\n"
        "3. Do not invent causal links; only cite the facts and their significance.\n"
        "4. Your output must be a valid JSON object with a single key 'top_alerts' containing a list of strings. "
        "Each string must end with the exact source article title, full date/time, and URL in the following format: (Article Title | YYYY-MM-DD HH:MM) {URL}."
    )

    user_payload = {
        "market_context": market_context,
        "developments": developments,
    }

    try:
        model = runtime.primary_model
        api_url = model.api_url or GROQ_CHAT_COMPLETIONS_URL
        session = runtime.get_session_for_model(model)
        
        response = session.post(
            api_url,
            json={
                "model": model.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = _parse_json_content(content)
        alerts_data = list(parsed.get("top_alerts") or [])[:3]
        
        results: list[str] = []
        for item in alerts_data:
            summary_text = str(item.get("summary") or "").strip()
            ref_id = str(item.get("ref_id") or "").strip()
            meta = article_metadata.get(ref_id)
            if meta and summary_text:
                title = meta["title"]
                date = meta["date"]
                url = meta["url"]
                results.append(f"{summary_text} ({title} | {date}) {{{url}}}")
            elif summary_text:
                results.append(summary_text)
        
        return results
    except Exception as e:
        LOGGER.warning("Alert generation LLM call failed: %s", e)
        # Simple fallback: pick first 3 from input if LLM fails
        return [d["text"] for d in developments[:3]]
