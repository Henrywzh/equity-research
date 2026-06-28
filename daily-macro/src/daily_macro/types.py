"""Shared type definitions, constants, and dataclasses for the daily-macro analysis pipeline.

This module is a pure leaf — it has no runtime imports from within `daily_macro`.
Everything here is safe to import from any other module without risk of circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

try:
    from typing import NotRequired
except ImportError:
    try:
        from typing_extensions import NotRequired
    except ImportError:
        # Fallback for Python < 3.11 without typing_extensions
        from typing import Optional as NotRequired  # type: ignore[assignment]

if TYPE_CHECKING:
    # Only resolved at type-check time; avoids circular import at runtime
    # (AnalysisRuntime lives in analysis.py / llm_client.py which imports from this module).
    from .llm_client import AnalysisRuntime  # noqa: F401

# ---------------------------------------------------------------------------
# API / provider constants
# ---------------------------------------------------------------------------

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_PROVIDER = "groq"
PRIMARY_MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_MODEL_IDS = ["qwen/qwen3-32b", "llama-3.1-8b-instant"]
DELAYED_RETRY_FINAL_MODEL_ID = "openai/gpt-oss-20b"

# ---------------------------------------------------------------------------
# Report constants
# ---------------------------------------------------------------------------

REPORT_FILE_NAME = "hkej-news-analysis.json"
REPORT_SCHEMA_VERSION = 6

# ---------------------------------------------------------------------------
# Timing / retry constants
# ---------------------------------------------------------------------------

DELAYED_RETRY_WAIT_SECONDS = 60.0
MAX_CATEGORY_SYNTHESIS_WAIT_SECONDS = 1800.0
MAX_CATEGORY_SYNTHESIS_RETRIES = 24
MAX_SYNTHESIS_MERGE_DEPTH = 3
DEFAULT_CHAT_RETRIES = 4

# ---------------------------------------------------------------------------
# Token / byte budget constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_TOKENS = 1200
DEFAULT_INPUT_BUDGET_TOKENS = 4500
DEFAULT_SYNTHESIS_INPUT_BUDGET_TOKENS = 2400
DEFAULT_PROMPT_OVERHEAD_TOKENS = 900
SHORT_ARTICLE_FULL_TEXT_THRESHOLD = 1500
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

# ---------------------------------------------------------------------------
# Category / section configuration
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# SectionProfile dataclass and instances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
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

# ---------------------------------------------------------------------------
# AnalysisGraphState TypedDict
# ---------------------------------------------------------------------------


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
    runtime: Any  # AnalysisRuntime at runtime; typed as Any to avoid circular import with llm_client
    category_reports: list[dict[str, Any]]
    previous_day_retry_successes: int
    updated_previous_report: NotRequired[dict[str, Any] | None]
    incremental: dict[str, int]
    market_context_string: str
    macro_release_digest: dict[str, Any]
    top_alerts: list[str]
    legacy_executive_summary: list[str]
    validation_issues: NotRequired[list[dict[str, Any]]]
    theme_memory: NotRequired[dict[str, Any]]
    report: dict[str, Any]
    total_scraped_articles: int
    newly_analyzed_keys: set[str]
    data_dir: NotRequired[Path]


# ---------------------------------------------------------------------------
# Small model / rate-limit dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    provider: str = DEFAULT_PROVIDER
    max_completion_tokens: int = DEFAULT_OUTPUT_TOKENS
    api_url: str | None = None
    api_key_env: str | None = None


@dataclass
class ModelRateLimitState:
    remaining_requests: int | None = None
    reset_requests_at: float | None = None
    remaining_tokens: int | None = None
    reset_tokens_at: float | None = None


@dataclass
class CategoryBudgetState:
    article_input_budget_tokens: int = DEFAULT_INPUT_BUDGET_TOKENS
    article_request_byte_budget: int = DEFAULT_REQUEST_BYTE_BUDGET
    synthesis_input_budget_tokens: int = DEFAULT_SYNTHESIS_INPUT_BUDGET_TOKENS
    synthesis_request_byte_budget: int = DEFAULT_SYNTHESIS_REQUEST_BYTE_BUDGET


# ---------------------------------------------------------------------------
# Diagnostics dataclasses
# ---------------------------------------------------------------------------


@dataclass
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
    request_timeout_count: int = 0
    avoided_rate_limit_wait_count: int = 0
    avoided_rate_limit_wait_seconds_total: float = 0.0
    degraded_mode_count: int = 0
    resolver_rejections: list[dict[str, Any]] = field(default_factory=list)
    model_task_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    model_substitutions: list[dict[str, Any]] = field(default_factory=list)

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
            "request_timeout_count": self.request_timeout_count,
            "avoided_rate_limit_wait_count": self.avoided_rate_limit_wait_count,
            "avoided_rate_limit_wait_seconds_total": round(self.avoided_rate_limit_wait_seconds_total, 3),
            "degraded_mode_count": self.degraded_mode_count,
            "resolver_rejections": list(self.resolver_rejections),
            "model_task_counts": {task: dict(counts) for task, counts in self.model_task_counts.items()},
            "model_substitutions": list(self.model_substitutions),
        }


@dataclass
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


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class SynthesisBudgetExceeded(RuntimeError):
    """Raised when a category has exhausted its allowed synthesis retry/wait budget."""


# ---------------------------------------------------------------------------
# BatchContext dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchContext:
    category_name: str
    batch_kind: str
    batch_label: str
    article_count: int
    estimated_input_tokens: int
    serialized_request_bytes: int
    content_shrunk: bool = False
    llm_task: str = "article_analysis"


# ---------------------------------------------------------------------------
# Section profile helper
# ---------------------------------------------------------------------------


def _section_profile(category_name: str) -> SectionProfile:
    """Return the SectionProfile for a given category name."""
    return LIGHT_SECTION_PROFILE if category_name in LIGHT_ANALYSIS_SECTIONS else STANDARD_SECTION_PROFILE
