from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from .config import DEFAULT_SOURCE_SITE, get_analysis_dir, get_data_dir, get_db_path, get_project_root
from .storage import Storage

LOGGER = logging.getLogger(__name__)

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
REPORT_FILE_NAME = "hkej-news-analysis.json"
REPORT_SCHEMA_VERSION = 4
DEFAULT_PROVIDER = "groq"
PRIMARY_MODEL_ID = "qwen/qwen3-32b"
FALLBACK_MODEL_ID = "llama-3.1-8b-instant"
DEFAULT_OUTPUT_TOKENS = 1200
DEFAULT_INPUT_BUDGET_TOKENS = 4500
DEFAULT_PROMPT_OVERHEAD_TOKENS = 900
SHORT_ARTICLE_FULL_TEXT_THRESHOLD = 1500
DEFAULT_CHAT_RETRIES = 4
RATE_LIMIT_REQUEST_FLOOR = 0
RATE_LIMIT_TOKEN_FLOOR = 800
DEFAULT_REQUEST_BYTE_BUDGET = 12000
MIN_REQUEST_BYTE_BUDGET = 4000
MIN_INPUT_BUDGET_TOKENS = 1200
CATEGORY_SHRINK_STEP_CHARS = 400
CATEGORY_MIN_CONTENT_CHARS = 0
CATEGORY_ORDER = [
    "港股直擊",
    "香港財經",
    "地產新聞",
    "中國財經",
    "國際財經",
    "時事脈搏",
    "即巿股評",
    "重要通告",
    "港交所通告",
]
ENTITY_TYPES = {"person", "company", "country", "institution", "index", "organization", "asset", "other"}
FAILURE_CLASSIFICATIONS = {
    "payload_too_large",
    "rate_limited",
    "invalid_json",
    "incomplete_model_output",
    "http_error",
    "unexpected_error",
}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    provider: str = DEFAULT_PROVIDER
    max_completion_tokens: int = DEFAULT_OUTPUT_TOKENS


@dataclass(slots=True)
class ModelRateLimitState:
    remaining_requests: int | None = None
    reset_requests_at: float | None = None
    remaining_tokens: int | None = None
    reset_tokens_at: float | None = None


@dataclass(slots=True)
class CategoryBudgetState:
    input_budget_tokens: int = DEFAULT_INPUT_BUDGET_TOKENS
    request_byte_budget: int = DEFAULT_REQUEST_BYTE_BUDGET


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "rate_limit_wait_count": self.rate_limit_wait_count,
            "rate_limit_wait_seconds_total": round(self.rate_limit_wait_seconds_total, 3),
            "fallback_switch_count": self.fallback_switch_count,
            "pre_send_split_count": self.pre_send_split_count,
            "response_413_split_count": self.response_413_split_count,
            "json_repair_retry_count": self.json_repair_retry_count,
            "batch_count": self.batch_count,
            "failed_batch_count": self.failed_batch_count,
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "sub_batch_count": self.sub_batch_count,
            "split_reasons": list(self.split_reasons),
            "models_attempted": list(self.models_attempted),
            "estimated_input_tokens_max": self.estimated_input_tokens_max,
            "serialized_request_bytes_max": self.serialized_request_bytes_max,
            "rate_limit_waits": self.rate_limit_waits,
            "partial_article_count": self.partial_article_count,
        }


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
        self._states: dict[str, ModelRateLimitState] = {}

    def before_request(self, model_id: str, estimated_input_tokens: int = 0) -> float:
        state = self._states.setdefault(model_id, ModelRateLimitState())
        now = self._time_fn()
        delays: list[float] = []

        if (
            state.remaining_requests is not None
            and state.remaining_requests <= RATE_LIMIT_REQUEST_FLOOR
            and state.reset_requests_at is not None
            and state.reset_requests_at > now
        ):
            delays.append(state.reset_requests_at - now)

        if (
            state.remaining_tokens is not None
            and state.remaining_tokens <= estimated_input_tokens + RATE_LIMIT_TOKEN_FLOOR
            and state.reset_tokens_at is not None
            and state.reset_tokens_at > now
        ):
            delays.append(state.reset_tokens_at - now)

        if delays:
            delay_seconds = max(delays)
            self._sleep_fn(delay_seconds)
            return delay_seconds
        return 0.0

    def record_response(self, model_id: str, response: requests.Response) -> None:
        state = self._states.setdefault(model_id, ModelRateLimitState())
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
    session: requests.Session
    governor: RateLimitGovernor
    primary_model: ModelConfig
    fallback_model: ModelConfig
    current_model: ModelConfig
    model_switches: list[dict[str, Any]] = field(default_factory=list)
    last_attempted_model: str | None = None
    diagnostics: RuntimeDiagnostics = field(default_factory=RuntimeDiagnostics)
    category_diagnostics: dict[str, CategoryDiagnostics] = field(default_factory=dict)
    category_budgets: dict[str, CategoryBudgetState] = field(default_factory=dict)

    def switch_to_fallback(self, reason: str) -> None:
        if self.current_model.model_id == self.fallback_model.model_id:
            return
        self.diagnostics.fallback_switch_count += 1
        self.model_switches.append(
            {
                "switched_at": datetime.now().astimezone().isoformat(),
                "from_model": self.current_model.model_id,
                "to_model": self.fallback_model.model_id,
                "reason": reason,
            }
        )
        LOGGER.info(
            "Switching Groq model from %s to %s: %s",
            self.current_model.model_id,
            self.fallback_model.model_id,
            reason,
        )
        self.current_model = self.fallback_model

    def get_category_diagnostics(self, category_name: str) -> CategoryDiagnostics:
        return self.category_diagnostics.setdefault(category_name, CategoryDiagnostics())

    def get_category_budget(self, category_name: str) -> CategoryBudgetState:
        return self.category_budgets.setdefault(category_name, CategoryBudgetState())

    def record_wait(self, category_name: str, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        self.diagnostics.rate_limit_wait_count += 1
        self.diagnostics.rate_limit_wait_seconds_total += delay_seconds
        self.get_category_diagnostics(category_name).rate_limit_waits += 1

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

    def tighten_category_budget(self, category_name: str, classification: str) -> None:
        state = self.get_category_budget(category_name)
        if classification == "payload_too_large":
            state.request_byte_budget = max(MIN_REQUEST_BYTE_BUDGET, state.request_byte_budget // 2)
        elif classification == "rate_limited":
            state.input_budget_tokens = max(MIN_INPUT_BUDGET_TOKENS, state.input_budget_tokens // 2)
            state.request_byte_budget = max(MIN_REQUEST_BYTE_BUDGET, int(state.request_byte_budget * 0.75))
        else:
            state.input_budget_tokens = max(MIN_INPUT_BUDGET_TOKENS, int(state.input_budget_tokens * 0.8))
            state.request_byte_budget = max(MIN_REQUEST_BYTE_BUDGET, int(state.request_byte_budget * 0.8))


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

    if report_path.exists() and not force:
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if cached.get("report_schema_version") == REPORT_SCHEMA_VERSION:
            cached["output_path"] = str(report_path)
            cached["cached"] = True
            return cached

    LOGGER.info("Starting daily analysis for %s.", target_date)

    storage = Storage(resolved_db_path)
    try:
        articles = storage.fetch_published_articles_for_date(target_date, source_site=DEFAULT_SOURCE_SITE)
    finally:
        storage.close()

    if not articles:
        empty_report = _build_empty_report(target_date)
        _write_report(report_path, empty_report)
        empty_report["output_path"] = str(report_path)
        empty_report["cached"] = False
        LOGGER.info("No published articles found for %s.", target_date)
        return empty_report

    runtime = AnalysisRuntime(
        session=_build_groq_session(load_groq_api_key()),
        governor=RateLimitGovernor(),
        primary_model=ModelConfig(PRIMARY_MODEL_ID),
        fallback_model=ModelConfig(FALLBACK_MODEL_ID),
        current_model=ModelConfig(PRIMARY_MODEL_ID),
    )

    category_reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        for category_name, category_articles in _group_source_articles(articles):
            prepared_articles = [_prepare_single_article(article) for article in category_articles]
            category_report, category_errors = _analyze_category(runtime, category_name, prepared_articles)
            category_reports.append(category_report)
            errors.extend(category_errors)
    finally:
        runtime.session.close()

    all_articles = [article for category in category_reports for article in category["articles"]]
    successful_article_analyses = sum(1 for article in all_articles if not article.get("error"))
    failed_article_analyses = len(all_articles) - successful_article_analyses
    full_text_count = sum(1 for article in all_articles if not article.get("content_truncated"))
    truncated_count = len(all_articles) - full_text_count
    successful_categories = sum(1 for category in category_reports if category["status"] == "success")
    partial_categories = sum(1 for category in category_reports if category["status"] == "partial")
    failed_categories = sum(1 for category in category_reports if category["status"] == "failed")

    status = "success"
    if errors:
        status = "partial"

    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_date": target_date,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_site": DEFAULT_SOURCE_SITE,
        "status": status,
        "model": {
            "provider": DEFAULT_PROVIDER,
            "primary_model": runtime.primary_model.model_id,
            "fallback_model": runtime.fallback_model.model_id,
        },
        "model_switches": runtime.model_switches,
        "input": {
            "article_count": len(articles),
            "category_count": len(category_reports),
        },
        "diagnostics": runtime.diagnostics.as_dict(),
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
        "categories": category_reports,
        "errors": errors,
    }

    _write_report(report_path, report)
    report["output_path"] = str(report_path)
    report["cached"] = False
    LOGGER.info(
        "Finished daily analysis for %s with status %s. Categories=%s, articles=%s.",
        target_date,
        status,
        len(category_reports),
        len(all_articles),
    )
    return report


def load_groq_api_key() -> str:
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key

    for config_path in _candidate_config_paths():
        if not config_path.exists():
            continue
        parsed = _parse_simple_env_file(config_path)
        key = parsed.get("GROQ_API_KEY")
        if key:
            return key

    raise RuntimeError(
        "GROQ_API_KEY is not set. Export GROQ_API_KEY or add it to the repo-root .config file."
    )


def _build_groq_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    )
    return session


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
    }


def _analyze_category(
    runtime: AnalysisRuntime,
    category_name: str,
    prepared_articles: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    LOGGER.info("Analyzing category %s with %s article(s).", category_name, len(prepared_articles))
    planned_batches = _plan_category_batches(runtime, category_name, prepared_articles)
    article_results: list[dict[str, Any]] = []
    article_errors: list[dict[str, Any]] = []
    sub_batch_count = 0
    for index, batch in enumerate(planned_batches, start=1):
        batch_results, batch_errors, batch_count = _process_batch_recursive(runtime, category_name, batch, str(index))
        article_results.extend(batch_results)
        article_errors.extend(batch_errors)
        sub_batch_count += batch_count
    successful_articles = [article for article in article_results if not article.get("error")]
    category_errors = list(article_errors)
    key_developments: list[str] = []
    named_entities: list[dict[str, str]] = _collect_entities_from_articles(successful_articles)
    synthesis_model = _category_model_used(article_results)
    category_status = "success"
    category_error_message: str | None = None
    diagnostics = runtime.get_category_diagnostics(category_name)
    diagnostics.sub_batch_count = sub_batch_count
    diagnostics.partial_article_count = sum(1 for article in article_results if article.get("error"))

    if successful_articles:
        try:
            synthesis_payload, synthesis_model = _invoke_synthesis(runtime, category_name, successful_articles)
            key_developments = _normalize_string_list(synthesis_payload.get("key_developments"), limit=5)
            named_entities = _normalize_entities(synthesis_payload.get("named_entities"))
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
    else:
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
            "model_used": synthesis_model,
            "sub_batch_count": sub_batch_count,
            "diagnostics": diagnostics.as_dict(),
            "error": category_error_message,
        },
        category_errors,
    )


def _process_batch_recursive(
    runtime: AnalysisRuntime,
    category_name: str,
    batch_articles: list[dict[str, Any]],
    batch_label: str = "1",
    *,
    allow_salvage: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    working_batch = [_clone_prepared_article(article) for article in batch_articles]
    budget_state = runtime.get_category_budget(category_name)
    content_shrunk = _shrink_batch_to_budget(
        category_name,
        working_batch,
        input_budget_tokens=budget_state.input_budget_tokens,
        request_byte_budget=budget_state.request_byte_budget,
        model_id=runtime.current_model.model_id,
    )
    estimated_input_tokens = _estimate_batch_request_tokens(category_name, working_batch)
    request_bytes = _estimate_batch_request_bytes(category_name, working_batch, runtime.current_model.model_id)

    if (
        (estimated_input_tokens > budget_state.input_budget_tokens or request_bytes > budget_state.request_byte_budget)
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
        left_results, left_errors, left_count = _process_batch_recursive(runtime, category_name, left, f"{batch_label}a")
        right_results, right_errors, right_count = _process_batch_recursive(runtime, category_name, right, f"{batch_label}b")
        return left_results + right_results, left_errors + right_errors, left_count + right_count

    try:
        payload, model_used = _invoke_article_batch(
            runtime,
            category_name,
            working_batch,
            estimated_input_tokens,
            batch_label=batch_label,
            content_shrunk=content_shrunk,
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
            runtime.tighten_category_budget(category_name, "incomplete_model_output")
            if len(missing_articles) == 1:
                salvage_results, salvage_errors, salvage_count = _process_batch_recursive(
                    runtime,
                    category_name,
                    missing_articles,
                    f"{batch_label}m",
                    allow_salvage=False,
                )
            else:
                left, right = _split_batch(missing_articles)
                left_results, left_errors, left_count = _process_batch_recursive(
                    runtime,
                    category_name,
                    left,
                    f"{batch_label}ma",
                )
                right_results, right_errors, right_count = _process_batch_recursive(
                    runtime,
                    category_name,
                    right,
                    f"{batch_label}mb",
                )
                salvage_results = left_results + right_results
                salvage_errors = left_errors + right_errors
                salvage_count = left_count + right_count
            combined = _order_results_like_input(working_batch, merged_results + salvage_results)
            return combined, salvage_errors, 1 + salvage_count
        return merged_results, [], 1
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        classification = _classify_exception(exc)
        if status_code == 413 and len(working_batch) > 1:
            runtime.record_split(category_name, "response_413")
            runtime.tighten_category_budget(category_name, classification)
            LOGGER.info(
                "Splitting category %s batch %s after HTTP 413: article_count=%s.",
                category_name,
                batch_label,
                len(working_batch),
            )
            left, right = _split_batch(working_batch)
            left_results, left_errors, left_count = _process_batch_recursive(runtime, category_name, left, f"{batch_label}a")
            right_results, right_errors, right_count = _process_batch_recursive(runtime, category_name, right, f"{batch_label}b")
            return left_results + right_results, left_errors + right_errors, left_count + right_count

        model_used = runtime.last_attempted_model or runtime.current_model.model_id
        message = str(exc)
        if len(working_batch) > 1 and classification in {"rate_limited", "http_error", "unexpected_error"}:
            runtime.tighten_category_budget(category_name, classification)
            LOGGER.info(
                "Retrying category %s batch %s as smaller sub-batches after %s.",
                category_name,
                batch_label,
                classification,
            )
            left, right = _split_batch(working_batch)
            left_results, left_errors, left_count = _process_batch_recursive(runtime, category_name, left, f"{batch_label}a")
            right_results, right_errors, right_count = _process_batch_recursive(runtime, category_name, right, f"{batch_label}b")
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
        model_used = runtime.last_attempted_model or runtime.current_model.model_id
        message = str(exc)
        classification = _classify_exception(exc)
        if len(working_batch) > 1 and classification in {"invalid_json", "unexpected_error"}:
            runtime.tighten_category_budget(category_name, classification)
            LOGGER.info(
                "Retrying category %s batch %s as smaller sub-batches after %s.",
                category_name,
                batch_label,
                classification,
            )
            left, right = _split_batch(working_batch)
            left_results, left_errors, left_count = _process_batch_recursive(runtime, category_name, left, f"{batch_label}a")
            right_results, right_errors, right_count = _process_batch_recursive(runtime, category_name, right, f"{batch_label}b")
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


def _invoke_article_batch(
    runtime: AnalysisRuntime,
    category_name: str,
    batch_articles: list[dict[str, Any]],
    estimated_input_tokens: int,
    *,
    batch_label: str,
    content_shrunk: bool,
) -> tuple[dict[str, Any], str]:
    messages = _build_article_batch_messages(category_name, batch_articles)
    context = BatchContext(
        category_name=category_name,
        batch_kind="article_batch",
        batch_label=batch_label,
        article_count=len(batch_articles),
        estimated_input_tokens=estimated_input_tokens,
        serialized_request_bytes=_estimate_request_payload_bytes(PRIMARY_MODEL_ID, messages, DEFAULT_OUTPUT_TOKENS),
        content_shrunk=content_shrunk,
    )
    return _invoke_json_with_retry(runtime, messages, estimated_input_tokens, context)


def _invoke_synthesis(
    runtime: AnalysisRuntime,
    category_name: str,
    successful_articles: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    messages = _build_synthesis_messages(category_name, successful_articles)
    estimated_input_tokens = _estimate_messages_tokens(messages)
    context = BatchContext(
        category_name=category_name,
        batch_kind="synthesis",
        batch_label="summary",
        article_count=len(successful_articles),
        estimated_input_tokens=estimated_input_tokens,
        serialized_request_bytes=_estimate_request_payload_bytes(PRIMARY_MODEL_ID, messages, DEFAULT_OUTPUT_TOKENS),
        content_shrunk=False,
    )
    return _invoke_json_with_retry(runtime, messages, estimated_input_tokens, context)


def _invoke_json_with_retry(
    runtime: AnalysisRuntime,
    messages: list[dict[str, Any]],
    estimated_input_tokens: int,
    context: BatchContext,
) -> tuple[dict[str, Any], str]:
    raw_text, model_used = _chat_completion(runtime, messages, estimated_input_tokens, context)
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
            serialized_request_bytes=_estimate_request_payload_bytes(PRIMARY_MODEL_ID, repair_messages, DEFAULT_OUTPUT_TOKENS),
            content_shrunk=context.content_shrunk,
        )
        repaired_text, repaired_model = _chat_completion(runtime, repair_messages, repaired_tokens, repair_context)
        return _parse_json_content(repaired_text), repaired_model


def _chat_completion(
    runtime: AnalysisRuntime,
    messages: list[dict[str, Any]],
    estimated_input_tokens: int,
    context: BatchContext,
) -> tuple[str, str]:
    response: requests.Response | None = None
    for attempt in range(DEFAULT_CHAT_RETRIES):
        model = runtime.current_model
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
        wait_seconds = runtime.governor.before_request(model.model_id, estimated_input_tokens)
        runtime.record_wait(context.category_name, wait_seconds)
        if wait_seconds > 0:
            LOGGER.info(
                "Waiting %.1f seconds before %s for category %s batch=%s on %s.",
                wait_seconds,
                attempt_context.batch_kind,
                attempt_context.category_name,
                attempt_context.batch_label,
                model.model_id,
            )
        response = runtime.session.post(
            GROQ_CHAT_COMPLETIONS_URL,
            json={
                "model": model.model_id,
                "messages": messages,
                "temperature": 0.1,
                "max_completion_tokens": model.max_completion_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        runtime.governor.record_response(model.model_id, response)
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

        if response.status_code == 429 and model.model_id == runtime.primary_model.model_id:
            LOGGER.info(
                "Groq returned HTTP 429 for category %s batch=%s on %s.",
                attempt_context.category_name,
                attempt_context.batch_label,
                model.model_id,
            )
            runtime.tighten_category_budget(attempt_context.category_name, "rate_limited")
            runtime.switch_to_fallback("Primary model returned 429 rate_limit_exceeded.")
            continue

        if response.status_code in {429, 500, 502, 503, 504}:
            if response.status_code == 429:
                LOGGER.info(
                    "Groq returned HTTP 429 for category %s batch=%s on %s.",
                    attempt_context.category_name,
                    attempt_context.batch_label,
                    model.model_id,
                )
                runtime.tighten_category_budget(attempt_context.category_name, "rate_limited")
            if attempt == DEFAULT_CHAT_RETRIES - 1:
                response.raise_for_status()
            delay_seconds = runtime.governor.apply_backoff(model.model_id, response, attempt)
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


def _build_article_batch_messages(category_name: str, batch_articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a financial news analyst. Return one valid JSON object only. "
                "Analyze every article in the batch. Use integer scores from 1 to 10. "
                "Do not omit any article."
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
                                "key_points": ["2 to 4 concise strings"],
                            }
                        ]
                    },
                    "rules": [
                        "Return one article result for every input article.",
                        "Match article results by source_article_id when available and also include canonical_url.",
                        "Do not include category-level summary in this response.",
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
                        }
                        for article in batch_articles
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]


def _build_synthesis_messages(category_name: str, successful_articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a financial news analyst. Return one valid JSON object only. "
                "Summarize the category using only the provided article analyses."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Synthesize one fixed news category from already-analyzed article results.",
                    "required_schema": {
                        "key_developments": ["3 to 5 concise category-level developments"],
                        "named_entities": [
                            {"name": "entity name", "type": "person|company|country|institution|index|organization|asset|other"}
                        ],
                    },
                    "category": category_name,
                    "articles": [
                        {
                            "source_article_id": article["source_article_id"],
                            "canonical_url": article["canonical_url"],
                            "title": article["title"],
                            "published_at": article["published_at"],
                            "scores": {
                                "novelty_score": article["novelty_score"],
                                "relevance_score": article["relevance_score"],
                                "urgency_score": article["urgency_score"],
                            },
                            "named_entities": article["named_entities"],
                            "key_points": article["key_points"],
                        }
                        for article in successful_articles
                    ],
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
    planned: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for article in prepared_articles:
        candidate = current + [article]
        if current and not _batch_within_budget(
            category_name,
            candidate,
            input_budget_tokens=budget_state.input_budget_tokens,
            request_byte_budget=budget_state.request_byte_budget,
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
                "named_entities": _normalize_entities(match.get("named_entities")),
                "key_points": _normalize_string_list(match.get("key_points"), limit=5),
                "content_truncated": article.get("content_truncated"),
                "original_content_length_chars": article.get("original_content_length_chars"),
                "analyzed_content_length_chars": article.get("analyzed_content_length_chars"),
                "original_content_token_estimate": article.get("original_content_token_estimate"),
                "analyzed_content_token_estimate": article.get("analyzed_content_token_estimate"),
                "truncation_reason": article.get("truncation_reason"),
                "analysis_method": article.get("analysis_method"),
                "model_used": model_used,
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
        "model_used": model_used,
        "error_classification": error_classification,
        "error": error_message,
    }


def _group_source_articles(articles: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article in articles:
        grouped[(article.get("article_section") or "未分類")].append(article)

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


def _build_empty_report(target_date: str) -> dict[str, Any]:
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_date": target_date,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_site": DEFAULT_SOURCE_SITE,
        "status": "empty",
        "model": {
            "provider": DEFAULT_PROVIDER,
            "primary_model": PRIMARY_MODEL_ID,
            "fallback_model": FALLBACK_MODEL_ID,
        },
        "model_switches": [],
        "input": {
            "article_count": 0,
            "category_count": 0,
        },
        "diagnostics": RuntimeDiagnostics().as_dict(),
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
        "categories": [],
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
