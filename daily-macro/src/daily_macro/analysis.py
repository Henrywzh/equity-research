from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import replace
from pathlib import Path
from typing import Any

import requests
from langgraph.graph import END, START, StateGraph

from .config import DEFAULT_SOURCE_SITE, get_analysis_dir, get_data_dir, get_db_path, get_project_root
from .storage import Storage

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-export everything from types.py so existing callers / tests that import
# from `daily_macro.analysis` continue to work without changes.
# ---------------------------------------------------------------------------
from .types import (  # noqa: E402
    GROQ_CHAT_COMPLETIONS_URL,
    OPENAI_CHAT_COMPLETIONS_URL,
    REPORT_FILE_NAME,
    REPORT_SCHEMA_VERSION,
    DEFAULT_PROVIDER,
    PRIMARY_MODEL_ID,
    FALLBACK_MODEL_IDS,
    DELAYED_RETRY_FINAL_MODEL_ID,
    DELAYED_RETRY_WAIT_SECONDS,
    MAX_CATEGORY_SYNTHESIS_WAIT_SECONDS,
    MAX_CATEGORY_SYNTHESIS_RETRIES,
    MAX_SYNTHESIS_MERGE_DEPTH,
    DEFAULT_OUTPUT_TOKENS,
    DEFAULT_INPUT_BUDGET_TOKENS,
    DEFAULT_SYNTHESIS_INPUT_BUDGET_TOKENS,
    DEFAULT_PROMPT_OVERHEAD_TOKENS,
    MEDIUM_ANALYSIS_MAX_CONTENT_TOKENS,
    SHORT_ARTICLE_FULL_TEXT_THRESHOLD,
    DEFAULT_CHAT_RETRIES,
    RATE_LIMIT_REQUEST_FLOOR,
    RATE_LIMIT_TOKEN_FLOOR,
    DEFAULT_REQUEST_BYTE_BUDGET,
    DEFAULT_SYNTHESIS_REQUEST_BYTE_BUDGET,
    MIN_REQUEST_BYTE_BUDGET,
    MIN_SYNTHESIS_REQUEST_BYTE_BUDGET,
    MIN_INPUT_BUDGET_TOKENS,
    MIN_SYNTHESIS_INPUT_BUDGET_TOKENS,
    CATEGORY_SHRINK_STEP_CHARS,
    CATEGORY_MIN_CONTENT_CHARS,
    CATEGORY_ORDER,
    ENTITY_TYPES,
    FAILURE_CLASSIFICATIONS,
    LIGHT_ANALYSIS_SECTIONS,
    ATTENTION_TIERS,
    ATTENTION_TIER_RANK,
    MARKET_CHANNELS,
    ROUTER_LLM_MIN_ARTICLES,
    HIGH_ATTENTION_THEME_KEYWORDS,
    SectionProfile,
    STANDARD_SECTION_PROFILE,
    LIGHT_SECTION_PROFILE,
    AnalysisGraphState,
    ModelConfig,
    ModelRateLimitState,
    CategoryBudgetState,
    RuntimeDiagnostics,
    CategoryDiagnostics,
    SynthesisBudgetExceeded,
    BatchContext,
    _section_profile,
)

from .llm_client import (  # noqa: E402
    RateLimitGovernor,
    AnalysisRuntime,
    LLMTask,
    ModelResolver,
    load_groq_api_keys,
    load_groq_api_key,
    _build_groq_session,
    _build_provider_session,
    _load_model_api_key,
    _invoke_json_with_retry,
    _chat_completion,
    _estimate_messages_tokens,
    _estimate_request_payload_bytes,
    _parse_json_content,
    _estimate_tokens,
    _candidate_config_paths,
    _parse_simple_env_file,
    _retry_delay_seconds,
    _parse_retry_after_seconds,
    _parse_duration_seconds,
    _parse_int,
    _classify_exception,
)

from .prompts import (  # noqa: E402
    _batch_attention_tier,
    _build_attention_routing_messages,
    _build_article_batch_messages,
    _build_article_quality_review_messages,
    _build_synthesis_messages,
    _build_grouping_messages,
)

from .model_registry import build_model_pool  # noqa: E402
from .budget import DailyBudgetLedger  # noqa: E402
from .provider_registry import config_value, load_provider_accounts, provider_model_ids  # noqa: E402

# ---------------------------------------------------------------------------
# Patch AnalysisRuntime methods to look up helpers through this module's
# namespace so that unit tests can mock `daily_macro.analysis._build_groq_session`
# (and similar names) and have the patches take effect even though the class
# now lives in llm_client.
# ---------------------------------------------------------------------------
import sys as _sys

def _analysis_get_groq_session(self, key_index=None):
    ki = key_index if key_index is not None else self.current_key_index
    if ki not in self.groq_sessions:
        _mod = _sys.modules[__name__]
        self.groq_sessions[ki] = _mod._build_groq_session(self.groq_api_keys[ki])
    return self.groq_sessions[ki]

def _analysis_get_session_for_model(self, model):
    from .types import DEFAULT_PROVIDER
    _mod = _sys.modules[__name__]
    if model.provider == DEFAULT_PROVIDER:
        return self.get_groq_session(self.current_key_index)
    session = self.provider_sessions.get(model.session_key)
    if session is not None:
        return session
    session = _mod._build_provider_session(_mod._load_model_api_key(model))
    self.provider_sessions[model.session_key] = session
    return session

AnalysisRuntime.get_groq_session = _analysis_get_groq_session
AnalysisRuntime.get_session_for_model = _analysis_get_session_for_model


def run_analysis(
    *,
    date_string: str | None = None,
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    run_started = time.monotonic()
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

    storage_started = time.monotonic()
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
    storage_elapsed = time.monotonic() - storage_started
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
        "event_packets": [],
        "review_queue": [],
        "previous_day_retry_successes": 0,
        "market_context_string": "",
        "market_snapshots": [],
        "macro_release_digest": {},
        "top_alerts": [],
        "validation_issues": [],
        "critic_issues": [],
        "theme_memory": {},
        "report": {},
        "total_scraped_articles": total_scraped_articles,
        "data_dir": resolved_data_dir,
    }
    final_state = graph.invoke(state)
    report = final_state["report"]
    runtime = final_state.get("runtime")
    wall_clock_seconds = time.monotonic() - run_started
    if runtime is not None:
        runtime.record_phase("storage_load", storage_elapsed)
        runtime.diagnostics.wall_clock_seconds = wall_clock_seconds
        report["diagnostics"] = runtime.diagnostics.as_dict()
    else:
        report.setdefault("diagnostics", {})["wall_clock_seconds"] = round(wall_clock_seconds, 3)
    report["output_path"] = str(report_path)
    report["cached"] = False
    _write_report(report_path, report)
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
    nodes = {
        "initialize": _graph_initialize,
        "fetch_market_data": _graph_fetch_market_data,
        "route_attention": _graph_route_attention,
        "analyze_today": _graph_analyze_today,
        "build_event_packets": _graph_build_event_packets,
        "retry_previous_day": _graph_retry_previous_day,
        "validate_outputs": _graph_validate_outputs,
        "update_theme_memory": _graph_update_theme_memory,
        "summarize_top_alerts": _graph_summarize_top_alerts,
        "critic_outputs": _graph_critic_outputs,
        "finalize": _graph_finalize,
    }
    for name, handler in nodes.items():
        graph.add_node(name, _timed_graph_node(name, handler))
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "fetch_market_data")
    graph.add_edge("fetch_market_data", "route_attention")
    graph.add_edge("route_attention", "analyze_today")
    graph.add_edge("analyze_today", "build_event_packets")
    graph.add_edge("build_event_packets", "retry_previous_day")
    graph.add_edge("retry_previous_day", "validate_outputs")
    graph.add_edge("validate_outputs", "update_theme_memory")
    graph.add_edge("update_theme_memory", "summarize_top_alerts")
    graph.add_edge("summarize_top_alerts", "critic_outputs")
    graph.add_edge("critic_outputs", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def _timed_graph_node(name: str, handler):
    """Measure graph-node wall time without changing the graph state contract."""
    def invoke(state: AnalysisGraphState) -> AnalysisGraphState:
        started = time.monotonic()
        result = state
        try:
            result = handler(state)
            return result
        finally:
            runtime = result.get("runtime") or state.get("runtime")
            if runtime is not None:
                runtime.record_phase(name, time.monotonic() - started)

    return invoke


def _graph_initialize(state: AnalysisGraphState) -> AnalysisGraphState:
    today_plan = _build_today_incremental_plan(state["articles"], state["existing_report"])
    previous_retry_plan = _build_previous_retry_plan(state["previous_articles"], state["previous_report"])
    runtime: AnalysisRuntime | None = None
    if today_plan["new_articles_analyzed"] > 0 or previous_retry_plan["retried_previous_day_articles"] > 0:
        provider_accounts = load_provider_accounts()
        configured_groq_keys = [account.api_key for account in provider_accounts if account.provider == DEFAULT_PROVIDER and account.api_key]
        try:
            groq_keys = configured_groq_keys or load_groq_api_keys()
        except RuntimeError:
            # Backward-compatible shim for tests and older single-key call sites.
            if provider_accounts and any(account.provider != DEFAULT_PROVIDER for account in provider_accounts):
                groq_keys = []
            else:
                groq_keys = [load_groq_api_key()]
        LOGGER.info(
            "Loaded LLM provider accounts: %s.",
            ", ".join(sorted({account.provider for account in provider_accounts})) or DEFAULT_PROVIDER,
        )
        pool = build_model_pool(
            groq_keys,
            data_dir=state.get("data_dir"),
            provider_accounts=provider_accounts or None,
        )
        fallback_model_ids = _groq_fallback_model_ids()
        fallback_chain = [ModelConfig(model_id) for model_id in fallback_model_ids]
        budget = DailyBudgetLedger.load(state.get("data_dir"))
        model_limits: dict[str, dict[str, int]] = {}
        for model in pool.models:
            cap = pool.capabilities.get(model.endpoint_id) or pool.capabilities.get(model.model_id)
            if cap and cap.limits:
                model_limits[model.quota_key] = cap.limits
                model_limits.setdefault(model.model_id, cap.limits)
        delayed_retry_final_model = None
        if config_value("OPENAI_API_KEY"):
            delayed_retry_final_model = ModelConfig(
                DELAYED_RETRY_FINAL_MODEL_ID,
                provider="openai",
                api_url=OPENAI_CHAT_COMPLETIONS_URL,
                api_key_env="OPENAI_API_KEY",
            )
        runtime = AnalysisRuntime(
            groq_api_keys=groq_keys,
            governor=RateLimitGovernor(model_limits=model_limits),
            model_chain=pool.models or fallback_chain,
            resolver=ModelResolver(active_model_ids=pool.active_ids, capabilities=pool.capabilities),
            budget=budget,
            delayed_retry_final_model=delayed_retry_final_model,
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
    from .release_calendar import build_release_digest

    snapshots = fetch_market_snapshot_for_date(state["target_date"])
    state["market_context_string"] = build_market_context_string(snapshots)
    state["market_snapshots"] = snapshots
    state["macro_release_digest"] = build_release_digest(start_date=state["target_date"], days_ahead=7, require_api_key=False)
    if state["runtime"] is not None:
        state["runtime"].market_context_string = state["market_context_string"]
        state["runtime"].macro_release_digest = state["macro_release_digest"]
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
        report_date=state["target_date"],
    )
    # Record which articles were actually newly analyzed in this run
    state["newly_analyzed_keys"].update(
        str(a.get("source_article_id") or a.get("canonical_url") or "")
        for a in state["today_plan"]["work_articles"]
    )
    return state


def _graph_build_event_packets(state: AnalysisGraphState) -> AnalysisGraphState:
    """Build an evidence-first event layer from successful article analyses.

    This is deliberately local and conservative. It reduces duplicate coverage
    before any future event-level LLM work, while still preserving the current
    article/category report as the compatibility surface.
    """
    packets, review_queue = _build_event_packets(
        state.get("category_reports") or [],
        target_date=state["target_date"],
    )
    state["event_packets"] = packets
    state["review_queue"] = review_queue
    return state


def _previous_retry_max_articles() -> int:
    raw = os.environ.get("DAILY_MACRO_PREVIOUS_RETRY_MAX_ARTICLES", "12")
    try:
        return max(0, int(raw))
    except ValueError:
        return 12


def _previous_retry_allowed(state: AnalysisGraphState) -> tuple[bool, str]:
    runtime = state.get("runtime")
    if runtime is None:
        return True, "no_runtime"
    unresolved = _collect_unresolved_articles(state.get("category_reports") or [])
    high_medium_unresolved = sum(
        1 for item in unresolved if str(item.get("attention_tier") or "medium") in {"high", "medium"}
    )
    if high_medium_unresolved:
        return False, "today_high_medium_unresolved"
    if runtime.diagnostics.synthesis_budget_exhausted_count:
        return False, "today_synthesis_budget_exhausted"
    max_wait = os.environ.get("DAILY_MACRO_PREVIOUS_RETRY_MAX_WAIT_SECONDS", "30")
    try:
        wait_budget = max(0.0, float(max_wait))
    except ValueError:
        wait_budget = 30.0
    if runtime.diagnostics.rate_limit_wait_seconds_total >= wait_budget:
        return False, "today_rate_limit_wait_budget_exhausted"
    return True, "capacity_available"


def _graph_retry_previous_day(state: AnalysisGraphState) -> AnalysisGraphState:
    runtime = state["runtime"]
    try:
        if state["previous_report"] is not None and state["previous_retry_plan"]["retried_previous_day_articles"] > 0:
            allowed, reason = _previous_retry_allowed(state)
            if not allowed:
                skipped = state["previous_retry_plan"]["retried_previous_day_articles"]
                LOGGER.info(
                    "Skipping previous-day retry for %s article(s): %s.",
                    skipped,
                    reason,
                )
                state["previous_retry_plan"] = {"work_articles": [], "retried_previous_day_articles": 0}
                state["incremental"]["retried_previous_day_articles"] = 0
                state["incremental"]["previous_day_retry_skipped"] = skipped
                if runtime is not None:
                    runtime.diagnostics.degraded_mode_count += 1
                return state
            max_articles = _previous_retry_max_articles()
            retry_articles = list(state["previous_retry_plan"].get("work_articles") or [])
            retry_articles.sort(
                key=lambda item: (
                    ATTENTION_TIER_RANK.get(str(item.get("attention_tier") or "medium"), ATTENTION_TIER_RANK["medium"]),
                    str(item.get("published_at") or ""),
                )
            )
            skipped = max(0, len(retry_articles) - max_articles)
            if skipped:
                retry_articles = retry_articles[:max_articles]
                state["previous_retry_plan"] = {
                    "work_articles": retry_articles,
                    "retried_previous_day_articles": len(retry_articles),
                }
                state["incremental"]["retried_previous_day_articles"] = len(retry_articles)
                state["incremental"]["previous_day_retry_skipped"] = skipped
                LOGGER.info(
                    "Capping previous-day retry to %s article(s); %s lower-priority article(s) deferred.",
                    max_articles,
                    skipped,
                )
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


def _graph_validate_outputs(state: AnalysisGraphState) -> AnalysisGraphState:
    issues: list[dict[str, Any]] = []
    valid_article_ids = {
        str(article.get("source_article_id") or article.get("canonical_url") or "")
        for category in state.get("category_reports") or []
        for article in category.get("articles") or []
        if not article.get("error")
    }
    for category in state.get("category_reports") or []:
        for article in category.get("articles") or []:
            if article.get("error"):
                continue
            key = article.get("source_article_id") or article.get("canonical_url")
            if not key:
                issues.append({"type": "article_validation", "category": category.get("category"), "reason": "missing_article_identifier"})
            if article.get("model_used") != "direct" and not article.get("key_points"):
                issues.append({"type": "article_validation", "category": category.get("category"), "reason": "missing_key_points", "target": key})
            if str(article.get("research_lane") or "").strip() == "":
                article["research_lane"] = _infer_research_lane(article)
    event_ids: set[str] = set()
    for event in state.get("event_packets") or []:
        event_id = str(event.get("event_id") or "")
        if not event_id:
            issues.append({"type": "event_validation", "reason": "missing_event_identifier"})
        else:
            event_ids.add(event_id)
        source_ids = [str(item) for item in event.get("source_article_ids") or []]
        invalid_ids = [item for item in source_ids if item not in valid_article_ids]
        if not source_ids:
            issues.append({"type": "event_validation", "reason": "missing_event_sources", "target": event_id})
        if invalid_ids:
            issues.append({"type": "event_validation", "reason": "invalid_event_sources", "target": event_id, "source_ids": invalid_ids})
        for evidence in event.get("evidence") or []:
            evidence_id = str(evidence.get("source_article_id") or "") if isinstance(evidence, dict) else ""
            if evidence_id not in source_ids:
                issues.append({"type": "event_validation", "reason": "invalid_event_evidence", "target": event_id, "source_id": evidence_id})
    for review_item in state.get("review_queue") or []:
        if str(review_item.get("event_id") or "") not in event_ids:
            issues.append({"type": "review_validation", "reason": "unknown_event", "event_id": review_item.get("event_id")})
    state["validation_issues"] = issues
    if issues and state.get("runtime") is not None:
        state["runtime"].diagnostics.degraded_mode_count += 1
    return state


def _graph_update_theme_memory(state: AnalysisGraphState) -> AnalysisGraphState:
    data_dir = state.get("data_dir")
    if data_dir is None:
        return state
    memory = _update_theme_memory_file(
        Path(data_dir),
        state["target_date"],
        state.get("category_reports") or [],
        event_packets=state.get("event_packets") or [],
    )
    state["theme_memory"] = memory
    return state


_CRITIC_NUMERIC_FACT_RE = re.compile(
    r"(?P<left>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:(?:[-–—~至到])\s*(?P<right>\d[\d,]*(?:\.\d+)?)\s*)?"
    r"(?P<unit>萬億|万亿|億元|亿元|億|亿|萬元|万元|萬|万|百萬|百万|"
    r"trillion|billion|bn|million|mn|thousand|k|%)",
    re.IGNORECASE,
)
_CRITIC_TICKER_RE = re.compile(r"^(?:\^?[A-Z]{3,8}(?:[._-][A-Z0-9]{1,8})?|\d{4,6}\.[A-Z]{2}|[A-Z]{2,8}=F)$")
_CRITIC_UNIT_SCALES = {
    "萬億": 1_000_000_000_000.0,
    "万亿": 1_000_000_000_000.0,
    "億元": 100_000_000.0,
    "亿元": 100_000_000.0,
    "億": 100_000_000.0,
    "亿": 100_000_000.0,
    "萬元": 10_000.0,
    "万元": 10_000.0,
    "萬": 10_000.0,
    "万": 10_000.0,
    "百萬": 1_000_000.0,
    "百万": 1_000_000.0,
    "trillion": 1_000_000_000_000.0,
    "billion": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "million": 1_000_000.0,
    "mn": 1_000_000.0,
    "thousand": 1_000.0,
    "k": 1_000.0,
    "%": 1.0,
}
_CANONICAL_ASSET_ALIASES = {
    "賽力斯": {"canonical_id": "09927.HK", "display_name": "Seres", "confidence": 0.98},
    "seres": {"canonical_id": "09927.HK", "display_name": "Seres", "confidence": 0.98},
    "三星電子": {"canonical_id": "005930.KS", "display_name": "Samsung Electronics", "confidence": 0.97},
    "samsung electronics": {"canonical_id": "005930.KS", "display_name": "Samsung Electronics", "confidence": 0.97},
    "恒指": {"canonical_id": "^HSI", "display_name": "Hang Seng Index", "confidence": 0.99},
    "恒生指數": {"canonical_id": "^HSI", "display_name": "Hang Seng Index", "confidence": 0.99},
    "hang seng index": {"canonical_id": "^HSI", "display_name": "Hang Seng Index", "confidence": 0.99},
}


def _critic_numeric_facts(text: str) -> list[float]:
    facts: list[float] = []
    for match in _CRITIC_NUMERIC_FACT_RE.finditer(text or ""):
        scale = _CRITIC_UNIT_SCALES.get(str(match.group("unit") or "").lower())
        if scale is None:
            scale = _CRITIC_UNIT_SCALES.get(str(match.group("unit") or ""))
        if scale is None:
            continue
        for value in (match.group("left"), match.group("right")):
            if value:
                facts.append(float(value.replace(",", "")) * scale)
    return facts


def _critic_numeric_issues(alert: dict[str, Any], evidence_text: str) -> list[dict[str, Any]]:
    alert_text = " ".join(
        str(alert.get(field) or "")
        for field in ("summary", "why_it_matters")
    )
    alert_facts = _critic_numeric_facts(alert_text)
    if not alert_facts:
        return []
    evidence_facts = _critic_numeric_facts(evidence_text)
    issues: list[dict[str, Any]] = []
    for value in alert_facts:
        supported = any(math.isclose(value, candidate, rel_tol=0.01, abs_tol=0.01) for candidate in evidence_facts)
        if not supported:
            issues.append(
                {
                    "type": "numeric_fact_unsupported",
                    "severity": "high",
                    "value": value,
                    "target": alert.get("summary"),
                    "reason": "A number with a financial scale or percentage was not found in the cited source evidence or market context.",
                }
            )
    return issues


def _critic_source_evidence(state: AnalysisGraphState) -> dict[str, str]:
    evidence: dict[str, list[str]] = defaultdict(list)
    for article in state.get("articles") or []:
        source_id = str(article.get("source_article_id") or "").strip()
        if not source_id:
            continue
        evidence[source_id].extend(
            str(article.get(field) or "")
            for field in ("title", "summary_snippet", "content_text")
        )
    for category in state.get("category_reports") or []:
        for article in category.get("articles") or []:
            source_id = str(article.get("source_article_id") or "").strip()
            if not source_id:
                continue
            evidence[source_id].extend(
                [
                    str(article.get("title") or ""),
                    *[str(point) for point in article.get("key_points") or []],
                    *[
                        str(entity.get("name") if isinstance(entity, dict) else entity)
                        for entity in article.get("named_entities") or []
                    ],
                ]
            )
    return {source_id: _normalize_whitespace(" ".join(parts)) for source_id, parts in evidence.items()}


def _critic_asset_check(
    raw_assets: Any,
    *,
    evidence_text: str,
    source_ids: list[str],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    assets = _normalize_string_list(raw_assets, limit=8)
    normalized_assets: list[str] = []
    details: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    evidence_folded = evidence_text.casefold()
    seen_canonical: set[str] = set()

    for raw_asset in assets:
        folded = raw_asset.casefold()
        alias = next(
            (value for key, value in _CANONICAL_ASSET_ALIASES.items() if key.casefold() in folded),
            None,
        )
        if alias is not None:
            canonical_id = str(alias["canonical_id"])
            if canonical_id not in seen_canonical:
                normalized_assets.append(canonical_id)
                seen_canonical.add(canonical_id)
            details.append(
                {
                    "input": raw_asset,
                    "canonical_id": canonical_id,
                    "display_name": alias["display_name"],
                    "mapping_confidence": alias["confidence"],
                    "status": "canonical",
                    "source_article_ids": source_ids,
                }
            )
            continue

        if _CRITIC_TICKER_RE.fullmatch(raw_asset.strip()) and folded not in evidence_folded:
            issues.append(
                {
                    "type": "unsupported_asset_identifier",
                    "severity": "medium",
                    "target": raw_asset,
                    "reason": "Ticker-like asset was not present in the cited source evidence or market context and was removed.",
                }
            )
            continue

        normalized_assets.append(raw_asset)
        details.append(
            {
                "input": raw_asset,
                "canonical_id": raw_asset if _CRITIC_TICKER_RE.fullmatch(raw_asset.strip()) else None,
                "display_name": raw_asset,
                "mapping_confidence": 0.8 if folded in evidence_folded else 0.45,
                "status": "evidence_supported" if folded in evidence_folded else "unmapped_name",
                "source_article_ids": source_ids,
            }
        )
    return list(dict.fromkeys(normalized_assets)), details, issues


def _critic_alerts(state: AnalysisGraphState, alerts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = {
        str(article.get("source_article_id") or article.get("canonical_url") or ""): article
        for category in state.get("category_reports") or []
        for article in category.get("articles") or []
        if not article.get("error")
    }
    valid_ids = set(metadata)
    evidence_by_id = _critic_source_evidence(state)
    market_context = str(state.get("market_context_string") or "")
    issues: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for alert in alerts:
        if not isinstance(alert, dict) or not str(alert.get("summary") or "").strip():
            issues.append({"type": "alert_critic", "severity": "high", "reason": "missing_summary"})
            continue
        source_ids = [str(item).strip() for item in alert.get("source_article_ids") or [] if str(item).strip()]
        valid_source_ids = list(dict.fromkeys(item for item in source_ids if item in valid_ids))
        if not valid_source_ids:
            issues.append({"type": "alert_critic", "severity": "high", "reason": "missing_or_invalid_citations", "target": alert.get("summary")})
            continue
        if len(valid_source_ids) != len(source_ids):
            issues.append({"type": "alert_critic", "severity": "high", "reason": "unsupported_citations_removed", "target": alert.get("summary")})

        evidence_text = " ".join([*(evidence_by_id.get(source_id, "") for source_id in valid_source_ids), market_context])
        normalized = dict(alert)
        normalized["source_article_ids"] = valid_source_ids
        assets, asset_details, asset_issues = _critic_asset_check(
            alert.get("affected_assets"),
            evidence_text=evidence_text,
            source_ids=valid_source_ids,
        )
        normalized["affected_assets"] = assets
        normalized["affected_asset_details"] = asset_details
        issues.extend({**item, "target": item.get("target") or alert.get("summary")} for item in asset_issues)

        numeric_issues = _critic_numeric_issues(alert, evidence_text)
        issues.extend(numeric_issues)
        if numeric_issues:
            rejected.append({**normalized, "critic_status": "needs_review", "confidence": min(float(normalized.get("confidence") or 0.5), 0.35)})
            continue

        causal_text = str(alert.get("summary") or "") + " " + str(alert.get("why_it_matters") or "")
        if re.search(r"\b(drove|caused|led to|triggered|underpinning|because of)\b", causal_text, re.IGNORECASE):
            issues.append(
                {
                    "type": "causal_claim_unverified",
                    "severity": "medium",
                    "target": alert.get("summary"),
                    "reason": "Causal wording should be reviewed against source evidence; co-movement alone does not establish causality.",
                }
            )
            normalized["critic_status"] = "needs_review"
        else:
            normalized["critic_status"] = "passed"
        checked.append(normalized)

    if not checked and rejected:
        checked = [rejected[0]]
        issues.append(
            {
                "type": "alert_critic",
                "severity": "high",
                "reason": "All candidate alerts had evidence issues; retained the highest-ranked candidate for human review.",
                "target": checked[0].get("summary"),
            }
        )
    return checked[:3], issues


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
            
            # Prefer per-development provenance (each development's own source
            # article ids) so an alert is tagged with the article it actually
            # came from — not every article in the subgroup. Fall back to the
            # subgroup-wide ref_ids for older reports or when the model gave no
            # (valid) ids, so behavior never regresses to empty.
            detailed = subgroup.get("key_developments_detailed")
            if detailed:
                dev_items = [
                    (
                        str(dev.get("text") or "").strip(),
                        [sid for sid in (dev.get("source_article_ids") or []) if sid in ref_ids],
                    )
                    for dev in detailed
                    if isinstance(dev, dict)
                ]
            else:
                dev_items = [(str(dev).strip(), []) for dev in (subgroup.get("key_developments") or [])]

            for dev_text, own_ref_ids in dev_items:
                if not dev_text:
                    continue
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
                        "ref_ids": own_ref_ids or ref_ids,
                    }
                )
    
    if not all_developments:
        return state

    try:
        alert_kwargs = {
            "runtime": runtime,
            "market_context": state["market_context_string"],
            "developments": all_developments,
            "article_metadata": article_metadata,
            "is_incremental": bool(state.get("legacy_executive_summary")),
        }
        if state.get("event_packets"):
            alert_kwargs["event_packets"] = state["event_packets"]
        top_alerts = _generate_top_alerts(
            **alert_kwargs,
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


def _graph_critic_outputs(state: AnalysisGraphState) -> AnalysisGraphState:
    """Run evidence, citation, numeric, and asset checks before rendering."""
    alerts = list(state.get("top_alerts") or [])
    metadata = {
        str(article.get("source_article_id") or article.get("canonical_url") or ""): article
        for category in state.get("category_reports") or []
        for article in category.get("articles") or []
        if not article.get("error")
    }
    checked, issues = _critic_alerts(state, alerts)
    for alert in checked:
        source_ids = [str(item) for item in alert.get("source_article_ids") or []]
        alert["source_articles"] = [
            {
                "source_article_id": source_id,
                "title": str(metadata[source_id].get("title") or ""),
                "date": str(metadata[source_id].get("published_at") or state["target_date"])[:10],
                "url": str(metadata[source_id].get("canonical_url") or ""),
            }
            for source_id in source_ids
            if source_id in metadata
        ]
    state["top_alerts"] = checked[:3]
    state["executive_summary"] = checked[:3]
    state["critic_issues"] = issues
    if state.get("runtime") is not None:
        state["runtime"].diagnostics.critic_checked_alert_count += len(alerts)
        if issues:
            state["runtime"].diagnostics.degraded_mode_count += 1
    return state


def _graph_finalize(state: AnalysisGraphState) -> AnalysisGraphState:
    if not state["articles"]:
        report = _build_empty_report(
            state["target_date"],
            incremental=state["incremental"],
            macro_release_digest=state.get("macro_release_digest"),
        )
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
        macro_release_digest=state.get("macro_release_digest"),
        legacy_executive_summary=state.get("legacy_executive_summary"),
        newly_analyzed_keys=state.get("newly_analyzed_keys"),
        validation_issues=state.get("validation_issues"),
        theme_memory=state.get("theme_memory"),
        event_packets=state.get("event_packets"),
        review_queue=state.get("review_queue"),
        critic_issues=state.get("critic_issues"),
    )
    _write_report(state["report_path"], report)
    state["report"] = report
    return state


def select_content_for_analysis(content_text: str, *, max_content_tokens: int | None = None) -> dict[str, Any]:
    normalized = _normalize_whitespace(content_text)
    original_length = len(normalized)
    original_token_estimate = _estimate_tokens(normalized)

    if original_length <= SHORT_ARTICLE_FULL_TEXT_THRESHOLD and (
        max_content_tokens is None or original_token_estimate <= max_content_tokens
    ):
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
    if (
        (max_content_tokens is None or original_token_estimate <= max_content_tokens)
        and projected_input_tokens <= DEFAULT_INPUT_BUDGET_TOKENS
    ):
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

    target_tokens = max_content_tokens or max(DEFAULT_INPUT_BUDGET_TOKENS - DEFAULT_PROMPT_OVERHEAD_TOKENS, 500)
    return _build_truncated_selection(
        normalized,
        original_length,
        original_token_estimate,
        _content_prefix_length_for_token_budget(normalized, target_tokens),
        "Medium-attention article was compacted to preserve token budget; the leading source content slice was analyzed."
        if max_content_tokens is not None
        else "Full article exceeded the working request budget after prompt overhead; the leading content slice was analyzed instead.",
    )


def _content_prefix_length_for_token_budget(text: str, max_tokens: int) -> int:
    """Return the longest leading prefix whose estimate fits the token budget.

    HKEJ is predominantly Chinese, where the estimator is approximately one
    token per character. A fixed ``tokens * 4`` character conversion would
    therefore recreate the exact over-budget failure this selector is meant to
    prevent.
    """
    if not text or _estimate_tokens(text) <= max_tokens:
        return len(text)
    lower, upper = 0, len(text)
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        if _estimate_tokens(text[:midpoint]) <= max_tokens:
            lower = midpoint
        else:
            upper = midpoint - 1
    return lower


def _prepare_single_article(article: dict[str, Any]) -> dict[str, Any]:
    attention = _article_attention_defaults(article)
    selected = select_content_for_analysis(
        article.get("content_text") or "",
        max_content_tokens=MEDIUM_ANALYSIS_MAX_CONTENT_TOKENS if attention["attention_tier"] == "medium" else None,
    )
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
        "research_lane": attention["research_lane"],
        "attention_reason": attention["attention_reason"],
        "must_keep": attention["must_keep"],
        "market_channel": attention["market_channel"],
        "routing_market_impact_score": attention["routing_market_impact_score"],
        "routing_urgency_score": attention["routing_urgency_score"],
        "routing_novelty_score": attention["routing_novelty_score"],
        "priority_score": attention["priority_score"],
    }


def _article_attention_defaults(article: dict[str, Any]) -> dict[str, Any]:
    return _normalize_attention_metadata(
        {
            "attention_tier": article.get("attention_tier"),
            "theme": article.get("theme"),
            "attention_reason": article.get("attention_reason"),
            "must_keep": article.get("must_keep"),
            "market_channel": article.get("market_channel"),
            "market_impact_score": article.get("routing_market_impact_score"),
            "urgency_score": article.get("routing_urgency_score"),
            "novelty_score": article.get("routing_novelty_score"),
            "priority_score": article.get("priority_score"),
        },
        article,
    )


_ROUTING_STATEMENT_MARKERS = (
    "宣布", "表示", "稱", "指出", "決定", "公布", "發布", "批准", "通過", "警告", "預計",
    "announce", "announced", "says", "said", "decides", "decision", "reports", "reported",
)
_ROUTING_STRONG_STOCK_EVENTS = (
    "盈喜", "盈警", "盈利預警", "回購", "配股", "供股", "集資", "併購", "收購", "出售", "重組",
    "停牌", "復牌", "上市", "退市", "發行股份", "派息", "減持", "增持", "同店銷售", "里程碑付款",
    "訂單", "搜查", "調查", "涉嫌", "交棒", "管理層變動", "profit warning", "buyback", "placement",
    "rights issue", "merger", "acquisition", "ipo", "delisting", "guidance", "investigation",
)
_ROUTING_MACRO_EVENTS = (
    "加息", "減息", "降息", "降準", "加準", "維持利率", "議息", "利率決議", "通脹數據", "就業數據", "非農", "褐皮書", "收緊", "放寬",
    "gdp", "cpi", "inflation data", "jobs data", "rate decision", "rate hike", "rate cut",
)
_ROUTING_GEOPOLITICAL_EVENTS = (
    "制裁", "關稅", "出口管制", "貿易限制", "軍事升級", "衝突升級", "襲擊", "打擊", "空襲", "入侵", "停火",
    "sanction", "tariff", "export control", "trade restriction", "military escalation", "attack",
    "invasion", "ceasefire",
)
_ROUTING_PROPERTY_EVENTS = (
    "房屋政策", "樓市政策", "按揭利率", "賣地", "土地拍賣", "樓價數據", "成交數據", "地產商業績",
    "housing policy", "mortgage rate", "land auction", "property data",
)
_ROUTING_MATERIAL_MOVE_MARKERS = (
    "暴跌", "急挫", "大跌", "大升", "飆升", "飆", "急升", "崩跌", "跌逾", "升逾", "創新高", "surge", "plunge",
    "soars", "slumps", "falls sharply", "rises sharply",
)


def _contains_routing_signal(text: str, signals: tuple[str, ...]) -> bool:
    return any(signal.lower() in text for signal in signals)


def _routing_market_channel(haystack: str, theme: str) -> str:
    channels: list[str] = []
    if theme == "stocks" or _contains_routing_signal(haystack, ("股", "ads", "shares", "stocks", "equity", "earnings")):
        channels.append("equity")
    if theme == "macro" or _contains_routing_signal(haystack, ("經濟", "央行", "聯儲", "通脹", "inflation", "economy", "gdp")):
        channels.append("macro")
    if _contains_routing_signal(haystack, ("利率", "息率", "收益率", "債券", "rate", "yield", "bond")):
        channels.append("rates")
    if _contains_routing_signal(haystack, ("匯率", "美元", "人民幣", "外匯", "fx", "currency")):
        channels.append("fx")
    if _contains_routing_signal(haystack, ("油價", "原油", "天然氣", "黃金", "商品", "oil", "gold", "commodity")):
        channels.append("commodity")
    if theme == "geopolitics" or _contains_routing_signal(haystack, ("戰爭", "制裁", "關稅", "軍事", "sanction", "tariff", "war")):
        channels.append("geopolitics")
    if theme == "property" or _contains_routing_signal(haystack, ("樓市", "地產", "樓價", "按揭", "property", "housing")):
        channels.append("property")
    deduplicated = list(dict.fromkeys(channels))
    if not deduplicated:
        return "none"
    specific_channels = [channel for channel in ("rates", "fx", "commodity") if channel in deduplicated]
    if len(deduplicated) == 2 and "macro" in deduplicated and len(specific_channels) == 1:
        return specific_channels[0]
    if len(deduplicated) > 1:
        return "multi"
    return deduplicated[0]


def _heuristic_attention_metadata(article: dict[str, Any]) -> dict[str, Any]:
    title = str(article.get("title") or "").strip()
    snippet = str(article.get("summary_snippet") or "").strip()
    section = str(article.get("article_section") or article.get("section") or "").strip()
    haystack = f"{title} {snippet} {section}".lower()

    theme_hits = {
        theme: sum(1 for keyword in keywords if keyword.lower() in haystack)
        for theme, keywords in HIGH_ATTENTION_THEME_KEYWORDS.items()
    }
    matched_theme = max(theme_hits, key=theme_hits.get) if max(theme_hits.values(), default=0) else "general"
    market_channel = _routing_market_channel(haystack, matched_theme)
    has_number = bool(re.search(r"(?:\d[\d,.]*\s*(?:%|％|點|個百分點|億|萬|m|bn|million|billion)?|逾\s*\d|超過\s*\d)", haystack))
    has_statement = _contains_routing_signal(haystack, _ROUTING_STATEMENT_MARKERS)
    has_material_move = _contains_routing_signal(haystack, _ROUTING_MATERIAL_MOVE_MARKERS)

    stock_high = _contains_routing_signal(haystack, _ROUTING_STRONG_STOCK_EVENTS) and (
        has_statement or has_number or _contains_routing_signal(haystack, ("盈喜", "盈警", "回購", "配股", "供股", "併購", "收購", "停牌", "上市", "ipo", "investigation"))
    )
    macro_high = (
        (_contains_routing_signal(haystack, ("聯儲", "央行", "人行", "政府", "federal reserve", "central bank", "treasury"))
         or _contains_routing_signal(haystack, _ROUTING_MACRO_EVENTS))
        and (_contains_routing_signal(haystack, _ROUTING_MACRO_EVENTS) or has_statement)
        and _contains_routing_signal(haystack, ("利率", "通脹", "經濟", "增長", "就業", "gdp", "inflation", "rate", "growth", "jobs"))
        and not (_contains_routing_signal(haystack, ("預期", "料", "可能", "或會", "預計")) and not has_statement)
    )
    geopolitical_high = _contains_routing_signal(haystack, _ROUTING_GEOPOLITICAL_EVENTS) and (
        has_statement or has_number or _contains_routing_signal(haystack, ("美國", "中國", "俄羅斯", "伊朗", "烏克蘭", "中東", "us", "china", "russia", "iran"))
    )
    property_high = _contains_routing_signal(haystack, _ROUTING_PROPERTY_EVENTS) and (has_statement or has_number)
    material_move_high = has_material_move and has_number and market_channel != "none"
    high_signal = stock_high or macro_high or geopolitical_high or property_high or material_move_high

    if high_signal:
        tier = "high"
        must_keep = True
        reason = f"Concrete {matched_theme} catalyst or material market event with a visible {market_channel} channel."
        market_impact_score, urgency_score, novelty_score = 5, 3, 2
    elif section in LIGHT_ANALYSIS_SECTIONS:
        tier = "light"
        must_keep = False
        if matched_theme == "general" and section == "地產新聞":
            matched_theme = "property"
        market_channel = _routing_market_channel(haystack, matched_theme)
        reason = f"Routine or low-catalyst story in {section}; no concrete market-moving event was detected."
        market_impact_score, urgency_score, novelty_score = 0, 0, 0
    else:
        tier = "medium"
        must_keep = False
        reason = "Relevant financial or market context, but no sufficiently concrete catalyst for high priority."
        market_impact_score, urgency_score, novelty_score = 2, 1, 1

    priority_score = 2 * market_impact_score + urgency_score + novelty_score

    return {
        "attention_tier": tier,
        "theme": matched_theme,
        "research_lane": _infer_research_lane({"theme": matched_theme, "market_channel": market_channel, "attention_tier": tier, "article_section": section, "title": title}),
        "attention_reason": reason,
        "must_keep": must_keep,
        "market_channel": market_channel,
        "routing_market_impact_score": market_impact_score,
        "routing_urgency_score": urgency_score,
        "routing_novelty_score": novelty_score,
        "priority_score": priority_score,
        "high_confidence": high_signal,
    }


def _normalize_attention_metadata(metadata: dict[str, Any], article: dict[str, Any]) -> dict[str, Any]:
    heuristic = _heuristic_attention_metadata(article)
    requested_tier = str(metadata.get("attention_tier") or heuristic["attention_tier"]).strip().lower()
    attention_tier = requested_tier
    if attention_tier not in ATTENTION_TIERS:
        attention_tier = heuristic["attention_tier"]
    theme = str(metadata.get("theme") or heuristic["theme"]).strip().lower() or heuristic["theme"]
    attention_reason = str(metadata.get("attention_reason") or heuristic["attention_reason"]).strip() or heuristic["attention_reason"]
    market_channel = str(metadata.get("market_channel") or heuristic["market_channel"]).strip().lower()
    if market_channel not in MARKET_CHANNELS:
        market_channel = heuristic["market_channel"]
    if market_channel == "none" and heuristic["market_channel"] != "none":
        market_channel = heuristic["market_channel"]
    market_impact_score = _coerce_bounded_score(
        metadata.get("market_impact_score", metadata.get("routing_market_impact_score")),
        0,
        5,
        heuristic["routing_market_impact_score"],
    )
    urgency_score = _coerce_bounded_score(
        metadata.get("urgency_score", metadata.get("routing_urgency_score")),
        0,
        3,
        heuristic["routing_urgency_score"],
    )
    novelty_score = _coerce_bounded_score(
        metadata.get("novelty_score", metadata.get("routing_novelty_score")),
        0,
        2,
        heuristic["routing_novelty_score"],
    )
    must_keep_value = metadata.get("must_keep")
    if isinstance(must_keep_value, bool):
        must_keep = must_keep_value
    elif must_keep_value is None:
        must_keep = bool(heuristic["must_keep"])
    else:
        must_keep = str(must_keep_value).strip().lower() in {"1", "true", "yes"}
    if heuristic.get("high_confidence"):
        attention_tier = "high"
        must_keep = True
        market_impact_score = max(market_impact_score, 4)
        urgency_score = max(urgency_score, 2)
        novelty_score = max(novelty_score, 1)
    elif attention_tier == "high":
        attention_tier = "medium"
        attention_reason = "Downgraded to medium because the title/summary did not show a concrete market-moving catalyst."
    if must_keep and attention_tier == "light":
        attention_tier = "medium"
    research_lane = str(metadata.get("research_lane") or heuristic.get("research_lane") or "").strip() or _infer_research_lane(
        {**article, "theme": theme, "market_channel": market_channel, "attention_tier": attention_tier}
    )
    priority_score = _coerce_bounded_score(
        metadata.get("priority_score"),
        0,
        15,
        2 * market_impact_score + urgency_score + novelty_score,
    )
    return {
        "attention_tier": attention_tier,
        "theme": theme,
        "research_lane": research_lane,
        "attention_reason": attention_reason,
        "must_keep": must_keep,
        "market_channel": market_channel,
        "routing_market_impact_score": market_impact_score,
        "routing_urgency_score": urgency_score,
        "routing_novelty_score": novelty_score,
        "priority_score": priority_score,
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
    # A section label is a weak prior, not a reason to miss an exceptional
    # story. Clearly routine light items can still use the deterministic title
    # heuristic; route a light section when any item needs disambiguation.
    if not category_articles or len(category_articles) < ROUTER_LLM_MIN_ARTICLES:
        return False
    if category_name in LIGHT_ANALYSIS_SECTIONS:
        return any(
            str(article.get("attention_tier") or "medium").lower() != "light"
            for article in category_articles
        )
    return True


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
        runtime.record_split(category_name, "pre_send_budget", batch_kind="routing")
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
        llm_task=LLMTask.ROUTING.value,
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
        runtime.record_failure_classification(classification)
        runtime.tighten_category_budget(category_name, classification, batch_kind="routing")
        if classification == "payload_too_large":
            runtime.record_split(category_name, "response_413", batch_kind="routing")
        if len(batch_articles) > 1 and depth < 1:
            left, right = _split_batch(batch_articles)
            return _route_batch_recursive(runtime, category_name, left, batch_label=f"{batch_label}a", depth=depth + 1) + _route_batch_recursive(
                runtime, category_name, right, batch_label=f"{batch_label}b", depth=depth + 1
            )
        return [_apply_attention_metadata(article, _heuristic_attention_metadata(article)) for article in batch_articles]


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
                    "market_channel": routed.get("market_channel"),
                    "market_impact_score": routed.get("market_impact_score"),
                    "urgency_score": routed.get("urgency_score"),
                    "novelty_score": routed.get("novelty_score"),
                    "priority_score": routed.get("priority_score"),
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
    report_date: str | None = None,
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
            category_report, _ = _analyze_category(
                runtime,
                category_name,
                prepared_articles,
                report_date=report_date,
            )
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
        report_date=previous_date,
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
                key_developments = [
                    dev["text"]
                    for dev in _normalize_developments(synthesis_payload.get("key_developments"), valid_ids=None, limit=profile.category_bullet_limit)
                ]
                named_entities = _normalize_entities(synthesis_payload.get("named_entities"))[: profile.entity_limit]
                subgroups = subgroup_reports
                model_used = synthesis_model
                sub_batch_count += synthesis_batch_count
            except Exception as exc:
                category_error_message = str(exc)
                runtime.mark_degraded_merge(category_name, "synthesis_failed_local_fallback")
                fallback_developments = _fallback_developments_from_articles(
                    successful_articles,
                    limit=profile.category_bullet_limit,
                )
                key_developments = [dev["text"] for dev in fallback_developments]
                subgroups = [_local_fallback_subgroup(category_name, successful_articles, fallback_developments)]
                model_used = "local_fallback"
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
    top_alerts: list[Any],
    runtime: AnalysisRuntime | None,
    incremental: dict[str, int],
    total_scraped_count: int,
    market_snapshots: list[dict[str, Any]] | None = None,
    macro_release_digest: dict[str, Any] | None = None,
    legacy_executive_summary: list[str] | None = None,
    newly_analyzed_keys: set[tuple[str | None, str]] | None = None,
    validation_issues: list[dict[str, Any]] | None = None,
    theme_memory: dict[str, Any] | None = None,
    event_packets: list[dict[str, Any]] | None = None,
    review_queue: list[dict[str, Any]] | None = None,
    critic_issues: list[dict[str, Any]] | None = None,
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
        "executive_summary": _format_top_alerts_for_legacy(top_alerts),
        "executive_summary_structured": _normalize_top_alerts(top_alerts, {}),
        "legacy_executive_summary": legacy_executive_summary or [],
        "model": _report_model_metadata(runtime),
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
        "macro_release_digest": macro_release_digest or {},
        "validation_issues": validation_issues or [],
        "critic_issues": critic_issues or [],
        "theme_memory": theme_memory or {},
        "events": event_packets or [],
        "review_queue": review_queue or [],
        "event_pipeline": {
            "mode": os.environ.get("DAILY_MACRO_EVENT_PIPELINE_MODE", "hybrid"),
            "event_count": len(event_packets or []),
            "review_count": len(review_queue or []),
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


def _format_top_alerts_for_legacy(top_alerts: list[Any]) -> list[str]:
    formatted: list[str] = []
    for alert in top_alerts:
        if isinstance(alert, str):
            formatted.append(alert)
            continue
        if not isinstance(alert, dict):
            continue
        summary = str(alert.get("summary") or "").strip()
        if not summary:
            continue
        sources = alert.get("source_articles") if isinstance(alert.get("source_articles"), list) else []
        first_source = sources[0] if sources and isinstance(sources[0], dict) else None
        if first_source:
            title = str(first_source.get("title") or "").strip()
            date = str(first_source.get("date") or "").strip()
            url = str(first_source.get("url") or "").strip()
            if title and date and url:
                formatted.append(f"{summary} ({title} | {date}) {{{url}}}")
                continue
        formatted.append(summary)
    return formatted


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
    if "synthesis wait budget" in message or "synthesis retry budget" in message:
        return "synthesis_budget_exhausted"
    if "NoEligibleEndpoint" in message or "not available within" in message:
        return "no_eligible_endpoint"
    if "timed out" in message.lower() or "deadline" in message.lower():
        return "provider_timeout"
    if "connection" in message.lower() or "unavailable" in message.lower():
        return "provider_unavailable"
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


def _quality_review_enabled() -> bool:
    raw = os.environ.get("DAILY_MACRO_ENABLE_LLM_CRITIC")
    if raw is None:
        raw = config_value("DAILY_MACRO_ENABLE_LLM_CRITIC")
    return str(raw or "0").strip().lower() in {"1", "true", "yes", "on"}


def _quality_review_max_articles() -> int:
    raw = os.environ.get("DAILY_MACRO_QUALITY_REVIEW_MAX_ARTICLES")
    if raw is None:
        raw = config_value("DAILY_MACRO_QUALITY_REVIEW_MAX_ARTICLES")
    try:
        return max(0, min(50, int(raw or 12)))
    except (TypeError, ValueError):
        return 12


def _quality_review_sample_rate() -> float:
    raw = os.environ.get("DAILY_MACRO_QUALITY_REVIEW_SAMPLE_RATE")
    if raw is None:
        raw = config_value("DAILY_MACRO_QUALITY_REVIEW_SAMPLE_RATE")
    try:
        return max(0.0, min(1.0, float(raw or 0.10)))
    except (TypeError, ValueError):
        return 0.10


def _quality_review_model(runtime: AnalysisRuntime) -> ModelConfig | None:
    preferred_id = os.environ.get("DAILY_MACRO_QUALITY_REVIEW_MODEL")
    if preferred_id is None:
        preferred_id = config_value("DAILY_MACRO_QUALITY_REVIEW_MODEL")
    preferred_id = str(preferred_id or "openai/gpt-oss-120b").strip()
    for model in runtime.model_chain:
        if model.provider == DEFAULT_PROVIDER and model.model_id == preferred_id:
            return replace(model, max_completion_tokens=min(model.max_completion_tokens, 1024))
    return None


def _stable_sample_fraction(article: dict[str, Any]) -> float:
    key = _article_key(article.get("source_article_id"), article.get("canonical_url"))
    digest = hashlib.sha256("|".join(str(value or "") for value in key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _select_quality_review_articles(
    prepared_articles: list[dict[str, Any]],
    article_results: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    result_by_key = _article_result_by_key(article_results)
    priority: list[tuple[dict[str, Any], dict[str, Any]]] = []
    sample: list[tuple[dict[str, Any], dict[str, Any]]] = []
    sample_rate = _quality_review_sample_rate()
    for article in prepared_articles:
        key = _article_key(article.get("source_article_id"), article.get("canonical_url"))
        result = result_by_key.get(key)
        if not result or result.get("error") or str(result.get("attention_tier") or "medium") == "light":
            continue
        if bool(article.get("must_keep")) or str(article.get("attention_tier") or "medium") == "high":
            priority.append((article, result))
        elif _stable_sample_fraction(article) < sample_rate:
            sample.append((article, result))
    return (priority + sample)[: _quality_review_max_articles()]


def _normalize_quality_review(payload: dict[str, Any]) -> dict[str, Any]:
    verdict = str(payload.get("verdict") or "needs_review").strip().lower()
    if verdict not in {"pass", "needs_correction", "needs_review"}:
        verdict = "needs_review"

    def score(name: str) -> int:
        try:
            return max(1, min(5, int(payload.get(name) or 3)))
        except (TypeError, ValueError):
            return 3

    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0

    def text_list(name: str) -> list[str]:
        values = payload.get(name)
        if not isinstance(values, list):
            return []
        return [str(value).strip() for value in values if str(value).strip()][:5]

    return {
        "verdict": verdict,
        "factuality_score": score("factuality_score"),
        "completeness_score": score("completeness_score"),
        "financial_usefulness_score": score("financial_usefulness_score"),
        "language_fit_score": score("language_fit_score"),
        "issues": text_list("issues"),
        "corrections": text_list("corrections"),
        "confidence": round(confidence, 3),
    }


def _run_quality_reviews(
    runtime: AnalysisRuntime,
    category_name: str,
    prepared_articles: list[dict[str, Any]],
    article_results: list[dict[str, Any]],
    *,
    report_date: str | None,
) -> list[dict[str, Any]]:
    if not _quality_review_enabled():
        return article_results
    quality_model = _quality_review_model(runtime)
    if quality_model is None:
        LOGGER.info("Skipping LLM quality review for %s: Groq GPT-OSS 120B is unavailable.", category_name)
        return article_results

    result_by_key = _article_result_by_key(article_results)
    selected = _select_quality_review_articles(prepared_articles, article_results)
    for index, (article, first_pass) in enumerate(selected, start=1):
        completed_reviews = (
            runtime.diagnostics.quality_review_count
            + runtime.diagnostics.quality_review_failed_count
            + runtime.diagnostics.quality_review_skipped_count
        )
        if completed_reviews >= _quality_review_max_articles():
            break
        messages = _build_article_quality_review_messages(
            category_name,
            article,
            first_pass,
            report_date=report_date,
        )
        estimated_input_tokens = _estimate_messages_tokens(messages)
        context = BatchContext(
            category_name=category_name,
            batch_kind="quality_review",
            batch_label=f"quality-{index}",
            article_count=1,
            estimated_input_tokens=estimated_input_tokens,
            serialized_request_bytes=_estimate_request_payload_bytes(
                quality_model.model_id,
                messages,
                quality_model.max_completion_tokens,
            ),
            content_shrunk=bool(article.get("content_truncated")),
            llm_task=LLMTask.CRITIC.value,
        )
        _, expected_wait = runtime.governor.peek_key(
            quality_model.model_id,
            runtime.key_index_for_model(quality_model),
            runtime.key_count_for_model(quality_model),
            estimated_input_tokens,
            quota_scope=quality_model.quota_scope,
        )
        critic_wait_budget = runtime.resolver.wait_budget_seconds(LLMTask.CRITIC.value) if runtime.resolver else 90.0
        if expected_wait > critic_wait_budget:
            first_pass["quality_review"] = {
                "verdict": "needs_review",
                "status": "skipped",
                "reason": "rate_limit_wait",
            }
            first_pass["quality_review_model"] = quality_model.model_id
            runtime.diagnostics.quality_review_skipped_count += 1
            LOGGER.info(
                "Skipping quality review for %s article %s: expected Groq wait %.1fs exceeds %.1fs budget.",
                category_name,
                article.get("source_article_id") or article.get("canonical_url"),
                expected_wait,
                critic_wait_budget,
            )
            # All Groq credentials share the same organization quota; later
            # candidates would see the same wait, so stop this category's lane.
            break
        try:
            payload, model_used = _invoke_json_with_retry(
                runtime,
                messages,
                estimated_input_tokens,
                context,
                model_override=quality_model,
            )
            first_pass["quality_review"] = _normalize_quality_review(payload)
            first_pass["quality_review_model"] = model_used
            runtime.diagnostics.quality_review_count += 1
        except Exception as exc:  # noqa: BLE001 - review must not fail the report
            first_pass["quality_review"] = {
                "verdict": "needs_review",
                "status": "failed",
                "error": _classify_exception(exc),
            }
            first_pass["quality_review_model"] = quality_model.model_id
            runtime.diagnostics.quality_review_failed_count += 1
            LOGGER.warning(
                "Quality review failed for %s article %s: %s",
                category_name,
                article.get("source_article_id") or article.get("canonical_url"),
                exc,
            )
        result_by_key[_article_key(article.get("source_article_id"), article.get("canonical_url"))] = first_pass
    return [
        result_by_key.get(_article_key(result.get("source_article_id"), result.get("canonical_url")), result)
        for result in article_results
    ]


def _analyze_category(
    runtime: AnalysisRuntime,
    category_name: str,
    prepared_articles: list[dict[str, Any]],
    *,
    report_date: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime.reset_model_for_category()
    LOGGER.info("Analyzing category %s with %s article(s).", category_name, len(prepared_articles))
    profile = _section_profile(category_name)
    planned_batches = _plan_category_batches(runtime, category_name, prepared_articles)
    article_results: list[dict[str, Any]] = []
    article_errors: list[dict[str, Any]] = []
    sub_batch_count = 0
    
    # Analyze independent article batches with bounded fan-out when enabled.
    batch_results, batch_errors, batch_count = _run_article_batches(
        runtime,
        category_name,
        planned_batches,
    )
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
                "theme": article.get("theme"),
                "research_lane": article.get("research_lane") or "low_relevance",
                "attention_reason": article.get("attention_reason"),
                "must_keep": article.get("must_keep"),
                "market_channel": article.get("market_channel"),
                "routing_market_impact_score": article.get("routing_market_impact_score"),
                "routing_urgency_score": article.get("routing_urgency_score"),
                "routing_novelty_score": article.get("routing_novelty_score"),
                "priority_score": article.get("priority_score"),
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
    article_results = _run_quality_reviews(
        runtime,
        category_name,
        prepared_articles,
        article_results,
        report_date=report_date,
    )
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
            key_developments = [
                dev["text"]
                for dev in _normalize_developments(synthesis_payload.get("key_developments"), valid_ids=None, limit=profile.category_bullet_limit)
            ]
            if not key_developments:
                runtime.mark_degraded_merge(category_name, "empty_synthesis_local_fallback")
                fallback_developments = _fallback_developments_from_articles(
                    successful_articles,
                    limit=profile.category_bullet_limit,
                )
                key_developments = [dev["text"] for dev in fallback_developments]
            named_entities = _normalize_entities(synthesis_payload.get("named_entities"))[: profile.entity_limit]
        except Exception as exc:
            category_status = "partial"
            category_error_message = str(exc)
            runtime.mark_degraded_merge(category_name, "synthesis_failed_local_fallback")
            fallback_developments = _fallback_developments_from_articles(
                successful_articles,
                limit=profile.category_bullet_limit,
            )
            key_developments = [dev["text"] for dev in fallback_developments]
            subgroup_reports = [_local_fallback_subgroup(category_name, successful_articles, fallback_developments)]
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

    diagnostics.sub_batch_count = sub_batch_count
    diagnostics.partial_article_count = sum(1 for article in article_results if article.get("error"))

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


def _llm_parallelism() -> int:
    """Return the opt-in worker count for independent LLM batches."""
    raw = os.environ.get("DAILY_MACRO_LLM_PARALLELISM", "1")
    try:
        return max(1, min(16, int(raw)))
    except ValueError:
        return 1


def _run_article_batches(
    runtime: AnalysisRuntime,
    category_name: str,
    planned_batches: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Run independent article batches with deterministic fan-in.

    Each worker receives isolated model cursors, sessions, budgets, and
    diagnostics. The governor and daily ledger remain shared, so provider
    reservations are coordinated across workers.
    """
    work = [(index, batch) for index, batch in enumerate(planned_batches, start=1) if batch]
    if not work:
        return [], [], 0

    parallelism = min(_llm_parallelism(), len(work))
    if parallelism <= 1:
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        count = 0
        for index, batch in work:
            batch_results, batch_errors, batch_count = _process_batch_recursive(
                runtime, category_name, batch, str(index)
            )
            results.extend(batch_results)
            errors.extend(batch_errors)
            count += batch_count
        return results, errors, count

    runtime.diagnostics.parallel_worker_count = max(
        runtime.diagnostics.parallel_worker_count,
        parallelism,
    )
    runtime.diagnostics.parallel_batch_count += len(work)
    LOGGER.info(
        "Running %d independent article batch(es) for %s with %d worker(s).",
        len(work),
        category_name,
        parallelism,
    )

    def run_one(index: int, batch: list[dict[str, Any]]) -> tuple[int, AnalysisRuntime, tuple[list[dict[str, Any]], list[dict[str, Any]], int], BaseException | None]:
        worker = runtime.fork_for_worker()
        try:
            result = _process_batch_recursive(worker, category_name, batch, str(index))
            return index, worker, result, None
        except BaseException as exc:  # preserve the sequential path's failure behavior
            return index, worker, ([], [], 0), exc
        finally:
            worker.close_worker_sessions()

    completed: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]], int]] = {}
    with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="daily-macro-llm") as executor:
        futures = [executor.submit(run_one, index, batch) for index, batch in work]
        for future in as_completed(futures):
            index, worker, result, error = future.result()
            runtime.merge_worker_diagnostics(worker)
            if error is not None:
                raise error
            completed[index] = result

    results = []
    errors = []
    count = 0
    for index, _batch in work:
        batch_results, batch_errors, batch_count = completed[index]
        results.extend(batch_results)
        errors.extend(batch_errors)
        count += batch_count
    return results, errors, count


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
        runtime.record_split(category_name, "pre_send_budget", batch_kind="article_batch")
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
                next_model = runtime.next_model_after(active_model)
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
                next_model = runtime.next_model_after(active_model)
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
        runtime.record_failure_classification(classification)
        if status_code == 413 and len(working_batch) > 1:
            runtime.record_split(category_name, "response_413", batch_kind="article_batch")
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
        runtime.record_failure_classification(classification)
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
            next_model = runtime.next_model_after(active_model)
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
        llm_task=LLMTask.ARTICLE_ANALYSIS.value,
    )
    return _invoke_json_with_retry(runtime, messages, estimated_input_tokens, context, model_override=model_override)


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
        detailed_developments = _normalize_developments(
            summary_payload.get("key_developments"),
            valid_ids=_subgroup_article_ids(successful_articles),
            limit=profile.subgroup_bullet_limit,
        )
        if not detailed_developments:
            runtime.mark_degraded_merge(category_name, "empty_synthesis_local_fallback")
            detailed_developments = _fallback_developments_from_articles(
                successful_articles,
                limit=profile.subgroup_bullet_limit,
            )
        subgroup = {
            "title": f"{category_name} overview",
            "theme_rationale": "Single subgroup because the category size did not require thematic splitting.",
            "article_count": len(successful_articles),
            "key_developments": [dev["text"] for dev in detailed_developments],
            "key_developments_detailed": detailed_developments,
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
        detailed_developments = _normalize_developments(
            subgroup_payload.get("key_developments"),
            valid_ids=_subgroup_article_ids(subgroup["articles"]),
            limit=profile.subgroup_bullet_limit,
        )
        if not detailed_developments:
            runtime.mark_degraded_merge(category_name, "empty_synthesis_local_fallback")
            detailed_developments = _fallback_developments_from_articles(
                subgroup["articles"],
                limit=profile.subgroup_bullet_limit,
            )
        subgroup_reports.append(
            {
                "title": subgroup["title"],
                "theme_rationale": subgroup["theme_rationale"],
                "article_count": len(subgroup["articles"]),
                "key_developments": [dev["text"] for dev in detailed_developments],
                "key_developments_detailed": detailed_developments,
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
                "key_developments": _normalize_developments(subgroup_payload.get("key_developments"), valid_ids=None, limit=profile.subgroup_bullet_limit),
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
    category_developments: list[dict[str, Any]] = []
    seen_development_text: set[str] = set()
    for subgroup in subgroup_reports:
        detailed = _normalize_developments(
            subgroup.get("key_developments_detailed"),
            valid_ids=_subgroup_article_ids(subgroup.get("articles") or []),
            limit=profile.subgroup_bullet_limit,
        )
        for development in detailed:
            if development["text"] in seen_development_text:
                continue
            seen_development_text.add(development["text"])
            category_developments.append(development)
            if len(category_developments) >= profile.category_bullet_limit:
                break
        if len(category_developments) >= profile.category_bullet_limit:
            break

    category_payload = {
        "key_developments": category_developments,
        "key_developments_detailed": category_developments,
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
        runtime.record_split(category_name, "pre_send_budget", batch_kind="synthesis_grouping")
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
        llm_task=LLMTask.CATEGORY_SYNTHESIS.value,
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
            next_model = runtime.next_model_after(active_model)
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
        runtime.record_failure_classification(classification)
        runtime.tighten_category_budget(category_name, classification, batch_kind="synthesis")
        if len(successful_articles) > 1 and depth + 1 < profile.salvage_max_depth:
            if classification == "payload_too_large":
                runtime.record_split(category_name, "response_413", batch_kind="synthesis_grouping")
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
            runtime.record_failure_classification(classification)
            if len(batch) > 1 and classification in {"payload_too_large", "invalid_json", "unexpected_error"}:
                if classification == "payload_too_large":
                    runtime.record_split(category_name, "response_413", batch_kind="synthesis")
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
        llm_task=LLMTask.CATEGORY_SYNTHESIS.value,
    )
    return _invoke_json_with_retry(runtime, messages, estimated_input_tokens, context, model_override=model_override)


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
            runtime.record_split(category_name, "planned_batch_boundary", batch_kind="article_batch")
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
    # Keep the fallback for an all-light category for compatibility with the
    # existing direct-analysis path and its partial-result recovery behavior.
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
            runtime.record_split(category_name, "planned_batch_boundary", batch_kind="synthesis")
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
        "market_channel": article.get("market_channel"),
        "priority_score": article.get("priority_score"),
        "research_lane": article.get("research_lane"),
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
        "market_channel": article.get("market_channel"),
        "priority_score": article.get("priority_score"),
        "research_lane": article.get("research_lane"),
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
        # Keep developments as {text, source_article_ids} objects so per-development
        # provenance survives batching/merging instead of being flattened to
        # strings (which would drop source ids and could stringify to reprs).
        "label": label,
        "article_count": article_count,
        "key_developments": _normalize_developments(payload.get("key_developments"), valid_ids=None, limit=5),
        "named_entities": _normalize_entities(payload.get("named_entities"))[:6],
    }


def _merge_summary_items_locally(summary_items: list[dict[str, Any]], bullet_limit: int) -> dict[str, Any]:
    key_developments: list[dict[str, Any]] = []
    seen_developments: set[str] = set()
    seen_entities: set[str] = set()
    named_entities: list[dict[str, str]] = []

    for item in summary_items:
        for development in _normalize_developments(item.get("key_developments"), valid_ids=None, limit=bullet_limit):
            text = development["text"]
            if text in seen_developments:
                continue
            seen_developments.add(text)
            key_developments.append(development)
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
                "research_lane": article.get("research_lane"),
                "attention_reason": article.get("attention_reason"),
                "must_keep": article.get("must_keep"),
                "market_channel": article.get("market_channel"),
                "routing_market_impact_score": article.get("routing_market_impact_score"),
                "routing_urgency_score": article.get("routing_urgency_score"),
                "routing_novelty_score": article.get("routing_novelty_score"),
                "priority_score": article.get("priority_score"),
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
        "research_lane": article.get("research_lane"),
        "attention_reason": article.get("attention_reason"),
        "must_keep": article.get("must_keep"),
        "market_channel": article.get("market_channel"),
        "routing_market_impact_score": article.get("routing_market_impact_score"),
        "routing_urgency_score": article.get("routing_urgency_score"),
        "routing_novelty_score": article.get("routing_novelty_score"),
        "priority_score": article.get("priority_score"),
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


def _groq_fallback_model_ids() -> list[str]:
    """Return the date-aware Groq emergency chain without widening policy."""
    return provider_model_ids(DEFAULT_PROVIDER) or [PRIMARY_MODEL_ID]


def _report_model_metadata(runtime: AnalysisRuntime | None) -> dict[str, Any]:
    """Expose provider/model choices without exposing credentials."""
    if runtime is None or not runtime.model_chain:
        fallback_model_ids = _groq_fallback_model_ids()
        return {
            "provider": DEFAULT_PROVIDER,
            "primary_model": fallback_model_ids[0],
            "fallback_models": fallback_model_ids[1:],
        }
    providers = sorted({model.provider for model in runtime.model_chain})
    return {
        "provider": providers[0] if len(providers) == 1 else "multi",
        "providers": providers,
        "primary_model": runtime.primary_model.model_id,
        "fallback_models": [model.model_id for model in runtime.fallback_models],
        "endpoints": [
            {
                "endpoint_id": model.endpoint_id,
                "provider": model.provider,
                "account_id": model.account_id,
                "model_id": model.model_id,
                "quota_scope": model.quota_scope,
            }
            for model in runtime.model_chain
        ],
    }


def _build_empty_report(
    target_date: str,
    *,
    incremental: dict[str, int] | None = None,
    macro_release_digest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback_model_ids = _groq_fallback_model_ids()
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_date": target_date,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_site": DEFAULT_SOURCE_SITE,
        "status": "empty",
        "model": {
            "provider": DEFAULT_PROVIDER,
            "primary_model": fallback_model_ids[0],
            "fallback_models": fallback_model_ids[1:],
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
        "macro_release_digest": macro_release_digest or {},
        "validation_issues": [],
        "critic_issues": [],
        "theme_memory": {},
        "events": [],
        "review_queue": [],
        "event_pipeline": {
            "mode": os.environ.get("DAILY_MACRO_EVENT_PIPELINE_MODE", "hybrid"),
            "event_count": 0,
            "review_count": 0,
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
        "research_lane": article.get("research_lane"),
        "attention_reason": article.get("attention_reason"),
        "must_keep": article.get("must_keep"),
        "market_channel": article.get("market_channel"),
        "routing_market_impact_score": article.get("routing_market_impact_score"),
        "routing_urgency_score": article.get("routing_urgency_score"),
        "routing_novelty_score": article.get("routing_novelty_score"),
        "priority_score": article.get("priority_score"),
    }


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


_DEVELOPMENT_TEXT_KEYS = ("text", "development", "summary", "point")
_DEVELOPMENT_ID_KEYS = ("source_article_ids", "ref_ids", "source_article_id")


def _subgroup_article_ids(articles: list[dict[str, Any]]) -> set[str]:
    """Source-article ids the synthesis model was shown, for provenance validation."""
    return {str(a.get("source_article_id")) for a in articles if a.get("source_article_id")}


def _normalize_developments(
    value: Any,
    *,
    valid_ids: set[str] | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    """Normalize model ``key_developments`` output to ``[{text, source_article_ids}]``.

    Defends against model drift: a development may arrive as a plain sentence
    string or as a structured object (``{"development": ..., "figure": ...}``).
    A dict is coerced to its text field — never stringified into a Python repr,
    which is how malformed dict-dumps used to leak into alerts. Source ids are
    intersected with ``valid_ids`` (when provided) to drop hallucinated ids.
    """
    if not isinstance(value, list):
        return []

    developments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, dict):
            text = ""
            for key in _DEVELOPMENT_TEXT_KEYS:
                candidate = str(item.get(key) or "").strip()
                if candidate:
                    text = candidate
                    break
            raw_ids: Any = None
            for key in _DEVELOPMENT_ID_KEYS:
                if item.get(key) is not None:
                    raw_ids = item.get(key)
                    break
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            source_ids = [str(x).strip() for x in raw_ids if str(x).strip()] if isinstance(raw_ids, list) else []
        else:
            text = str(item).strip()
            source_ids = []

        if not text or text in seen:
            continue
        if valid_ids is not None:
            source_ids = [sid for sid in source_ids if sid in valid_ids]
        # Dedupe source ids while preserving order.
        source_ids = list(dict.fromkeys(source_ids))
        seen.add(text)
        developments.append({"text": text, "source_article_ids": source_ids})
        if len(developments) >= limit:
            break
    return developments


def _fallback_developments_from_articles(
    articles: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Build a provenance-preserving summary when synthesis returns no bullets.

    Historical runs frequently had complete article analysis but an empty
    category summary. A short local rollup is preferable to publishing a
    successful section with no usable developments.
    """
    developments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for article in articles:
        source_id = str(article.get("source_article_id") or "").strip()
        points = _normalize_string_list(article.get("key_points"), limit=4)
        if not points:
            title = str(article.get("title") or "").strip()
            points = [title] if title else []
        for point in points:
            text = _normalize_whitespace(point)
            if not text or text in seen:
                continue
            seen.add(text)
            developments.append(
                {
                    "text": text,
                    "source_article_ids": [source_id] if source_id else [],
                }
            )
            if len(developments) >= max(1, limit):
                return developments
    return developments


def _local_fallback_subgroup(
    category_name: str,
    articles: list[dict[str, Any]],
    developments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Represent a degraded local rollup as a normal subgroup for downstream use."""
    return {
        "title": f"{category_name} local rollup",
        "theme_rationale": "Local evidence rollup used because category synthesis was unavailable within the run budget.",
        "article_count": len(articles),
        "key_developments": [item["text"] for item in developments],
        "key_developments_detailed": developments,
        "named_entities": _collect_entities_from_articles(articles)[: _section_profile(category_name).entity_limit],
        "articles": articles,
        "model_used": "local_fallback",
    }


def _coerce_score(value: Any) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 5
    return max(1, min(10, numeric))


def _coerce_bounded_score(value: Any, lower: int, upper: int, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    return max(lower, min(upper, numeric))


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _article_key(source_article_id: Any, canonical_url: Any) -> tuple[str | None, str]:
    normalized_id = str(source_article_id) if source_article_id not in {None, ""} else None
    normalized_url = str(canonical_url or "")
    return normalized_id, normalized_url


def _article_group_key(article: dict[str, Any]) -> str:
    source_article_id = str(article.get("source_article_id") or "").strip()
    if source_article_id:
        return source_article_id
    return str(article.get("canonical_url") or "").strip()


def _infer_research_lane(article: dict[str, Any]) -> str:
    theme = str(article.get("theme") or "").lower()
    market_channel = str(article.get("market_channel") or "").lower()
    tier = str(article.get("attention_tier") or "").lower()
    section = str(article.get("article_section") or article.get("section") or "")
    title = str(article.get("title") or "").lower()
    haystack = f"{theme} {section} {title}"
    if tier == "light":
        return "low_relevance"
    if market_channel in {"rates", "fx", "macro"}:
        return "macro_policy"
    if market_channel == "commodity":
        return "commodities"
    if market_channel == "geopolitics":
        return "geopolitical_risk"
    if market_channel in {"equity", "property"}:
        return "hk_china_equity"
    if theme == "geopolitics":
        return "geopolitical_risk"
    if theme == "macro":
        if any(word in haystack for word in ("oil", "brent", "gold", "copper", "commodity", "油", "金", "銅")):
            return "commodities"
        return "macro_policy"
    if theme == "property":
        return "hk_china_equity"
    if theme == "stocks" or any(name in section for name in ("港股", "中國財經", "香港財經", "即巿股評")):
        return "hk_china_equity"
    if any(word in haystack for word in ("業績", "盈利", "guidance", "earnings", "profit")):
        return "company_specific"
    return "general_research"


_EVENT_STOPWORDS = {
    "about", "after", "amid", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "into", "is", "its", "of", "on", "or", "over", "that", "the",
    "to", "with", "will", "hong", "kong", "china", "market", "news", "今日", "香港", "中國",
}


def _event_tokens(value: Any) -> set[str]:
    text = str(value or "").casefold()
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", text)
    return {
        token for token in tokens
        if token not in _EVENT_STOPWORDS and (len(token) > 1 or token.isdigit())
    }


def _event_entities(article: dict[str, Any]) -> set[str]:
    entities: set[str] = set()
    for entity in article.get("named_entities") or []:
        if isinstance(entity, dict):
            name = str(entity.get("name") or "").strip().casefold()
        else:
            name = str(entity).strip().casefold()
        if name:
            entities.add(name)
    return entities


def _event_datetime(value: Any, fallback: str) -> datetime:
    raw = str(value or fallback).strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None)
    except ValueError:
        return datetime.fromisoformat(fallback).replace(tzinfo=None)


def _event_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_tokens = _event_tokens(
        " ".join([str(left.get("title") or ""), *(str(item) for item in left.get("key_points") or [])])
    )
    right_tokens = _event_tokens(
        " ".join([str(right.get("title") or ""), *(str(item) for item in right.get("key_points") or [])])
    )
    union = left_tokens | right_tokens
    overlap = len(left_tokens & right_tokens) / max(len(union), 1)
    shared_entities = _event_entities(left) & _event_entities(right)
    if shared_entities and overlap >= 0.30:
        return max(overlap, 0.7)
    return overlap


def _event_source_quality(article: dict[str, Any]) -> float:
    tier_score = {"high": 1.0, "medium": 0.75, "light": 0.45}.get(
        str(article.get("attention_tier") or "medium").lower(),
        0.65,
    )
    try:
        score = max(float(article.get("relevance_score") or 0), float(article.get("urgency_score") or 0)) / 10.0
    except (TypeError, ValueError):
        score = 0.5
    return round(min(1.0, max(tier_score, score)), 2)


def _build_event_packets(
    category_reports: list[dict[str, Any]],
    *,
    target_date: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Conservatively cluster analyzed articles into evidence-backed events."""
    candidates: list[dict[str, Any]] = []
    for category in category_reports:
        category_name = str(category.get("category") or "Uncategorized")
        for article in category.get("articles") or []:
            if article.get("error"):
                continue
            source_id = str(article.get("source_article_id") or article.get("canonical_url") or "").strip()
            if not source_id:
                continue
            candidate = dict(article)
            candidate["_category"] = category_name
            candidate["_published_dt"] = _event_datetime(article.get("published_at"), target_date)
            candidate["_event_tokens"] = _event_tokens(
                " ".join([str(article.get("title") or ""), *(str(item) for item in article.get("key_points") or [])])
            )
            candidate["_event_entities"] = _event_entities(article)
            candidates.append(candidate)

    clusters: list[dict[str, Any]] = []
    for candidate in candidates:
        best_cluster: dict[str, Any] | None = None
        best_similarity = 0.0
        for cluster in clusters:
            if candidate["_category"] != cluster["category"]:
                continue
            if candidate.get("research_lane") != cluster.get("research_lane"):
                continue
            if candidate.get("theme") != cluster.get("theme"):
                continue
            for existing in cluster["articles"]:
                hours = abs((candidate["_published_dt"] - existing["_published_dt"]).total_seconds()) / 3600.0
                if hours > 72:
                    continue
                similarity = _event_similarity(candidate, existing)
                if similarity >= 0.55 and similarity > best_similarity:
                    best_cluster = cluster
                    best_similarity = similarity
        if best_cluster is None:
            clusters.append(
                {
                    "category": candidate["_category"],
                    "theme": candidate.get("theme") or "general",
                    "research_lane": candidate.get("research_lane") or _infer_research_lane(candidate),
                    "articles": [candidate],
                }
            )
        else:
            best_cluster["articles"].append(candidate)

    packets: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    for cluster in clusters:
        articles = sorted(cluster["articles"], key=lambda item: item["_published_dt"], reverse=True)
        source_ids = [str(item.get("source_article_id") or item.get("canonical_url") or "") for item in articles]
        source_ids = list(dict.fromkeys(item for item in source_ids if item))
        anchor = articles[0]
        entity_names = []
        for article in articles:
            for entity in article.get("named_entities") or []:
                name = str(entity.get("name") if isinstance(entity, dict) else entity or "").strip()
                if name and name not in entity_names:
                    entity_names.append(name)
        facts: list[str] = []
        evidence: list[dict[str, Any]] = []
        best_sources: list[dict[str, Any]] = []
        for article in articles:
            article_id = str(article.get("source_article_id") or article.get("canonical_url") or "")
            for point in article.get("key_points") or []:
                text = str(point).strip()
                if text and text not in facts:
                    facts.append(text)
            article_evidence = [str(point).strip() for point in article.get("key_points") or [] if str(point).strip()]
            evidence.append(
                {
                    "source_article_id": article_id,
                    "title": str(article.get("title") or ""),
                    "url": str(article.get("canonical_url") or ""),
                    "claims": article_evidence[:3],
                }
            )
            best_sources.append(
                {
                    "source_article_id": article_id,
                    "title": str(article.get("title") or ""),
                    "url": str(article.get("canonical_url") or ""),
                    "published_at": str(article.get("published_at") or ""),
                    "quality": _event_source_quality(article),
                }
            )
        try:
            novelty = max(float(article.get("novelty_score") or 0) for article in articles) / 10.0
            market_relevance = max(
                max(float(article.get("relevance_score") or 0), float(article.get("urgency_score") or 0))
                for article in articles
            ) / 10.0
        except (TypeError, ValueError):
            novelty, market_relevance = 0.5, 0.5
        review_reasons: list[str] = []
        if len(source_ids) == 1 and market_relevance >= 0.7:
            review_reasons.append("single_source_high_impact")
        if not facts:
            review_reasons.append("missing_evidence_claims")
        if market_relevance >= 0.85 and len(source_ids) < 2:
            review_reasons.append("high_market_relevance_needs_confirmation")
        confidence = min(0.96, 0.52 + 0.10 * min(len(source_ids) - 1, 3) + 0.12 * max(_event_source_quality(anchor) - 0.5, 0))
        signature = "|".join(
            [
                str(cluster["category"]),
                str(cluster["research_lane"]),
                str(cluster["theme"]),
                ",".join(sorted(entity_names[:4])),
                " ".join(sorted(_event_tokens(anchor.get("title")))[0:5]),
            ]
        )
        event_id = "evt_" + hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
        packet = {
            "event_id": event_id,
            "event_title": str(anchor.get("title") or "Untitled event"),
            "category": cluster["category"],
            "research_lane": cluster["research_lane"],
            "theme": cluster["theme"],
            "source_article_ids": source_ids,
            "source_count": len(source_ids),
            "best_sources": best_sources[:5],
            "facts": facts[:8],
            "evidence": evidence,
            "affected_assets": entity_names[:12],
            "novelty": round(min(1.0, max(0.0, novelty)), 2),
            "market_relevance": round(min(1.0, max(0.0, market_relevance)), 2),
            "confidence": round(confidence, 2),
            "review_required": bool(review_reasons),
            "review_reasons": review_reasons,
        }
        packets.append(packet)
        if review_reasons:
            review_queue.append(
                {
                    "event_id": event_id,
                    "priority": "high" if market_relevance >= 0.8 else "normal",
                    "reasons": review_reasons,
                    "source_article_ids": source_ids,
                }
            )
    packets.sort(key=lambda item: (item.get("market_relevance", 0), item.get("confidence", 0)), reverse=True)
    review_queue.sort(key=lambda item: (item.get("priority") != "high", item.get("event_id") or ""))
    return packets, review_queue


def _theme_memory_file(data_dir: Path) -> Path:
    return data_dir / "theme_memory.json"


def _update_theme_memory_file(
    data_dir: Path,
    target_date: str,
    category_reports: list[dict[str, Any]],
    *,
    event_packets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path = _theme_memory_file(data_dir)
    try:
        memory = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"themes": {}}
    except (OSError, json.JSONDecodeError):
        memory = {"themes": {}}
    themes = memory.setdefault("themes", {})
    today_counts: dict[str, int] = defaultdict(int)
    today_articles: dict[str, list[str]] = defaultdict(list)
    today_events: dict[str, list[str]] = defaultdict(list)

    for category in category_reports:
        for article in category.get("articles") or []:
            if article.get("error"):
                continue
            theme = str(article.get("theme") or article.get("research_lane") or "general").strip() or "general"
            lane = str(article.get("research_lane") or _infer_research_lane(article))
            key = f"{lane}:{theme}"
            article_id = str(article.get("source_article_id") or article.get("canonical_url") or "")
            today_counts[key] += 1
            if article_id:
                today_articles[key].append(article_id)

    for event in event_packets or []:
        lane = str(event.get("research_lane") or "general_research")
        theme = str(event.get("theme") or "general")
        key = f"{lane}:{theme}"
        event_id = str(event.get("event_id") or "")
        if event_id:
            today_events[key].append(event_id)

    for key, count in today_counts.items():
        existing = themes.get(key) if isinstance(themes.get(key), dict) else {}
        previous_count = int(existing.get("last_count") or 0)
        related = list(dict.fromkeys([*(existing.get("related_articles") or []), *today_articles[key]]))[-50:]
        related_events = list(dict.fromkeys([*(existing.get("related_events") or []), *today_events[key]]))[-50:]
        themes[key] = {
            "theme": key.split(":", 1)[1],
            "research_lane": key.split(":", 1)[0],
            "first_seen": existing.get("first_seen") or target_date,
            "last_updated": target_date,
            "related_articles": related,
            "related_events": related_events,
            "trend": "strengthening" if count > previous_count else "unchanged",
            "confidence": round(min(0.95, 0.55 + 0.05 * count), 2),
            "last_count": count,
            "status": "open",
            "last_seen": target_date,
            "inactive_days": 0,
        }

    close_after_raw = os.environ.get("DAILY_MACRO_THEME_CLOSE_DAYS", "7")
    try:
        close_after_days = max(1, int(close_after_raw))
    except ValueError:
        close_after_days = 7
    target_dt = datetime.fromisoformat(target_date).date()
    for key, existing in themes.items():
        if key in today_counts or not isinstance(existing, dict):
            continue
        last_seen = str(existing.get("last_seen") or existing.get("last_updated") or "")
        try:
            inactive_days = max(0, (target_dt - datetime.fromisoformat(last_seen).date()).days)
        except ValueError:
            inactive_days = close_after_days
        existing["inactive_days"] = inactive_days
        existing["status"] = "closed" if inactive_days >= close_after_days else "cooling"
        existing["trend"] = "weakening"

    memory["last_updated"] = target_date
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Could not write theme memory at %s: %s", path, exc)
    return memory


def _build_market_context_for_report(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from .market import build_market_context_for_report

    return build_market_context_for_report(snapshots)


def _generate_top_alerts(
    runtime: AnalysisRuntime,
    market_context: str,
    developments: list[dict[str, Any]],
    article_metadata: dict[str, dict[str, str]],
    is_incremental: bool = False,
    event_packets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Use the shared LLM path to select 1-3 structured top alerts."""
    briefing_style = os.environ.get("DAILY_MACRO_BRIEFING_STYLE") or "CIO briefing"
    session_label = "intraday update" if is_incremental else "morning briefing"
    messages = [
        {
            "role": "system",
            "content": (
                f"You are preparing a {briefing_style} for a {session_label}. "
                "Return one valid JSON object only. Pick 1 to 3 genuinely important developments; fewer than 3 is allowed. "
                "Do not invent causal links or source ids."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Select the highest-value macro/equity alerts from pre-synthesized developments.",
                    "required_schema": {
                        "top_alerts": [
                            {
                                "summary": "one concise alert",
                                "why_it_matters": "specific market or portfolio implication",
                                "affected_assets": ["asset, index, FX pair, sector, or ticker"],
                                "time_horizon": "intraday|1w|1m|structural",
                                "confidence": "number from 0.0 to 1.0",
                                "source_article_ids": ["ids from ref_ids only"],
                            }
                        ]
                    },
                    "rules": [
                        "Return 1 to 3 alerts.",
                        "Use source_article_ids from the provided ref_ids only.",
                        "Prefer developments with concrete market, macro, or asset-price implications.",
                        "Preserve every source number's currency, unit, and scale exactly; do not convert 億/億元 into billions without checking the conversion.",
                        "Only emit ticker-like identifiers when they appear in the provided evidence or market context; otherwise use the company, sector, or commodity name.",
                        "Use cautious language for causal claims: co-movement is not proof that one event caused another.",
                    ],
                    "market_context": market_context,
                    "developments": developments,
                    "event_packets": [
                        {
                            "event_id": event.get("event_id"),
                            "event_title": event.get("event_title"),
                            "theme": event.get("theme"),
                            "research_lane": event.get("research_lane"),
                            "facts": list(event.get("facts") or [])[:4],
                            "affected_assets": list(event.get("affected_assets") or [])[:8],
                            "source_article_ids": list(event.get("source_article_ids") or []),
                            "confidence": event.get("confidence"),
                            "market_relevance": event.get("market_relevance"),
                            "review_required": event.get("review_required"),
                        }
                        for event in (event_packets or [])[:40]
                    ],
                    "article_metadata": article_metadata,
                },
                ensure_ascii=False,
            ),
        },
    ]
    estimated_input_tokens = _estimate_messages_tokens(messages)
    active_model = runtime.current_model
    context = BatchContext(
        category_name="top_alerts",
        batch_kind="top_alerts",
        batch_label="top",
        article_count=len(developments),
        estimated_input_tokens=estimated_input_tokens,
        serialized_request_bytes=_estimate_request_payload_bytes(active_model.model_id, messages, active_model.max_completion_tokens),
        content_shrunk=False,
        llm_task=LLMTask.TOP_ALERTS.value,
    )
    try:
        payload, _model_used = _invoke_json_with_retry(runtime, messages, estimated_input_tokens, context)
        alerts = _normalize_top_alerts(payload.get("top_alerts") or [], article_metadata)
        if alerts:
            return alerts
    except Exception as e:
        LOGGER.warning("Alert generation LLM call failed: %s", e)
    runtime.diagnostics.degraded_mode_count += 1
    return _fallback_top_alerts(developments, article_metadata)


def _normalize_top_alerts(alerts_data: Any, article_metadata: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    if not isinstance(alerts_data, list):
        return []
    alerts: list[dict[str, Any]] = []
    for raw in alerts_data[:3]:
        if isinstance(raw, str):
            raw = {"summary": raw}
        if not isinstance(raw, dict):
            continue
        summary = str(raw.get("summary") or "").strip()
        if not summary:
            continue
        raw_ids = raw.get("source_article_ids")
        if raw_ids is None and raw.get("ref_id"):
            raw_ids = [raw.get("ref_id")]
        source_ids = [str(item).strip() for item in (raw_ids if isinstance(raw_ids, list) else []) if str(item).strip()]
        source_articles = [
            {
                "source_article_id": source_id,
                "title": str(article_metadata.get(source_id, {}).get("title") or ""),
                "date": str(article_metadata.get(source_id, {}).get("date") or ""),
                "url": str(article_metadata.get(source_id, {}).get("url") or ""),
            }
            for source_id in source_ids
            if source_id in article_metadata
        ]
        if not source_articles and isinstance(raw.get("source_articles"), list):
            source_articles = [item for item in raw.get("source_articles") if isinstance(item, dict)]
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        normalized = {
            "summary": summary,
            "why_it_matters": str(raw.get("why_it_matters") or "").strip(),
            "affected_assets": _normalize_string_list(raw.get("affected_assets"), limit=8),
            "time_horizon": str(raw.get("time_horizon") or "1w").strip() or "1w",
            "confidence": max(0.0, min(1.0, confidence)),
            "source_article_ids": source_ids,
            "source_articles": source_articles,
        }
        if raw.get("critic_status"):
            normalized["critic_status"] = str(raw.get("critic_status"))
        if isinstance(raw.get("affected_asset_details"), list):
            normalized["affected_asset_details"] = [
                detail for detail in raw.get("affected_asset_details") if isinstance(detail, dict)
            ]
        alerts.append(normalized)
    return alerts


def _fallback_top_alerts(
    developments: list[dict[str, Any]],
    article_metadata: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    fallback: list[dict[str, Any]] = []
    for development in developments[:3]:
        ref_ids = [str(item).strip() for item in development.get("ref_ids", []) if str(item).strip()]
        fallback.extend(
            _normalize_top_alerts(
                [
                    {
                        "summary": str(development.get("text") or "").strip(),
                        "why_it_matters": "Selected by deterministic fallback after alert generation degraded.",
                        "affected_assets": [],
                        "time_horizon": "1w",
                        "confidence": 0.4,
                        "source_article_ids": ref_ids,
                    }
                ],
                article_metadata,
            )
        )
    return fallback[:3]
