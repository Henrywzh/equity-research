"""Compact, provider-neutral run-health notes for email and site output."""

from __future__ import annotations

from typing import Any


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    rounded = int(round(seconds))
    if rounded < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _phase_label(name: str) -> str:
    return {
        "storage_load": "storage",
        "fetch_market_data": "market data",
        "route_attention": "routing",
        "analyze_today": "article analysis",
        "retry_previous_day": "previous-day retry",
        "validate_outputs": "validation",
        "update_theme_memory": "theme memory",
        "summarize_top_alerts": "top alerts",
        "finalize": "finalization",
        "initialize": "initialization",
    }.get(name, name.replace("_", " "))


def build_run_notes(summary: dict[str, Any]) -> list[str]:
    diagnostics = summary.get("diagnostics") or {}
    totals = summary.get("totals") or {}
    unresolved = list(summary.get("unresolved_articles") or [])
    notes: list[str] = []

    status = str(summary.get("status") or "").lower()
    article_count = int(totals.get("article_count") or 0)
    failed_articles = int(totals.get("failed_article_analyses") or len(unresolved))
    partial_categories = int(totals.get("partial_categories") or 0)
    successful_categories = int(totals.get("successful_categories") or 0)
    if status == "partial":
        if failed_articles:
            notes.append(
                f"Digest is partial: {failed_articles}/{article_count} article(s) remained unresolved after salvage."
            )
        elif partial_categories:
            category_label = "category" if partial_categories == 1 else "categories"
            notes.append(
                f"Digest is partial at synthesis level: {partial_categories} {category_label} were partial; "
                f"article analysis covered {article_count}/{article_count}."
            )
        else:
            notes.append("Digest is partial; no article-level losses were recorded.")
    elif not failed_articles:
        notes.append(f"Article coverage was complete: {article_count}/{article_count} analyzed.")

    wall_clock = float(diagnostics.get("wall_clock_seconds") or 0.0)
    request_seconds = float(diagnostics.get("llm_request_seconds_total") or 0.0)
    wait_seconds = float(diagnostics.get("rate_limit_wait_seconds_total") or 0.0)
    wait_count = int(diagnostics.get("rate_limit_wait_count") or 0)
    timeout_seconds = float(diagnostics.get("timeout_seconds_total") or 0.0)
    if wall_clock:
        notes.append(f"Wall-clock runtime: {_format_duration(wall_clock)}.")
        if request_seconds or wait_seconds or timeout_seconds:
            llm_label = f"LLM calls {_format_duration(request_seconds)}" if request_seconds else "LLM calls unmeasured"
            if timeout_seconds:
                llm_label += f" (including {_format_duration(timeout_seconds)} timeout time)"
            breakdown = [llm_label]
            if wait_seconds:
                breakdown.append(
                    f"rate-limit/backoff waits {_format_duration(wait_seconds)} "
                    f"({wait_count} event(s))"
                )
            parallel_workers = int(diagnostics.get("parallel_worker_count") or 1)
            if parallel_workers <= 1:
                other_seconds = max(0.0, wall_clock - request_seconds - wait_seconds)
                breakdown.append(f"other work {_format_duration(other_seconds)}")
            else:
                breakdown.append(f"parallel workers {parallel_workers}")
            notes.append("Time breakdown: " + "; ".join(breakdown) + ".")

        phases = diagnostics.get("phase_seconds") or {}
        phase_items = sorted(
            ((name, float(seconds)) for name, seconds in phases.items() if float(seconds) >= 0.5),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        if phase_items:
            notes.append(
                "Graph phases: "
                + "; ".join(f"{_phase_label(name)} {_format_duration(seconds)}" for name, seconds in phase_items)
                + "."
            )

    if wait_count and not wall_clock:
        notes.append(f"Rate-limit/backoff waits: {wait_count} event(s), {_format_duration(wait_seconds)} total.")
    timeout_count = int(diagnostics.get("request_timeout_count") or 0)
    if timeout_count:
        cooldown_count = int(diagnostics.get("endpoint_cooldown_count") or 0)
        suffix = f"; {cooldown_count} endpoint(s) cooled down" if cooldown_count else ""
        notes.append(f"LLM request timeouts: {timeout_count}{suffix}.")

    truncated_count = int(totals.get("truncated_article_count") or 0)
    if truncated_count:
        notes.append(f"{truncated_count} article(s) were truncated to fit the working request budget.")

    pre_send_splits = int(diagnostics.get("pre_send_split_count") or 0)
    response_413_splits = int(diagnostics.get("response_413_split_count") or 0)
    if pre_send_splits or response_413_splits:
        notes.append(
            "Batch handling: "
            f"{pre_send_splits} pre-send split(s), {response_413_splits} HTTP 413 split(s), "
            f"{int(diagnostics.get('llm_request_count') or 0)} LLM request attempt(s)."
        )

    fallback_switches = int(diagnostics.get("fallback_switch_count") or 0)
    switches = list(summary.get("model_switches") or [])
    if fallback_switches:
        if switches:
            same_model_switches = sum(
                1 for item in switches if item.get("from_model") and item.get("from_model") == item.get("to_model")
            )
            evictions = max(fallback_switches - len(switches), 0)
            eviction_suffix = f"; {evictions} pool eviction(s)" if evictions else ""
            notes.append(
                f"Model switch summary: {len(switches)} endpoint transition(s) "
                f"({same_model_switches} same-model account move(s)){eviction_suffix}."
            )
        else:
            notes.append(f"Model switch summary: fallback endpoints used {fallback_switches} time(s).")

    error_classes = sorted({
        str(item.get("classification") or "unclassified")
        for item in summary.get("errors") or []
    })
    if error_classes:
        notes.append("Failure classifications: " + ", ".join(error_classes) + ".")

    if not notes:
        notes.append(f"All {successful_categories or article_count} report unit(s) completed successfully.")

    output_path = summary.get("output_path")
    if output_path:
        notes.append(f"Stored analysis report: {output_path}")
    return notes
