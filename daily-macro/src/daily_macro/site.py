from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

from .analysis import REPORT_FILE_NAME
from .config import get_analysis_dir, get_project_root

LOGGER = logging.getLogger(__name__)


def build_site(
    *,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    analysis_dir = get_analysis_dir(data_dir)
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve() if output_dir is not None else (get_project_root() / "site").resolve()
    )
    report_paths = sorted(
        analysis_dir.glob(f"*/{REPORT_FILE_NAME}"),
        key=lambda path: path.parent.name,
        reverse=True,
    )

    if resolved_output_dir.exists():
        shutil.rmtree(resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    if not report_paths:
        return {
            "status": "empty",
            "report_count": 0,
            "latest_report_date": None,
            "output_dir": str(resolved_output_dir),
            "generated_files": [],
        }

    reports: list[dict[str, Any]] = []
    skipped_reports: list[str] = []
    for path in report_paths:
        try:
            reports.append(_sanitize_report(json.loads(path.read_text(encoding="utf-8"))))
        except json.JSONDecodeError as exc:
            skipped_reports.append(str(path))
            LOGGER.warning("Skipping unreadable analysis report at %s: %s", path, exc)

    if not reports:
        return {
            "status": "empty",
            "report_count": 0,
            "latest_report_date": None,
            "output_dir": str(resolved_output_dir),
            "generated_files": [],
            "skipped_reports": skipped_reports,
        }

    reports.sort(key=lambda item: str(item.get("report_date") or ""), reverse=True)

    generated_files: list[str] = []
    _write_text_file(resolved_output_dir / ".nojekyll", "", generated_files)
    _write_text_file(resolved_output_dir / "assets" / "styles.css", _render_stylesheet(), generated_files)

    archive_entries = [_build_archive_entry(report) for report in reports]
    _write_json_file(resolved_output_dir / "archive.json", archive_entries, generated_files)

    for report in reports:
        report_date = str(report.get("report_date") or "unknown-date")
        report_dir = resolved_output_dir / "reports" / report_date
        _write_json_file(report_dir / "report.json", report, generated_files)
        _write_text_file(
            report_dir / "index.html",
            _render_report_page(report, page_title=f"Daily Macro {report_date}", nav_prefix="../../"),
            generated_files,
        )

    latest_report = reports[0]
    latest_html = _render_report_page(latest_report, page_title="Daily Macro Today", nav_prefix="")
    _write_text_file(resolved_output_dir / "index.html", latest_html, generated_files)
    _write_text_file(
        resolved_output_dir / "today" / "index.html",
        _render_report_page(latest_report, page_title="Daily Macro Today", nav_prefix="../"),
        generated_files,
    )
    _write_text_file(
        resolved_output_dir / "archive" / "index.html",
        _render_archive_page(archive_entries),
        generated_files,
    )

    return {
        "status": "success",
        "report_count": len(reports),
        "latest_report_date": latest_report.get("report_date"),
        "output_dir": str(resolved_output_dir),
        "generated_files": generated_files,
        "skipped_reports": skipped_reports,
    }


def _sanitize_report(report: dict[str, Any]) -> dict[str, Any]:
    totals = report.get("totals") or {}
    diagnostics = report.get("diagnostics") or {}
    safe_report = {
        "report_date": report.get("report_date"),
        "generated_at": report.get("generated_at"),
        "status": report.get("status"),
        "source_site": report.get("source_site"),
        "report_schema_version": report.get("report_schema_version"),
        "executive_summary": _string_list(report.get("executive_summary")),
        "market_context": _string_list(report.get("market_context")),
        "daily_stats": {
            "total_scraped": (report.get("daily_stats") or {}).get("total_scraped", 0),
            "analyzed": (report.get("daily_stats") or {}).get("analyzed", 0),
            "success_rate": (report.get("daily_stats") or {}).get("success_rate", 0),
        },
        "totals": {
            "article_count": totals.get("article_count", 0),
            "successful_article_analyses": totals.get("successful_article_analyses", 0),
            "failed_article_analyses": totals.get("failed_article_analyses", 0),
            "full_text_article_count": totals.get("full_text_article_count", 0),
            "truncated_article_count": totals.get("truncated_article_count", 0),
            "successful_categories": totals.get("successful_categories", 0),
            "partial_categories": totals.get("partial_categories", 0),
            "failed_categories": totals.get("failed_categories", 0),
        },
        "diagnostics": {
            "rate_limit_wait_count": diagnostics.get("rate_limit_wait_count", 0),
            "rate_limit_wait_seconds_total": diagnostics.get("rate_limit_wait_seconds_total", 0),
            "pre_send_split_count": diagnostics.get("pre_send_split_count", 0),
            "response_413_split_count": diagnostics.get("response_413_split_count", 0),
            "fallback_switch_count": diagnostics.get("fallback_switch_count", 0),
            "delayed_retry_candidate_count": diagnostics.get("delayed_retry_candidate_count", 0),
            "delayed_retry_attempted_count": diagnostics.get("delayed_retry_attempted_count", 0),
            "delayed_retry_recovered_count": diagnostics.get("delayed_retry_recovered_count", 0),
            "delayed_retry_failed_count": diagnostics.get("delayed_retry_failed_count", 0),
            "high_medium_unresolved_count": diagnostics.get("high_medium_unresolved_count", 0),
            "light_unresolved_count": diagnostics.get("light_unresolved_count", 0),
            "delayed_retry_skipped_final_model_count": diagnostics.get("delayed_retry_skipped_final_model_count", 0),
        },
        "model": _sanitize_model(report.get("model") or {}),
        "model_switches": [_sanitize_model_switch(item) for item in report.get("model_switches") or []],
        "categories": [_sanitize_category(item) for item in report.get("categories") or []],
        "unresolved_articles": [_sanitize_unresolved_article(item) for item in report.get("unresolved_articles") or []],
    }
    return safe_report


def _sanitize_model(model: dict[str, Any]) -> dict[str, Any]:
    fallback_models = model.get("fallback_models")
    if fallback_models is None and model.get("fallback_model") is not None:
        fallback_models = [model.get("fallback_model")]
    return {
        "provider": model.get("provider"),
        "primary_model": model.get("primary_model"),
        "fallback_models": [str(item) for item in fallback_models or []],
    }


def _sanitize_model_switch(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "from_model": item.get("from_model"),
        "to_model": item.get("to_model"),
        "reason": item.get("reason"),
    }


def _sanitize_category(category: dict[str, Any]) -> dict[str, Any]:
    diagnostics = category.get("diagnostics") or {}
    return {
        "category": category.get("category"),
        "status": category.get("status"),
        "analysis_profile": category.get("analysis_profile") or "standard",
        "article_count": category.get("article_count", 0),
        "key_developments": _string_list(category.get("key_developments")),
        "named_entities": [_sanitize_entity(item) for item in category.get("named_entities") or []],
        "diagnostics": {
            "sub_batch_count": diagnostics.get("sub_batch_count", category.get("sub_batch_count", 0)),
            "split_reasons": [str(item) for item in diagnostics.get("split_reasons") or []],
            "partial_article_count": diagnostics.get("partial_article_count", 0),
        },
        "subgroups": [_sanitize_subgroup(item) for item in category.get("subgroups") or []],
        "articles": [_sanitize_article(item) for item in category.get("articles") or []],
    }


def _sanitize_subgroup(subgroup: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": subgroup.get("title"),
        "theme_rationale": subgroup.get("theme_rationale"),
        "article_count": subgroup.get("article_count", 0),
        "key_developments": _string_list(subgroup.get("key_developments")),
        "named_entities": [_sanitize_entity(item) for item in subgroup.get("named_entities") or []],
        "articles": [_sanitize_article(item) for item in subgroup.get("articles") or []],
    }


def _sanitize_article(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": article.get("title"),
        "canonical_url": article.get("canonical_url"),
        "published_at": article.get("published_at"),
        "attention_tier": article.get("attention_tier") or "medium",
        "theme": article.get("theme"),
        "section": article.get("section"),
        "key_points": _string_list(article.get("key_points")),
        "error": article.get("error"),
        "error_classification": article.get("error_classification"),
    }


def _sanitize_unresolved_article(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": article.get("category"),
        "title": article.get("title"),
        "canonical_url": article.get("canonical_url"),
        "source_article_id": article.get("source_article_id"),
        "attention_tier": article.get("attention_tier") or "medium",
        "theme": article.get("theme"),
        "must_keep": bool(article.get("must_keep")),
        "error_classification": article.get("error_classification"),
        "error": article.get("error"),
        "model_used": article.get("model_used"),
        "delayed_retry_attempted": bool(article.get("delayed_retry_attempted")),
        "delayed_retry_model_chain": [str(item) for item in article.get("delayed_retry_model_chain") or []],
        "delayed_retry_final_model": article.get("delayed_retry_final_model"),
        "published_at": article.get("published_at"),
    }


def _sanitize_entity(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": entity.get("name"),
        "type": entity.get("type"),
    }


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value or [] if str(item).strip()]


def _build_archive_entry(report: dict[str, Any]) -> dict[str, Any]:
    totals = report.get("totals") or {}
    unresolved = list(report.get("unresolved_articles") or [])
    return {
        "report_date": report.get("report_date"),
        "status": report.get("status"),
        "article_count": totals.get("article_count", 0),
        "successful_categories": totals.get("successful_categories", 0),
        "partial_categories": totals.get("partial_categories", 0),
        "failed_categories": totals.get("failed_categories", 0),
        "unresolved_count": len(unresolved),
        "relative_url": f"reports/{report.get('report_date')}/",
        "executive_summary": _string_list(report.get("executive_summary"))[:2],
    }


def _write_text_file(path: Path, content: str, generated_files: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    generated_files.append(str(path))


def _write_json_file(path: Path, payload: Any, generated_files: list[str]) -> None:
    _write_text_file(path, json.dumps(payload, ensure_ascii=False, indent=2), generated_files)


def _render_report_page(report: dict[str, Any], *, page_title: str, nav_prefix: str) -> str:
    report_date = escape(str(report.get("report_date") or ""))
    categories = list(report.get("categories") or [])
    totals = report.get("totals") or {}
    diagnostics = report.get("diagnostics") or {}
    unresolved_articles = list(report.get("unresolved_articles") or [])
    executive_summary = _render_bullet_list(report.get("executive_summary") or [], fallback="No executive summary was produced.")
    market_context = _render_bullet_list(report.get("market_context") or [], fallback="No additional market context was captured.")
    unresolved_section = _render_unresolved_section(unresolved_articles)
    category_sections = "".join(_render_category_block(category) for category in categories)
    notes = _build_run_notes(report)
    cards = [
        ("Status", str(report.get("status") or "unknown").upper()),
        ("Articles", str(totals.get("article_count", 0))),
        ("Coverage", f"{(report.get('daily_stats') or {}).get('analyzed', 0)} / {(report.get('daily_stats') or {}).get('total_scraped', 0)}"),
        ("Unresolved", str(len(unresolved_articles))),
        ("Rate-limit waits", str(diagnostics.get("rate_limit_wait_count", 0))),
        ("Truncated", str(totals.get("truncated_article_count", 0))),
    ]
    cards_html = "".join(
        f"<div class='metric-card'><div class='metric-label'>{escape(label)}</div><div class='metric-value'>{escape(value)}</div></div>"
        for label, value in cards
    )
    model_chain = [str((report.get("model") or {}).get("primary_model") or "")]
    model_chain.extend(str(item) for item in (report.get("model") or {}).get("fallback_models") or [])
    model_chain = [item for item in model_chain if item]
    model_html = " → ".join(escape(item) for item in model_chain) if model_chain else "Unavailable"
    site_title = escape(page_title)
    nav = _render_nav(nav_prefix)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{site_title}</title>
  <link rel="stylesheet" href="{escape(nav_prefix)}assets/styles.css">
</head>
<body>
  <div class="page-shell">
    {nav}
    <header class="hero">
      <p class="eyebrow">Daily Macro</p>
      <h1>{site_title}</h1>
      <p class="hero-meta">Report date {report_date} • Source {escape(str(report.get("source_site") or "hkej"))}</p>
      <p class="hero-copy">One shared daily pipeline, published as a read-only archive.</p>
    </header>
    <section class="metrics-grid">{cards_html}</section>
    <section class="panel">
      <h2>Run notes</h2>
      <ul class="compact-list">{''.join(f'<li>{escape(item)}</li>' for item in notes)}</ul>
      <p class="muted"><strong>Models:</strong> {model_html}</p>
    </section>
    {unresolved_section}
    <section class="panel">
      <h2>Executive summary</h2>
      {executive_summary}
    </section>
    <section class="panel">
      <h2>Market context</h2>
      {market_context}
    </section>
    <section class="panel">
      <h2>Sections</h2>
      {category_sections}
    </section>
  </div>
</body>
</html>
"""


def _render_archive_page(entries: list[dict[str, Any]]) -> str:
    cards = []
    for entry in entries:
        summary = _render_bullet_list(entry.get("executive_summary") or [], fallback="No executive summary.")
        cards.append(
            f"""
            <article class="archive-card">
              <div class="archive-card-header">
                <h2><a href="../{escape(str(entry.get('relative_url') or ''))}">{escape(str(entry.get('report_date') or 'Unknown date'))}</a></h2>
                <span class="status-badge status-{escape(str(entry.get('status') or 'unknown').lower())}">{escape(str(entry.get('status') or 'unknown').upper())}</span>
              </div>
              <p class="muted">{escape(str(entry.get('article_count', 0)))} articles • {escape(str(entry.get('unresolved_count', 0)))} unresolved</p>
              {summary}
            </article>
            """
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Daily Macro Archive</title>
  <link rel="stylesheet" href="../assets/styles.css">
</head>
<body>
  <div class="page-shell">
    {_render_nav('../')}
    <header class="hero">
      <p class="eyebrow">Daily Macro</p>
      <h1>Archive</h1>
      <p class="hero-copy">Browse previously published shared reports.</p>
    </header>
    <section class="archive-grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>
"""


def _render_nav(prefix: str) -> str:
    return (
        "<nav class='top-nav'>"
        f"<a href='{escape(prefix)}index.html'>Today</a>"
        f"<a href='{escape(prefix)}archive/'>Archive</a>"
        "</nav>"
    )


def _render_category_block(category: dict[str, Any]) -> str:
    category_name = escape(str(category.get("category") or "Untitled section"))
    status = escape(str(category.get("status") or "unknown").lower())
    profile = escape(str(category.get("analysis_profile") or "standard"))
    key_developments = _render_bullet_list(category.get("key_developments") or [], fallback="No section summary was produced.")
    named_entities = ", ".join(
        escape(str(entity.get("name") or ""))
        for entity in category.get("named_entities") or []
        if str(entity.get("name") or "").strip()
    )
    subgroups = list(category.get("subgroups") or [])
    if not subgroups:
        subgroups = [
            {
                "title": category.get("category"),
                "theme_rationale": "",
                "article_count": category.get("article_count", 0),
                "key_developments": category.get("key_developments") or [],
                "named_entities": category.get("named_entities") or [],
                "articles": category.get("articles") or [],
            }
        ]
    subgroup_html = "".join(_render_subgroup_block(subgroup, force_plain_title=(len(subgroups) == 1)) for subgroup in subgroups)
    entity_line = f"<p class='muted'><strong>Named entities:</strong> {named_entities}</p>" if named_entities else ""
    diagnostics = category.get("diagnostics") or {}
    diagnostics_line = (
        "<p class='muted'><strong>Diagnostics:</strong> "
        f"sub-batches {escape(str(diagnostics.get('sub_batch_count', 0)))}, "
        f"partial articles {escape(str(diagnostics.get('partial_article_count', 0)))}, "
        f"splits {escape(', '.join(diagnostics.get('split_reasons') or []) or 'none')}"
        "</p>"
    )
    return f"""
    <section class="category-card">
      <div class="section-header">
        <div>
          <h3>{category_name}</h3>
          <p class="muted">Profile: {profile}</p>
        </div>
        <span class="status-badge status-{status}">{status.upper()}</span>
      </div>
      {key_developments}
      {entity_line}
      {diagnostics_line}
      <div class="subgroup-stack">
        {subgroup_html}
      </div>
    </section>
    """


def _render_subgroup_block(subgroup: dict[str, Any], *, force_plain_title: bool) -> str:
    title = escape(str(subgroup.get("title") or "Subgroup"))
    rationale = escape(str(subgroup.get("theme_rationale") or ""))
    rationale_html = f"<p class='muted'>{rationale}</p>" if rationale and not force_plain_title else ""
    key_developments = _render_bullet_list(subgroup.get("key_developments") or [], fallback="No subgroup summary was produced.")
    articles = "".join(_render_article_line(article) for article in subgroup.get("articles") or [])
    return f"""
    <article class="subgroup-card">
      <h4>{title}</h4>
      {rationale_html}
      {key_developments}
      <ul class="article-list">{articles}</ul>
    </article>
    """


def _render_article_line(article: dict[str, Any]) -> str:
    title = escape(str(article.get("title") or "Untitled article"))
    url = escape(str(article.get("canonical_url") or "#"))
    published_at = escape(str(article.get("published_at") or ""))
    attention = escape(str(article.get("attention_tier") or "medium").upper())
    error = str(article.get("error") or "")
    error_classification = str(article.get("error_classification") or "")
    badges = [f"<span class='pill pill-{attention.lower()}'>{attention}</span>"]
    if error:
        badges.append("<span class='pill pill-unresolved'>UNRESOLVED</span>")
    badges_html = "".join(badges)
    points = _render_bullet_list(article.get("key_points") or [], fallback="", compact=True)
    error_html = ""
    if error:
        reason = escape(error_classification or error)
        detail = escape(error)
        error_html = f"<p class='article-error'><strong>{reason}</strong>: {detail}</p>"
    return (
        "<li class='article-item'>"
        f"<div class='article-main'><a href='{url}' target='_blank' rel='noopener noreferrer'>{title}</a>{badges_html}</div>"
        f"<div class='article-meta'>{published_at}</div>"
        f"{points}"
        f"{error_html}"
        "</li>"
    )


def _render_unresolved_section(unresolved_articles: list[dict[str, Any]]) -> str:
    if not unresolved_articles:
        return ""

    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        tier = str(item.get("attention_tier") or "medium").lower()
        order = {"high": 0, "medium": 1, "light": 2}.get(tier, 3)
        return (order, str(item.get("category") or ""), str(item.get("title") or ""))

    items_html = []
    for article in sorted(unresolved_articles, key=sort_key):
        tier = escape(str(article.get("attention_tier") or "medium").upper())
        category = escape(str(article.get("category") or "Unknown section"))
        title = escape(str(article.get("title") or "Untitled article"))
        url = escape(str(article.get("canonical_url") or "#"))
        classification = escape(str(article.get("error_classification") or "unknown"))
        detail = escape(str(article.get("error") or ""))
        delayed_retry = ""
        if article.get("delayed_retry_attempted"):
            final_model = escape(str(article.get("delayed_retry_final_model") or "normal chain"))
            delayed_retry = f"<p class='article-error'>Delayed retry attempted and still unresolved. Final model: {final_model}.</p>"
        items_html.append(
            "<li class='unresolved-item'>"
            f"<div class='article-main'><a href='{url}' target='_blank' rel='noopener noreferrer'>{title}</a>"
            f"<span class='pill pill-{tier.lower()}'>{tier}</span>"
            f"<span class='pill pill-unresolved'>{category}</span></div>"
            f"<p class='article-error'><strong>{classification}</strong>: {detail}</p>"
            f"{delayed_retry}"
            "</li>"
        )
    return (
        "<section class='panel warning-panel'>"
        "<h2>Unresolved articles</h2>"
        "<p class='muted'>These stories remained unresolved after the normal salvage path. Review the links directly if they matter to you.</p>"
        f"<ul class='unresolved-list'>{''.join(items_html)}</ul>"
        "</section>"
    )


def _build_run_notes(report: dict[str, Any]) -> list[str]:
    totals = report.get("totals") or {}
    diagnostics = report.get("diagnostics") or {}
    unresolved = list(report.get("unresolved_articles") or [])
    notes: list[str] = []
    if unresolved:
        notes.append(f"This digest is partial. {len(unresolved)} article(s) remained unresolved.")
    truncated = int(totals.get("truncated_article_count", 0) or 0)
    if truncated:
        notes.append(f"{truncated} article(s) were truncated to fit the working request budget.")
    wait_count = int(diagnostics.get("rate_limit_wait_count", 0) or 0)
    if wait_count:
        wait_seconds = float(diagnostics.get("rate_limit_wait_seconds_total", 0.0) or 0.0)
        notes.append(f"Analysis paused {wait_count} time(s) for rate limits, totaling about {wait_seconds:.1f} second(s).")
    pre_send = int(diagnostics.get("pre_send_split_count", 0) or 0)
    after_413 = int(diagnostics.get("response_413_split_count", 0) or 0)
    if pre_send or after_413:
        notes.append(f"Oversized batches were split (pre-send: {pre_send}, after HTTP 413: {after_413}).")
    switches = int(diagnostics.get("fallback_switch_count", 0) or 0)
    if switches:
        notes.append(f"Fallback models were used {switches} time(s) during this run.")
    skipped_final = int(diagnostics.get("delayed_retry_skipped_final_model_count", 0) or 0)
    if skipped_final:
        notes.append(f"The final delayed-retry model was skipped {skipped_final} time(s) because OPENAI_API_KEY was unavailable.")
    classifications = Counter(
        str(article.get("error_classification") or "unknown")
        for article in unresolved
        if str(article.get("error_classification") or "").strip()
    )
    if classifications:
        labels = ", ".join(sorted(classifications))
        notes.append(f"Failure classifications present: {labels}.")
    if not notes:
        notes.append("All articles were published successfully with no unresolved items.")
    return notes


def _render_bullet_list(items: list[str], *, fallback: str, compact: bool = False) -> str:
    if not items:
        return f"<p class='muted'>{escape(fallback)}</p>" if fallback else ""
    class_name = "compact-list" if compact else "bullet-list"
    return f"<ul class='{class_name}'>{''.join(f'<li>{escape(item)}</li>' for item in items)}</ul>"


def _render_stylesheet() -> str:
    return """
:root {
  --bg: #f5f1e7;
  --panel: #fffdf7;
  --ink: #1e2a22;
  --muted: #5f6a64;
  --border: #d8d0bd;
  --accent: #1f4f46;
  --accent-soft: #dbe9e3;
  --warn: #8a3b12;
  --warn-soft: #f7e2d6;
  --high: #2443a6;
  --medium: #256f5a;
  --light: #866a1b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
  background:
    radial-gradient(circle at top left, rgba(31,79,70,0.12), transparent 30%),
    linear-gradient(180deg, #f8f3e8 0%, var(--bg) 100%);
  color: var(--ink);
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.page-shell {
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 20px 56px;
}
.top-nav {
  display: flex;
  gap: 14px;
  margin-bottom: 20px;
  font-size: 15px;
}
.hero {
  background: linear-gradient(135deg, rgba(31,79,70,0.95), rgba(19,34,53,0.95));
  color: #f7f4ec;
  padding: 28px;
  border-radius: 22px;
  margin-bottom: 20px;
}
.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 12px;
  margin: 0 0 10px;
  opacity: 0.82;
}
.hero h1 { margin: 0 0 8px; font-size: clamp(32px, 5vw, 52px); line-height: 1.05; }
.hero-meta, .hero-copy { margin: 0; color: rgba(247,244,236,0.88); }
.hero-copy { margin-top: 8px; max-width: 720px; }
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 20px;
}
.metric-card, .panel, .category-card, .archive-card, .subgroup-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 12px 24px rgba(38, 44, 40, 0.05);
}
.metric-card { padding: 16px; }
.metric-label { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { font-size: 28px; margin-top: 8px; }
.panel { padding: 20px; margin-bottom: 18px; }
.warning-panel { border-color: rgba(138,59,18,0.35); background: linear-gradient(180deg, #fff9f4, var(--panel)); }
.muted { color: var(--muted); }
.compact-list, .bullet-list, .article-list, .unresolved-list { margin: 0; padding-left: 20px; }
.compact-list li, .bullet-list li, .article-list li, .unresolved-list li { margin: 6px 0; }
.category-card { padding: 20px; margin-top: 16px; }
.section-header, .archive-card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.section-header h3, .archive-card h2, .subgroup-card h4 { margin: 0; }
.subgroup-stack {
  display: grid;
  gap: 12px;
  margin-top: 16px;
}
.subgroup-card { padding: 16px; background: linear-gradient(180deg, #fffdf8, #fbf8ef); }
.article-item, .unresolved-item { margin: 12px 0; }
.article-main {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.article-meta { color: var(--muted); font-size: 13px; margin-top: 4px; }
.article-error { color: var(--warn); margin: 8px 0 0; font-size: 14px; }
.pill, .status-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.03em;
}
.status-badge { background: var(--accent-soft); color: var(--accent); }
.status-success { background: #ddefe7; color: #23614d; }
.status-partial { background: #f8e4d9; color: #8a3b12; }
.status-failed, .status-empty { background: #ead8d4; color: #7a2525; }
.pill-high { background: rgba(36,67,166,0.12); color: var(--high); }
.pill-medium { background: rgba(37,111,90,0.12); color: var(--medium); }
.pill-light { background: rgba(134,106,27,0.12); color: var(--light); }
.pill-unresolved { background: var(--warn-soft); color: var(--warn); }
.archive-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
}
.archive-card { padding: 18px; }
@media (max-width: 720px) {
  .page-shell { padding: 18px 14px 40px; }
  .hero { padding: 22px; border-radius: 18px; }
}
""".strip() + "\n"
