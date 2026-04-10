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
        "legacy_executive_summary": _string_list(report.get("legacy_executive_summary")),
        "top_alerts": _string_list(report.get("top_alerts")),
        "synthesis_alerts": _string_list(report.get("synthesis_alerts")),
        "previous_top_alerts": _string_list(report.get("previous_top_alerts")),
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
    unresolved_section = _render_unresolved_section(unresolved_articles)
    
    # Handle sectioned executive summary
    # Handle sectioned executive summary with robust fallbacks
    new_alerts = report.get("executive_summary") or report.get("top_alerts") or report.get("synthesis_alerts") or []
    legacy_alerts = report.get("legacy_executive_summary") or report.get("previous_top_alerts") or []
    
    # Combined Summary (Always visible outside of tabs)
    summary_html_blocks = []
    if new_alerts:
        summary_html_blocks.append('<div style="margin-bottom: 20px;"><span class="badge badge-cio" style="padding: 4px 12px; font-size: 13px;">CIO BRIEFING</span></div>')
        summary_html_blocks.extend(f'<div class="glass-card alert-box"><strong>Top Alert</strong>{escape(item)}</div>' for item in new_alerts)
    
    if legacy_alerts:
        summary_html_blocks.append('<div style="margin: 32px 0 16px; font-size: 11px; font-weight: 800; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px;">PREVIOUSLY TODAY</div>')
        summary_html_blocks.extend(f'<div class="glass-card alert-box alert-box-legacy" style="background: #f8fafc; border-left-color: #cbd5e1; color: #64748b; opacity: 0.8;"><strong>Previous Alert</strong>{escape(item)}</div>' for item in legacy_alerts)
    
    summary_section_html = ""
    if summary_html_blocks:
        summary_section_html = f"""
        <section class="panel">
          {''.join(summary_html_blocks)}
        </section>
        """

    # Tabbed Interface Generation
    tab_btns = []
    tab_contents = []
    
    for i, category in enumerate(categories):
        cat_title = str(category.get("category") or f"Section {i+1}")
        art_count = sum(len(sg.get("articles") or []) for sg in (category.get("subgroups") or []))
        if not art_count: art_count = category.get("article_count", 0) # Fallback for old schema
        
        # Default to active
        is_active = (i == 0)
        active_class = "active" if is_active else ""
        hidden_class = "" if is_active else "hidden"
        
        tab_btns.append(f"""
        <button class="tab-btn {active_class}" onclick="switchTab('{i}')">
          {escape(cat_title)} <span class="tab-count">({art_count})</span>
        </button>
        """)
        
        tab_contents.append(f"""
        <div id="cat-tab-{i}" class="category-tab-content {hidden_class}">
          {_render_category_block(category)}
        </div>
        """)

    tab_bar_html = f"<div class='tab-bar'>{''.join(tab_btns)}</div>" if tab_btns else ""
    category_sections_html = "".join(tab_contents)

    # Model and Status Context
    model_chain = [str((report.get("model") or {}).get("primary_model") or "")]
    model_chain.extend(str(item) for item in (report.get("model") or {}).get("fallback_models") or [])
    model_chain = [item for item in model_chain if item]
    model_html = " → ".join(escape(item) for item in model_chain) if model_chain else "Unavailable"
    status_val = str(report.get("status") or "unknown").upper()

    notes = _build_run_notes(report)
    main_note = notes[0] if notes else "Run completed successfully."
    
    # Style 1: Status Banner
    status_variant = "banner-success" if status_val == "SUCCESS" else "banner-warn"
    status_banner = f"""
    <div class="banner-status {status_variant}">
      <span class="banner-icon">{"✓" if status_val == "SUCCESS" else "⚠"}</span>
      <span class="banner-msg">{escape(main_note)}</span>
      <div class="metadata-row" style="margin-left: auto; opacity: 0.8;">
        {model_html}
      </div>
    </div>
    """
    cards = [
        ("Status", f"<span class='status-{status_val}'>{status_val}</span>"),
        ("Articles", str(totals.get("article_count", 0))),
        ("Coverage", f"{(report.get('daily_stats') or {}).get('analyzed', 0)} / {(report.get('daily_stats') or {}).get('total_scraped', 0)}"),
        ("Unresolved", str(len(unresolved_articles))),
        ("Rate-limit waits", str(diagnostics.get("rate_limit_wait_count", 0))),
        ("Truncated", str(totals.get("truncated_article_count", 0))),
    ]
    cards_html = "".join(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>"
        for label, value in cards
    )
    site_title = escape(page_title)
    nav = _render_nav(nav_prefix)
    
    market_bullets = list(report.get("market_context") or [])
    market_context_html = ""
    if market_bullets:
        market_context_html = f"""
        <section class="panel">
          <h2>Market context</h2>
          {_render_bullet_list(market_bullets, fallback="")}
        </section>
        """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <!-- PREMIUM_UI_V4 -->
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
      <p class="hero-meta">Report date {report_date} • Source <a href="https://www.hkej.com/" target="_blank">hkej.com</a></p>
    </header>
    {status_banner}
    {market_context_html}
    <section class="metrics-grid">{cards_html}</section>
    {summary_section_html}
    {unresolved_section}
    <section class="panel">
      <h2>Briefing Items</h2>
      {tab_bar_html}
      {category_sections_html}
    </section>
  </div>
  <script>
    function switchTab(idx) {{
      // Hide all contents
      document.querySelectorAll('.category-tab-content').forEach(c => c.classList.add('hidden'));
      // Deactivate all buttons
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      
      // Show target
      const target = document.getElementById('cat-tab-' + idx);
      if (target) target.classList.remove('hidden');
      // Activate button
      const btns = document.querySelectorAll('.tab-btn');
      if (btns[idx]) btns[idx].classList.add('active');
    }}
  </script>
</body>
</html>
"""


def _render_archive_page(entries: list[dict[str, Any]]) -> str:
    cards = []
    for entry in entries:
        summary = _render_bullet_list(entry.get("executive_summary") or [], fallback="No executive summary.")
        cards.append(
            f"""
            <div class="glass-card archive-card">
              <div class="archive-card-header">
                <h2><a href="../{escape(str(entry.get('relative_url') or ''))}">{escape(str(entry.get('report_date') or 'Unknown date'))}</a></h2>
                <span class="status-badge status-{escape(str(entry.get('status') or 'unknown').lower())}">{escape(str(entry.get('status') or 'unknown').upper())}</span>
              </div>
              <p class="muted">{escape(str(entry.get('article_count', 0)))} articles • {escape(str(entry.get('unresolved_count', 0)))} unresolved</p>
              {summary}
            </div>
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
    article_count = category.get("article_count", 0)
    return f"""
    <section class="category-card">
      <div class="section-header" style="margin-bottom: 20px;">
        <h3 style="font-size: 24px;">{category_name}</h3>
        <div style="display: flex; gap: 8px;">
          <span class="badge badge-info">{article_count} article(s)</span>
          <span class="badge badge-success">{status.upper()}</span>
        </div>
      </div>
      <div class="glass-card" style="margin-bottom: 20px;">{key_developments}</div>
      {entity_line}
      {subgroup_html}
      <div class="metadata-row" style="margin-top: 24px; border-top: 1px solid var(--border); padding-top: 12px;">
        <span>Profile: {profile}</span>
        <span>{escape(str(diagnostics.get('sub_batch_count', 0)))} batches • {escape(', '.join(diagnostics.get('split_reasons') or []) or 'no splits')}</span>
      </div>
    </section>
    """


def _render_subgroup_block(subgroup: dict[str, Any], *, force_plain_title: bool) -> str:
    title = escape(str(subgroup.get("title") or "Subgroup"))
    title_html = title if force_plain_title else f"<strong>{title}</strong>"
    theme = str(subgroup.get("theme_rationale") or "")
    articles = subgroup.get("articles") or []
    
    theme_line = f"<div class='glass-card' style='margin-bottom: 16px;'><p class='theme-text'><strong>Theme:</strong> {escape(theme)}</p></div>" if theme else ""
    
    # Sort articles: Importance (HIGH > MEDIUM > LIGHT), then Recency (Newest first)
    articles_to_sort = list(articles)
    # High Importance (2) > Medium (1) > Light (0)
    prio_map = {"HIGH": 2, "MEDIUM": 1, "LIGHT": 0}
    articles_to_sort.sort(
        key=lambda x: (
            prio_map.get(str(x.get("attention_tier") or "medium").upper(), -1),
            str(x.get("published_at") or "")
        ),
        reverse=True
    )
    
    articles_html = "".join(_render_article_line(art, is_expanded=False) for art in articles_to_sort)
    
    developments_html = _render_bullet_list(subgroup.get("key_developments") or [], fallback="", compact=True)
    
    return f"""
    <div class="subgroup-card">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
        <h4>{title_html}</h4>
        <span class="badge badge-info" style="background: #f1f5f9; color: #64748b;">{len(articles)} articles</span>
      </div>
      {theme_line}
      <div style="margin-bottom: 16px;">{developments_html}</div>
      <div class="article-list">{articles_html}</div>
    </div>
    """


def _render_article_line(article: dict[str, Any], *, is_expanded: bool = False) -> str:
    title = escape(str(article.get("title") or "Untitled article"))
    snippet = escape(str(article.get("summary_snippet") or ""))
    url = escape(str(article.get("canonical_url") or "#"))
    raw_pub = str(article.get("published_at") or "")
    # Format: 2026-04-09 15:46
    date_display = raw_pub.replace("T", " ")[:16] if "T" in raw_pub else raw_pub[:16]
    
    attention = escape(str(article.get("attention_tier") or "medium").upper())
    is_new = bool(article.get("is_new"))
    error = str(article.get("error") or "")
    error_classification = str(article.get("error_classification") or "")
    
    badges = []
    if is_new:
        badges.append("<span class='pill pill-new'>NEW</span>")
    badges.append(f"<span class='pill pill-{attention.lower()}'>{attention}</span>")
    
    points = _render_bullet_list(article.get("key_points") or [], fallback="", compact=True)
    error_html = ""
    if error:
        reason = escape(error_classification or error)
        detail = escape(error)
        error_html = f"<p class='article-error'><strong>{reason}</strong>: {detail}</p>"
        
    open_attr = "open" if is_expanded else ""
    return f"""
    <div class="article-item">
      <details {open_attr}>
        <summary>
          <div class="article-main">
            {"".join(badges)}
            <span class="article-title">{title}</span>
            {f'<span class="article-snippet">— {snippet}</span>' if snippet else ""}
          </div>
          <span class="article-meta">{date_display}</span>
          <span class="fold-trigger">▼</span>
        </summary>
        <div class="article-details">
          <p><a href="{url}" target="_blank" rel="noopener noreferrer">View source article</a></p>
          {points}
          {error_html}
        </div>
      </details>
    </div>
    """


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
  --bg: #f3f4f6;
  --panel: #ffffff;
  --ink: #111827;
  --muted: #6b7280;
  --border: #e5e7eb;
  --accent: #2563eb;
  --accent-soft: #dbeafe;
  --warn: #dc2626;
  --warn-soft: #fee2e2;
  --high: #991b1b;
  --high-soft: #fee2e2;
  --medium: #1d4ed8;
  --medium-soft: #dbeafe;
  --light: #374151;
  --light-soft: #f3f4f6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  background-color: var(--bg);
  color: #374151;
  line-height: 1.5;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.page-shell {
  max-width: 800px;
  margin: 0 auto;
  padding: 40px 20px;
}
.top-nav {
  display: flex;
  gap: 14px;
  margin-bottom: 20px;
  font-size: 15px;
}
.hero {
  background: white;
  padding: 32px;
  margin-bottom: 24px;
  border: 1px solid var(--border);
  border-radius: 12px;
}
.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.15em;
  font-size: 11px;
  font-weight: 800;
  color: var(--accent);
  margin: 0 0 8px;
}
.hero h1 { margin: 0 0 12px; font-size: 32px; font-weight: 800; color: var(--ink); letter-spacing: -0.02em; }
.hero-meta { margin: 0; color: var(--muted); font-size: 14px; font-weight: 500; }
.hero-meta a { color: var(--accent); font-weight: 600; }
.metrics-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 32px;
  padding: 16px 20px;
  background: white;
  border: 1px solid var(--border);
  border-radius: 12px;
}
.metric-card {
  flex: 1;
  min-width: 100px;
  border: none;
  padding: 0;
}
.metric-label { color: var(--muted); font-size: 10px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.1em; margin-bottom: 4px; }
.metric-value { font-size: 18px; font-weight: 800; color: var(--ink); }

.status-SUCCESS { color: #059669; }
.status-PARTIAL { color: #d97706; }
.status-FAILED { color: #dc2626; }

.banner-status {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  border-radius: 12px;
  margin-bottom: 24px;
  font-size: 14px;
}
.banner-success { background: #ecfdf5; border: 1px solid #10b981; color: #065f46; }
.banner-warn { background: #fffbeb; border: 1px solid #f59e0b; color: #92400e; }
.banner-icon { font-size: 18px; font-weight: bold; }

.glass-card {
  background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid rgba(0, 0, 0, 0.05);
  border-radius: 12px;
  padding: 16px;
}

.badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 700;
  white-space: nowrap;
}
.badge-success { background: #d1fae5; color: #065f46; }
.badge-info { background: #e0f2fe; color: #0369a1; }
.badge-cio { background: #fee2e2; color: #991b1b; }

.metadata-row {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  font-size: 12px;
  color: var(--muted);
}

.panel { padding: 20px; margin-bottom: 24px; }
.panel h2 { margin-top: 0; font-size: 18px; font-weight: 700; color: var(--ink); margin-bottom: 16px; display: flex; align-items: center; gap: 10px; }
.alert-box { margin-bottom: 12px; border-left: 4px solid #ef4444; border-radius: 0 12px 12px 0; font-size: 15px; color: #111827; }
.alert-box strong { color: #991b1b; display: block; font-size: 12px; margin-bottom: 4px; text-transform: uppercase; }
.alert-box-legacy { border-left-color: #cbd5e1; color: #64748b; background: #f8fafc; }
.alert-box-legacy strong { color: #94a3b8; }
.category-tab-content.hidden { display: none; }
.warning-panel { border-color: #fecaca; background: #fff7f7; }
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
.subgroup-card { padding: 20px; margin-bottom: 24px; }
.subgroup-card h4 { margin: 0 0 12px; font-size: 18px; font-weight: 700; color: var(--ink); }
.theme-box { font-size: 13px; color: var(--muted); margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px dotted var(--border); }
.article-list { margin: 0; padding: 0; list-style: none; }
.article-item { border-bottom: 1px solid #f1f5f9; padding: 8px 0; }
.article-item:last-child { border-bottom: none; }
.article-item:hover { background-color: #f8fafc; }
details { display: block; width: 100%; }
summary {
  display: flex !important;
  align-items: center;
  padding: 10px 20px;
  cursor: pointer;
  list-style: none;
  outline: none;
}
summary::-webkit-details-marker { display: none; }

.article-main {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
}
.article-title { font-weight: 600; color: var(--ink); }
.article-snippet { color: var(--muted); font-weight: 400; flex: 1; overflow: hidden; text-overflow: ellipsis; margin-left: 4px; }
.article-meta { color: var(--muted); font-size: 12px; margin-left: 12px; white-space: nowrap; font-weight: 500; }
.fold-trigger { 
  color: var(--muted); 
  font-size: 10px; 
  margin-left: 10px; 
  opacity: 0.6; 
  transition: transform 0.2s;
  display: inline-block;
}
details .fold-trigger { transform: rotate(90deg); }
details[open] .fold-trigger { transform: rotate(0deg); }
.tab-bar {
  display: flex;
  overflow-x: auto;
  border-bottom: 2px solid var(--border);
  margin-bottom: 24px;
  gap: 8px;
  scrollbar-width: none;
}
.tab-bar::-webkit-scrollbar { display: none; }
.tab-btn {
  background: none;
  border: none;
  padding: 12px 16px;
  font-size: 14px;
  font-weight: 600;
  color: var(--muted);
  cursor: pointer;
  white-space: nowrap;
  border-bottom: 3px solid transparent;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 6px;
}
.tab-btn:hover { color: var(--ink); background: #f1f5f9; }
.tab-btn.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}
.tab-count {
  font-size: 11px;
  background: #f1f5f9;
  color: var(--muted);
  padding: 2px 6px;
  border-radius: 10px;
  font-weight: 700;
}
.tab-btn.active .tab-count {
  background: var(--accent-soft);
  color: var(--accent);
}
.category-tab-content.hidden { display: none; }
.article-details { padding: 0 20px 16px 52px; font-size: 14px; background: #ffffff; }
.pill, .status-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 700;
}
.status-badge { background: var(--accent-soft); color: var(--accent); }
.status-success { background: #ddefe7; color: #23614d; }
.status-partial { background: #fef2f2; color: #991b1b; }
.status-failed, .status-empty { background: #fee2e2; color: #991b1b; }
.pill-high { background: var(--high-soft); color: var(--high); }
.pill-medium { background: var(--medium-soft); color: var(--medium); }
.pill-light { background: var(--light-soft); color: var(--light); }
.pill-new { background: #dcfce7; color: #166534; }
.pill-unresolved { background: var(--warn-soft); color: var(--warn); }
.archive-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
  margin-top: 24px;
}
.archive-card { 
  display: flex;
  flex-direction: column;
  height: 100%;
}
.archive-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.archive-card h2 { margin: 0; font-size: 20px; }
.archive-card h2 a { color: var(--ink); text-decoration: none; }
.archive-card h2 a:hover { color: var(--accent); }
.archive-card ul { margin: 12px 0 0; padding-left: 18px; color: var(--muted); font-size: 13px; line-height: 1.5; }
@media (max-width: 720px) {
  .page-shell { padding: 18px 14px 40px; }
  .hero { padding: 22px; border-radius: 18px; }
  .archive-grid { grid-template-columns: 1fr; }
}
""".strip() + "\n"
