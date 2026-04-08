from __future__ import annotations

import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from .config import get_project_root

ATTENTION_TIER_RANK = {"high": 0, "medium": 1, "light": 2}

GMAIL_SENDER_ENV = "DAILY_MACRO_GMAIL_SENDER"
GMAIL_APP_PASSWORD_ENV = "DAILY_MACRO_GMAIL_APP_PASSWORD"
GMAIL_RECIPIENT_ENV = "DAILY_MACRO_GMAIL_RECIPIENT"
LEGACY_GMAIL_SENDER_ENV = "GMAIL_SENDER"
LEGACY_GMAIL_APP_PASSWORD_ENV = "GMAIL_APP_PASSWORD"
LEGACY_GMAIL_RECIPIENT_ENV = "GMAIL_RECIPIENT"


def load_analysis_result(result_path: str | Path) -> dict[str, Any]:
    path = Path(result_path).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def send_analysis_summary_email(summary: dict[str, Any]) -> tuple[bool, str]:
    status = str(summary.get("status") or "").strip().lower()
    if status not in {"success", "partial"}:
        return False, f"Skipped email because report status is {status or 'unknown'}."

    categories = list(summary.get("categories") or [])
    article_count = int((summary.get("totals") or {}).get("article_count") or 0)
    if not categories or article_count <= 0:
        return False, "Skipped email because the analysis report has no analyzable content."

    sender, app_password, recipient = _get_gmail_credentials()
    if not sender or not app_password or not recipient:
        raise RuntimeError(
            "Gmail credentials not set. Expected either "
            f"{GMAIL_SENDER_ENV}, {GMAIL_APP_PASSWORD_ENV}, and {GMAIL_RECIPIENT_ENV} "
            "or local .config equivalents."
        )

    message = MIMEMultipart("alternative")
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = _build_subject(summary)
    message.attach(MIMEText(_build_plain_body(summary), "plain", "utf-8"))
    message.attach(MIMEText(_build_html_body(summary), "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, recipient, message.as_string())

    return True, f"Sent daily macro Gmail summary to {recipient}."


def send_test_email() -> tuple[bool, str]:
    return send_analysis_summary_email(
        {
            "report_date": "2026-04-04",
            "generated_at": "2026-04-04T07:05:00+00:00",
            "status": "partial",
            "model": {
                "provider": "groq",
                "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "fallback_models": ["qwen/qwen3-32b", "llama-3.1-8b-instant"],
            },
            "model_switches": [
                {
                    "from_model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "to_model": "qwen/qwen3-32b",
                    "reason": "Model meta-llama/llama-4-scout-17b-16e-instruct returned 429 rate_limit_exceeded.",
                }
            ],
            "input": {"article_count": 4, "category_count": 2},
            "diagnostics": {
                "rate_limit_wait_count": 1,
                "rate_limit_wait_seconds_total": 22.5,
                "fallback_switch_count": 1,
                "pre_send_split_count": 1,
                "response_413_split_count": 0,
                "json_repair_retry_count": 0,
                "batch_count": 3,
                "failed_batch_count": 0,
            },
            "totals": {
                "article_count": 4,
                "successful_article_analyses": 3,
                "failed_article_analyses": 1,
                "full_text_article_count": 3,
                "truncated_article_count": 1,
                "successful_categories": 1,
                "partial_categories": 1,
                "failed_categories": 0,
            },
            "errors": [
                {
                    "type": "article",
                    "target": "https://example.com/2",
                    "classification": "incomplete_model_output",
                    "message": "Model response omitted this article from the category batch.",
                }
            ],
            "categories": [
                {
                    "category": "國際財經",
                    "article_count": 2,
                    "status": "partial",
                    "key_developments": [
                        "Markets focused on fresh geopolitical risk around Iran-related escalation.",
                        "Major global banks tightened operations and contingency planning in Europe.",
                    ],
                    "named_entities": [
                        {"name": "伊朗", "type": "country"},
                        {"name": "Amazon", "type": "company"},
                    ],
                    "subgroups": [
                        {
                            "title": "Geopolitical risk",
                            "theme_rationale": "Articles clustered around conflict-sensitive international macro developments.",
                            "article_count": 2,
                            "key_developments": [
                                "Markets repriced geopolitical risk around Iran-linked developments."
                            ],
                            "named_entities": [{"name": "伊朗", "type": "country"}],
                            "articles": [
                                {
                                    "title": "Global risk headline",
                                    "canonical_url": "https://example.com/global-risk",
                                    "published_at": "2026-04-04T07:00:00+00:00",
                                    "error": None,
                                },
                                {
                                    "title": "Amazon contingency note",
                                    "canonical_url": "https://example.com/amazon",
                                    "published_at": "2026-04-04T07:10:00+00:00",
                                    "error": "Model response omitted this article from the category batch.",
                                },
                            ],
                        }
                    ],
                    "diagnostics": {
                        "sub_batch_count": 1,
                        "split_reasons": [],
                        "models_attempted": ["qwen/qwen3-32b"],
                        "estimated_input_tokens_max": 2100,
                        "serialized_request_bytes_max": 8400,
                        "rate_limit_waits": 0,
                        "partial_article_count": 0,
                    },
                },
                {
                    "category": "時事脈搏",
                    "article_count": 2,
                    "status": "success",
                    "key_developments": [
                        "The day’s pulse coverage centered on fast-moving geopolitical and macro-sensitive developments.",
                    ],
                    "named_entities": [
                        {"name": "Oracle", "type": "company"},
                    ],
                    "subgroups": [
                        {
                            "title": "Corporate pulse",
                            "theme_rationale": "Fast-moving company headlines dominated this lighter section.",
                            "article_count": 2,
                            "key_developments": [
                                "The day’s pulse coverage centered on fast-moving company-sensitive updates."
                            ],
                            "named_entities": [{"name": "Oracle", "type": "company"}],
                            "articles": [
                                {
                                    "title": "Oracle pulse headline",
                                    "canonical_url": "https://example.com/oracle",
                                    "published_at": "2026-04-04T07:15:00+00:00",
                                    "error": None,
                                }
                            ],
                        }
                    ],
                    "diagnostics": {
                        "sub_batch_count": 2,
                        "split_reasons": ["pre_send_budget"],
                        "models_attempted": ["qwen/qwen3-32b", "llama-3.1-8b-instant"],
                        "estimated_input_tokens_max": 3200,
                        "serialized_request_bytes_max": 12600,
                        "rate_limit_waits": 1,
                        "partial_article_count": 0,
                    },
                },
            ],
            "output_path": "/tmp/hkej-news-analysis.json",
        }
    )


def _build_subject(summary: dict[str, Any]) -> str:
    report_date = summary.get("report_date") or "Unknown date"
    category_count = int((summary.get("input") or {}).get("category_count") or 0)
    article_count = int((summary.get("totals") or {}).get("article_count") or 0)
    status = str(summary.get("status") or "unknown").upper()
    suffix = f"[{status}] " if status != "SUCCESS" else ""
    return f"[DAILY MACRO] {suffix}{report_date} | {article_count} article(s), {category_count} categorie(s)"


def _build_plain_body(summary: dict[str, Any]) -> str:
    lines = [
        "DAILY MACRO ANALYST",
        "",
        f"Report date: {summary.get('report_date') or 'Unknown'}",
        f"Generated at: {summary.get('generated_at') or 'Unknown'}",
        f"Status: {summary.get('status') or 'Unknown'}",
        f"Primary model: {_model_label(summary, 'primary_model')}",
        f"Fallback models: {_fallback_model_labels(summary)}",
        f"Article count: {(summary.get('totals') or {}).get('article_count') or 0}",
        f"Category count: {(summary.get('input') or {}).get('category_count') or 0}",
        "",
    ]

    notes = _build_run_notes(summary)
    if notes:
        lines.append("Run notes:")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")

    market_lines = _build_plain_market_lines(summary)
    if market_lines:
        lines.append("Market snapshot:")
        lines.extend(market_lines)
        lines.append("")

    daily_stats = summary.get("daily_stats") or {}
    scraped = daily_stats.get("total_scraped") or 0
    analyzed = daily_stats.get("analyzed") or 0
    if scraped:
        lines.append(f"MARKET COVERAGE: {analyzed} analyzed / {scraped} scraped ({daily_stats.get('success_rate', 0)}% success)")
        lines.append("")

    executive_summary = summary.get("executive_summary") or []
    if executive_summary:
        lines.append("EXECUTIVE SUMMARY (TOP ALERTS):")
        for alert in executive_summary:
            lines.append(f"!!! {alert}")
        lines.append("")

    unresolved_lines = _build_plain_unresolved_lines(summary)
    if unresolved_lines:
        lines.append("Unresolved articles:")
        lines.extend(unresolved_lines)
        lines.append("")

    for category in summary.get("categories") or []:
        lines.append(f"{category.get('category') or 'Uncategorized'} ({category.get('article_count') or 0})")
        for item in (category.get("key_developments") or [])[:5]:
            lines.append(f"- {item}")
        entities = category.get("named_entities") or []
        if entities:
            lines.append("Named entities: " + ", ".join(entity["name"] for entity in entities[:8] if entity.get("name")))
        subgroup_lines = _build_plain_subgroup_lines(category)
        if subgroup_lines:
            lines.append("")
            lines.extend(subgroup_lines)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _build_html_body(summary: dict[str, Any]) -> str:
    category_blocks = []
    for category in summary.get("categories") or []:
        category_name = str(category.get("category") or "")
        category_articles = category.get("articles") or []
        # Check if all articles are low importance (LIGHT)
        importance_levels = {str(a.get("attention_tier") or "").lower() for a in category_articles}
        all_light = importance_levels == {"light"} or not importance_levels
        
        is_property = (category_name in {"二手市場", "新盤情報", "地產新聞"}) and not all_light
        is_light_only = _is_light_only(category_articles) and not is_property
        
        items_html = ""
        if not is_light_only:
            developments = category.get("key_developments") or []
            if is_property and developments:
                items_html = _build_html_property_table(developments)
            elif developments:
                bullets = "".join(
                    f"<li>{_escape_html(str(item))}</li>"
                    for item in developments[:5]
                )
                items_html = f"<ul style='margin:0;padding-left:18px;color:#374151;line-height:1.6;'>{bullets}</ul>"

        entities = category.get("named_entities") or []
        entities_html = ""
        if entities and not is_light_only:
            labels = ", ".join(_escape_html(str(entity.get("name") or "")) for entity in entities[:8] if entity.get("name"))
            entities_html = f"<p style='margin:10px 0 0;color:#4b5563;'><strong>Named entities:</strong> {labels}</p>"
        
        subgroup_html = _build_html_subgroups(category, category_name)
        category_blocks.append(
            f"""
            <div style="margin:0 0 16px;padding:12px;border:1px solid #e5e7eb;border-radius:12px;background:#ffffff;">
              <div style="font-size:17px;font-weight:700;color:#111827;margin-bottom:6px;">
                {_escape_html(category_name or "Uncategorized")}
                <span style="font-size:12px;font-weight:500;color:#6b7280;">({int(category.get("article_count") or 0)} article(s))</span>
              </div>
              {items_html}
              {entities_html}
              {subgroup_html}
            </div>
            """
        )

    notes = _build_run_notes(summary)
    notes_html = ""
    if notes:
        notes_html = (
            "<div style='margin:0 0 20px;padding:16px;border:1px solid #f3f4f6;border-radius:12px;background:#f9fafb;'>"
            "<div style='font-size:16px;font-weight:700;color:#111827;margin-bottom:8px;'>Run notes</div>"
            "<ul style='margin:0;padding-left:18px;color:#4b5563;line-height:1.6;'>"
            + "".join(f"<li>{_escape_html(note)}</li>" for note in notes)
            + "</ul></div>"
        )

    unresolved_html = _build_html_unresolved_section(summary)
    market_html = _build_html_market_section(summary)
    executive_summary_html = _build_html_executive_summary(summary)

    daily_stats = summary.get("daily_stats") or {}
    scraped = daily_stats.get("total_scraped") or 0
    analyzed = daily_stats.get("analyzed") or 0
    market_coverage_html = ""
    if scraped:
        market_coverage_html = (
            f"<div style='margin-bottom:16px;font-size:14px;color:#6b7280;'>"
            f"<b>Market Coverage:</b> {analyzed} analyzed / {scraped} scraped "
            f"({daily_stats.get('success_rate', 0)}% success)"
            f"</div>"
        )

    return f"""
    <html>
      <body style="margin:0;padding:24px;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
        <div style="max-width:760px;margin:0 auto;background:#ffffff;border-radius:16px;padding:28px 28px 16px;border:1px solid #e5e7eb;">
          <div style="font-size:30px;font-weight:800;color:#111827;">Daily Macro Analyst</div>
          <p style="margin:8px 0 20px;color:#4b5563;">
            Report date: {_escape_html(str(summary.get("report_date") or "Unknown"))}
          </p>
          {market_coverage_html}
          {executive_summary_html}
          {market_html}
          {unresolved_html}
          {"".join(category_blocks)}
        </div>
      </body>
    </html>
    """


def _build_plain_subgroup_lines(category: dict[str, Any]) -> list[str]:
    subgroups = list(category.get("subgroups") or [])
    category_name = str(category.get("category") or "")
    if not subgroups:
        return _build_plain_article_lines(category.get("articles") or [])

    lines: list[str] = []
    for subgroup in subgroups:
        subgroup_articles = subgroup.get("articles") or []
        is_light_only = _is_light_only(subgroup_articles)
        
        lines.append(f"Subgroup: {subgroup.get('title') or 'Overview'}")
        if not is_light_only:
            rationale = str(subgroup.get("theme_rationale") or "").strip()
            if rationale:
                lines.append(f"  Theme: {rationale}")
            for item in (subgroup.get("key_developments") or [])[:4]:
                lines.append(f"  - {item}")
            entities = subgroup.get("named_entities") or []
            if entities:
                labels = ", ".join(entity["name"] for entity in entities[:6] if entity.get("name"))
                if labels:
                    lines.append(f"  Named entities: {labels}")
        lines.extend(f"  - {line}" for line in _build_plain_article_lines(subgroup_articles))
    return lines


def _build_plain_article_lines(articles: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for article in articles:
        title = str(article.get("title") or "Untitled")
        url = str(article.get("canonical_url") or "").strip()
        unresolved = " [UNRESOLVED]" if article.get("error") else ""
        attention = _attention_badge(article)
        if url:
            lines.append(f"{attention}{title}{unresolved}: {url}")
        else:
            lines.append(f"{attention}{title}{unresolved}")
    return lines


def _build_html_subgroups(category: dict[str, Any], category_name: str = "") -> str:
    subgroups = list(category.get("subgroups") or [])
    if not subgroups:
        articles_html = _build_html_articles(category.get("articles") or [])
        return f"<div style='margin-top:14px;'>{articles_html}</div>" if articles_html else ""

    blocks = []
    is_property = (category_name in {"二手市場", "新盤情報", "地產新聞"})
    importance_levels = {str(a.get("attention_tier") or "").lower() for a in subgroups[0].get("articles", [])} if subgroups else set()
    all_light = importance_levels == {"light"} or not importance_levels
    is_property = is_property and not all_light
    
    for subgroup in subgroups:
        subgroup_articles = subgroup.get("articles") or []
        is_light_only = _is_light_only(subgroup_articles) and not is_property
        
        items_html = ""
        if not is_light_only:
            developments = subgroup.get("key_developments") or []
            if is_property and developments:
                items_html = _build_html_property_table(developments)
            elif developments:
                bullets = "".join(
                    f"<li>{_escape_html(str(item))}</li>"
                    for item in developments[:4]
                )
                items_html = f"<ul style='margin:0;padding-left:18px;color:#374151;line-height:1.6;'>{bullets}</ul>"

        rationale_html = ""
        if not is_light_only:
            rationale = str(subgroup.get("theme_rationale") or "").strip()
            if rationale:
                rationale_html = f"<p style='margin:6px 0 8px;color:#6b7280;'><strong>Theme:</strong> {_escape_html(rationale)}</p>"

        entities_html = ""
        if not is_light_only:
            entities = subgroup.get("named_entities") or []
            entity_labels = ", ".join(_escape_html(str(entity.get("name") or "")) for entity in entities[:6] if entity.get("name"))
            if entity_labels:
                entities_html = f"<p style='margin:8px 0 0;color:#4b5563;'><strong>Named entities:</strong> {entity_labels}</p>"

        article_links = _build_html_articles(subgroup_articles)
        blocks.append(
            f"""
            <div style="margin-top:16px;padding-top:14px;border-top:1px solid #f3f4f6;">
              <div style="font-size:15px;font-weight:700;color:#111827;">{_escape_html(str(subgroup.get("title") or "Overview"))}</div>
              {rationale_html}
              {items_html}
              {entities_html}
              {article_links}
            </div>
            """
        )
    return "".join(blocks)


def _build_html_articles(articles: list[dict[str, Any]]) -> str:
    if not articles:
        return ""
    
    # Sort: High/Medium/Light, then Newest Date
    sorted_articles = sorted(
        articles,
        key=lambda a: (
            ATTENTION_TIER_RANK.get(str(a.get("attention_tier") or "medium").lower(), 1),
            -(_to_epoch(a.get("published_at")) or 0)
        )
    )

    items = []
    
    if len(sorted_articles) < 5:
        top_n = sorted_articles
        remaining = []
    else:
        high_articles = [a for a in sorted_articles if str(a.get("attention_tier") or "").lower() == "high"]
        if len(high_articles) >= 3:
            top_n = high_articles
            remaining = [a for a in sorted_articles if str(a.get("attention_tier") or "").lower() != "high"]
        else:
            top_n = sorted_articles[:3]
            remaining = sorted_articles[3:]

    def _render_item(article, compact=False):
        title = _escape_html(str(article.get("title") or "Untitled"))
        url = str(article.get("canonical_url") or "").strip()
        pub_at = (article.get("published_at") or "").replace("T", " ")
        
        if compact:
            # titles only for shadow lists to save space in Gmail
            if url:
                return f"<li style='margin-bottom:2px;font-size:13px;color:#4b5563;'><a href=\"{_escape_html(url)}\" style=\"color:#4b5563;text-decoration:none;\">{title}</a></li>"
            return f"<li style='margin-bottom:2px;font-size:13px;color:#4b5563;'>{title}</li>"

        date_str = f" <span style='color:#6b7280;font-size:12px;'>({pub_at[:16]})</span>" if pub_at else ""
        unresolved = " <strong style='color:#b91c1c;'>[UNRESOLVED]</strong>" if article.get("error") else ""
        attention = _attention_badge_html(article)
        if url:
            return f"<li style='margin-bottom:4px;'>{attention}<a href=\"{_escape_html(url)}\" style=\"color:#2563eb;text-decoration:none;\">{title}</a>{date_str}{unresolved}</li>"
        return f"<li style='margin-bottom:4px;'>{attention}{title}{date_str}{unresolved}</li>"

    for article in top_n:
        items.append(_render_item(article))

    articles_list = "<ul style='margin:0;padding-left:18px;line-height:1.7;'>" + "".join(items)
    
    if remaining:
        # Switch from <details> to a 'Shadow List' (Compact headlines) for better Gmail compatibility
        folded_items = "".join([_render_item(a, compact=True) for a in remaining])
        articles_list += (
            f"<li style='list-style:none;margin-top:12px;padding-top:8px;border-top:1px solid #f3f4f6;'>"
            f"<div style='color:#6b7280;font-size:12px;font-weight:600;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.025em;'>"
            f"And {len(remaining)} more headline(s):"
            f"</div>"
            f"<ul style='margin:0;padding-left:18px;list-style:circle;'>{folded_items}</ul>"
            f"</li>"
        )
    
    articles_list += "</ul>"
    return f"<div style='margin-top:10px;'><div style='font-weight:600;color:#111827;margin-bottom:6px;'>Articles</div>{articles_list}</div>"


def _attention_badge(article: dict[str, Any]) -> str:
    tier = str(article.get("attention_tier") or "").strip().lower()
    if tier == "high":
        return "[HIGH] "
    if tier == "light":
        return "[LIGHT] "
    if tier == "medium":
        return "[MEDIUM] "
    return ""


def _attention_badge_html(article: dict[str, Any]) -> str:
    tier = str(article.get("attention_tier") or "").strip().lower()
    if tier == "high":
        return "<span style='display:inline-block;margin-right:6px;padding:1px 6px;border-radius:999px;background:#fee2e2;color:#991b1b;font-size:11px;font-weight:700;'>HIGH</span>"
    if tier == "light":
        return "<span style='display:inline-block;margin-right:6px;padding:1px 6px;border-radius:999px;background:#e5e7eb;color:#374151;font-size:11px;font-weight:700;'>LIGHT</span>"
    if tier == "medium":
        return "<span style='display:inline-block;margin-right:6px;padding:1px 6px;border-radius:999px;background:#dbeafe;color:#1d4ed8;font-size:11px;font-weight:700;'>MEDIUM</span>"
    return ""


def _build_run_notes(summary: dict[str, Any]) -> list[str]:
    diagnostics = summary.get("diagnostics") or {}
    totals = summary.get("totals") or {}
    notes: list[str] = []
    if str(summary.get("status") or "").lower() == "partial":
        unresolved_count = int(totals.get("failed_article_analyses") or 0)
        notes.append(f"This digest is partial. {unresolved_count} article(s) remained unresolved after analysis salvage.")
    truncated_count = int((summary.get("totals") or {}).get("truncated_article_count") or 0)
    if truncated_count > 0:
        notes.append(f"{truncated_count} article(s) were truncated to fit the working request budget.")
    wait_count = int(diagnostics.get("rate_limit_wait_count") or 0)
    wait_seconds = diagnostics.get("rate_limit_wait_seconds_total") or 0
    if wait_count > 0:
        notes.append(f"Analysis paused {wait_count} time(s) for Groq rate limits, totaling about {wait_seconds} second(s).")
    fallback_switches = int(diagnostics.get("fallback_switch_count") or 0)
    if fallback_switches > 0:
        notes.append(f"The run switched to the fallback model {fallback_switches} time(s).")
    pre_send_splits = int(diagnostics.get("pre_send_split_count") or 0)
    response_413_splits = int(diagnostics.get("response_413_split_count") or 0)
    if pre_send_splits > 0 or response_413_splits > 0:
        notes.append(
            "Oversized category batches were split "
            f"(pre-send: {pre_send_splits}, after HTTP 413: {response_413_splits})."
        )
    if summary.get("model_switches"):
        notes.extend(
            f"Model switch: {item.get('from_model')} -> {item.get('to_model')} ({item.get('reason')})"
            for item in summary.get("model_switches") or []
        )
    error_classes = sorted({str(item.get("classification") or "unclassified") for item in summary.get("errors") or []})
    if error_classes:
        notes.append("Failure classifications present: " + ", ".join(error_classes))
    
    output_path = summary.get("output_path")
    if output_path:
        notes.append(f"Stored analysis report: {output_path}")
    return notes


def _build_plain_unresolved_lines(summary: dict[str, Any]) -> list[str]:
    unresolved = list(summary.get("unresolved_articles") or [])
    lines: list[str] = []
    for item in unresolved:
        attention = _attention_badge(item)
        category = str(item.get("category") or "Uncategorized")
        title = str(item.get("title") or "Untitled")
        url = str(item.get("canonical_url") or "").strip()
        reason = str(item.get("error_classification") or "unclassified")
        message = str(item.get("error") or "").strip()
        retry_note = ""
        if item.get("delayed_retry_attempted"):
            retry_note = " | delayed retry failed"
        base = f"- {attention}[{category}] {title}"
        if url:
            base += f": {url}"
        details = f" ({reason}"
        if message:
            details += f"; {message}"
        details += f"{retry_note})"
        lines.append(base + details)
    return lines


def _build_html_unresolved_section(summary: dict[str, Any]) -> str:
    unresolved = list(summary.get("unresolved_articles") or [])
    if not unresolved:
        return ""
    items = []
    for item in unresolved:
        category = _escape_html(str(item.get("category") or "Uncategorized"))
        title = _escape_html(str(item.get("title") or "Untitled"))
        url = str(item.get("canonical_url") or "").strip()
        reason = _escape_html(str(item.get("error_classification") or "unclassified"))
        message = _escape_html(str(item.get("error") or ""))
        retry_note = ""
        if item.get("delayed_retry_attempted"):
            retry_note = " <span style='color:#b91c1c;font-weight:600;'>Delayed retry failed</span>"
        label = _attention_badge_html(item)
        body = f"{label}<strong>[{category}]</strong> "
        if url:
            body += f"<a href=\"{_escape_html(url)}\" style=\"color:#2563eb;text-decoration:none;\">{title}</a>"
        else:
            body += title
        body += f"<div style='color:#6b7280;margin-top:2px;font-size:13px;'>{reason}"
        if message:
            body += f" | {message}"
        body += f"{retry_note}</div>"
        items.append(f"<li style='margin-bottom:10px;'>{body}</li>")
    return (
        "<div style='margin:0 0 20px;padding:16px;border:1px solid #fecaca;border-radius:12px;background:#fff7f7;'>"
        "<div style='font-size:16px;font-weight:700;color:#991b1b;margin-bottom:8px;'>Unresolved articles</div>"
        "<ul style='margin:0;padding-left:18px;line-height:1.6;'>" + "".join(items) + "</ul></div>"
    )


def _build_plain_market_lines(summary: dict[str, Any]) -> list[str]:
    items = list(summary.get("market_context") or [])
    lines: list[str] = []
    for item in items:
        ticker = item.get("ticker", "?")
        price = item.get("price")
        pct = item.get("pct_change")
        if price is None:
            continue
        pct_str = f" ({pct:+.2f}%)" if pct is not None else ""
        lines.append(f"- {ticker}: {price:.2f}{pct_str}")
    return lines


def _build_html_market_section(summary: dict[str, Any]) -> str:
    items = list(summary.get("market_context") or [])
    if not items:
        return ""
    rows = []
    for item in items:
        ticker = _escape_html(str(item.get("ticker", "?")))
        price = item.get("price")
        pct = item.get("pct_change")
        if price is None:
            continue
        price_str = f"{price:.2f}"
        date_str = _escape_html(str(item.get("data_timestamp", "")[:10]))
        if pct is not None:
            color = "#059669" if pct >= 0 else "#dc2626"
            pct_str = f"<span style='color:{color};font-weight:600;'>{pct:+.2f}%</span>"
        else:
            pct_str = "<span style='color:#6b7280;'>N/A</span>"
        rows.append(
            f"<tr><td style='padding:4px 12px 4px 0;'>{ticker}</td>"
            f"<td style='padding:4px 12px 4px 0;text-align:right;'>{price_str}</td>"
            f"<td style='padding:4px 12px 4px 0;text-align:right;'>{pct_str}</td>"
            f"<td style='padding:4px 0;text-align:right;color:#6b7280;font-size:12px;'>{date_str}</td></tr>"
        )
    table = (
        "<table style='width:100%;border-collapse:collapse;color:#374151;font-size:14px;'>"
        "<tr style='border-bottom:1px solid #e5e7eb;'>"
        "<th style='text-align:left;padding:4px 12px 4px 0;font-weight:600;'>Ticker</th>"
        "<th style='text-align:right;padding:4px 12px 4px 0;font-weight:600;'>Price</th>"
        "<th style='text-align:right;padding:4px 12px 4px 0;font-weight:600;'>Change</th>"
        "<th style='text-align:right;padding:4px 0;font-weight:600;'>Date</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    return (
        "<div style='margin:0 0 20px;padding:16px;border:1px solid #e5e7eb;border-radius:12px;background:#f9fafb;'>"
        "<div style='font-size:16px;font-weight:700;color:#111827;margin-bottom:10px;'>Market Snapshot</div>"
        + table
        + "</div>"
    )


def _model_label(summary: dict[str, Any], key: str) -> str:
    model = summary.get("model") or {}
    provider = str(model.get("provider") or "groq")
    model_id = str(model.get(key) or "unknown")
    return f"{provider}/{model_id}"


def _fallback_model_labels(summary: dict[str, Any]) -> str:
    model = summary.get("model") or {}
    provider = str(model.get("provider") or "groq")
    fallback_models = list(model.get("fallback_models") or [])
    if fallback_models:
        return ", ".join(f"{provider}/{model_id}" for model_id in fallback_models)
    legacy = model.get("fallback_model")
    if legacy:
        return f"{provider}/{legacy}"
    return f"{provider}/unknown"


def _get_gmail_credentials() -> tuple[str, str, str]:
    config_values = _load_local_config()
    sender = (
        _read_env(GMAIL_SENDER_ENV)
        or _read_env(LEGACY_GMAIL_SENDER_ENV)
        or config_values.get(GMAIL_SENDER_ENV)
        or config_values.get(LEGACY_GMAIL_SENDER_ENV)
        or ""
    ).strip()
    app_password = (
        _read_env(GMAIL_APP_PASSWORD_ENV)
        or _read_env(LEGACY_GMAIL_APP_PASSWORD_ENV)
        or config_values.get(GMAIL_APP_PASSWORD_ENV)
        or config_values.get(LEGACY_GMAIL_APP_PASSWORD_ENV)
        or ""
    ).strip()
    recipient = (
        _read_env(GMAIL_RECIPIENT_ENV)
        or _read_env(LEGACY_GMAIL_RECIPIENT_ENV)
        or config_values.get(GMAIL_RECIPIENT_ENV)
        or config_values.get(LEGACY_GMAIL_RECIPIENT_ENV)
        or ""
    ).strip()
    return sender, app_password, recipient


def _load_local_config() -> dict[str, str]:
    for path in _candidate_config_paths():
        if path.exists():
            return _parse_simple_env_file(path)
    return {}


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


def _read_env(name: str) -> str:
    import os

    return str(os.environ.get(name) or "")


def _is_light_only(articles: list[dict[str, Any]]) -> bool:
    if not articles:
        return False
    return all(str(a.get("attention_tier") or "").strip().lower() == "light" for a in articles)


def _build_html_property_table(developments: list[str]) -> str:
    import re

    rows = []
    # Heuristic extraction
    for text in developments:
        # Try to split by common Chinese indicators for price/action
        parts = re.split(r"([以以])", text, 1)
        if len(parts) == 3:
            project = parts[0].strip()
            details = parts[1] + parts[2].strip()
        else:
            # Try splitting by "獲利" or "易手" or "成交"
            parts = re.split(r"(獲利|易手|成交|招標售出)", text, 1)
            if len(parts) == 3:
                project = parts[0].strip()
                details = parts[1] + parts[2].strip()
            else:
                project = text
                details = ""
        
        rows.append(
            f"<tr>"
            f"<td style='padding:6px 12px 6px 0;border-bottom:1px solid #f3f4f6;font-weight:500;'>{_escape_html(project)}</td>"
            f"<td style='padding:6px 0;border-bottom:1px solid #f3f4f6;color:#6b7280;'>{_escape_html(details)}</td>"
            f"</tr>"
        )

    return (
        f"<details style='margin-top:10px;border:1px solid #f3f4f6;border-radius:8px;padding:8px;'>"
        f"<summary style='font-size:14px;font-weight:600;color:#374151;cursor:pointer;padding:4px 10px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:6px;display:inline-block;'>"
        f"Property Market Transactions ({len(developments)} units)"
        f"</summary>"
        f"<div style='margin-top:8px;overflow-x:hidden;'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:13px;line-height:1.4;'>"
        f"<tr style='background:#f9fafb;'>"
        f"<th style='text-align:left;padding:4px 12px 4px 0;border-bottom:2px solid #e5e7eb;color:#4b5563;'>Project/Unit</th>"
        f"<th style='text-align:left;padding:4px 0;border-bottom:2px solid #e5e7eb;color:#4b5563;'>Transaction</th>"
        f"</tr>"
        + "".join(rows)
        + "</table></div></details>"
    )


def _build_html_executive_summary(summary: dict[str, Any]) -> str:
    import re
    alerts = summary.get("executive_summary") or []
    if not alerts:
        return ""
    
    items = []
    for alert in alerts:
        # Extract URL in braces if present
        url_match = re.search(r'\{(https?://[^\}]+)\}', alert)
        url = url_match.group(1) if url_match else None
        clean_alert = re.sub(r'\s*\{https?://[^\}]+\}', '', alert).strip()
        
        if url:
            alert_html = f"<a href=\"{_escape_html(url)}\" style=\"color:#111827;text-decoration:none;\">{_escape_html(clean_alert)}</a>"
        else:
            alert_html = _escape_html(clean_alert)
            
        items.append(
            f"<div style='margin-bottom:12px;padding:12px;border-left:4px solid #ef4444;background:#fff5f5;border-radius:0 8px 8px 0;'>"
            f"<div style='font-weight:700;color:#991b1b;font-size:14px;margin-bottom:4px;'>TOP ALERT</div>"
            f"<div style='line-height:1.5;'>{alert_html}</div>"
            f"</div>"
        )
    
    return (
        "<div style='margin-bottom:24px;'>"
        "<div style='font-size:18px;font-weight:700;color:#111827;margin-bottom:12px;display:flex;align-items:center;'>"
        "<span style='background:#ef4444;color:white;padding:2px 8px;border-radius:4px;font-size:12px;margin-right:8px;'>CIO BRIEFING</span>"
        "Executive Summary"
        "</div>"
        + "".join(items)
        + "</div>"
    )


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
def _to_epoch(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        from datetime import datetime

        normalized = date_str.strip().replace("Z", "+00:00")
        return int(datetime.fromisoformat(normalized).timestamp())
    except Exception:
        try:
            from datetime import datetime

            # Backward-compatible fallback for naive timestamps like "2026-04-07 15:33:21".
            return int(datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S").timestamp())
        except Exception:
            return None
