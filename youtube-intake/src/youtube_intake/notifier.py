from __future__ import annotations

import smtplib
from collections import defaultdict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from .profiles import SYNTHESIS_SECTION_ORDER
from .runtime_env import load_local_config, merge_config_file, read_env
from .storage import load_json_document


GMAIL_SENDER_ENV = "GMAIL_SENDER"
GMAIL_APP_PASSWORD_ENV = "GMAIL_APP_PASSWORD"
GMAIL_RECIPIENT_ENV = "GMAIL_RECIPIENT"
LEGACY_GMAIL_SENDER_ENV = "YOUTUBE_INTAKE_GMAIL_SENDER"
LEGACY_GMAIL_APP_PASSWORD_ENV = "YOUTUBE_INTAKE_GMAIL_APP_PASSWORD"
LEGACY_GMAIL_RECIPIENT_ENV = "YOUTUBE_INTAKE_GMAIL_RECIPIENT"


def load_analysis_result(result_path: str | Path) -> dict[str, Any]:
    return load_json_document(result_path)


def load_run_result(result_path: str | Path) -> dict[str, Any]:
    return load_analysis_result(result_path)


def send_analysis_summary_email(summary: dict[str, Any]) -> tuple[bool, str]:
    videos = list(summary.get("videos") or [])
    has_retry_note = bool(summary.get("queued_retry_count")) or bool(summary.get("retryable_failure_count"))
    if not videos and not has_retry_note:
        return False, "No analyzed items to email."

    sender, app_password, recipient = _get_gmail_credentials()
    if not sender or not app_password or not recipient:
        raise RuntimeError(
            "Gmail credentials not set. Expected either "
            f"{GMAIL_SENDER_ENV}, {GMAIL_APP_PASSWORD_ENV}, and {GMAIL_RECIPIENT_ENV} "
            "or local .config equivalents."
        )

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = _build_subject(summary)
    msg.attach(MIMEText(_build_plain_body(summary), "plain", "utf-8"))
    msg.attach(MIMEText(_build_html_body(summary), "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, recipient, msg.as_string())

    return True, f"Sent analyst Gmail summary to {recipient}."


def send_run_summary_email(summary: dict[str, Any]) -> tuple[bool, str]:
    return send_analysis_summary_email(summary)


def send_test_email() -> tuple[bool, str]:
    summary = {
        "status": "success",
        "run_started_at": "2026-04-03T01:00:00+00:00",
        "analysis_started_at": "2026-04-03T01:02:00+00:00",
        "analysis_model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "analysis_models_used": ["meta-llama/llama-4-scout-17b-16e-instruct"],
        "videos": [
            {
                "video_id": "abc",
                "channel_slug": "top3pct",
                "channel_name": "3% 財富覺醒",
                "title": "SPX key support levels and trade setups",
                "webpage_url": "https://www.youtube.com/watch?v=abc",
                "published_at": "2026-04-03T00:00:00+00:00",
                "source_kind": "video",
                "transcript_status": "fetched",
                "source_basis": "transcript",
                "profile": "technical_trading",
                "synthesis_section": "📈 Trading Desk",
                "executive_summary": "Speaker identifies key SPX support at 5,200 and resistance at 5,450. Recommends staged entries on pullbacks.",
                "tickers_mentioned": ["SPX", "QQQ", "NVDA"],
                "profile_data": {
                    "trading_signals": [
                        "SPX bullish setup above 5,200 support with target 5,450",
                        "NVDA swing long on earnings dip below $850",
                    ],
                    "key_price_levels": [
                        "SPX support 5,200 / resistance 5,450",
                    ],
                },
                "key_timestamps": [
                    {
                        "timestamp": "00:02:14",
                        "label": "SPX support thesis",
                        "snippet": "Explains why 5,200 is the line in the sand.",
                        "why_it_matters": "Frames the entire bullish positioning.",
                    }
                ],
                "topic_tags": [{"tag": "market structure", "score": 93}, {"tag": "trade setups", "score": 84}],
                "confidence": 0.82,
            },
            {
                "video_id": "def",
                "channel_slug": "goldman-sachs",
                "channel_name": "Goldman Sachs",
                "title": "Weekly macro outlook: rates and inflation",
                "webpage_url": "https://www.youtube.com/watch?v=def",
                "published_at": "2026-04-03T01:00:00+00:00",
                "source_kind": "video",
                "transcript_status": "fetched",
                "source_basis": "transcript",
                "profile": "institutional_macro",
                "synthesis_section": "🏦 Institutional Research",
                "executive_summary": "GS macro team expects two rate cuts in H2, with inflation moderating to 2.4% by year-end.",
                "tickers_mentioned": ["TLT", "US10Y"],
                "profile_data": {
                    "investment_themes": ["Duration extension as rates peak"],
                    "asset_allocation_views": ["Overweight fixed income vs equities"],
                    "risk_assessments": ["Tariff escalation remains tail risk"],
                    "market_outlook": "Cautiously constructive with bias toward bonds.",
                },
                "key_timestamps": [],
                "topic_tags": [{"tag": "rates", "score": 95}, {"tag": "inflation", "score": 88}],
                "confidence": 0.90,
            },
        ],
        "channels": {
            "top3pct": {
                "channel_name": "3% 財富覺醒",
                "video_count": 1,
                "summary": "Bullish tactical stance with emphasis on support levels and staged entries.",
                "top_topics": ["market structure", "trade setups"],
            },
            "goldman-sachs": {
                "channel_name": "Goldman Sachs",
                "video_count": 1,
                "summary": "Institutional view: rates peaking, duration extension favored.",
                "top_topics": ["rates", "inflation"],
            },
        },
        "run_summary": {
            "overall_day_summary": "Trading sources are bullish near-term on SPX support levels, while institutional research flags a pivot toward fixed income duration.",
            "cross_video_themes": ["rates trajectory", "equity support levels"],
            "agreements": ["Both sources see near-term support holding."],
            "disagreements": [],
            "top_claims_worth_watching": ["Watch whether SPX 5,200 holds on the next pullback."],
            "crowded_trades": ["Long duration / TLT mentioned by institutional sources."],
            "contrarian_flags": [],
            "run_notes": [],
        },
        "errors": [],
    }
    return send_analysis_summary_email(summary)


def _build_subject(summary: dict[str, Any]) -> str:
    videos = list(summary.get("videos") or [])
    queued_retry_count = int(summary.get("queued_retry_count") or 0)
    channels = sorted({item.get("channel_name") or item.get("channel_slug") or "Unknown" for item in videos})
    if len(channels) == 1:
        channel_label = channels[0]
    else:
        channel_label = f"{len(channels)} channels"
    if not videos and queued_retry_count:
        return f"[YOUTUBE ANALYST] retry scheduled for {queued_retry_count} video(s)"
    if queued_retry_count:
        return f"[YOUTUBE ANALYST] {len(videos)} video(s) across {channel_label} + {queued_retry_count} retry"
    return f"[YOUTUBE ANALYST] {len(videos)} video(s) across {channel_label}"


def _group_videos_by_section(videos: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group videos by synthesis_section, preserving SYNTHESIS_SECTION_ORDER."""
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in videos:
        section = item.get("synthesis_section") or "📰 News Digest"
        by_section[section].append(item)
    ordered: list[tuple[str, list[dict[str, Any]]]] = []
    for section in SYNTHESIS_SECTION_ORDER:
        if section in by_section:
            ordered.append((section, by_section.pop(section)))
    for section in sorted(by_section.keys()):
        ordered.append((section, by_section[section]))
    return ordered


def _build_plain_body(summary: dict[str, Any]) -> str:
    run_summary = summary.get("run_summary") or {}
    lines = [
        "YOUTUBE ANALYST",
        "",
        f"Run started: {summary.get('run_started_at') or 'Unknown'}",
        f"Analysis model: {summary.get('analysis_model') or 'Unknown'}",
        (
            "Models used: "
            + ", ".join(str(model) for model in (summary.get("analysis_models_used") or []) if str(model).strip())
        )
        if summary.get("analysis_models_used")
        else "Models used: (not recorded)",
        "",
        "Top run summary:",
        str(run_summary.get("overall_day_summary") or "No summary available."),
        "",
    ]

    if summary.get("retryable_failure_count") or summary.get("non_retryable_failure_count"):
        lines.extend(
            [
                "Partial analysis status:",
                f"- Analyzed videos: {len(summary.get('videos') or [])}",
                f"- Queued retries: {summary.get('queued_retry_count') or 0}",
                f"- Non-retryable failures: {summary.get('non_retryable_failure_count') or 0}",
                "",
            ]
        )

    failed_items = list(summary.get("failed_items") or [])
    if failed_items:
        lines.append("Failed videos:")
        for failed_item in failed_items:
            label = f"{failed_item.get('channel_slug') or 'unknown'} / {failed_item.get('video_id') or 'unknown'}"
            if failed_item.get("retry_exhausted"):
                status = "retry budget exhausted"
            elif failed_item.get("retryable"):
                status = f"queued for retry after {failed_item.get('next_retry_after') or 'unknown time'}"
            else:
                status = failed_item.get("failure_kind") or "analysis failure"
            lines.append(f"- {label}: {status}")
        lines.append("")

    if run_summary.get("cross_video_themes"):
        lines.append("Cross-video themes:")
        lines.extend(f"- {item}" for item in run_summary.get("cross_video_themes") or [])
        lines.append("")

    if run_summary.get("crowded_trades"):
        lines.append("Crowded trades:")
        lines.extend(f"- {item}" for item in run_summary.get("crowded_trades") or [])
        lines.append("")

    if run_summary.get("contrarian_flags"):
        lines.append("Contrarian flags:")
        lines.extend(f"- {item}" for item in run_summary.get("contrarian_flags") or [])
        lines.append("")

    grouped = _group_videos_by_section(list(summary.get("videos") or []))
    for section_title, section_videos in grouped:
        lines.append(f"--- {section_title} ---")
        lines.append("")
        for item in section_videos:
            lines.extend(
                [
                    f"  {item.get('channel_name') or item.get('channel_slug')}",
                    f"  Title: {item.get('title')}",
                    f"  Link: {item.get('webpage_url')}",
                    f"  Published: {item.get('published_at') or 'Unknown'}",
                    f"  Confidence: {item.get('confidence')}",
                    f"  Takeaway: {item.get('executive_summary')}",
                ]
            )
            tickers = item.get("tickers_mentioned") or []
            if tickers:
                lines.append(f"  Tickers: {', '.join(tickers)}")
            profile_data = item.get("profile_data") or {}
            for field_name, values in profile_data.items():
                label = field_name.replace("_", " ").title()
                if isinstance(values, list):
                    lines.append(f"  {label}:")
                    lines.extend(f"    - {v}" for v in values[:4])
                elif isinstance(values, str) and values:
                    lines.append(f"  {label}: {values}")
            timestamps = item.get("key_timestamps") or []
            if timestamps:
                lines.append("  Key timestamps:")
                for timestamp in timestamps[:3]:
                    lines.append(
                        f"    - {timestamp.get('timestamp')} | {timestamp.get('label')}: "
                        f"{timestamp.get('snippet')} ({timestamp.get('why_it_matters')})"
                    )
            tags = item.get("topic_tags") or []
            if tags:
                lines.append(
                    "  Tags: "
                    + ", ".join(f"{tag.get('tag')} ({tag.get('score')})" for tag in tags[:4] if tag.get("tag"))
                )
            lines.append("")

    notes = _build_notes(summary)
    if notes:
        lines.append("Run notes:")
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines)


def _build_html_body(summary: dict[str, Any]) -> str:
    run_summary = summary.get("run_summary") or {}

    themes_html = ""
    if run_summary.get("cross_video_themes"):
        themes_html = (
            "<div style='margin:18px 0 22px;'>"
            "<div style='font-weight:700;color:#111827;margin-bottom:8px;'>Cross-video themes</div>"
            "<ul style='margin:0;padding-left:18px;color:#374151;'>"
            + "".join(f"<li>{_escape_html(theme)}</li>" for theme in run_summary.get("cross_video_themes") or [])
            + "</ul></div>"
        )

    crowded_html = ""
    if run_summary.get("crowded_trades"):
        crowded_html = (
            "<div style='margin:0 0 18px;padding:14px;border:1px solid #f59e0b;background:#fffbeb;border-radius:10px;'>"
            "<div style='font-weight:700;margin-bottom:6px;color:#92400e;'>🔥 Crowded trades</div>"
            "<ul style='margin:0;padding-left:18px;color:#78350f;'>"
            + "".join(f"<li>{_escape_html(item)}</li>" for item in run_summary.get("crowded_trades") or [])
            + "</ul></div>"
        )

    contrarian_html = ""
    if run_summary.get("contrarian_flags"):
        contrarian_html = (
            "<div style='margin:0 0 18px;padding:14px;border:1px solid #ef4444;background:#fef2f2;border-radius:10px;'>"
            "<div style='font-weight:700;margin-bottom:6px;color:#991b1b;'>⚡ Contrarian flags</div>"
            "<ul style='margin:0;padding-left:18px;color:#7f1d1d;'>"
            + "".join(f"<li>{_escape_html(item)}</li>" for item in run_summary.get("contrarian_flags") or [])
            + "</ul></div>"
        )

    sections_html = ""
    grouped = _group_videos_by_section(list(summary.get("videos") or []))
    for section_title, section_videos in grouped:
        cards_html = "".join(_build_video_card(item) for item in section_videos)
        sections_html += f"""
<div style="margin-top:28px;">
  <div style="font-size:18px;font-weight:700;color:#1d4ed8;margin-bottom:4px;border-bottom:2px solid #dbeafe;padding-bottom:6px;">{_escape_html(section_title)}</div>
  {cards_html}
</div>
"""

    notes = _build_notes(summary)
    notes_html = ""
    if notes:
        notes_html = (
            "<div style='margin-top:20px;padding:16px;border:1px solid #f59e0b;background:#fffbeb;border-radius:10px;'>"
            "<div style='font-weight:700;margin-bottom:8px;color:#92400e;'>Run notes</div>"
            "<ul style='margin:0;padding-left:18px;color:#78350f;'>"
            + "".join(f"<li>{_escape_html(note)}</li>" for note in notes)
            + "</ul></div>"
        )

    return f"""
<html>
  <body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;padding:24px;color:#111827;">
    <div style="max-width:860px;margin:0 auto;background:#ffffff;border-radius:12px;padding:28px 32px;border:1px solid #e5e7eb;">
      <div style="font-size:12px;font-weight:700;letter-spacing:0.08em;color:#2563eb;text-transform:uppercase;">YouTube Analyst</div>
      <h1 style="margin:10px 0 6px;font-size:28px;line-height:1.2;">{len(summary.get('videos') or [])} video analysis digest</h1>
      <p style="margin:0 0 12px;color:#4b5563;">Run started at {_escape_html(str(summary.get('run_started_at') or 'Unknown'))}.</p>
      <p style="margin:0 0 12px;color:#4b5563;">Models used: {_escape_html(', '.join(str(model) for model in (summary.get('analysis_models_used') or []) if str(model).strip()) or '(not recorded)')}</p>
      <p style="margin:0;color:#111827;line-height:1.7;"><strong>Top run summary:</strong> {_escape_html(run_summary.get('overall_day_summary') or 'No summary available.')}</p>
      {_build_partial_status_html(summary)}
      {_build_failed_items_html(summary)}
      {themes_html}
      {crowded_html}
      {contrarian_html}
      {sections_html}
      {notes_html}
    </div>
  </body>
</html>
"""


def _build_video_card(item: dict[str, Any]) -> str:
    timestamps = item.get("key_timestamps") or []
    tags = item.get("topic_tags") or []
    tickers = item.get("tickers_mentioned") or []
    profile_data = item.get("profile_data") or {}

    tickers_html = ""
    if tickers:
        tickers_html = (
            "<div style='margin-top:10px;font-size:13px;'>"
            "<strong>Tickers:</strong> "
            + ", ".join(f"<span style='background:#dbeafe;color:#1e40af;padding:2px 6px;border-radius:4px;font-weight:600;'>{_escape_html(t)}</span>" for t in tickers[:8])
            + "</div>"
        )

    profile_html = ""
    for field_name, values in profile_data.items():
        label = field_name.replace("_", " ").title()
        if isinstance(values, list) and values:
            profile_html += (
                f"<div style='margin-top:14px;'>"
                f"<div style='font-weight:700;color:#111827;margin-bottom:6px;'>{_escape_html(label)}</div>"
                f"<ul style='margin:0;padding-left:18px;color:#374151;'>"
                + "".join(f"<li>{_escape_html(v)}</li>" for v in values[:4])
                + "</ul></div>"
            )
        elif isinstance(values, str) and values:
            profile_html += (
                f"<div style='margin-top:10px;color:#374151;'>"
                f"<strong>{_escape_html(label)}:</strong> {_escape_html(values)}"
                f"</div>"
            )

    timestamp_html = ""
    if timestamps:
        timestamp_html = (
            "<div style='margin-top:14px;'>"
            "<div style='font-weight:700;color:#111827;margin-bottom:6px;'>Key timestamps</div>"
            "<ul style='margin:0;padding-left:18px;color:#374151;'>"
            + "".join(
                "<li><strong>"
                + _escape_html(timestamp.get("timestamp") or "")
                + "</strong> "
                + _escape_html(timestamp.get("label") or "")
                + ": "
                + _escape_html(timestamp.get("snippet") or "")
                + " <span style='color:#6b7280;'>("
                + _escape_html(timestamp.get("why_it_matters") or "")
                + ")</span></li>"
                for timestamp in timestamps[:3]
            )
            + "</ul></div>"
        )

    tags_html = ""
    if tags:
        tags_html = (
            "<div style='margin-top:14px;font-size:13px;color:#4b5563;'>"
            "<strong>Topic tags:</strong> "
            + ", ".join(
                f"{_escape_html(tag.get('tag') or '')} ({_escape_html(tag.get('score') or '')})"
                for tag in tags[:4]
                if tag.get("tag")
            )
            + "</div>"
        )

    return f"""
<div style="padding:18px 20px;border:1px solid #e5e7eb;border-radius:10px;margin-top:16px;background:#fafafa;">
  <div style="font-size:12px;font-weight:700;color:#1d4ed8;text-transform:uppercase;letter-spacing:0.05em;">
    {_escape_html(item.get("channel_name") or item.get("channel_slug") or "Unknown Channel")}
  </div>
  <div style="margin-top:6px;font-size:22px;font-weight:700;line-height:1.3;">
    <a href="{_escape_html(item.get("webpage_url") or "#")}" style="color:#1d4ed8;text-decoration:underline;">{_escape_html(item.get("title") or "Untitled")}</a>
  </div>
  <div style="margin-top:10px;font-size:13px;color:#4b5563;">
    Published: {_escape_html(item.get("published_at") or "Unknown")}<br />
    Type: {_escape_html(item.get("source_kind") or "unknown")}<br />
    Confidence: {_escape_html(item.get("confidence") or "unknown")}
  </div>
  <div style="margin-top:12px;font-size:14px;color:#111827;line-height:1.7;">
    {_escape_html(item.get("executive_summary") or "No takeaway available.")}
  </div>
  {tickers_html}
  {profile_html}
  {timestamp_html}
  {tags_html}
</div>
"""


def _build_notes(summary: dict[str, Any]) -> list[str]:
    notes = list(summary.get("errors") or [])
    run_summary = summary.get("run_summary") or {}
    notes.extend(str(item) for item in (run_summary.get("run_notes") or []))
    for failed_item in summary.get("failed_items") or []:
        if failed_item.get("retry_exhausted"):
            notes.append(
                f"{failed_item.get('channel_slug')}/{failed_item.get('video_id')}: retry budget exhausted."
            )
        elif failed_item.get("retryable"):
            notes.append(
                f"{failed_item.get('channel_slug')}/{failed_item.get('video_id')}: queued for retry after "
                f"{failed_item.get('next_retry_after') or 'unknown time'}."
            )
        else:
            notes.append(
                f"{failed_item.get('channel_slug')}/{failed_item.get('video_id')}: "
                f"{failed_item.get('failure_kind') or 'analysis failure'}."
            )

    metadata_only = [
        f"{item.get('channel_name') or item.get('channel_slug')}: analyzed from metadata only."
        for item in summary.get("videos") or []
        if item.get("source_basis") == "metadata_only"
    ]
    notes.extend(metadata_only)
    if summary.get("fallback_activated"):
        notes.append("Automatic fallback model was activated during this run.")
    unique_notes: list[str] = []
    seen: set[str] = set()
    for note in notes:
        if note in seen:
            continue
        seen.add(note)
        unique_notes.append(note)
    return unique_notes


def _build_partial_status_html(summary: dict[str, Any]) -> str:
    retryable_failure_count = int(summary.get("retryable_failure_count") or 0)
    non_retryable_failure_count = int(summary.get("non_retryable_failure_count") or 0)
    queued_retry_count = int(summary.get("queued_retry_count") or 0)
    if not (retryable_failure_count or non_retryable_failure_count or queued_retry_count):
        return ""
    return (
        "<div style='margin:18px 0;padding:14px;border:1px solid #f59e0b;background:#fffbeb;border-radius:10px;'>"
        "<div style='font-weight:700;margin-bottom:6px;color:#92400e;'>Partial analysis status</div>"
        f"<div style='color:#78350f;'>Analyzed videos: {len(summary.get('videos') or [])}<br />"
        f"Queued retries: {queued_retry_count}<br />"
        f"Non-retryable failures: {non_retryable_failure_count}</div></div>"
    )


def _build_failed_items_html(summary: dict[str, Any]) -> str:
    failed_items = list(summary.get("failed_items") or [])
    if not failed_items:
        return ""

    list_items = []
    for failed_item in failed_items:
        label = f"{failed_item.get('channel_slug') or 'unknown'} / {failed_item.get('video_id') or 'unknown'}"
        if failed_item.get("retry_exhausted"):
            status = "retry budget exhausted"
        elif failed_item.get("retryable"):
            status = f"queued for retry after {failed_item.get('next_retry_after') or 'unknown time'}"
        else:
            status = failed_item.get("failure_kind") or "analysis failure"
        list_items.append(f"<li><strong>{_escape_html(label)}</strong>: {_escape_html(status)}</li>")

    return (
        "<div style='margin:18px 0;padding:14px;border:1px solid #d1d5db;background:#f9fafb;border-radius:10px;'>"
        "<div style='font-weight:700;margin-bottom:6px;color:#111827;'>Failed videos</div>"
        "<ul style='margin:0;padding-left:18px;color:#374151;'>"
        + "".join(list_items)
        + "</ul></div>"
    )


def _escape_html(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _get_gmail_credentials() -> tuple[str, str, str]:
    load_local_config()
    sender = read_env(GMAIL_SENDER_ENV, LEGACY_GMAIL_SENDER_ENV)
    app_password = read_env(GMAIL_APP_PASSWORD_ENV, LEGACY_GMAIL_APP_PASSWORD_ENV)
    recipient = read_env(GMAIL_RECIPIENT_ENV, LEGACY_GMAIL_RECIPIENT_ENV)
    return sender, app_password, recipient


_merge_config_file = merge_config_file
