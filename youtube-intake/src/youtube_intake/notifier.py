from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from .runtime_env import load_local_config, merge_config_file, read_env
from .storage import load_json_document


GMAIL_SENDER_ENV = "YOUTUBE_INTAKE_GMAIL_SENDER"
GMAIL_APP_PASSWORD_ENV = "YOUTUBE_INTAKE_GMAIL_APP_PASSWORD"
GMAIL_RECIPIENT_ENV = "YOUTUBE_INTAKE_GMAIL_RECIPIENT"
LEGACY_GMAIL_SENDER_ENV = "GMAIL_SENDER"
LEGACY_GMAIL_APP_PASSWORD_ENV = "GMAIL_APP_PASSWORD"
LEGACY_GMAIL_RECIPIENT_ENV = "GMAIL_RECIPIENT"


def load_analysis_result(result_path: str | Path) -> dict[str, Any]:
    return load_json_document(result_path)


def load_run_result(result_path: str | Path) -> dict[str, Any]:
    return load_analysis_result(result_path)


def send_analysis_summary_email(summary: dict[str, Any]) -> tuple[bool, str]:
    videos = list(summary.get("videos") or [])
    if not videos:
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
        "videos": [
            {
                "video_id": "abc",
                "channel_slug": "top3pct",
                "channel_name": "3% 財富覺醒",
                "title": "First title",
                "webpage_url": "https://www.youtube.com/watch?v=abc",
                "published_at": "2026-04-03T00:00:00+00:00",
                "source_kind": "video",
                "transcript_status": "fetched",
                "source_basis": "transcript",
                "executive_summary": "Speaker argues the current selloff is fear-driven and points to dealer positioning as the main support.",
                "notable_claims": [
                    "Market makers are absorbing panic selling, limiting downside despite weak sentiment.",
                    "A sharper pullback later in the month would still be treated as a buying opportunity.",
                ],
                "notable_opinions": [
                    "The host prefers staged entries over trying to call the exact bottom.",
                ],
                "key_timestamps": [
                    {
                        "timestamp": "00:02:14",
                        "label": "Support thesis",
                        "snippet": "Explains why panic has not turned into outright breakdown yet.",
                        "why_it_matters": "Frames the entire bullish-near-term positioning.",
                    }
                ],
                "topic_tags": [{"tag": "market structure", "score": 93}, {"tag": "dip buying", "score": 84}],
                "confidence": 0.82,
            }
        ],
        "channels": {
            "top3pct": {
                "channel_name": "3% 財富覺醒",
                "video_count": 1,
                "summary": "Bullish tactical stance with emphasis on dealer support and staged entries.",
                "top_topics": ["market structure", "dip buying"],
            }
        },
        "run_summary": {
            "overall_day_summary": "Today’s finance videos leaned constructive on near-term market resilience but still warned that a sharper reset could arrive before a cleaner entry.",
            "cross_video_themes": ["buying fear", "dealer positioning"],
            "agreements": ["Several hosts treat pullbacks as tactical entries instead of regime breaks."],
            "disagreements": [],
            "top_claims_worth_watching": ["Watch whether downside continues to stall despite negative sentiment."],
            "run_notes": [],
        },
        "errors": [],
    }
    return send_analysis_summary_email(summary)


def _build_subject(summary: dict[str, Any]) -> str:
    videos = list(summary.get("videos") or [])
    channels = sorted({item.get("channel_name") or item.get("channel_slug") or "Unknown" for item in videos})
    if len(channels) == 1:
        channel_label = channels[0]
    else:
        channel_label = f"{len(channels)} channels"
    return f"[YOUTUBE ANALYST] {len(videos)} video(s) across {channel_label}"


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

    if run_summary.get("cross_video_themes"):
        lines.append("Cross-video themes:")
        lines.extend(f"- {item}" for item in run_summary.get("cross_video_themes") or [])
        lines.append("")

    for index, item in enumerate(summary.get("videos") or [], start=1):
        lines.extend(
            [
                f"{index}. {item.get('channel_name') or item.get('channel_slug')}",
                f"Title: {item.get('title')}",
                f"Link: {item.get('webpage_url')}",
                f"Published: {item.get('published_at') or 'Unknown'}",
                f"Type: {item.get('source_kind')}",
                f"Source basis: {item.get('source_basis')}",
                f"Confidence: {item.get('confidence')}",
                f"Takeaway: {item.get('executive_summary')}",
            ]
        )
        claims = item.get("notable_claims") or []
        opinions = item.get("notable_opinions") or []
        timestamps = item.get("key_timestamps") or []
        tags = item.get("topic_tags") or []
        if claims:
            lines.append("Claims:")
            lines.extend(f"- {claim}" for claim in claims[:3])
        if opinions:
            lines.append("Opinions:")
            lines.extend(f"- {opinion}" for opinion in opinions[:3])
        if timestamps:
            lines.append("Key timestamps:")
            for timestamp in timestamps[:3]:
                lines.append(
                    f"- {timestamp.get('timestamp')} | {timestamp.get('label')}: "
                    f"{timestamp.get('snippet')} ({timestamp.get('why_it_matters')})"
                )
        if tags:
            lines.append(
                "Tags: "
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
    video_cards = "".join(_build_video_card(item) for item in summary.get("videos") or [])
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

    themes_html = ""
    if run_summary.get("cross_video_themes"):
        themes_html = (
            "<div style='margin:18px 0 22px;'>"
            "<div style='font-weight:700;color:#111827;margin-bottom:8px;'>Cross-video themes</div>"
            "<ul style='margin:0;padding-left:18px;color:#374151;'>"
            + "".join(f"<li>{_escape_html(theme)}</li>" for theme in run_summary.get("cross_video_themes") or [])
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
      {themes_html}
      {video_cards}
      {notes_html}
    </div>
  </body>
</html>
"""


def _build_video_card(item: dict[str, Any]) -> str:
    claims = item.get("notable_claims") or []
    opinions = item.get("notable_opinions") or []
    timestamps = item.get("key_timestamps") or []
    tags = item.get("topic_tags") or []

    claims_html = _build_bullets("Notable claims", claims[:3], text_color="#374151")
    opinions_html = _build_bullets("Notable opinions", opinions[:3], text_color="#374151")
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
    <a href="{_escape_html(item.get("webpage_url") or "#")}" style="color:#111827;text-decoration:none;">{_escape_html(item.get("title") or "Untitled")}</a>
  </div>
  <div style="margin-top:10px;font-size:13px;color:#4b5563;">
    Published: {_escape_html(item.get("published_at") or "Unknown")}<br />
    Type: {_escape_html(item.get("source_kind") or "unknown")}<br />
    Source basis: {_escape_html(item.get("source_basis") or "unknown")}<br />
    Confidence: {_escape_html(item.get("confidence") or "unknown")}
  </div>
  <div style="margin-top:12px;font-size:14px;color:#111827;line-height:1.7;">
    {_escape_html(item.get("executive_summary") or "No takeaway available.")}
  </div>
  {claims_html}
  {opinions_html}
  {timestamp_html}
  {tags_html}
</div>
"""


def _build_bullets(title: str, values: list[str], *, text_color: str) -> str:
    if not values:
        return ""
    return (
        "<div style='margin-top:14px;'>"
        f"<div style='font-weight:700;color:#111827;margin-bottom:6px;'>{_escape_html(title)}</div>"
        f"<ul style='margin:0;padding-left:18px;color:{text_color};'>"
        + "".join(f"<li>{_escape_html(value)}</li>" for value in values)
        + "</ul></div>"
    )


def _build_notes(summary: dict[str, Any]) -> list[str]:
    notes = list(summary.get("errors") or [])
    run_summary = summary.get("run_summary") or {}
    notes.extend(str(item) for item in (run_summary.get("run_notes") or []))

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
