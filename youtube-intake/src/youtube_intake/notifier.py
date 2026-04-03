from __future__ import annotations

import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any


GMAIL_SENDER_ENV = "YOUTUBE_INTAKE_GMAIL_SENDER"
GMAIL_APP_PASSWORD_ENV = "YOUTUBE_INTAKE_GMAIL_APP_PASSWORD"
GMAIL_RECIPIENT_ENV = "YOUTUBE_INTAKE_GMAIL_RECIPIENT"
LEGACY_GMAIL_SENDER_ENV = "GMAIL_SENDER"
LEGACY_GMAIL_APP_PASSWORD_ENV = "GMAIL_APP_PASSWORD"
LEGACY_GMAIL_RECIPIENT_ENV = "GMAIL_RECIPIENT"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_run_result(result_path: str | Path) -> dict[str, Any]:
    path = Path(result_path).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def send_run_summary_email(summary: dict[str, Any]) -> tuple[bool, str]:
    new_items = list(summary.get("new_items") or [])
    if not new_items:
        return False, "No new items to email."

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

    return True, f"Sent Gmail summary to {recipient}."


def send_test_email() -> tuple[bool, str]:
    summary = {
        "status": "success",
        "run_started_at": "TEST",
        "archived_count": 1,
        "bootstrap_count": 0,
        "transcript_unavailable_count": 0,
        "channels": {},
        "errors": [],
        "new_items": [
            {
                "archive_path": "TEST",
                "channel_slug": "test-channel",
                "channel_handle": "@testchannel",
                "channel_name": "Test Channel",
                "video_id": "test-video",
                "title": "YouTube Intake Gmail Connectivity Test",
                "webpage_url": "https://www.youtube.com/watch?v=test-video",
                "published_at": "TEST",
                "source_kind": "video",
                "transcript_status": "fetched",
                "description_excerpt": "This is a test message from youtube-intake.",
            }
        ],
    }
    return send_run_summary_email(summary)


def _build_subject(summary: dict[str, Any]) -> str:
    new_items = list(summary.get("new_items") or [])
    channels = sorted({item.get("channel_name") or item.get("channel_slug") or "Unknown" for item in new_items})
    if len(channels) == 1:
        channel_label = channels[0]
    else:
        channel_label = f"{len(channels)} channels"
    return f"[YOUTUBE INTAKE] {len(new_items)} new item(s) from {channel_label}"


def _build_plain_body(summary: dict[str, Any]) -> str:
    lines = [
        "YOUTUBE INTAKE",
        "",
        f"Run started: {summary.get('run_started_at')}",
        f"New items archived: {summary.get('archived_count', 0)}",
        "",
    ]
    for index, item in enumerate(summary.get("new_items") or [], start=1):
        lines.extend(
            [
                f"{index}. {item.get('channel_name') or item.get('channel_slug')}",
                f"Title: {item.get('title')}",
                f"Link: {item.get('webpage_url')}",
                f"Published: {item.get('published_at') or 'Unknown'}",
                f"Type: {item.get('source_kind')}",
                f"Transcript: {item.get('transcript_status')}",
                f"Summary: {item.get('description_excerpt') or '(no description excerpt)'}",
                "",
            ]
        )
    errors = list(summary.get("errors") or [])
    if errors:
        lines.append("Run notes:")
        lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines)


def _build_html_body(summary: dict[str, Any]) -> str:
    items_html = "".join(_build_item_card(item) for item in summary.get("new_items") or [])
    errors = list(summary.get("errors") or [])
    notes_html = ""
    if errors:
        notes_html = (
            "<div style='margin-top:20px;padding:16px;border:1px solid #f59e0b;background:#fffbeb;border-radius:8px;'>"
            "<div style='font-weight:700;margin-bottom:8px;color:#92400e;'>Run notes</div>"
            "<ul style='margin:0;padding-left:18px;color:#78350f;'>"
            + "".join(f"<li>{_escape_html(error)}</li>" for error in errors)
            + "</ul></div>"
        )

    return f"""
<html>
  <body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;padding:24px;color:#111827;">
    <div style="max-width:760px;margin:0 auto;background:#ffffff;border-radius:12px;padding:28px 32px;border:1px solid #e5e7eb;">
      <div style="font-size:12px;font-weight:700;letter-spacing:0.08em;color:#2563eb;text-transform:uppercase;">YouTube Intake</div>
      <h1 style="margin:10px 0 6px;font-size:28px;line-height:1.2;">{len(summary.get('new_items') or [])} new item(s) archived</h1>
      <p style="margin:0 0 24px;color:#4b5563;">Run started at { _escape_html(str(summary.get('run_started_at') or 'Unknown')) }.</p>
      {items_html}
      {notes_html}
    </div>
  </body>
</html>
"""


def _build_item_card(item: dict[str, Any]) -> str:
    return f"""
<div style="padding:18px 20px;border:1px solid #e5e7eb;border-radius:10px;margin-bottom:16px;background:#fafafa;">
  <div style="font-size:12px;font-weight:700;color:#1d4ed8;text-transform:uppercase;letter-spacing:0.05em;">
    {_escape_html(item.get("channel_name") or item.get("channel_slug") or "Unknown Channel")}
  </div>
  <div style="margin-top:6px;font-size:22px;font-weight:700;line-height:1.3;">
    <a href="{_escape_html(item.get("webpage_url") or "#")}" style="color:#111827;text-decoration:none;">{_escape_html(item.get("title") or "Untitled")}</a>
  </div>
  <div style="margin-top:10px;font-size:13px;color:#4b5563;">
    Published: {_escape_html(item.get("published_at") or "Unknown")}<br />
    Type: {_escape_html(item.get("source_kind") or "unknown")}<br />
    Transcript: {_escape_html(item.get("transcript_status") or "unknown")}
  </div>
  <div style="margin-top:12px;font-size:14px;color:#374151;line-height:1.6;">
    {_escape_html(item.get("description_excerpt") or "(no description excerpt)")}
  </div>
</div>
"""


def _escape_html(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _get_gmail_credentials() -> tuple[str, str, str]:
    _load_local_config()

    sender = _read_env(GMAIL_SENDER_ENV, LEGACY_GMAIL_SENDER_ENV)
    app_password = _read_env(GMAIL_APP_PASSWORD_ENV, LEGACY_GMAIL_APP_PASSWORD_ENV)
    recipient = _read_env(GMAIL_RECIPIENT_ENV, LEGACY_GMAIL_RECIPIENT_ENV)
    return sender, app_password, recipient


def _read_env(primary: str, legacy: str) -> str:
    return os.getenv(primary, "").strip() or os.getenv(legacy, "").strip()


def _load_local_config() -> None:
    for path in (_REPO_ROOT / ".config", _PROJECT_ROOT / ".config"):
        _merge_config_file(path)


def _merge_config_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        os.environ.setdefault(key, value)
