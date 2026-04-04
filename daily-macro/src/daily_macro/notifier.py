from __future__ import annotations

import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from .config import get_project_root

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
    if status != "success":
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
            "status": "success",
            "model": {
                "provider": "groq",
                "primary_model": "qwen/qwen3-32b",
                "fallback_model": "llama-3.1-8b-instant",
            },
            "model_switches": [
                {
                    "from_model": "qwen/qwen3-32b",
                    "to_model": "llama-3.1-8b-instant",
                    "reason": "Primary model returned 429 rate_limit_exceeded.",
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
                "successful_article_analyses": 4,
                "failed_article_analyses": 0,
                "full_text_article_count": 3,
                "truncated_article_count": 1,
                "successful_categories": 2,
                "partial_categories": 0,
                "failed_categories": 0,
            },
            "categories": [
                {
                    "category": "國際財經",
                    "article_count": 2,
                    "status": "success",
                    "key_developments": [
                        "Markets focused on fresh geopolitical risk around Iran-related escalation.",
                        "Major global banks tightened operations and contingency planning in Europe.",
                    ],
                    "named_entities": [
                        {"name": "伊朗", "type": "country"},
                        {"name": "Amazon", "type": "company"},
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
    return f"[DAILY MACRO] {report_date} | {article_count} article(s), {category_count} categorie(s)"


def _build_plain_body(summary: dict[str, Any]) -> str:
    diagnostics = summary.get("diagnostics") or {}
    lines = [
        "DAILY MACRO ANALYST",
        "",
        f"Report date: {summary.get('report_date') or 'Unknown'}",
        f"Generated at: {summary.get('generated_at') or 'Unknown'}",
        f"Status: {summary.get('status') or 'Unknown'}",
        f"Primary model: {_model_label(summary, 'primary_model')}",
        f"Fallback model: {_model_label(summary, 'fallback_model')}",
        f"Article count: {(summary.get('totals') or {}).get('article_count') or 0}",
        f"Category count: {(summary.get('input') or {}).get('category_count') or 0}",
        "",
    ]

    notes = _build_run_notes(summary)
    if notes:
        lines.append("Run notes:")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")

    for category in summary.get("categories") or []:
        lines.append(f"{category.get('category') or 'Uncategorized'} ({category.get('article_count') or 0})")
        for item in (category.get("key_developments") or [])[:5]:
            lines.append(f"- {item}")
        entities = category.get("named_entities") or []
        if entities:
            lines.append("Named entities: " + ", ".join(entity["name"] for entity in entities[:8] if entity.get("name")))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _build_html_body(summary: dict[str, Any]) -> str:
    category_blocks = []
    for category in summary.get("categories") or []:
        items = "".join(
            f"<li>{_escape_html(str(item))}</li>"
            for item in (category.get("key_developments") or [])[:5]
        )
        entities = category.get("named_entities") or []
        entities_html = ""
        if entities:
            labels = ", ".join(_escape_html(str(entity.get("name") or "")) for entity in entities[:8] if entity.get("name"))
            entities_html = f"<p style='margin:10px 0 0;color:#4b5563;'><strong>Named entities:</strong> {labels}</p>"
        category_blocks.append(
            f"""
            <div style="margin:0 0 18px;padding:16px;border:1px solid #e5e7eb;border-radius:12px;background:#ffffff;">
              <div style="font-size:18px;font-weight:700;color:#111827;margin-bottom:8px;">
                {_escape_html(str(category.get("category") or "Uncategorized"))}
                <span style="font-size:13px;font-weight:500;color:#6b7280;">({int(category.get("article_count") or 0)} article(s))</span>
              </div>
              <ul style="margin:0;padding-left:18px;color:#374151;line-height:1.6;">{items}</ul>
              {entities_html}
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

    return f"""
    <html>
      <body style="margin:0;padding:24px;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
        <div style="max-width:760px;margin:0 auto;background:#ffffff;border-radius:16px;padding:28px 28px 16px;border:1px solid #e5e7eb;">
          <div style="font-size:30px;font-weight:800;color:#111827;">Daily Macro Analyst</div>
          <p style="margin:8px 0 0;color:#4b5563;">
            Report date: {_escape_html(str(summary.get("report_date") or "Unknown"))}
          </p>
          <p style="margin:6px 0 0;color:#4b5563;">
            Articles: {int((summary.get("totals") or {}).get("article_count") or 0)} |
            Categories: {int((summary.get("input") or {}).get("category_count") or 0)}
          </p>
          <p style="margin:6px 0 20px;color:#4b5563;">
            Primary model: {_escape_html(_model_label(summary, "primary_model"))} |
            Fallback model: {_escape_html(_model_label(summary, "fallback_model"))}
          </p>
          {notes_html}
          {"".join(category_blocks)}
        </div>
      </body>
    </html>
    """


def _build_run_notes(summary: dict[str, Any]) -> list[str]:
    diagnostics = summary.get("diagnostics") or {}
    notes: list[str] = []
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
    output_path = summary.get("output_path")
    if output_path:
        notes.append(f"Stored analysis report: {output_path}")
    return notes


def _model_label(summary: dict[str, Any], key: str) -> str:
    model = summary.get("model") or {}
    provider = str(model.get("provider") or "groq")
    model_id = str(model.get(key) or "unknown")
    return f"{provider}/{model_id}"


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


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
