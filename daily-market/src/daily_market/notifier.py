from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from .config import get_project_root

# Same env var names as daily-macro (shared credentials — no separate prefix needed)
GMAIL_SENDER_ENV = "GMAIL_SENDER"
GMAIL_APP_PASSWORD_ENV = "GMAIL_APP_PASSWORD"
GMAIL_RECIPIENT_ENV = "GMAIL_RECIPIENT"


def send_market_summary(
    summary: dict[str, Any],
    session: str,
    date_str: str,
) -> tuple[bool, str]:
    """Send the market summary email. Returns (success, message)."""
    sender, app_password, recipient = _get_gmail_credentials()
    if not sender or not app_password or not recipient:
        raise RuntimeError(
            "Gmail credentials not set. Expected "
            f"{GMAIL_SENDER_ENV}, {GMAIL_APP_PASSWORD_ENV}, and {GMAIL_RECIPIENT_ENV} "
            "as environment variables or in the .config file."
        )

    message = MIMEMultipart("alternative")
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = _build_subject(summary, session, date_str)
    message.attach(MIMEText(summary.get("plain") or "", "plain", "utf-8"))
    message.attach(MIMEText(summary.get("html") or "", "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, recipient, message.as_string())

    return True, f"Sent daily market summary to {recipient}."


def send_test_email() -> tuple[bool, str]:
    """Send a test email to verify Gmail connectivity."""
    dummy_summary = {
        "plain": "This is a test email from the daily-market pipeline.\n",
        "html": "<html><body><p>This is a <strong>test email</strong> from the daily-market pipeline.</p></body></html>",
        "status": "success",
        "ticker_count": 0,
        "success_count": 0,
    }
    return send_market_summary(dummy_summary, session="morning", date_str="TEST")


def _build_subject(summary: dict[str, Any], session: str, date_str: str) -> str:
    status = str(summary.get("status") or "unknown").upper()
    total = summary.get("ticker_count") or 0
    ok = summary.get("success_count") or 0
    session_label = "Morning Brief" if session == "morning" else "Evening Brief"
    suffix = f" [{status}]" if status != "SUCCESS" else ""
    return f"[DAILY MARKET]{suffix} {session_label} | {date_str} | {ok}/{total} symbols"


# -------------------------------------------------------------------
# Credential helpers  (mirrors daily-macro/notifier.py exactly)
# -------------------------------------------------------------------

def _get_gmail_credentials() -> tuple[str, str, str]:
    config_values = _load_local_config()
    sender = (
        _read_env(GMAIL_SENDER_ENV)
        or config_values.get(GMAIL_SENDER_ENV)
        or ""
    ).strip()
    app_password = (
        _read_env(GMAIL_APP_PASSWORD_ENV)
        or config_values.get(GMAIL_APP_PASSWORD_ENV)
        or ""
    ).strip()
    recipient = (
        _read_env(GMAIL_RECIPIENT_ENV)
        or config_values.get(GMAIL_RECIPIENT_ENV)
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
