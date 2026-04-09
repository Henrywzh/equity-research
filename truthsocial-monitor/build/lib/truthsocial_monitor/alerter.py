from __future__ import annotations

import json
import logging
import re
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from .models import TruthPost

logger = logging.getLogger(__name__)

ALERT_PATTERNS: dict[str, str] = {
    r"\btariff\w*": "TRADE",
    r"\btrade war\b": "TRADE",
    r"\bfed\b|\bfederal reserve\b": "FED",
    r"\brate\b|\binterest rate\b": "FED",
    r"\bchina\b|\bchinese\b": "GEOPOLITICAL",
    r"\bsanction\w*": "GEOPOLITICAL",
    r"\bwar\b|\bmilitary\b|\btroops\b": "GEOPOLITICAL",
    r"\$[A-Z]{1,5}\b": "TICKER",
    r"\bstock\b|\bwall street\b|\bmarket\b": "MARKET",
    r"\bimpeach\w*|\bindictment\w*": "POLITICAL",
}

_COOLDOWN_FILE = "last_alert.json"


def _load_cooldown(state_dir: Path) -> datetime | None:
    path = state_dir / _COOLDOWN_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return datetime.fromisoformat(data["last_alert_at"])
    except Exception:
        return None


def _save_cooldown(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / _COOLDOWN_FILE
    path.write_text(
        json.dumps({"last_alert_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )


def _scan_post(post: TruthPost) -> list[tuple[str, str]]:
    """Returns list of (matched_text, category) for all keyword hits in a post."""
    hits: list[tuple[str, str]] = []
    text = post.content.lower()
    for pattern, category in ALERT_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hits.append((match.group(0), category))
    return hits


def check_and_alert(
    posts: list[TruthPost],
    *,
    state_dir: Path,
    sender: str,
    app_password: str,
    recipient: str,
    cooldown_secs: int = 900,
) -> bool:
    """Scan posts for keywords. If any hit, send alert email (subject to cooldown). Returns True if email sent."""
    triggered: list[tuple[TruthPost, list[tuple[str, str]]]] = []
    for post in posts:
        hits = _scan_post(post)
        if hits:
            triggered.append((post, hits))

    if not triggered:
        return False

    last = _load_cooldown(state_dir)
    now = datetime.now(timezone.utc)
    if last is not None:
        elapsed = (now - last).total_seconds()
        if elapsed < cooldown_secs:
            logger.info(
                "Alert cooldown active (%ds remaining) — suppressing alert for %d triggered post(s).",
                int(cooldown_secs - elapsed),
                len(triggered),
            )
            return False

    _send_alert_email(triggered, sender=sender, app_password=app_password, recipient=recipient)
    _save_cooldown(state_dir)
    logger.info("Alert email sent for %d triggered post(s).", len(triggered))
    return True


def send_test_email(*, sender: str, app_password: str, recipient: str) -> tuple[bool, str]:
    """Send a test alert email with a fake post."""
    fake_post = TruthPost(
        post_id="000000000000000",
        created_at=datetime.now(timezone.utc).isoformat(),
        content="TEST: This is a test alert. The tariff on China just increased to 145%. $TSLA is down 10%.",
        reblogs_count=0,
        favourites_count=0,
        replies_count=0,
        url="https://truthsocial.com/@realDonaldTrump",
        sensitive=False,
        raw_json="{}",
    )
    hits = _scan_post(fake_post)
    try:
        _send_alert_email([(fake_post, hits)], sender=sender, app_password=app_password, recipient=recipient)
        return True, f"Test alert sent to {recipient}."
    except Exception as exc:
        return False, f"Failed to send test alert: {exc}"


def _send_alert_email(
    triggered: list[tuple[TruthPost, list[tuple[str, str]]]],
    *,
    sender: str,
    app_password: str,
    recipient: str,
) -> None:
    categories = sorted({cat for _, hits in triggered for _, cat in hits})
    subject = f"[TRUMP ALERT] {', '.join(categories)} — {len(triggered)} post(s)"

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(_build_plain(triggered), "plain", "utf-8"))
    msg.attach(MIMEText(_build_html(triggered), "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, recipient, msg.as_string())


def _build_plain(triggered: list[tuple[TruthPost, list[tuple[str, str]]]]) -> str:
    lines = ["TRUMP ALERT", "=" * 40, ""]
    for post, hits in triggered:
        categories = ", ".join(sorted({cat for _, cat in hits}))
        lines += [
            f"[{categories}]",
            f"Posted: {post.created_at}",
            f"URL: {post.url}",
            "",
            post.content,
            "",
            f"Engagement: {post.reblogs_count} reblogs / {post.favourites_count} likes / {post.replies_count} replies",
            "-" * 40,
            "",
        ]
    return "\n".join(lines)


def _build_html(triggered: list[tuple[TruthPost, list[tuple[str, str]]]]) -> str:
    cards = ""
    for post, hits in triggered:
        categories = sorted({cat for _, cat in hits})
        tags_html = "".join(
            f"<span style='background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:12px;"
            f"font-size:12px;font-weight:700;margin-right:4px;'>{_esc(cat)}</span>"
            for cat in categories
        )
        content_html = _esc(post.content).replace("\n", "<br>")
        cards += f"""
<div style="border:1px solid #e5e7eb;border-radius:10px;padding:18px 20px;margin-bottom:16px;background:#fff;">
  <div style="margin-bottom:8px;">{tags_html}</div>
  <div style="font-size:14px;color:#4b5563;margin-bottom:10px;">
    {_esc(post.created_at)} &nbsp;|&nbsp;
    <a href="{_esc(post.url)}" style="color:#2563eb;">{_esc(post.url)}</a>
  </div>
  <div style="font-size:16px;color:#111827;line-height:1.7;">{content_html}</div>
  <div style="margin-top:10px;font-size:12px;color:#6b7280;">
    {post.reblogs_count} reblogs &nbsp;·&nbsp; {post.favourites_count} likes &nbsp;·&nbsp; {post.replies_count} replies
  </div>
</div>"""

    return f"""
<html>
  <body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;padding:24px;">
    <div style="max-width:700px;margin:0 auto;background:#ffffff;border-radius:12px;padding:28px 32px;border:1px solid #e5e7eb;">
      <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;color:#dc2626;text-transform:uppercase;">
        Truth Social Monitor
      </div>
      <h1 style="margin:8px 0 20px;font-size:24px;color:#111827;">
        {len(triggered)} new post(s) triggered alert
      </h1>
      {cards}
    </div>
  </body>
</html>"""


def _esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
