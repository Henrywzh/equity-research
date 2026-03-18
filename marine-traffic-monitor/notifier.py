"""
notifier.py — Gmail SMTP Alert Notifier
========================================
Sends an email alert when the LLM analyst returns a significant threat level.
Also sends a daily digest email once per day (no LLM required).

Fires on  : CRITICAL, ELEVATED, DISPUTED
Silent on : NONE, LOW

Alert cooldown: max 1 alert email per N minutes (default 5) to prevent spam
                during rapid ship count oscillations.

Emails are sent as multipart/alternative (plain-text fallback + rich HTML).
Gmail renders the HTML version; other clients fall back to plain text.

Setup (one-time, ~2 min):
  1. myaccount.google.com → Security → 2-Step Verification (must be ON)
  2. App Passwords → Select "Mail" + "Other device" → name "marine-monitor"
  3. Copy the 16-char password and add to .config:
       GMAIL_SENDER=yourname@gmail.com
       GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
       GMAIL_RECIPIENT=yourname@gmail.com

Usage:
  from notifier import send_alert, send_digest
  send_alert(briefing, model_a="llama_4_scout", model_b="llama_3_3_70b",
             reported_count=3,
             image_paths=["screenshots/raw_map_<ts>.png",
                          "screenshots/detected_ships_<ts>.png"],
             csv_snapshot="...", alert_cooldown=5, news="...")
  send_digest(csv_snapshot="...", reported_today=12)

Standalone test:
  python notifier.py --test
  python notifier.py --test --level DISPUTED
  python notifier.py --test-digest
"""

import os
import sys
import json
import smtplib
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load .config from project root (one level up)
_config_path = os.path.join(os.path.dirname(__file__), "..", ".config")
load_dotenv(dotenv_path=_config_path)

_HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_HERE, "state"), exist_ok=True)
_LAST_ALERT_PATH = os.path.join(_HERE, "state", "last_alert.json")

# Threat levels that trigger an alert email
ALERT_LEVELS = {"CRITICAL", "ELEVATED", "ESCALATED", "DISPUTED", "REVIEW"}

# Emoji prefix per level
_LEVEL_EMOJI = {
    "CRITICAL":  "🚨",
    "ELEVATED":  "⚠️",
    "ESCALATED": "⚡",
    "DISPUTED":  "❓",
    "REVIEW":    "🔬",
}


# ------------------------------------------------------------------
# Cooldown state helpers
# ------------------------------------------------------------------

def _read_last_alert_ts() -> datetime | None:
    """Read the timestamp of the last successfully sent alert email."""
    try:
        with open(_LAST_ALERT_PATH) as f:
            data = json.load(f)
        return datetime.fromisoformat(data["ts"])
    except (FileNotFoundError, KeyError, ValueError):
        return None


def _write_last_alert_ts() -> None:
    """Persist the current UTC timestamp as the last alert send time."""
    with open(_LAST_ALERT_PATH, "w") as f:
        json.dump({"ts": datetime.now(timezone.utc).isoformat()}, f)


# ------------------------------------------------------------------
# HTML helpers
# ------------------------------------------------------------------

def _threat_colour(level: str) -> str:
    """Return the hex background colour for a given threat level."""
    return {
        "CRITICAL":  "#dc2626",
        "ELEVATED":  "#d97706",
        "ESCALATED": "#d97706",
        "DISPUTED":  "#7c3aed",
        "REVIEW":    "#0891b2",
    }.get(level, "#374151")


def _csv_to_html_table(csv_text: str) -> str:
    """Convert a CSV snapshot string into a styled HTML table."""
    if not csv_text.strip():
        return "<p style='color:#6b7280;font-style:italic;'>No traffic data available.</p>"

    lines = [l for l in csv_text.strip().splitlines() if l.strip()]
    if not lines:
        return "<p style='color:#6b7280;font-style:italic;'>No traffic data available.</p>"

    header = lines[0]
    rows   = lines[1:]

    th_style = (
        "padding:8px 12px;text-align:left;font-size:11px;"
        "font-weight:600;text-transform:uppercase;letter-spacing:0.05em;"
        "color:#ffffff;background:#1e3a5f;"
    )
    table_html = (
        "<table style='width:100%;border-collapse:collapse;"
        "font-size:12px;font-family:monospace;'>"
        "<thead><tr>"
    )
    for col in header.split(","):
        table_html += f"<th style='{th_style}'>{col.strip()}</th>"
    table_html += "</tr></thead><tbody>"

    for i, row in enumerate(rows):
        cols = row.split(",")
        # Highlight rows where ships_in_zone > 0 (column index 1)
        try:
            ship_count = int(cols[1].strip()) if len(cols) > 1 else 0
        except ValueError:
            ship_count = 0

        if ship_count > 0:
            row_bg     = "#fffbeb"
            row_border = "border-left:3px solid #d97706;"
        else:
            row_bg     = "#f9fafb" if i % 2 == 0 else "#ffffff"
            row_border = "border-left:3px solid transparent;"

        td_style = f"padding:6px 12px;{row_border}border-bottom:1px solid #e5e7eb;"
        table_html += f"<tr style='background:{row_bg};'>"
        for col in cols:
            table_html += f"<td style='{td_style}'>{col.strip()}</td>"
        table_html += "</tr>"

    table_html += "</tbody></table>"
    return table_html


def _card(title: str, content: str, accent: str = "#6b7280") -> str:
    """Return a styled card section for the HTML email."""
    return f"""
<div style="margin:16px 0;background:#ffffff;border-radius:6px;
            border:1px solid #e5e7eb;border-left:4px solid {accent};
            padding:16px 20px;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;
              letter-spacing:0.08em;color:{accent};margin-bottom:8px;">
    {title}
  </div>
  <div style="font-size:13px;color:#374151;line-height:1.6;">
    {content}
  </div>
</div>"""


def _fmt_evidence_section(evidence: dict, title: str, accent: str) -> str:
    """Render one model's evidence dict as a collapsible HTML block."""
    if not evidence or "parse_error" in evidence:
        err = evidence.get("parse_error", "No evidence available.") if evidence else "No evidence available."
        return _card(title, f"<em style='color:#9ca3af;'>{err}</em>", accent)

    obs   = evidence.get("direct_observations", [])
    ctx   = evidence.get("historical_context", [])
    risk  = evidence.get("risk_signals", [])
    unc   = evidence.get("uncertainties", [])
    hyps  = evidence.get("hypotheses", [])
    conf  = evidence.get("overall_confidence", 0.0)
    abs_  = evidence.get("abstain", False)
    abs_r = evidence.get("abstain_reason", "")
    rec   = evidence.get("recommended_state", "?")

    def _ul(items: list) -> str:
        if not items:
            return "<em style='color:#9ca3af;font-size:11px;'>None noted.</em>"
        return "<ul style='margin:4px 0 0;padding-left:16px;'>" + \
               "".join(f"<li style='font-size:12px;color:#374151;margin-bottom:3px;'>{i}</li>" for i in items) + \
               "</ul>"

    hyp_html = ""
    if hyps:
        hyp_html = "<div style='margin-top:6px;'>"
        for h in hyps:
            c = float(h.get("confidence", 0))
            bar_w = int(c * 100)
            bar_col = "#16a34a" if c >= 0.6 else "#d97706" if c >= 0.4 else "#6b7280"
            hyp_html += (
                f"<div style='margin-bottom:6px;'>"
                f"<div style='font-size:11px;color:#374151;'>{h.get('statement','?')}</div>"
                f"<div style='height:4px;background:#e5e7eb;border-radius:2px;margin-top:2px;'>"
                f"<div style='width:{bar_w}%;height:100%;background:{bar_col};border-radius:2px;'></div>"
                f"</div>"
                f"<div style='font-size:10px;color:#6b7280;'>{c:.0%} confidence</div>"
                f"</div>"
            )
        hyp_html += "</div>"

    abstain_badge = (
        f"<span style='background:#fef3c7;color:#92400e;padding:2px 8px;"
        f"border-radius:3px;font-size:10px;font-weight:600;'>ABSTAINED: {abs_r}</span>"
        if abs_ else ""
    )
    conf_col = "#16a34a" if conf >= 0.6 else "#d97706" if conf >= 0.4 else "#dc2626"

    inner = f"""
<div style='display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap;'>
  <span style='background:{accent};color:#fff;padding:3px 10px;border-radius:3px;
               font-size:10px;font-weight:700;text-transform:uppercase;'>Rec: {rec}</span>
  <span style='background:#f3f4f6;color:{conf_col};padding:3px 10px;border-radius:3px;
               font-size:10px;font-weight:700;'>Conf: {conf:.0%}</span>
  {abstain_badge}
</div>
<details>
  <summary style='cursor:pointer;font-size:11px;font-weight:600;color:{accent};
                  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;'>
    Observations &amp; Context
  </summary>
  <div style='margin-top:6px;'>
    <div style='font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;
                letter-spacing:0.05em;margin-bottom:2px;'>Direct Observations</div>
    {_ul(obs)}
    <div style='font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;
                letter-spacing:0.05em;margin:8px 0 2px;'>Historical Context</div>
    {_ul(ctx)}
    <div style='font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;
                letter-spacing:0.05em;margin:8px 0 2px;'>Risk Signals</div>
    {_ul(risk)}
    <div style='font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;
                letter-spacing:0.05em;margin:8px 0 2px;'>Uncertainties</div>
    {_ul(unc)}
  </div>
</details>
<details style='margin-top:6px;'>
  <summary style='cursor:pointer;font-size:11px;font-weight:600;color:{accent};
                  text-transform:uppercase;letter-spacing:0.05em;'>
    Hypotheses ({len(hyps)})
  </summary>
  {hyp_html}
</details>"""

    return _card(title, inner, accent)


def _fmt_policy_reasoning(policy_decision: dict) -> str:
    """Render the policy engine's reasoning chain as a styled card."""
    if not policy_decision:
        return ""
    rules     = policy_decision.get("reasoning", [])
    avg_conf  = policy_decision.get("avg_confidence", 0.0)
    alert     = policy_decision.get("alert_level", "?")
    email_ok  = policy_decision.get("email_decision", False)
    human_rev = policy_decision.get("human_review_needed", False)
    transition = policy_decision.get("proposed_transition", "NONE")

    colour = _threat_colour(alert)

    pills_html = (
        f"<div style='margin-bottom:10px;display:flex;gap:6px;flex-wrap:wrap;'>"
        f"<span style='background:{colour};color:#fff;padding:3px 10px;border-radius:3px;"
        f"font-size:10px;font-weight:700;'>{alert}</span>"
        f"<span style='background:#f3f4f6;color:#374151;padding:3px 10px;border-radius:3px;"
        f"font-size:10px;font-weight:700;'>Conf: {avg_conf:.0%}</span>"
        f"<span style='background:{'#dcfce7' if not email_ok else '#fef3c7'};"
        f"color:{'#166534' if not email_ok else '#92400e'};padding:3px 10px;border-radius:3px;"
        f"font-size:10px;font-weight:700;'>Email: {'YES' if email_ok else 'NO'}</span>"
        f"<span style='background:{'#fee2e2' if human_rev else '#f3f4f6'};"
        f"color:{'#991b1b' if human_rev else '#6b7280'};padding:3px 10px;border-radius:3px;"
        f"font-size:10px;font-weight:700;'>Human Review: {'REQUIRED' if human_rev else 'No'}</span>"
        f"</div>"
    )

    if transition and transition.upper() != "NONE":
        pills_html += (
            f"<div style='font-size:11px;color:#374151;margin-bottom:8px;'>"
            f"&#8594; Suggested transition: <strong>{transition}</strong>"
            f"</div>"
        )

    rules_html = ""
    if rules:
        rules_html = "<ol style='margin:4px 0;padding-left:18px;'>"
        for r in rules:
            rules_html += f"<li style='font-size:12px;color:#374151;font-family:monospace;margin-bottom:3px;'>{r}</li>"
        rules_html += "</ol>"
    else:
        rules_html = "<em style='color:#9ca3af;font-size:11px;'>No rules logged.</em>"

    inner = pills_html + "<div style='font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;'>Rules Fired</div>" + rules_html
    return _card("⚙️ Policy Engine Reasoning", inner, "#374151")


def _build_html_body(briefing: dict, model_a: str, model_b: str,
                     reported_count: int, csv_snapshot: str = "",
                     news: str = "", policy_decision: dict = None) -> str:
    """Build a rich HTML email body for the alert."""
    level       = briefing.get("threat_level", "UNKNOWN")
    cls_        = briefing.get("event_classification", "N/A")
    is_tp       = briefing.get("is_true_positive")
    consensus   = briefing.get("consensus")
    cons_note   = briefing.get("consensus_note", "")
    brief_text  = briefing.get("analyst_briefing", "").replace("\n", "<br>")
    verify_note = briefing.get("verification_note", "").replace("\n", "<br>")
    macro_state = briefing.get("current_macro_state", "—")
    transition  = briefing.get("proposed_state_transition", "NONE")
    ts          = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    emoji       = _LEVEL_EMOJI.get(level, "⚠️")
    colour      = _threat_colour(level)
    models_str  = f"{model_a} + {model_b}" if model_b else model_a

    # Stats pill helper
    def pill(label, value, bg="#f3f4f6", fg="#111827"):
        return (
            f"<div style='display:inline-block;background:{bg};color:{fg};"
            f"border-radius:4px;padding:6px 14px;margin:4px;'>"
            f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;"
            f"letter-spacing:0.06em;color:{fg};opacity:0.7;'>{label}</div>"
            f"<div style='font-size:15px;font-weight:700;'>{value}</div>"
            f"</div>"
        )

    # Build news bullets
    news_lines = [l for l in (news or "").strip().splitlines() if l.strip()]
    if news_lines:
        news_html = "<ul style='margin:0;padding-left:18px;'>"
        for line in news_lines:
            news_html += f"<li style='margin-bottom:6px;font-size:12px;color:#1e40af;'>{line}</li>"
        news_html += "</ul>"
    else:
        news_html = "<p style='font-size:12px;color:#6b7280;font-style:italic;margin:0;'>No Hormuz/Iran maritime news in the last 24h.</p>"

    # Traffic table
    table_html = _csv_to_html_table(csv_snapshot) if csv_snapshot else (
        "<p style='color:#6b7280;font-style:italic;'>No traffic data available.</p>"
    )

    # Compact JSON block
    json_str = json.dumps(briefing, indent=2, default=str).replace("<", "&lt;").replace(">", "&gt;")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<div style="max-width:640px;margin:24px auto;background:#f3f4f6;">

  <!-- THREAT BANNER -->
  <div style="background:{colour};border-radius:8px 8px 0 0;padding:28px 28px 20px;">
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:rgba(255,255,255,0.75);">
      Hormuz Transit Corridor · Maritime OSINT
    </div>
    <div style="font-size:28px;font-weight:800;color:#ffffff;margin:8px 0 4px;">
      {emoji} {level}
    </div>
    <div style="font-size:13px;color:rgba(255,255,255,0.85);">{cls_}</div>
    <div style="font-size:11px;color:rgba(255,255,255,0.6);margin-top:10px;">{ts}</div>
  </div>

  <!-- BODY CONTAINER -->
  <div style="background:#f3f4f6;padding:20px 24px 28px;">

    <!-- STATS ROW -->
    <div style="text-align:center;margin-bottom:4px;">
      {pill("Ships Detected", reported_count, "#fef3c7", "#92400e")}
      {pill("True Positive", "Yes" if is_tp else ("No" if is_tp is False else "?"), "#f0fdf4" if is_tp else "#fef2f2", "#166534" if is_tp else "#991b1b")}
      {pill("Consensus", "Yes" if consensus else "No", "#eff6ff" if consensus else "#faf5ff", "#1e40af" if consensus else "#6d28d9")}
      {pill("Macro State", macro_state, "#f8fafc", "#374151")}
    </div>
    <div style="text-align:center;margin-bottom:12px;font-size:11px;color:#6b7280;">
      Models: <strong>{models_str}</strong>
      {"&nbsp;·&nbsp;Transition: <strong>" + transition + "</strong>" if transition and transition.upper() != "NONE" else ""}
    </div>

    {_card("🔍 Verification Note", verify_note, "#6b7280")}
    {_card("📋 Analyst Briefing", brief_text, "#1e3a5f")}

    <!-- EVIDENCE SECTIONS -->
    {_fmt_evidence_section(briefing.get("evidence_a"), f"🔭 Model A Evidence ({model_a})", "#1e3a5f")}
    {_fmt_evidence_section(briefing.get("evidence_b"), f"📈 Model B Evidence ({model_b})", "#1e40af")}

    <!-- POLICY REASONING (replaces consensus note) -->
    {_fmt_policy_reasoning(policy_decision or briefing.get("_policy_decision"))}
    {"" if not cons_note else _card("⚖️ Consensus Note", cons_note.replace(chr(10), "<br>"), "#7c3aed")}

    <!-- GEO-POLITICAL NEWS -->
    <div style="margin:16px 0;background:#eff6ff;border-radius:6px;
                border:1px solid #bfdbfe;border-left:4px solid #2563eb;padding:16px 20px;">
      <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:#2563eb;margin-bottom:10px;">
        🌍 Geo-Political Context &nbsp;<span style="font-weight:400;color:#6b7280;">(headlines at time of analysis)</span>
      </div>
      {news_html}
    </div>

    <!-- TRAFFIC DATA TABLE -->
    {"" if not csv_snapshot else f'''
    <div style="margin:16px 0;">
      <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:#374151;margin-bottom:8px;">
        📊 Recent Traffic Data (last 12 intervals)
      </div>
      <div style="border-radius:6px;overflow:hidden;border:1px solid #e5e7eb;">
        {table_html}
      </div>
    </div>'''}

    <!-- FULL JSON (collapsible) -->
    <details style="margin:16px 0;">
      <summary style="cursor:pointer;font-size:11px;font-weight:600;color:#6b7280;
                      text-transform:uppercase;letter-spacing:0.06em;padding:8px 0;">
        🔎 Full JSON Result
      </summary>
      <pre style="background:#1e293b;color:#e2e8f0;border-radius:6px;
                  padding:16px;font-size:11px;line-height:1.5;overflow-x:auto;
                  white-space:pre-wrap;word-break:break-all;margin:8px 0 0;">
{json_str}</pre>
    </details>

    <!-- FOOTER -->
    <div style="margin-top:20px;padding-top:16px;border-top:1px solid #e5e7eb;
                font-size:11px;color:#9ca3af;text-align:center;">
      Sent by <strong>marine-traffic-monitor</strong> · Audit log: analyst_audit.jsonl
    </div>

  </div>
</div>
</body>
</html>"""

    return html


def _build_html_digest(csv_snapshot: str = "", reported_today: int = 0) -> str:
    """Build a rich HTML email body for the daily digest."""
    ts       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
    table_html = _csv_to_html_table(csv_snapshot)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<div style="max-width:640px;margin:24px auto;background:#f3f4f6;">

  <!-- HEADER BANNER -->
  <div style="background:#1e3a5f;border-radius:8px 8px 0 0;padding:28px 28px 20px;">
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:rgba(255,255,255,0.6);">
      Hormuz Transit Corridor · Maritime OSINT
    </div>
    <div style="font-size:26px;font-weight:800;color:#ffffff;margin:8px 0 4px;">
      📅 Daily Digest
    </div>
    <div style="font-size:13px;color:rgba(255,255,255,0.7);">{date_str}</div>
    <div style="font-size:11px;color:rgba(255,255,255,0.5);margin-top:8px;">Generated {ts}</div>
  </div>

  <!-- BODY -->
  <div style="background:#f3f4f6;padding:20px 24px 28px;">

    <!-- SUMMARY STAT -->
    <div style="text-align:center;margin:12px 0 20px;">
      <div style="display:inline-block;background:#ffffff;border-radius:8px;
                  border:1px solid #e5e7eb;padding:20px 36px;">
        <div style="font-size:11px;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.06em;color:#6b7280;margin-bottom:6px;">
          Total Ship Detections Today
        </div>
        <div style="font-size:42px;font-weight:800;
                    color:{'#dc2626' if reported_today > 5 else '#d97706' if reported_today > 0 else '#16a34a'};">
          {reported_today}
        </div>
        <div style="font-size:11px;color:#9ca3af;margin-top:4px;">vessel(s) in zone</div>
      </div>
    </div>

    <!-- TRAFFIC TABLE -->
    <div style="margin:16px 0;">
      <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:#374151;margin-bottom:8px;">
        📊 Last 24h Traffic Data
      </div>
      <div style="border-radius:6px;overflow:hidden;border:1px solid #e5e7eb;">
        {table_html}
      </div>
    </div>

    <!-- FOOTER -->
    <div style="margin-top:20px;padding-top:16px;border-top:1px solid #e5e7eb;
                font-size:11px;color:#9ca3af;text-align:center;">
      Sent by <strong>marine-traffic-monitor</strong> (no LLM — digest only) ·
      Full alert log: analyst_audit.jsonl
    </div>

  </div>
</div>
</body>
</html>"""

    return html


# ------------------------------------------------------------------
# Plain-text email body builder (fallback for non-HTML clients)
# ------------------------------------------------------------------

def _build_email_body(briefing: dict, model_a: str, model_b: str,
                      reported_count: int, csv_snapshot: str = "",
                      news: str = "") -> str:
    """Build a plain-text email body from the briefing dict (fallback)."""
    level       = briefing.get("threat_level", "UNKNOWN")
    cls_        = briefing.get("event_classification", "N/A")
    is_tp       = briefing.get("is_true_positive")
    consensus   = briefing.get("consensus")
    cons_note   = briefing.get("consensus_note", "")
    brief_text  = briefing.get("analyst_briefing", "")
    verify_note = briefing.get("verification_note", "")
    ts          = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    emoji       = _LEVEL_EMOJI.get(level, "⚠️")

    models_str = f"{model_a} + {model_b}" if model_b else model_a

    sep  = "━" * 42
    sep2 = "─" * 42

    lines = [
        sep,
        " MARITIME ALERT — HORMUZ TRANSIT CORRIDOR",
        sep,
        f"Timestamp      : {ts}",
        f"Threat Level   : {emoji}  {level}",
        f"Classification : {cls_}",
        f"Is True Pos.   : {is_tp}",
        f"Consensus      : {consensus}",
        f"Ships Reported : {reported_count}",
        f"Models Used    : {models_str}",
        sep,
        "",
        "VERIFICATION NOTE:",
        verify_note,
        "",
        "ANALYST BRIEFING:",
        brief_text,
        "",
    ]

    if cons_note:
        lines += ["CONSENSUS NOTE:", cons_note, ""]

    # Geo-political news context (headlines the LLM saw when making this decision)
    news_body = news.strip() if news.strip() else "No Hormuz/Iran maritime news in the last 24h."
    lines += [
        sep2,
        "GEO-POLITICAL CONTEXT (headlines at time of analysis):",
        news_body,
        "",
    ]

    if csv_snapshot:
        lines += [
            "RECENT TRAFFIC DATA (last 12 intervals):",
            csv_snapshot,
            "",
        ]

    lines += [
        "FULL JSON RESULT:",
        json.dumps(briefing, indent=2, default=str),
        "",
        sep,
        "Sent by marine-traffic-monitor",
        "Audit log: analyst_audit.jsonl",
    ]

    return "\n".join(lines)


# ------------------------------------------------------------------
# Public API: alert email
# ------------------------------------------------------------------

def send_alert(
    briefing:        dict,
    model_a:         str  = "",
    model_b:         str  = "",
    reported_count:  int  = 0,
    image_paths:     list = None,   # list of image file paths to attach
    csv_snapshot:    str  = "",     # recent CSV rows as plain text for email body
    alert_cooldown:  int  = 5,      # minutes between alert emails; 0 = no cooldown
    news:            str  = "",     # geo-political headlines the LLM saw (from news_fetcher)
    policy_decision: dict = None,   # full policy engine decision dict (for email evidence sections)
) -> bool:
    """
    Send a Gmail SMTP alert for significant threat levels.

    Emails are sent as multipart/alternative (plain-text fallback + rich HTML).

    Args:
        briefing:       The analyst briefing dict (output of run_consensus_check).
        model_a:        Primary model key (for display in subject / body).
        model_b:        Secondary model key (empty string for single-model calls).
        reported_count: Number of ships OpenCV reported (for context in email).
        image_paths:    Optional list of image paths to attach
                        (e.g. [raw_map.png, detected_ships.png]).
        csv_snapshot:   Recent CSV rows as a plain-text string for the email body.
        alert_cooldown: Minimum minutes between alert emails (default 5; 0 = disabled).
        news:           Recent geo-political headlines (from news_fetcher) injected
                        into the email body for context.

    Returns:
        True if email was sent, False otherwise (never raises).
    """
    level = briefing.get("threat_level", "")
    if level not in ALERT_LEVELS:
        return False  # Silent — don't spam on NONE or LOW

    # Cooldown gate — prevent alert spam during rapid oscillation
    if alert_cooldown > 0:
        last_ts = _read_last_alert_ts()
        if last_ts:
            elapsed_min = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60
            if elapsed_min < alert_cooldown:
                remaining = int(alert_cooldown - elapsed_min) + 1
                print(f"[NOTIFIER] Cooldown active — ~{remaining} min until next alert allowed. Skipping.")
                return False

    # Load credentials
    sender       = os.getenv("GMAIL_SENDER", "").strip()
    app_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    recipient    = os.getenv("GMAIL_RECIPIENT", "").strip()

    if not sender or not app_password or not recipient:
        print(
            "[NOTIFIER] Gmail credentials not set. Add to .config:\n"
            "  GMAIL_SENDER=yourname@gmail.com\n"
            "  GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx\n"
            "  GMAIL_RECIPIENT=yourname@gmail.com"
        )
        return False

    try:
        cls_    = briefing.get("event_classification", "Alert")
        emoji   = _LEVEL_EMOJI.get(level, "⚠️")
        models  = f"{model_a} + {model_b}" if model_b else model_a
        subject = f"[HORMUZ ALERT] {emoji} {level} — {cls_} | {models}"

        # Outer container: mixed (holds alternative block + attachments)
        msg = MIMEMultipart("mixed")
        msg["From"]    = sender
        msg["To"]      = recipient
        msg["Subject"] = subject

        # Inner alternative block: plain text + HTML
        alt = MIMEMultipart("alternative")
        plain_body = _build_email_body(briefing, model_a, model_b, reported_count,
                                       csv_snapshot=csv_snapshot, news=news)
        html_body  = _build_html_body(briefing, model_a, model_b, reported_count,
                                      csv_snapshot=csv_snapshot, news=news,
                                      policy_decision=policy_decision)
        alt.attach(MIMEText(plain_body, "plain", "utf-8"))
        alt.attach(MIMEText(html_body,  "html",  "utf-8"))
        msg.attach(alt)

        # Attach each image directly onto the outer msg
        for img_path in (image_paths or []):
            if not img_path:
                continue
            abs_img = img_path if os.path.isabs(img_path) \
                      else os.path.join(_HERE, img_path)
            if not os.path.exists(abs_img):
                print(f"[NOTIFIER] Image not found, skipping: {abs_img}")
                continue
            with open(abs_img, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            fname = os.path.basename(abs_img)
            part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
            msg.attach(part)
            print(f"[NOTIFIER] Attaching: {fname}")

        # Send via Gmail SMTP SSL (port 465)
        print(f"[NOTIFIER] Sending alert: {level} → {recipient}")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_password)
            server.sendmail(sender, recipient, msg.as_string())

        _write_last_alert_ts()   # record successful send for cooldown tracking
        print(f"[NOTIFIER] ✓ Alert sent successfully.")
        return True

    except smtplib.SMTPAuthenticationError:
        print(
            "[NOTIFIER] ✗ Authentication failed. Check your App Password in .config.\n"
            "  Reminder: use an App Password, NOT your regular Gmail password.\n"
            "  Get one at: myaccount.google.com → Security → App Passwords"
        )
        return False
    except smtplib.SMTPException as e:
        print(f"[NOTIFIER] ✗ SMTP error: {e}")
        return False
    except Exception as e:
        print(f"[NOTIFIER] ✗ Unexpected error: {e}")
        return False


# ------------------------------------------------------------------
# Public API: daily digest email (no LLM required)
# ------------------------------------------------------------------

def send_digest(csv_snapshot: str = "", reported_today: int = 0) -> bool:
    """
    Send a daily digest email with the last 24h of CSV traffic data.
    No threat analysis — just confirms the system is alive and shows the day's data.

    Args:
        csv_snapshot:   Last ~96 CSV rows (24h at 15-min intervals) as plain text.
        reported_today: Total ship detections recorded today (for the summary line).

    Returns:
        True if sent, False otherwise (never raises).
    """
    sender       = os.getenv("GMAIL_SENDER", "").strip()
    app_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    recipient    = os.getenv("GMAIL_RECIPIENT", "").strip()

    if not sender or not app_password or not recipient:
        print("[NOTIFIER] Gmail credentials not set — digest skipped.")
        return False

    try:
        ts       = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_str = ts[:10]
        subject  = f"[HORMUZ DIGEST] Daily Summary — {date_str}"

        # Outer container
        msg = MIMEMultipart("mixed")
        msg["From"]    = sender
        msg["To"]      = recipient
        msg["Subject"] = subject

        # Plain-text fallback
        sep = "━" * 42
        plain_body = "\n".join([
            sep,
            " HORMUZ DAILY DIGEST",
            sep,
            f"Generated             : {ts}",
            f"Ship detections today : {reported_today}",
            sep,
            "",
            "LAST 24h TRAFFIC DATA:",
            csv_snapshot if csv_snapshot else "No data recorded today.",
            "",
            sep,
            "Sent by marine-traffic-monitor  (no LLM — digest only)",
            "Full alert log: analyst_audit.jsonl",
        ])

        # HTML version
        html_body = _build_html_digest(csv_snapshot=csv_snapshot, reported_today=reported_today)

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_body, "plain", "utf-8"))
        alt.attach(MIMEText(html_body,  "html",  "utf-8"))
        msg.attach(alt)

        print(f"[NOTIFIER] Sending daily digest → {recipient}")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_password)
            server.sendmail(sender, recipient, msg.as_string())

        print(f"[NOTIFIER] ✓ Daily digest sent.")
        return True

    except smtplib.SMTPAuthenticationError:
        print("[NOTIFIER] ✗ Digest auth failed. Check App Password in .config.")
        return False
    except Exception as e:
        print(f"[NOTIFIER] ✗ Digest error: {e}")
        return False


# ------------------------------------------------------------------
# Standalone test entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the Gmail notifier")
    parser.add_argument("--test", action="store_true",
                        help="Send a test alert email (default level: CRITICAL)")
    parser.add_argument("--test-digest", action="store_true",
                        help="Send a test daily digest email")
    parser.add_argument("--level", default="CRITICAL",
                        choices=["CRITICAL", "ELEVATED", "ESCALATED", "DISPUTED", "REVIEW"],
                        help="Threat level for the test alert (default: CRITICAL)")
    args = parser.parse_args()

    if not args.test and not args.test_digest:
        print("Usage: python notifier.py --test [--level CRITICAL|ELEVATED|DISPUTED]")
        print("       python notifier.py --test-digest")
        sys.exit(0)

    if args.test_digest:
        dummy_csv = (
            "Timestamp,Detected_Ships,Status_Note\n"
            "2026-03-15_07-30-00,0,Active Blockade / Clear Zone\n"
            "2026-03-15_07-45-00,0,Active Blockade / Clear Zone\n"
            "2026-03-15_08-00-00,1,WARNING: 1 dark/escorted vessel(s) in choke point.\n"
            "2026-03-15_08-15-00,3,WARNING: 3 dark/escorted vessel(s) in choke point.\n"
            "2026-03-15_08-30-00,2,WARNING: 2 dark/escorted vessel(s) in choke point.\n"
            "2026-03-15_08-45-00,0,Active Blockade / Clear Zone"
        )
        print("[NOTIFIER TEST] Sending dummy daily digest...")
        success = send_digest(csv_snapshot=dummy_csv, reported_today=6)
        sys.exit(0 if success else 1)

    _is_disputed = args.level in ("DISPUTED", "REVIEW")
    dummy_briefing = {
        "is_true_positive": True if args.level in ("CRITICAL", "ESCALATED") else None if _is_disputed else False,
        "verification_note": (
            "TEST — Bounding boxes show 3 distinct triangular vessel icons inside the red zone polygon. "
            f"[{args.level}] model_a conf=0.72 abstain=False | model_b conf=0.65 abstain=False | avg_conf=0.69"
        ),
        "event_classification": (
            "TEST — Escorted Convoy" if args.level in ("CRITICAL", "ESCALATED")
            else "TEST — Low Confidence Review Required" if args.level == "REVIEW"
            else "TEST — Cross-Model Disagreement"
        ),
        "threat_level": args.level,
        "alert_level":  args.level,
        "analyst_briefing": (
            "TEST EMAIL — This is a notifier connectivity check. "
            "Sudden spike of 3 vessels detected after 6 hours of zero traffic. "
            "Formation suggests coordinated transit. Temporal pattern consistent with blockade breach."
        ),
        "consensus": not _is_disputed,
        "consensus_note": (
            "TEST — Both models agree on threat level." if not _is_disputed
            else "TEST — Scout says CRITICAL; 70B says NORMAL. Gap ≥ 2 urgency levels → DISPUTED."
        ),
        "current_macro_state": "SURGE",
        "proposed_state_transition": "SURGE -> BLOCKADE_ACTIVE" if args.level in ("CRITICAL", "ESCALATED") else "NONE",
        "applied_macro_state": "BLOCKADE_ACTIVE" if args.level in ("CRITICAL", "ESCALATED") else "SURGE",
        "policy_reasoning": [f"p4_both_escalated final={args.level} conf=0.69"],
        "avg_confidence": 0.69,
        "human_review_needed": _is_disputed,
        # Dummy evidence for new email sections
        "evidence_a": {
            "model_role": "visual_analyst",
            "direct_observations": [
                "3 green bounding boxes inside red polygon zone",
                "Triangular vessel icons pointing NE — transit heading",
                "Formation spacing consistent with escorted convoy"
            ],
            "historical_context": [
                "Spike from 0 to 3 ships in 1 interval (15 min)",
                "Prior 6 intervals all showed 0 ships"
            ],
            "news_context": ["Iran warns tankers — may explain sudden transit surge"],
            "hypotheses": [
                {"statement": "Escorted blockade breach convoy", "confidence": 0.72},
                {"statement": "Routine merchant transit — icons misidentified", "confidence": 0.21},
                {"statement": "CV false positives (map artefacts)", "confidence": 0.07},
            ],
            "risk_signals": [
                "Sudden count spike after sustained zero baseline",
                "Convoy formation geometry visible"
            ],
            "uncertainties": [
                "Cannot confirm AIS status from screenshot alone",
                "Icon resolution low — vessel type ambiguous"
            ],
            "recommended_state": "ESCALATED" if args.level in ("CRITICAL", "ESCALATED") else "WATCH",
            "recommended_action": "escalate" if args.level in ("CRITICAL", "ESCALATED") else "watch_closely",
            "abstain": False,
            "abstain_reason": None,
            "overall_confidence": 0.72,
        },
        "evidence_b": {
            "model_role": "context_analyst",
            "direct_observations": ["No image available — visual QA delegated to visual_analyst model."],
            "historical_context": [
                "CSV shows 0→3 spike in 1 interval — statistically unusual",
                "No gradual build-up: discrete step change suggests coordinated event",
                "Prior 4 hours showed zero activity"
            ],
            "news_context": [
                "Iran warning to tankers published 4 hours before spike — plausible causal link",
                "US carrier group arrival may be triggering compensatory transit"
            ],
            "hypotheses": [
                {"statement": "Geopolitical event driving escorted transit", "confidence": 0.65},
                {"statement": "Scheduled merchant convoy — timing coincidence", "confidence": 0.28},
            ],
            "risk_signals": [
                "Count spike correlates temporally with recent Iran warning",
                "No gradual increase — suggests pre-planned movement"
            ],
            "uncertainties": [
                "CSV count only — no vessel identity or AIS status available",
                "Short history (< 12 intervals) limits baseline comparison"
            ],
            "recommended_state": "ESCALATED" if args.level in ("CRITICAL", "ESCALATED") else "WATCH",
            "recommended_action": "escalate" if args.level in ("CRITICAL", "ESCALATED") else "watch_closely",
            "abstain": False,
            "abstain_reason": None,
            "overall_confidence": 0.65,
        },
    }

    dummy_policy_decision = {
        "alert_level":          args.level,
        "proposed_transition":  "SURGE -> BLOCKADE_ACTIVE" if args.level in ("CRITICAL", "ESCALATED") else "NONE",
        "email_decision":       True,
        "human_review_needed":  _is_disputed,
        "reasoning":            [f"p4_both_escalated final={args.level} conf=0.69", "avg_conf=0.69>=0.55"],
        "avg_confidence":       0.69,
        "ships_in_zone":        3,
        "applied_state":        "BLOCKADE_ACTIVE" if args.level in ("CRITICAL", "ESCALATED") else "SURGE",
    }

    dummy_csv = (
        "Timestamp,Detected_Ships,Status_Note\n"
        "2026-03-15_07-30-00,0,Active Blockade / Clear Zone\n"
        "2026-03-15_07-45-00,0,Active Blockade / Clear Zone\n"
        "2026-03-15_08-00-00,1,WARNING: 1 dark/escorted vessel(s) in choke point.\n"
        "2026-03-15_08-15-00,3,WARNING: 3 dark/escorted vessel(s) in choke point."
    )

    dummy_news = (
        "1. [Mon, 16 Mar 2026 14:22:00 GMT] Iran warns tankers after US sanctions - Reuters\n"
        "2. [Mon, 16 Mar 2026 10:08:00 GMT] US Navy carrier group enters Persian Gulf - AP\n"
        "3. [Mon, 16 Mar 2026 07:45:00 GMT] Oman reports vessel harassment near strait - BBC"
    )

    print(f"[NOTIFIER TEST] Sending dummy {args.level} alert (cooldown bypassed)...")
    success = send_alert(
        briefing=dummy_briefing,
        model_a="llama_4_scout",
        model_b="llama_3_3_70b",
        reported_count=3,
        csv_snapshot=dummy_csv,
        alert_cooldown=0,   # bypass cooldown for testing
        news=dummy_news,
        policy_decision=dummy_policy_decision,
    )
    sys.exit(0 if success else 1)
