from __future__ import annotations

from typing import Any

from .models import PriceSnapshot

# -------------------------------------------------------------------
# Bilingual display labels per ticker
# -------------------------------------------------------------------
TICKER_LABELS: dict[str, dict[str, str]] = {
    # Indices
    "^HSI":      {"en": "Hang Seng Index",  "zh": "恒生指數"},
    "000001.SS": {"en": "SSE Composite",    "zh": "上海綜指"},
    "^SPX":      {"en": "S&P 500",          "zh": "標普500"},
    "^NDX":      {"en": "NASDAQ 100",       "zh": "納斯達克100"},
    "^DJI":      {"en": "Dow Jones",        "zh": "道瓊斯工業"},
    "^N225":     {"en": "Nikkei 225",       "zh": "日經225"},
    "^KS11":     {"en": "KOSPI",            "zh": "韓國綜指"},
    "^STOXX50E": {"en": "Euro Stoxx 50",    "zh": "歐洲斯托克50"},
    "^FTSE":     {"en": "FTSE 100",         "zh": "富時100"},
    # Commodities
    "GC=F":      {"en": "Gold",             "zh": "黃金"},
    "CL=F":      {"en": "WTI Crude Oil",    "zh": "WTI原油"},
    # FX
    "USDCNH=X":  {"en": "USD/CNH",          "zh": "美元/離岸人民幣"},
    "GBPHKD=X":  {"en": "GBP/HKD",         "zh": "英鎊/港元"},
    "USDJPY=X":  {"en": "USD/JPY",          "zh": "美元/日元"},
    "DX-Y.NYB":  {"en": "DXY (USD Index)",  "zh": "美元指數"},
    # Crypto
    "BTC-USD":   {"en": "Bitcoin",          "zh": "比特幣"},
}

_SECTION_LABELS: dict[str, dict[str, str]] = {
    "index":     {"en": "EQUITY INDICES",  "zh": "股票指數"},
    "commodity": {"en": "COMMODITIES",     "zh": "商品"},
    "fx":        {"en": "FX RATES",        "zh": "外匯"},
    "crypto":    {"en": "CRYPTO",          "zh": "加密貨幣"},
    "custom":    {"en": "CUSTOM SYMBOLS",  "zh": "自選標的"},
}

_SESSION_LABELS: dict[str, dict[str, str]] = {
    "morning": {"en": "Morning Brief", "zh": "早市簡報"},
    "evening": {"en": "Evening Brief", "zh": "晚市簡報"},
}

_ASSET_CLASS_ORDER = ["index", "commodity", "fx", "crypto", "custom"]


def format_summary(
    snapshots: list[PriceSnapshot],
    session: str,
    date_str: str,
    fetch_time_utc: str,
) -> dict[str, Any]:
    """Build a bilingual market summary from price snapshots.

    Returns a dict with keys:
      - plain: plain-text email body (str)
      - html:  HTML email body (str)
      - structured: machine-readable dict of sections/rows
      - session: morning | evening
      - date: YYYY-MM-DD
      - status: success | partial | failed

    NOTE: No LLM call is made here. AI commentary is a future placeholder.
    """
    sections = _group_by_asset_class(snapshots)
    success_count = sum(1 for s in snapshots if s.error is None)
    status = (
        "failed" if success_count == 0
        else "partial" if success_count < len(snapshots)
        else "success"
    )

    structured = {
        "session": session,
        "date": date_str,
        "fetch_time_utc": fetch_time_utc,
        "status": status,
        "ticker_count": len(snapshots),
        "success_count": success_count,
        "sections": [
            {
                "asset_class": ac,
                "label_en": _SECTION_LABELS.get(ac, {}).get("en", ac.upper()),
                "label_zh": _SECTION_LABELS.get(ac, {}).get("zh", ac),
                "rows": [
                    _snapshot_to_row(s)
                    for s in sorted(
                        snaps,
                        key=lambda x: str(x.data_timestamp or "0000-00-00"),
                        reverse=True,
                    )
                ],
            }
            for ac, snaps in sections
        ],
        # TODO: AI commentary goes here in a future version
        "ai_commentary": None,
    }

    return {
        "plain": _build_plain(structured, session, date_str),
        "html": _build_html(structured, session, date_str),
        "structured": structured,
        "session": session,
        "date": date_str,
        "status": status,
        "ticker_count": len(snapshots),
        "success_count": success_count,
    }


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _group_by_asset_class(
    snapshots: list[PriceSnapshot],
) -> list[tuple[str, list[PriceSnapshot]]]:
    buckets: dict[str, list[PriceSnapshot]] = {}
    for s in snapshots:
        buckets.setdefault(s.asset_class, []).append(s)
    result = []
    for ac in _ASSET_CLASS_ORDER:
        if ac in buckets:
            result.append((ac, buckets[ac]))
    for ac, snaps in buckets.items():
        if ac not in _ASSET_CLASS_ORDER:
            result.append((ac, snaps))
    return result


def _snapshot_to_row(s: PriceSnapshot) -> dict[str, Any]:
    labels = TICKER_LABELS.get(s.ticker, {"en": s.ticker, "zh": s.ticker})
    arrow, color_class = _arrow(s.pct_change)
    return {
        "ticker": s.ticker,
        "label_en": labels["en"],
        "label_zh": labels["zh"],
        "price": s.price,
        "price_fmt": _fmt_price(s.price, s.asset_class),
        "pct_change": s.pct_change,
        "pct_fmt": _fmt_pct(s.pct_change),
        "arrow": arrow,
        "color_class": color_class,
        "data_timestamp": s.data_timestamp,
        "error": s.error,
    }


def _arrow(pct: float | None) -> tuple[str, str]:
    if pct is None:
        return "—", "neutral"
    if pct > 0:
        return "▲", "up"
    if pct < 0:
        return "▼", "down"
    return "—", "neutral"


def _fmt_price(price: float | None, asset_class: str) -> str:
    if price is None:
        return "N/A"
    if asset_class == "fx":
        return f"{price:,.4f}"
    if asset_class == "crypto":
        return f"{price:,.2f}"
    if price >= 10_000:
        return f"{price:,.0f}"
    if price >= 100:
        return f"{price:,.2f}"
    return f"{price:,.4f}"


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "N/A"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


# -------------------------------------------------------------------
# Plain-text builder
# -------------------------------------------------------------------

def _build_plain(structured: dict[str, Any], session: str, date_str: str) -> str:
    sess_labels = _SESSION_LABELS.get(session, {"en": session.title(), "zh": session})
    header = (
        f"DAILY MARKET — {sess_labels['en']} 每日市場{sess_labels['zh']}\n"
        f"{date_str}  |  Fetched at {structured['fetch_time_utc']} UTC\n"
        f"Status: {structured['status'].upper()}  "
        f"({structured['success_count']}/{structured['ticker_count']} symbols)\n"
    )
    lines = [header]

    for section in structured["sections"]:
        ac = section["asset_class"]
        en = section["label_en"]
        zh = section["label_zh"]
        lines.append(f"\n=== {en} {zh} ===")
        for row in section["rows"]:
            if row["error"]:
                lines.append(f"  {row['label_en']} ({row['ticker']})  ERROR: {row['error']}")
            else:
                name = f"{row['label_en']} {row['label_zh']}"
                lines.append(
                    f"  {name:<36}  {row['ticker']:<12}  "
                    f"{row['price_fmt']:>12}  {row['arrow']} {row['pct_fmt']}"
                )
        if ac == "index":
            # TODO: AI macro commentary placeholder
            pass

    lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------
# HTML builder
# -------------------------------------------------------------------

_COLOR = {
    "up":      "#16a34a",
    "down":    "#dc2626",
    "neutral": "#6b7280",
}

_BG = {
    "up":      "#f0fdf4",
    "down":    "#fef2f2",
    "neutral": "#f9fafb",
}


def _build_html(structured: dict[str, Any], session: str, date_str: str) -> str:
    sess_labels = _SESSION_LABELS.get(session, {"en": session.title(), "zh": session})
    section_blocks = "".join(_build_section_html(s) for s in structured["sections"])

    return f"""
    <html>
      <body style="margin:0;padding:24px;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
        <div style="max-width:760px;margin:0 auto;background:#ffffff;border-radius:16px;padding:28px 28px 16px;border:1px solid #e5e7eb;">
          <div style="font-size:26px;font-weight:800;color:#111827;">
            Daily Market
            <span style="font-size:14px;font-weight:500;color:#6b7280;margin-left:10px;">
              {_esc(sess_labels['en'])} &nbsp;每日市場{_esc(sess_labels['zh'])}
            </span>
          </div>
          <p style="margin:6px 0 2px;color:#4b5563;font-size:14px;">
            {_esc(date_str)} &nbsp;·&nbsp; Fetched {_esc(structured['fetch_time_utc'])} UTC
          </p>
          <p style="margin:0 0 20px;color:#4b5563;font-size:13px;">
            {structured['success_count']}/{structured['ticker_count']} symbols fetched successfully
          </p>
          {section_blocks}
          <p style="margin:20px 0 0;color:#9ca3af;font-size:11px;">
            Data via yfinance — latest available (may be prior session close for closed markets).
            <!-- TODO: AI commentary will be inserted here in a future version -->
          </p>
        </div>
      </body>
    </html>
    """


def _build_section_html(section: dict[str, Any]) -> str:
    rows_html = "".join(_build_row_html(row) for row in section["rows"])
    return f"""
    <div style="margin:0 0 20px;">
      <div style="font-size:15px;font-weight:700;color:#374151;padding:8px 0 6px;
                  border-bottom:2px solid #e5e7eb;margin-bottom:4px;">
        {_esc(section['label_en'])}
        <span style="font-weight:500;color:#9ca3af;margin-left:8px;">{_esc(section['label_zh'])}</span>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
          <tr style="color:#6b7280;font-size:11px;text-transform:uppercase;">
            <th style="text-align:left;padding:4px 8px;font-weight:600;">Asset 資產</th>
            <th style="text-align:right;padding:4px 8px;font-weight:600;">Price 價格</th>
            <th style="text-align:right;padding:4px 8px;font-weight:600;">Change 變動</th>
            <th style="text-align:right;padding:4px 8px;font-weight:600;">As of 數據日期</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """


def _build_row_html(row: dict[str, Any]) -> str:
    if row["error"]:
        return (
            f"<tr style='background:#fefce8;'>"
            f"<td style='padding:6px 8px;'><strong>{_esc(row['label_en'])}</strong> "
            f"<span style='color:#9ca3af;font-size:11px;'>{_esc(row['ticker'])}</span></td>"
            f"<td colspan='3' style='padding:6px 8px;color:#b45309;font-size:12px;'>"
            f"Error: {_esc(row['error'])}</td></tr>"
        )

    cc = row["color_class"]
    color = _COLOR[cc]
    bg = _BG[cc] if cc != "neutral" else "#ffffff"
    pct_html = (
        f"<span style='color:{color};font-weight:600;'>"
        f"{_esc(row['arrow'])} {_esc(row['pct_fmt'])}</span>"
    )
    ts = _esc(row["data_timestamp"] or "")
    return (
        f"<tr style='background:{bg};border-bottom:1px solid #f3f4f6;'>"
        f"<td style='padding:7px 8px;'>"
        f"<strong style='color:#111827;'>{_esc(row['label_en'])}</strong> "
        f"<span style='color:#6b7280;font-size:11px;'>{_esc(row['label_zh'])}</span><br>"
        f"<span style='color:#9ca3af;font-size:11px;'>{_esc(row['ticker'])}</span></td>"
        f"<td style='text-align:right;padding:7px 8px;font-weight:600;color:#111827;'>"
        f"{_esc(row['price_fmt'])}</td>"
        f"<td style='text-align:right;padding:7px 8px;'>{pct_html}</td>"
        f"<td style='text-align:right;padding:7px 8px;color:#9ca3af;font-size:11px;'>{ts}</td>"
        f"</tr>"
    )


def _esc(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
