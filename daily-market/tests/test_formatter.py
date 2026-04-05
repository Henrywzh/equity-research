from __future__ import annotations

from daily_market.formatter import (
    TICKER_LABELS,
    _arrow,
    _fmt_pct,
    _fmt_price,
    format_summary,
)
from daily_market.models import PriceSnapshot


def _snap(ticker: str, pct: float | None = 1.5, error: str | None = None) -> PriceSnapshot:
    return PriceSnapshot(
        ticker=ticker,
        asset_class="index",
        fetched_at="2026-04-05T07:00:00+00:00",
        price=20000.0 if not error else None,
        prev_close=19703.0 if not error else None,
        pct_change=pct if not error else None,
        open=19800.0,
        high=20100.0,
        low=19750.0,
        volume=1_200_000.0,
        data_timestamp="2026-04-04",
        error=error,
    )


def test_all_default_tickers_have_labels():
    from daily_market.watchlist import load_watchlist
    import json, tempfile
    from pathlib import Path

    default_tickers = [
        "^HSI", "000001.SS", "^SPX", "^NDX", "^DJI", "^N225", "^KS11", "^STOXX50E", "^FTSE",
        "GC=F", "CL=F",
        "USDCNH=X", "GBPHKD=X", "USDJPY=X", "DX-Y.NYB",
        "BTC-USD",
    ]
    for t in default_tickers:
        assert t in TICKER_LABELS, f"Missing label for {t}"
        assert TICKER_LABELS[t].get("en")
        assert TICKER_LABELS[t].get("zh")


def test_arrow():
    assert _arrow(1.5) == ("▲", "up")
    assert _arrow(-0.5) == ("▼", "down")
    assert _arrow(0.0) == ("—", "neutral")
    assert _arrow(None) == ("—", "neutral")


def test_fmt_pct():
    assert _fmt_pct(1.5) == "+1.50%"
    assert _fmt_pct(-2.0) == "-2.00%"
    assert _fmt_pct(None) == "N/A"


def test_fmt_price_fx():
    assert _fmt_price(7.2345, "fx") == "7.2345"


def test_format_summary_structure():
    snaps = [_snap("^HSI", pct=1.5), _snap("^SPX", pct=-0.5)]
    result = format_summary(snaps, session="morning", date_str="2026-04-05",
                            fetch_time_utc="2026-04-05T07:00:00+00:00")
    assert "plain" in result
    assert "html" in result
    assert "structured" in result
    assert result["session"] == "morning"
    assert result["status"] == "success"


def test_format_summary_partial_status():
    snaps = [_snap("^HSI", pct=1.5), _snap("^SPX", error="fetch failed")]
    result = format_summary(snaps, session="morning", date_str="2026-04-05",
                            fetch_time_utc="2026-04-05T07:00:00+00:00")
    assert result["status"] == "partial"


def test_format_summary_failed_status():
    snaps = [
        _snap("^HSI", error="err1"),
        _snap("^SPX", error="err2"),
    ]
    result = format_summary(snaps, session="morning", date_str="2026-04-05",
                            fetch_time_utc="2026-04-05T07:00:00+00:00")
    assert result["status"] == "failed"


def test_html_contains_ticker():
    snaps = [_snap("^HSI", pct=1.5)]
    result = format_summary(snaps, session="morning", date_str="2026-04-05",
                            fetch_time_utc="2026-04-05T07:00:00+00:00")
    assert "^HSI" in result["html"]
    assert "恒生指數" in result["html"]


def test_ai_commentary_placeholder():
    """AI commentary must be None until implemented."""
    snaps = [_snap("^HSI")]
    result = format_summary(snaps, session="morning", date_str="2026-04-05",
                            fetch_time_utc="2026-04-05T07:00:00+00:00")
    assert result["structured"]["ai_commentary"] is None
