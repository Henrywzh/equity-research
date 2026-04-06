"""Market data enrichment for daily-macro analysis.

Provides two data sources:
- check_portfolio_impact(ticker): reads from daily-market's SQLite database
- get_live_market_data(ticker): fetches live via yfinance

Plus helpers to build LLM-friendly context strings and report-ready structures.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _resolve_market_db_path() -> Path:
    """Locate daily-market/data/market.sqlite relative to the repo root."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "daily-market" / "data" / "market.sqlite"
        if candidate.exists():
            return candidate
    return Path.cwd().parent / "daily-market" / "data" / "market.sqlite"


def check_portfolio_impact(ticker: str) -> dict[str, Any] | None:
    """Read latest snapshot for a tracked ticker from market.sqlite.

    Returns None if the ticker is not found or the database is unavailable.
    """
    db_path = _resolve_market_db_path()
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """SELECT ps.*, fr.started_at AS run_started_at
               FROM price_snapshots ps
               JOIN fetch_runs fr ON fr.id = ps.run_id
               WHERE ps.ticker = ?
               ORDER BY ps.fetched_at DESC
               LIMIT 1""",
            (ticker,),
        ).fetchone()
        return dict(row) if row else None
    except Exception:
        LOGGER.debug("Failed to read market.sqlite for ticker %s", ticker, exc_info=True)
        return None
    finally:
        conn.close()


def get_live_market_data(ticker: str) -> dict[str, Any] | None:
    """Fetch a live snapshot for an arbitrary ticker via yfinance.

    Returns None on failure.
    """
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=True)
        if hist.empty:
            return None
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) >= 2 else None
        price = float(latest["Close"])
        prev_close = float(prev["Close"]) if prev is not None else None
        pct = ((price - prev_close) / prev_close * 100) if prev_close else None
        return {
            "ticker": ticker,
            "price": round(price, 4),
            "prev_close": round(prev_close, 4) if prev_close else None,
            "pct_change": round(pct, 2) if pct is not None else None,
            "data_timestamp": str(latest.name.date()),
        }
    except Exception:
        LOGGER.debug("Failed to fetch live data for ticker %s", ticker, exc_info=True)
        return None


def fetch_market_snapshot_for_date(target_date: str) -> list[dict[str, Any]]:
    """Read all tracked tickers from market.sqlite for a given date.

    Falls back to the latest available data if the target date has no snapshots.
    """
    db_path = _resolve_market_db_path()
    if not db_path.exists():
        LOGGER.info("market.sqlite not found at %s; skipping market snapshot.", db_path)
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT ps.* FROM price_snapshots ps
               WHERE ps.data_timestamp = ?
               ORDER BY ps.asset_class, ps.ticker""",
            (target_date,),
        ).fetchall()
        if rows:
            return [dict(r) for r in rows]
        rows = conn.execute(
            """SELECT ps.* FROM price_snapshots ps
               INNER JOIN (
                   SELECT ticker, MAX(fetched_at) AS max_fetched
                   FROM price_snapshots GROUP BY ticker
               ) latest ON ps.ticker = latest.ticker
                       AND ps.fetched_at = latest.max_fetched
               ORDER BY ps.asset_class, ps.ticker""",
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        LOGGER.warning("Failed to read market snapshots from %s", db_path, exc_info=True)
        return []
    finally:
        conn.close()


def build_market_context_string(snapshots: list[dict[str, Any]]) -> str:
    """Format snapshots into a compact string suitable for LLM injection."""
    if not snapshots:
        return ""
    lines = ["Market snapshot (latest available):"]
    for s in snapshots:
        ticker = s.get("ticker", "?")
        price = s.get("price")
        pct = s.get("pct_change")
        ts = s.get("data_timestamp", "?")
        if price is None:
            continue
        pct_str = f" ({pct:+.2f}%)" if pct is not None else ""
        lines.append(f"  {ticker}: {price:.2f}{pct_str} [{ts}]")
    return "\n".join(lines)


def build_market_context_for_report(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format snapshots into a structured list for the report JSON."""
    return [
        {
            "ticker": s["ticker"],
            "asset_class": s.get("asset_class"),
            "price": s.get("price"),
            "prev_close": s.get("prev_close"),
            "pct_change": s.get("pct_change"),
            "data_timestamp": s.get("data_timestamp"),
        }
        for s in snapshots
        if s.get("price") is not None
    ]
