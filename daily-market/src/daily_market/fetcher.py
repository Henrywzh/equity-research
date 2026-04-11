from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from .models import PriceSnapshot

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError("yfinance is required: pip install yfinance") from exc


def fetch_snapshots(
    tickers: list[str],
    asset_class_map: dict[str, str],
) -> list[PriceSnapshot]:
    """Fetch latest price snapshots for all tickers.

    Continues on per-ticker failure — partial results are returned.
    Data is "latest available" (may be T-1 for closed markets).
    """
    fetched_at = datetime.now(timezone.utc).isoformat()
    snapshots: list[PriceSnapshot] = []
    for ticker in tickers:
        asset_class = asset_class_map.get(ticker, "custom")
        snapshot = _fetch_one(ticker, asset_class, fetched_at)
        snapshots.append(snapshot)
    return snapshots


def _fetch_one(ticker: str, asset_class: str, fetched_at: str) -> PriceSnapshot:
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=True)
    except Exception as exc:
        return PriceSnapshot(
            ticker=ticker,
            asset_class=asset_class,
            fetched_at=fetched_at,
            price=None,
            prev_close=None,
            pct_change=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            data_timestamp=None,
            error=f"yfinance fetch failed: {exc}",
        )

    if hist is None or len(hist) == 0:
        return PriceSnapshot(
            ticker=ticker,
            asset_class=asset_class,
            fetched_at=fetched_at,
            price=None,
            prev_close=None,
            pct_change=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            data_timestamp=None,
            error="No data returned by yfinance",
        )

    price = _safe_float(hist["Close"].iloc[-1])
    data_timestamp = str(hist.index[-1].date())

    if len(hist) >= 2:
        prev_close = _safe_float(hist["Close"].iloc[-2])
    else:
        prev_close = _safe_float(hist["Open"].iloc[-1])

    if price is not None and prev_close is not None and prev_close != 0:
        pct_change = (price - prev_close) / prev_close * 100
    else:
        pct_change = None

    open_ = _safe_float(hist["Open"].iloc[-1])
    high = _safe_float(hist["High"].iloc[-1])
    low = _safe_float(hist["Low"].iloc[-1])

    raw_volume = _safe_float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else None
    volume = raw_volume if (raw_volume is not None and raw_volume > 0) else None

    return PriceSnapshot(
        ticker=ticker,
        asset_class=asset_class,
        fetched_at=fetched_at,
        price=price,
        prev_close=prev_close,
        pct_change=pct_change,
        open=open_,
        high=high,
        low=low,
        volume=volume,
        data_timestamp=data_timestamp,
        error=None,
    )


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None
