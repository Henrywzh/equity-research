from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from daily_market.fetcher import fetch_snapshots, _safe_float


def _mock_history(closes: list[float]) -> MagicMock:
    """Build a mock DataFrame mimicking yf.Ticker.history() output."""
    import pandas as pd
    idx = pd.date_range("2026-04-01", periods=len(closes), freq="D")
    df = pd.DataFrame(
        {
            "Open":   closes,
            "High":   [c * 1.01 for c in closes],
            "Low":    [c * 0.99 for c in closes],
            "Close":  closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=idx,
    )
    return df


def test_fetch_success():
    mock_df = _mock_history([100.0, 105.0])
    with patch("daily_market.fetcher.yf.Ticker") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf.return_value = mock_ticker

        snapshots = fetch_snapshots(["^HSI"], {"^HSI": "index"})

    assert len(snapshots) == 1
    s = snapshots[0]
    assert s.ticker == "^HSI"
    assert s.asset_class == "index"
    assert s.price == 105.0
    assert s.prev_close == 100.0
    assert abs(s.pct_change - 5.0) < 0.001
    assert s.error is None


def test_fetch_empty_history():
    empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    with patch("daily_market.fetcher.yf.Ticker") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = empty_df
        mock_yf.return_value = mock_ticker

        snapshots = fetch_snapshots(["^HSI"], {"^HSI": "index"})

    assert snapshots[0].error == "No data returned by yfinance"
    assert snapshots[0].price is None


def test_fetch_exception():
    with patch("daily_market.fetcher.yf.Ticker") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = RuntimeError("network error")
        mock_yf.return_value = mock_ticker

        snapshots = fetch_snapshots(["^HSI"], {"^HSI": "index"})

    assert snapshots[0].error is not None
    assert "network error" in snapshots[0].error


def test_fetch_partial_continues():
    """A single ticker failure should not stop other tickers."""
    good_df = _mock_history([50.0, 51.0])
    with patch("daily_market.fetcher.yf.Ticker") as mock_yf:
        def side_effect(ticker):
            m = MagicMock()
            if ticker == "FAIL":
                m.history.side_effect = RuntimeError("bad")
            else:
                m.history.return_value = good_df
            return m
        mock_yf.side_effect = side_effect

        snapshots = fetch_snapshots(["^HSI", "FAIL"], {"^HSI": "index", "FAIL": "index"})

    assert len(snapshots) == 2
    assert snapshots[0].error is None
    assert snapshots[1].error is not None


def test_safe_float():
    import math
    assert _safe_float(1.5) == 1.5
    assert _safe_float(None) is None
    assert _safe_float(float("nan")) is None
    assert _safe_float(float("inf")) is None
    assert _safe_float("bad") is None
