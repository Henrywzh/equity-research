from __future__ import annotations

import pandas as pd
import pytest

from market_analysis.regime_monitor import BreadthMonitor


class StubEngine:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices

    def fetch_data(self, *args, **kwargs):
        return self.prices


def test_breadth_monitor_runs_successfully() -> None:
    index = pd.date_range("2025-01-01", periods=260, freq="B")
    prices = pd.DataFrame(
        {
            "RSP": pd.Series(range(100, 360), index=index, dtype=float),
            "SPY": pd.Series(range(110, 370), index=index, dtype=float),
        }
    )

    results = BreadthMonitor(StubEngine(prices)).analyze()

    assert results is not None
    assert "regime" in results
    assert "z_score" in results
    assert "ratio_history" in results
    assert "ma_history" in results
    assert "z_history" in results


def test_breadth_monitor_fails_cleanly_when_ticker_is_missing() -> None:
    index = pd.date_range("2025-01-01", periods=260, freq="B")
    prices = pd.DataFrame(
        {
            "RSP": pd.Series(range(100, 360), index=index, dtype=float),
        }
    )

    with pytest.raises(ValueError, match="No RSP or SPY tickers available"):
        BreadthMonitor(StubEngine(prices)).analyze()
