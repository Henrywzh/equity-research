from __future__ import annotations

import pandas as pd
import pytest

from market_analysis.regime_monitor import RegimeMonitor


class StubEngine:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices

    def fetch_data(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self.prices


def test_regime_monitor_uses_available_tickers_when_some_are_missing() -> None:
    index = pd.date_range("2025-01-01", periods=260, freq="B")
    prices = pd.DataFrame(
        {
            "XLK": pd.Series(range(100, 360), index=index, dtype=float),
            "XLY": pd.Series(range(110, 370), index=index, dtype=float),
            "XLC": pd.Series(range(120, 380), index=index, dtype=float),
            "XLU": pd.Series(range(90, 350), index=index, dtype=float),
            "XLV": pd.Series(range(95, 355), index=index, dtype=float),
        }
    )

    results = RegimeMonitor(StubEngine(prices)).analyze()

    assert results is not None
    assert "regime" in results
    assert "z_score" in results


def test_regime_monitor_fails_cleanly_when_group_is_completely_missing() -> None:
    index = pd.date_range("2025-01-01", periods=260, freq="B")
    prices = pd.DataFrame(
        {
            "XLK": pd.Series(range(100, 360), index=index, dtype=float),
            "XLY": pd.Series(range(110, 370), index=index, dtype=float),
            "XLC": pd.Series(range(120, 380), index=index, dtype=float),
        }
    )

    with pytest.raises(ValueError, match="No defensive tickers available"):
        RegimeMonitor(StubEngine(prices)).analyze()
