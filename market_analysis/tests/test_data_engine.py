from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_analysis.data_engine import DataEngine


def _multiindex_prices(index: pd.DatetimeIndex, series_map: dict[str, list[float | None]]) -> pd.DataFrame:
    close_map = {ticker: [value * 10 if value is not None else None for value in values] for ticker, values in series_map.items()}
    return pd.concat(
        {
            "Adj Close": pd.DataFrame(series_map, index=index),
            "Close": pd.DataFrame(close_map, index=index),
        },
        axis=1,
    )


def test_get_close_prices_prefers_adjusted_close_without_chained_assignment() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    data = pd.concat(
        {
            "Adj Close": pd.DataFrame({"AAA": [10.0, 11.0, None], "BBB": [None, None, None]}, index=index),
            "Close": pd.DataFrame({"AAA": [100.0, 110.0, 120.0], "BBB": [20.0, 21.0, 22.0]}, index=index),
        },
        axis=1,
    )

    prices = DataEngine().get_close_prices(data)

    assert prices["AAA"].tolist() == [10.0, 11.0, 120.0]
    assert prices["BBB"].tolist() == [20.0, 21.0, 22.0]


def test_refresh_market_data_bootstraps_when_cache_missing(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    responses = [_multiindex_prices(index, {"AAA": [1.0, 2.0, 3.0], "BBB": [4.0, 5.0, 6.0]})]

    def fake_download(tickers, period, interval, progress, auto_adjust, threads):  # noqa: ANN001
        assert period == "5y"
        return responses.pop(0)

    monkeypatch.setattr("market_analysis.data_engine.yf.download", fake_download)

    engine = DataEngine(cache_dir=str(tmp_path), chunk_size=10)
    prices, meta = engine.refresh_market_data(["AAA", "BBB"])

    assert prices is not None
    assert list(prices.columns) == ["AAA", "BBB"]
    assert meta["mode"] == "bootstrap"
    assert meta["refreshed_tickers"] == ["AAA", "BBB"]
    assert Path(meta["cache_file"]).exists()


def test_refresh_market_data_merges_incremental_overlap_and_keeps_cache_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    engine = DataEngine(cache_dir=str(cache_dir), chunk_size=10)

    cached_index = pd.date_range(pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=4), periods=3, freq="D")
    cached = pd.DataFrame(
        {
            "AAA": [1.0, 2.0, 3.0],
            "BBB": [10.0, 11.0, 12.0],
        },
        index=cached_index,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.to_pickle(engine.cache_file)

    fresh_index = pd.date_range(cached_index[-1], periods=3, freq="D")
    fresh = _multiindex_prices(fresh_index, {"AAA": [30.0, 40.0, 50.0]})

    def fake_download(tickers, period, interval, progress, auto_adjust, threads):  # noqa: ANN001
        assert period == "5d"
        return fresh

    monkeypatch.setattr("market_analysis.data_engine.yf.download", fake_download)

    prices, meta = engine.refresh_market_data(["AAA", "BBB"])

    assert prices is not None
    expected_index = pd.date_range(cached_index[0].tz_localize(None), periods=5, freq="D")
    assert prices.index.tolist() == expected_index.tolist()
    overlap_day = cached_index[-1].tz_localize(None)
    assert prices.loc[overlap_day, "AAA"] == 30.0
    assert prices.loc[expected_index[-1], "AAA"] == 50.0
    assert prices.loc[overlap_day, "BBB"] == 12.0
    assert meta["mode"] == "incremental"
    assert meta["refreshed_tickers"] == ["AAA"]
    assert meta["fallback_tickers"] == ["BBB"]
    assert meta["unavailable_tickers"] == []


def test_refresh_market_data_bootstraps_missing_ticker_and_then_runs_incremental(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    engine = DataEngine(cache_dir=str(cache_dir), chunk_size=10)

    cached_index = pd.date_range(pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=4), periods=3, freq="D")
    cached = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=cached_index)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.to_pickle(engine.cache_file)

    bootstrap_index = pd.date_range("2020-01-01", periods=3, freq="D")
    incremental_index = pd.date_range(cached_index[-1], periods=2, freq="D")
    responses = [
        _multiindex_prices(bootstrap_index, {"BBB": [7.0, 8.0, 9.0]}),
        _multiindex_prices(incremental_index, {"AAA": [30.0, 40.0], "BBB": [50.0, 60.0]}),
    ]

    def fake_download(tickers, period, interval, progress, auto_adjust, threads):  # noqa: ANN001
        return responses.pop(0)

    monkeypatch.setattr("market_analysis.data_engine.yf.download", fake_download)

    prices, meta = engine.refresh_market_data(["AAA", "BBB"])

    assert prices is not None
    assert sorted(prices.columns.tolist()) == ["AAA", "BBB"]
    assert meta["mode"] == "bootstrap+incremental"
    assert meta["refreshed_tickers"] == ["AAA", "BBB"]
