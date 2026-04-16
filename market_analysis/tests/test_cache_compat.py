from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_analysis.data_engine import DataEngine


def test_load_cache_falls_back_to_readable_legacy_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    (cache_dir / "etf_data_5y.pkl").write_bytes(b"not a pickle")

    index = pd.date_range("2025-01-01", periods=3, freq="D")
    legacy = pd.concat(
        {
            "Adj Close": pd.DataFrame({"AAA": [1.0, 2.0, 3.0], "BBB": [4.0, 5.0, 6.0]}, index=index),
            "Close": pd.DataFrame({"AAA": [10.0, 20.0, 30.0], "BBB": [40.0, 50.0, 60.0]}, index=index),
        },
        axis=1,
    )
    legacy.columns = legacy.columns.map(str)
    legacy.to_pickle(cache_dir / "etf_data_2y.pkl")

    engine = DataEngine(cache_dir=str(cache_dir))
    prices = engine._load_cache()

    assert prices is not None
    assert list(prices.columns) == ["AAA", "BBB"]
    assert engine.loaded_cache_file == str(cache_dir / "etf_data_2y.pkl")
