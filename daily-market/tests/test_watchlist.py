from __future__ import annotations

import json
import tempfile
from pathlib import Path

from daily_market.watchlist import load_watchlist


def _write_config(tmp_dir: Path, data: dict) -> Path:
    path = tmp_dir / "watchlist.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_default_symbols():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_config(
            Path(tmp),
            {
                "default": {
                    "indices": ["^HSI", "^SPX"],
                    "commodities": ["GC=F"],
                    "fx": ["USDCNH=X"],
                    "crypto": ["BTC-USD"],
                },
                "users": {},
            },
        )
        wl = load_watchlist(cfg_path)
        assert wl.tickers == ["^HSI", "^SPX", "GC=F", "USDCNH=X", "BTC-USD"]


def test_asset_class_map():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_config(
            Path(tmp),
            {
                "default": {
                    "indices": ["^HSI"],
                    "crypto": ["BTC-USD"],
                },
                "users": {},
            },
        )
        wl = load_watchlist(cfg_path)
        ac_map = wl.asset_class_map()
        assert ac_map["^HSI"] == "index"
        assert ac_map["BTC-USD"] == "crypto"


def test_user_extra_symbols_string():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_config(
            Path(tmp),
            {
                "default": {"indices": ["^HSI"]},
                "users": {
                    "user@example.com": {"extra_symbols": ["0700.HK"]},
                },
            },
        )
        wl = load_watchlist(cfg_path, user_email="user@example.com")
        assert "0700.HK" in wl.tickers
        ac_map = wl.asset_class_map()
        assert ac_map["0700.HK"] == "custom"


def test_user_extra_symbols_dict():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_config(
            Path(tmp),
            {
                "default": {"indices": ["^HSI"]},
                "users": {
                    "user@example.com": {
                        "extra_symbols": [{"ticker": "0700.HK", "asset_class": "index"}]
                    }
                },
            },
        )
        wl = load_watchlist(cfg_path, user_email="user@example.com")
        assert wl.asset_class_map()["0700.HK"] == "index"


def test_no_duplicate_symbols():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_config(
            Path(tmp),
            {
                "default": {"indices": ["^HSI", "^HSI"]},
                "users": {
                    "user@example.com": {"extra_symbols": ["^HSI"]}
                },
            },
        )
        wl = load_watchlist(cfg_path, user_email="user@example.com")
        assert wl.tickers.count("^HSI") == 1


def test_missing_config_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_watchlist("/nonexistent/watchlist.json")
