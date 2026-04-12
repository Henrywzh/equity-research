from __future__ import annotations

import io
from contextlib import redirect_stdout
from unittest.mock import patch

from daily_market import cli


def test_fetch_polymarket_cli_prints_json():
    argv = ["daily_market", "fetch-polymarket", "--json"]
    payload = {
        "run_id": "poly-1",
        "status": "success",
        "date": "2026-04-12",
        "market_count": 2,
        "snapshot_count": 2,
        "error_count": 0,
        "errors": [],
        "output_path": "/tmp/poly.json",
        "db_path": "/tmp/market.sqlite",
        "data_dir": "/tmp",
    }

    with patch("daily_market.pipeline.run_fetch_polymarket", return_value=payload):
        with patch("sys.argv", argv):
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main()

    assert '"run_id": "poly-1"' in buf.getvalue()


def test_query_polymarket_group_cli_uses_pipeline():
    argv = ["daily_market", "query-polymarket", "group", "qqq_daily", "--json"]
    payload = [{"market_slug": "qqq-up-or-down-on-april-13-2026", "implied_probability": 0.51}]

    with patch("daily_market.pipeline.query_polymarket_group", return_value=payload) as mocked:
        with patch("sys.argv", argv):
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main()

    mocked.assert_called_once()
    assert "qqq-up-or-down-on-april-13-2026" in buf.getvalue()
