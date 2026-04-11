from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from daily_market.models import FetchRun, PriceSnapshot
from daily_market.storage import Storage


def _make_run(session: str = "morning") -> FetchRun:
    return FetchRun(
        id="test-run-001",
        started_at=datetime.now(timezone.utc).isoformat(),
        session=session,
        ticker_count=2,
    )


def _make_snapshot(ticker: str = "^HSI", error: str | None = None) -> PriceSnapshot:
    return PriceSnapshot(
        ticker=ticker,
        asset_class="index",
        fetched_at=datetime.now(timezone.utc).isoformat(),
        price=20000.0 if not error else None,
        prev_close=19800.0 if not error else None,
        pct_change=1.01 if not error else None,
        open=19900.0 if not error else None,
        high=20100.0 if not error else None,
        low=19750.0 if not error else None,
        volume=1_200_000.0,
        data_timestamp="2026-04-04",
        error=error,
    )


def test_record_and_fetch_run():
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "test.sqlite")
        run = _make_run()
        storage.record_run(run)
        result = storage.fetch_latest_run()
        assert result is not None
        assert result["id"] == run.id
        assert result["session"] == "morning"
        storage.close()


def test_update_run():
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "test.sqlite")
        run = _make_run()
        storage.record_run(run)
        storage.update_run(run.id, status="success", success_count=2)
        result = storage.fetch_latest_run()
        assert result["status"] == "success"
        assert result["success_count"] == 2
        storage.close()


def test_insert_and_fetch_snapshots():
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "test.sqlite")
        run = _make_run()
        storage.record_run(run)
        snaps = [_make_snapshot("^HSI"), _make_snapshot("^SPX")]
        storage.insert_snapshots(run.id, snaps)
        rows = storage.fetch_snapshots_by_run(run.id)
        assert len(rows) == 2
        tickers = {r["ticker"] for r in rows}
        assert "^HSI" in tickers
        assert "^SPX" in tickers
        storage.close()


def test_fetch_snapshots_by_date():
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "test.sqlite")
        run = _make_run()
        storage.record_run(run)
        storage.insert_snapshots(run.id, [_make_snapshot("^HSI")])
        rows = storage.fetch_snapshots_by_date("2026-04-04")
        assert len(rows) == 1
        assert rows[0]["ticker"] == "^HSI"
        storage.close()


def test_fetch_ticker_history():
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "test.sqlite")
        for i in range(3):
            run = FetchRun(
                id=f"run-{i}",
                started_at=f"2026-04-0{i+1}T07:00:00+00:00",
                session="morning",
            )
            storage.record_run(run)
            storage.insert_snapshots(run.id, [_make_snapshot("^HSI")])
        rows = storage.fetch_ticker_history("^HSI", limit=10)
        assert len(rows) == 3
        storage.close()


def test_write_snapshot_json(tmp_path: Path):
    storage = Storage(tmp_path / "test.sqlite")
    run = _make_run()
    storage.record_run(run)
    snaps = [_make_snapshot("^HSI")]
    out = storage.write_snapshot_json(run, snaps, tmp_path / "snapshots")
    assert out.exists()
    import json
    data = json.loads(out.read_text())
    assert data["run_id"] == run.id
    assert len(data["snapshots"]) == 1
    storage.close()
