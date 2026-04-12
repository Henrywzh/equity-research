from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from daily_market.models import (
    FetchRun,
    PolymarketFetchRun,
    PolymarketMarket,
    PolymarketSnapshot,
    PriceSnapshot,
)
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


def _make_polymarket_run() -> PolymarketFetchRun:
    return PolymarketFetchRun(
        id="poly-run-001",
        started_at="2026-04-12T09:00:00+00:00",
        market_count=2,
    )


def _make_polymarket_market() -> PolymarketMarket:
    return PolymarketMarket(
        market_id="616903",
        event_id="evt-1",
        event_slug="how-many-fed-rate-cuts-in-2026",
        event_title="How many Fed rate cuts in 2026?",
        market_slug="will-1-fed-rate-cut-happen-in-2026",
        question="Will 1 Fed rate cut happen in 2026?",
        group_key="fed_rates",
        asset="fed",
        horizon="year_end",
        end_date="2026-12-31T00:00:00Z",
        active=True,
        closed=False,
        archived=False,
        yes_label="Yes",
        no_label="No",
        outcomes_json='["Yes","No"]',
        outcome_prices_json='["0.52","0.48"]',
        clob_token_ids_json='["1","2"]',
        source_url="https://polymarket.com/event/how-many-fed-rate-cuts-in-2026/will-1-fed-rate-cut-happen-in-2026",
        last_metadata_refresh_at="2026-04-12T09:00:00+00:00",
    )


def _make_polymarket_snapshot() -> PolymarketSnapshot:
    return PolymarketSnapshot(
        market_id="616903",
        market_slug="will-1-fed-rate-cut-happen-in-2026",
        group_key="fed_rates",
        asset="fed",
        horizon="year_end",
        fetched_at="2026-04-12T09:05:00+00:00",
        bucket_hour="2026-04-12T09:00:00+00:00",
        implied_probability=0.52,
        best_bid=0.51,
        best_ask=0.53,
        midpoint=0.52,
        spread=0.02,
        last_trade_price=0.52,
        volume=1234.0,
        volume_24h=345.0,
        liquidity=4567.0,
        open_interest=None,
        expiry_timestamp="2026-12-31T00:00:00Z",
        market_status="active",
        error=None,
    )


def test_polymarket_storage_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "test.sqlite")
        run = _make_polymarket_run()
        market = _make_polymarket_market()
        snapshot = _make_polymarket_snapshot()

        storage.record_polymarket_run(run)
        storage.upsert_polymarket_markets([market])
        storage.insert_polymarket_snapshots(run.id, [snapshot])
        storage.update_polymarket_run(run.id, status="success", snapshot_count=1)

        latest = storage.fetch_latest_polymarket_run()
        rows = storage.fetch_polymarket_snapshots_by_run(run.id)
        history = storage.fetch_polymarket_market_history(market.market_slug, limit=5)
        storage.close()

        assert latest is not None
        assert latest["status"] == "success"
        assert len(rows) == 1
        assert rows[0]["market_slug"] == market.market_slug
        assert len(history) == 1


def test_polymarket_snapshot_dedupes_same_hour():
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "test.sqlite")
        run = _make_polymarket_run()
        market = _make_polymarket_market()
        snapshot = _make_polymarket_snapshot()

        storage.record_polymarket_run(run)
        storage.upsert_polymarket_markets([market])
        storage.insert_polymarket_snapshots(run.id, [snapshot, snapshot])
        rows = storage.fetch_polymarket_snapshots_by_run(run.id)
        storage.close()

        assert len(rows) == 1
