from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import (
    FetchRun,
    PolymarketFetchRun,
    PolymarketMarket,
    PolymarketSnapshot,
    PriceSnapshot,
)


class Storage:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS fetch_runs (
                id           TEXT PRIMARY KEY,
                started_at   TEXT NOT NULL,
                finished_at  TEXT,
                session      TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'running',
                ticker_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                error_summary TEXT
            );

            CREATE TABLE IF NOT EXISTS price_snapshots (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id         TEXT NOT NULL REFERENCES fetch_runs(id),
                ticker         TEXT NOT NULL,
                asset_class    TEXT NOT NULL,
                fetched_at     TEXT NOT NULL,
                price          REAL,
                prev_close     REAL,
                pct_change     REAL,
                open           REAL,
                high           REAL,
                low            REAL,
                volume         REAL,
                data_timestamp TEXT,
                error          TEXT,
                UNIQUE(run_id, ticker)
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_ticker
                ON price_snapshots(ticker);
            CREATE INDEX IF NOT EXISTS idx_snapshots_data_timestamp
                ON price_snapshots(data_timestamp);

            CREATE TABLE IF NOT EXISTS polymarket_fetch_runs (
                id             TEXT PRIMARY KEY,
                started_at     TEXT NOT NULL,
                finished_at    TEXT,
                status         TEXT NOT NULL DEFAULT 'running',
                market_count   INTEGER NOT NULL DEFAULT 0,
                snapshot_count INTEGER NOT NULL DEFAULT 0,
                error_summary  TEXT
            );

            CREATE TABLE IF NOT EXISTS polymarket_markets (
                market_id                 TEXT PRIMARY KEY,
                event_id                  TEXT,
                event_slug                TEXT,
                event_title               TEXT,
                market_slug               TEXT NOT NULL,
                question                  TEXT NOT NULL,
                group_key                 TEXT NOT NULL,
                asset                     TEXT NOT NULL,
                horizon                   TEXT NOT NULL,
                end_date                  TEXT,
                active                    INTEGER NOT NULL DEFAULT 0,
                closed                    INTEGER NOT NULL DEFAULT 0,
                archived                  INTEGER NOT NULL DEFAULT 0,
                yes_label                 TEXT,
                no_label                  TEXT,
                outcomes_json             TEXT,
                outcome_prices_json       TEXT,
                clob_token_ids_json       TEXT,
                source_url                TEXT,
                last_metadata_refresh_at  TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_polymarket_markets_slug
                ON polymarket_markets(market_slug);
            CREATE INDEX IF NOT EXISTS idx_polymarket_markets_group
                ON polymarket_markets(group_key);

            CREATE TABLE IF NOT EXISTS polymarket_snapshots (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id              TEXT NOT NULL REFERENCES polymarket_fetch_runs(id),
                market_id           TEXT NOT NULL REFERENCES polymarket_markets(market_id),
                market_slug         TEXT NOT NULL,
                group_key           TEXT NOT NULL,
                asset               TEXT NOT NULL,
                horizon             TEXT NOT NULL,
                fetched_at          TEXT NOT NULL,
                bucket_hour         TEXT NOT NULL,
                implied_probability REAL,
                best_bid            REAL,
                best_ask            REAL,
                midpoint            REAL,
                spread              REAL,
                last_trade_price    REAL,
                volume              REAL,
                volume_24h          REAL,
                liquidity           REAL,
                open_interest       REAL,
                expiry_timestamp    TEXT,
                market_status       TEXT NOT NULL,
                error               TEXT,
                UNIQUE(market_id, bucket_hour)
            );

            CREATE INDEX IF NOT EXISTS idx_polymarket_snapshots_market
                ON polymarket_snapshots(market_id, fetched_at DESC);
            CREATE INDEX IF NOT EXISTS idx_polymarket_snapshots_group
                ON polymarket_snapshots(group_key, fetched_at DESC);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def record_run(self, run: FetchRun) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO fetch_runs
                (id, started_at, finished_at, session, status, ticker_count, success_count, error_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.started_at,
                run.finished_at,
                run.session,
                run.status,
                run.ticker_count,
                run.success_count,
                run.error_summary,
            ),
        )
        self._conn.commit()

    def update_run(self, run_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        columns = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [run_id]
        self._conn.execute(f"UPDATE fetch_runs SET {columns} WHERE id = ?", values)
        self._conn.commit()

    def fetch_latest_run(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM fetch_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def record_polymarket_run(self, run: PolymarketFetchRun) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO polymarket_fetch_runs
                (id, started_at, finished_at, status, market_count, snapshot_count, error_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.started_at,
                run.finished_at,
                run.status,
                run.market_count,
                run.snapshot_count,
                run.error_summary,
            ),
        )
        self._conn.commit()

    def update_polymarket_run(self, run_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        columns = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [run_id]
        self._conn.execute(f"UPDATE polymarket_fetch_runs SET {columns} WHERE id = ?", values)
        self._conn.commit()

    def fetch_latest_polymarket_run(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM polymarket_fetch_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def insert_snapshots(self, run_id: str, snapshots: list[PriceSnapshot]) -> None:
        for s in snapshots:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO price_snapshots
                    (run_id, ticker, asset_class, fetched_at, price, prev_close,
                     pct_change, open, high, low, volume, data_timestamp, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    s.ticker,
                    s.asset_class,
                    s.fetched_at,
                    s.price,
                    s.prev_close,
                    s.pct_change,
                    s.open,
                    s.high,
                    s.low,
                    s.volume,
                    s.data_timestamp,
                    s.error,
                ),
            )
        self._conn.commit()

    def fetch_snapshots_by_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM price_snapshots WHERE run_id = ? ORDER BY asset_class, ticker",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def fetch_snapshots_by_date(self, date_str: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT ps.*
            FROM price_snapshots ps
            JOIN fetch_runs fr ON ps.run_id = fr.id
            WHERE ps.data_timestamp = ?
            ORDER BY ps.asset_class, ps.ticker
            """,
            (date_str,),
        ).fetchall()
        return [dict(r) for r in rows]

    def fetch_ticker_history(
        self, ticker: str, limit: int = 30
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT ps.*
            FROM price_snapshots ps
            WHERE ps.ticker = ?
            ORDER BY ps.fetched_at DESC
            LIMIT ?
            """,
            (ticker, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def upsert_polymarket_markets(self, markets: list[PolymarketMarket]) -> None:
        for market in markets:
            self._conn.execute(
                """
                INSERT INTO polymarket_markets (
                    market_id, event_id, event_slug, event_title, market_slug, question,
                    group_key, asset, horizon, end_date, active, closed, archived,
                    yes_label, no_label, outcomes_json, outcome_prices_json,
                    clob_token_ids_json, source_url, last_metadata_refresh_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    event_slug = excluded.event_slug,
                    event_title = excluded.event_title,
                    market_slug = excluded.market_slug,
                    question = excluded.question,
                    group_key = excluded.group_key,
                    asset = excluded.asset,
                    horizon = excluded.horizon,
                    end_date = excluded.end_date,
                    active = excluded.active,
                    closed = excluded.closed,
                    archived = excluded.archived,
                    yes_label = excluded.yes_label,
                    no_label = excluded.no_label,
                    outcomes_json = excluded.outcomes_json,
                    outcome_prices_json = excluded.outcome_prices_json,
                    clob_token_ids_json = excluded.clob_token_ids_json,
                    source_url = excluded.source_url,
                    last_metadata_refresh_at = excluded.last_metadata_refresh_at
                """,
                (
                    market.market_id,
                    market.event_id,
                    market.event_slug,
                    market.event_title,
                    market.market_slug,
                    market.question,
                    market.group_key,
                    market.asset,
                    market.horizon,
                    market.end_date,
                    int(market.active),
                    int(market.closed),
                    int(market.archived),
                    market.yes_label,
                    market.no_label,
                    market.outcomes_json,
                    market.outcome_prices_json,
                    market.clob_token_ids_json,
                    market.source_url,
                    market.last_metadata_refresh_at,
                ),
            )
        self._conn.commit()

    def insert_polymarket_snapshots(
        self, run_id: str, snapshots: list[PolymarketSnapshot]
    ) -> None:
        for snapshot in snapshots:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO polymarket_snapshots (
                    run_id, market_id, market_slug, group_key, asset, horizon,
                    fetched_at, bucket_hour, implied_probability, best_bid, best_ask,
                    midpoint, spread, last_trade_price, volume, volume_24h,
                    liquidity, open_interest, expiry_timestamp, market_status, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    snapshot.market_id,
                    snapshot.market_slug,
                    snapshot.group_key,
                    snapshot.asset,
                    snapshot.horizon,
                    snapshot.fetched_at,
                    snapshot.bucket_hour,
                    snapshot.implied_probability,
                    snapshot.best_bid,
                    snapshot.best_ask,
                    snapshot.midpoint,
                    snapshot.spread,
                    snapshot.last_trade_price,
                    snapshot.volume,
                    snapshot.volume_24h,
                    snapshot.liquidity,
                    snapshot.open_interest,
                    snapshot.expiry_timestamp,
                    snapshot.market_status,
                    snapshot.error,
                ),
            )
        self._conn.commit()

    def fetch_polymarket_snapshots_by_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT ps.*, pm.question, pm.source_url
            FROM polymarket_snapshots ps
            JOIN polymarket_markets pm ON pm.market_id = ps.market_id
            WHERE ps.run_id = ?
            ORDER BY ps.group_key, ps.market_slug
            """,
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def fetch_polymarket_snapshots_by_date(self, date_str: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT ps.*, pm.question, pm.source_url
            FROM polymarket_snapshots ps
            JOIN polymarket_markets pm ON pm.market_id = ps.market_id
            WHERE substr(ps.fetched_at, 1, 10) = ?
            ORDER BY ps.group_key, ps.market_slug, ps.fetched_at DESC
            """,
            (date_str,),
        ).fetchall()
        return [dict(r) for r in rows]

    def fetch_polymarket_group_history(
        self, group_key: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT ps.*, pm.question, pm.source_url
            FROM polymarket_snapshots ps
            JOIN polymarket_markets pm ON pm.market_id = ps.market_id
            WHERE ps.group_key = ?
            ORDER BY ps.fetched_at DESC, ps.market_slug
            LIMIT ?
            """,
            (group_key, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def fetch_polymarket_market_history(
        self,
        market_ref: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT ps.*, pm.question, pm.source_url
            FROM polymarket_snapshots ps
            JOIN polymarket_markets pm ON pm.market_id = ps.market_id
            WHERE ps.market_id = ? OR ps.market_slug = ?
            ORDER BY ps.fetched_at DESC
            LIMIT ?
            """,
            (market_ref, market_ref, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def write_snapshot_json(
        self,
        run: FetchRun,
        snapshots: list[PriceSnapshot],
        snapshots_dir: Path,
    ) -> Path:
        date_str = run.started_at[:10]
        out_dir = snapshots_dir / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run.session}.json"
        payload = {
            "run_id": run.id,
            "session": run.session,
            "started_at": run.started_at,
            "snapshots": [asdict(s) for s in snapshots],
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path

    def write_summary_json(
        self,
        summary: dict[str, Any],
        summaries_dir: Path,
        date_str: str,
        session: str,
    ) -> Path:
        out_dir = summaries_dir / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{session}.json"
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path

    def write_polymarket_run_json(
        self,
        payload: dict[str, Any],
        out_dir: Path,
        started_at: str,
    ) -> Path:
        date_str = started_at[:10]
        target_dir = out_dir / date_str
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "polymarket.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path

    def close(self) -> None:
        self._conn.close()
