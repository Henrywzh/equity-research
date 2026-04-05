from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import FetchRun, PriceSnapshot


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

    def close(self) -> None:
        self._conn.close()
