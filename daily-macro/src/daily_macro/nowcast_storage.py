from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class NowcastStorage:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS indicator_series (
                series_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                name TEXT NOT NULL,
                unit TEXT NOT NULL,
                description TEXT
            );

            CREATE TABLE IF NOT EXISTS indicator_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                series_id TEXT NOT NULL,
                target_period TEXT NOT NULL,
                value REAL NOT NULL,
                as_of_date TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                FOREIGN KEY (series_id) REFERENCES indicator_series(series_id) ON DELETE CASCADE,
                UNIQUE (series_id, target_period, as_of_date)
            );
            """
        )
        self.connection.commit()

    def ensure_series(
        self, series_id: str, source: str, name: str, unit: str, description: str = ""
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO indicator_series (series_id, source, name, unit, description)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(series_id) DO UPDATE SET
                source=excluded.source,
                name=excluded.name,
                unit=excluded.unit,
                description=excluded.description
            """,
            (series_id, source, name, unit, description),
        )
        self.connection.commit()

    def upsert_observation(
        self, series_id: str, target_period: str, value: float, as_of_date: str
    ) -> None:
        fetched_at = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            """
            INSERT INTO indicator_observations (series_id, target_period, value, as_of_date, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(series_id, target_period, as_of_date) DO UPDATE SET
                value=excluded.value,
                fetched_at=excluded.fetched_at
            """,
            (series_id, target_period, value, as_of_date, fetched_at),
        )
        self.connection.commit()

    def fetch_latest(self, series_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM indicator_observations
            WHERE series_id = ?
            ORDER BY as_of_date DESC, fetched_at DESC
            LIMIT 1
            """,
            (series_id,),
        ).fetchone()
        return dict(row) if row else None

    def fetch_all_latest(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT s.name, s.source, s.unit, o.*
            FROM indicator_series s
            JOIN (
                SELECT series_id, target_period, MAX(as_of_date) as max_as_of
                FROM indicator_observations
                GROUP BY series_id, target_period
            ) latest_obs ON s.series_id = latest_obs.series_id
            JOIN indicator_observations o ON o.series_id = latest_obs.series_id
                AND o.target_period = latest_obs.target_period
                AND o.as_of_date = latest_obs.max_as_of
            ORDER BY s.series_id ASC, o.target_period ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def fetch_revision_history(
        self, series_id: str, target_period: str
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM indicator_observations
            WHERE series_id = ? AND target_period = ?
            ORDER BY as_of_date ASC
            """,
            (series_id, target_period),
        ).fetchall()
        return [dict(r) for r in rows]
