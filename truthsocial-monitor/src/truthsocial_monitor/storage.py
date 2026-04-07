from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .models import TruthPost


class Storage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS posts (
                post_id          TEXT PRIMARY KEY,
                created_at       TEXT NOT NULL,
                content          TEXT,
                reblogs_count    INTEGER DEFAULT 0,
                favourites_count INTEGER DEFAULT 0,
                replies_count    INTEGER DEFAULT 0,
                url              TEXT,
                sensitive        INTEGER DEFAULT 0,
                first_seen_at    TEXT NOT NULL,
                raw_json         TEXT
            );

            CREATE TABLE IF NOT EXISTS scrape_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at      TEXT NOT NULL,
                finished_at     TEXT,
                status          TEXT,
                new_posts_count INTEGER DEFAULT 0,
                error_summary   TEXT
            );
            """
        )
        self.connection.commit()

    # ── Posts ─────────────────────────────────────────────────────────────────

    def insert_posts(self, posts: list[TruthPost], seen_at: str) -> int:
        """Insert new posts (INSERT OR IGNORE — never overwrites). Returns count of rows inserted."""
        if not posts:
            return 0
        inserted = 0
        for post in posts:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO posts
                    (post_id, created_at, content, reblogs_count, favourites_count,
                     replies_count, url, sensitive, first_seen_at, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    post.post_id,
                    post.created_at,
                    post.content,
                    post.reblogs_count,
                    post.favourites_count,
                    post.replies_count,
                    post.url,
                    1 if post.sensitive else 0,
                    seen_at,
                    post.raw_json,
                ),
            )
            inserted += cursor.rowcount
        self.connection.commit()
        return inserted

    def fetch_recent_posts(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM posts ORDER BY created_at DESC, post_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def fetch_post_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM posts").fetchone()[0])

    # ── Scrape runs ───────────────────────────────────────────────────────────

    def start_run(self, started_at: str) -> int:
        cursor = self.connection.execute(
            "INSERT INTO scrape_runs (started_at, status) VALUES (?, ?)",
            (started_at, "running"),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        finished_at: str,
        status: str,
        new_posts_count: int,
        error_summary: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE scrape_runs
            SET finished_at = ?, status = ?, new_posts_count = ?, error_summary = ?
            WHERE id = ?
            """,
            (finished_at, status, new_posts_count, error_summary, run_id),
        )
        self.connection.commit()

    def fetch_recent_runs(self, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM scrape_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
