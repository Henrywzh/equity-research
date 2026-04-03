from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import ArticleDetails, PlacementCandidate


class Storage:
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
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_url TEXT NOT NULL UNIQUE,
                source_site TEXT NOT NULL,
                source_article_id TEXT,
                title TEXT NOT NULL,
                article_section TEXT,
                published_at TEXT,
                summary_snippet TEXT,
                content_text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scrape_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                article_count INTEGER NOT NULL DEFAULT 0,
                placement_count INTEGER NOT NULL DEFAULT 0,
                error_summary TEXT
            );

            CREATE TABLE IF NOT EXISTS article_placements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                run_id INTEGER NOT NULL,
                collection TEXT NOT NULL,
                rank INTEGER NOT NULL,
                homepage_section TEXT,
                summary_snippet TEXT,
                collected_at TEXT NOT NULL,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
                FOREIGN KEY (run_id) REFERENCES scrape_runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS article_backups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER,
                run_id INTEGER NOT NULL,
                backup_type TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                scraped_at TEXT NOT NULL,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
                FOREIGN KEY (run_id) REFERENCES scrape_runs(id) ON DELETE CASCADE
            );
            """
        )
        self.connection.commit()

    def start_run(self, started_at: str) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO scrape_runs (started_at, status)
            VALUES (?, ?)
            """,
            (started_at, "running"),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        finished_at: str,
        status: str,
        article_count: int,
        placement_count: int,
        error_summary: list[str] | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE scrape_runs
            SET finished_at = ?, status = ?, article_count = ?, placement_count = ?, error_summary = ?
            WHERE id = ?
            """,
            (
                finished_at,
                status,
                article_count,
                placement_count,
                json.dumps(error_summary, ensure_ascii=False) if error_summary else None,
                run_id,
            ),
        )
        self.connection.commit()

    def upsert_article(self, article: ArticleDetails, seen_at: str) -> int:
        existing = self.connection.execute(
            """
            SELECT id, first_seen_at
            FROM articles
            WHERE canonical_url = ?
            """,
            (article.canonical_url,),
        ).fetchone()

        if existing is None:
            cursor = self.connection.execute(
                """
                INSERT INTO articles (
                    canonical_url,
                    source_site,
                    source_article_id,
                    title,
                    article_section,
                    published_at,
                    summary_snippet,
                    content_text,
                    content_hash,
                    first_seen_at,
                    last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.canonical_url,
                    article.source_site,
                    article.source_article_id,
                    article.title,
                    article.article_section,
                    article.published_at,
                    article.summary_snippet,
                    article.content_text,
                    article.content_hash,
                    seen_at,
                    seen_at,
                ),
            )
            self.connection.commit()
            return int(cursor.lastrowid)

        article_id = int(existing["id"])
        self.connection.execute(
            """
            UPDATE articles
            SET
                source_site = ?,
                source_article_id = ?,
                title = ?,
                article_section = ?,
                published_at = ?,
                summary_snippet = COALESCE(?, summary_snippet),
                content_text = ?,
                content_hash = ?,
                last_seen_at = ?
            WHERE id = ?
            """,
            (
                article.source_site,
                article.source_article_id,
                article.title,
                article.article_section,
                article.published_at,
                article.summary_snippet,
                article.content_text,
                article.content_hash,
                seen_at,
                article_id,
            ),
        )
        self.connection.commit()
        return article_id

    def record_placement(
        self,
        article_id: int,
        run_id: int,
        placement: PlacementCandidate,
        collected_at: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO article_placements (
                article_id,
                run_id,
                collection,
                rank,
                homepage_section,
                summary_snippet,
                collected_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_id,
                run_id,
                placement.collection,
                placement.rank,
                placement.homepage_section,
                placement.summary_snippet,
                collected_at,
            ),
        )
        self.connection.commit()

    def record_backup(
        self,
        run_id: int,
        backup_type: str,
        relative_path: str,
        content_hash: str,
        scraped_at: str,
        article_id: int | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO article_backups (
                article_id,
                run_id,
                backup_type,
                relative_path,
                content_hash,
                scraped_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (article_id, run_id, backup_type, relative_path, content_hash, scraped_at),
        )
        self.connection.commit()

    def get_backups_older_than(self, cutoff_iso: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT id, relative_path
            FROM article_backups
            WHERE scraped_at < ?
            ORDER BY id
            """,
            (cutoff_iso,),
        ).fetchall()

    def delete_backups(self, backup_ids: list[int]) -> None:
        if not backup_ids:
            return
        placeholders = ", ".join("?" for _ in backup_ids)
        self.connection.execute(
            f"DELETE FROM article_backups WHERE id IN ({placeholders})",
            backup_ids,
        )
        self.connection.commit()

    def fetch_articles_by_date(self, date_string: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT *
            FROM articles
            WHERE substr(COALESCE(published_at, first_seen_at), 1, 10) = ?
            ORDER BY published_at DESC, id DESC
            """,
            (date_string,),
        ).fetchall()
        return [dict(row) for row in rows]

    def fetch_latest_run(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT *
            FROM scrape_runs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        return dict(row) if row is not None else None

    def fetch_placements_by_collection(self, collection: str, limit: int | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT
                article_placements.*,
                articles.canonical_url,
                articles.title,
                articles.article_section,
                articles.published_at
            FROM article_placements
            JOIN articles ON articles.id = article_placements.article_id
            WHERE article_placements.collection = ?
            ORDER BY article_placements.collected_at DESC, article_placements.rank ASC
        """
        params: list[Any] = [collection]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def fetch_article_content(
        self,
        *,
        url: str | None = None,
        source_article_id: str | None = None,
    ) -> dict[str, Any] | None:
        if bool(url) == bool(source_article_id):
            raise ValueError("Provide exactly one of url or source_article_id.")

        if url:
            row = self.connection.execute(
                "SELECT * FROM articles WHERE canonical_url = ?",
                (url,),
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT * FROM articles WHERE source_article_id = ?",
                (source_article_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def fetch_total_counts(self) -> dict[str, int]:
        article_count = self.connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        backup_count = self.connection.execute("SELECT COUNT(*) FROM article_backups").fetchone()[0]
        return {
            "article_count": int(article_count),
            "backup_count": int(backup_count),
        }

    def fetch_latest_run_backup_count(self, run_id: int) -> int:
        count = self.connection.execute(
            "SELECT COUNT(*) FROM article_backups WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        return int(count)

    def fetch_latest_run_items(self, run_id: int, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT
                article_placements.collection,
                article_placements.rank,
                article_placements.collected_at,
                articles.title,
                articles.canonical_url,
                articles.article_section,
                articles.published_at
            FROM article_placements
            JOIN articles ON articles.id = article_placements.article_id
            WHERE article_placements.run_id = ?
            ORDER BY
                CASE article_placements.collection
                    WHEN 'head_news' THEN 0
                    WHEN 'latest' THEN 1
                    ELSE 2
                END,
                article_placements.rank ASC,
                article_placements.id ASC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
