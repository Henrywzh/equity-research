from __future__ import annotations

import tempfile
import unittest
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from daily_macro.models import ArticleDetails, PlacementCandidate
from daily_macro.pipeline import cleanup_old_snapshots, run_scrape
from daily_macro.storage import Storage


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data"
        self.backup_dir = self.data_dir / "article_backups"
        self.db_path = self.data_dir / "news.sqlite"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.storage = Storage(self.db_path)

    def tearDown(self) -> None:
        self.storage.close()
        self.temp_dir.cleanup()

    def _build_article(self, url: str, summary: str | None = None) -> ArticleDetails:
        return ArticleDetails(
            canonical_url=url,
            title="Example Article",
            source_site="hkej",
            source_article_id="1234",
            article_section="時事脈搏",
            published_at="2026-04-03T08:00:00+08:00",
            summary_snippet=summary,
            content_text="content body",
            content_hash="hash-1",
            extraction_method="jsonld",
            malformed_jsonld_recovered=False,
        )

    def test_upsert_article_keeps_single_row(self) -> None:
        seen_at = "2026-04-03T00:00:00+00:00"
        first_id = self.storage.upsert_article(
            self._build_article("https://www.hkej.com/instantnews/current/article/1234/story", summary="first"),
            seen_at,
        )
        second_id = self.storage.upsert_article(
            self._build_article("https://www.hkej.com/instantnews/current/article/1234/story", summary="updated"),
            "2026-04-03T01:00:00+00:00",
        )

        self.assertEqual(first_id, second_id)
        row = self.storage.fetch_article_content(
            url="https://www.hkej.com/instantnews/current/article/1234/story"
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["summary_snippet"], "updated")

    def test_same_article_can_have_multiple_collections(self) -> None:
        run_id = self.storage.start_run("2026-04-03T00:00:00+00:00")
        article_id = self.storage.upsert_article(
            self._build_article("https://www.hkej.com/instantnews/current/article/1234/story"),
            "2026-04-03T00:00:00+00:00",
        )

        self.storage.record_placement(
            article_id,
            run_id,
            PlacementCandidate(
                collection="head_news",
                rank=1,
                title="Hero",
                url="https://www.hkej.com/instantnews/current/article/1234/story",
                homepage_section="時事脈搏",
            ),
            "2026-04-03T00:00:00+00:00",
        )
        self.storage.record_placement(
            article_id,
            run_id,
            PlacementCandidate(
                collection="latest",
                rank=1,
                title="Latest",
                url="https://www.hkej.com/instantnews/current/article/1234/story",
                homepage_section="時事脈搏",
                summary_snippet="摘要",
            ),
            "2026-04-03T00:00:00+00:00",
        )

        head_rows = self.storage.fetch_placements_by_collection("head_news")
        latest_rows = self.storage.fetch_placements_by_collection("latest")
        self.assertEqual(len(head_rows), 1)
        self.assertEqual(len(latest_rows), 1)
        self.assertEqual(head_rows[0]["article_id"], latest_rows[0]["article_id"])

    def test_cleanup_only_removes_old_backups(self) -> None:
        run_id = self.storage.start_run("2026-04-03T00:00:00+00:00")
        article_id = self.storage.upsert_article(
            self._build_article("https://www.hkej.com/instantnews/current/article/1234/story"),
            "2026-04-03T00:00:00+00:00",
        )

        old_rel = Path("2026/02/01/run_1/article-old.json")
        new_rel = Path("2026/04/03/run_1/article-new.json")
        old_path = self.backup_dir / old_rel
        new_path = self.backup_dir / new_rel
        old_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.write_text("old", encoding="utf-8")
        new_path.write_text("new", encoding="utf-8")

        old_time = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        new_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        self.storage.record_backup(
            run_id, "parsed_article_json", str(old_rel), "oldhash", old_time, article_id=article_id
        )
        self.storage.record_backup(
            run_id, "parsed_article_json", str(new_rel), "newhash", new_time, article_id=article_id
        )
        self.storage.close()

        result = cleanup_old_snapshots(data_dir=self.data_dir, db_path=self.db_path, retention_days=30)

        self.assertEqual(result["deleted_backup_rows"], 1)
        self.assertFalse(old_path.exists())
        self.assertTrue(new_path.exists())

    def test_scrape_writes_article_json_backups(self) -> None:
        self.storage.close()

        class FakeAdapter:
            homepage_url = "https://example.com/home"

            def parse_homepage(self, html: str) -> list[PlacementCandidate]:
                return [
                    PlacementCandidate(
                        collection="head_news",
                        rank=1,
                        title="Example Title",
                        url="https://example.com/article/1234",
                        homepage_section="Section A",
                        summary_snippet="Summary A",
                    ),
                    PlacementCandidate(
                        collection="latest",
                        rank=1,
                        title="Example Title",
                        url="https://example.com/article/1234",
                        homepage_section="Section B",
                        summary_snippet="Summary B",
                    ),
                ]

            def parse_latest_page(self, html: str, start_rank: int = 1):
                class Snapshot:
                    active_title = "昨日"
                    items = []

                return Snapshot()

            def parse_article(self, html: str, url: str) -> ArticleDetails:
                return ArticleDetails(
                    canonical_url=url,
                    title="Example Title",
                    source_site="hkej",
                    source_article_id="1234",
                    article_section="時事脈搏",
                    published_at="2026-04-03T08:00:00+08:00",
                    summary_snippet=None,
                    content_text="Body text",
                    content_hash="body-hash",
                    extraction_method="jsonld",
                    malformed_jsonld_recovered=True,
                )

        responses = {
            "https://example.com/home": "<html>home</html>",
            "https://example.com/article/1234": "<html>article</html>",
        }

        from unittest.mock import patch

        with patch(
            "daily_macro.pipeline.fetch_text",
            side_effect=lambda session, url: responses.get(url, "<html>stop</html>"),
        ):
            result = run_scrape(data_dir=self.data_dir, db_path=self.db_path, adapter=FakeAdapter())

        self.assertEqual(result["status"], "success")
        backups = list(self.backup_dir.rglob("*.json"))
        self.assertEqual(len(backups), 1)
        payload = json.loads(backups[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["canonical_url"], "https://example.com/article/1234")
        self.assertEqual(payload["parser_metadata"]["extraction_method"], "jsonld")
        self.assertTrue(payload["parser_metadata"]["malformed_jsonld_recovered"])
        self.assertEqual(len(payload["placements"]), 2)
        self.assertEqual(payload["placements"][0]["collection"], "head_news")

    def test_latest_run_inspect_helpers(self) -> None:
        run_id = self.storage.start_run("2026-04-03T00:00:00+00:00")
        article_id = self.storage.upsert_article(
            self._build_article("https://www.hkej.com/instantnews/current/article/1234/story"),
            "2026-04-03T00:00:00+00:00",
        )
        self.storage.record_placement(
            article_id,
            run_id,
            PlacementCandidate(
                collection="head_news",
                rank=1,
                title="Hero",
                url="https://www.hkej.com/instantnews/current/article/1234/story",
                homepage_section="時事脈搏",
            ),
            "2026-04-03T00:00:00+00:00",
        )
        self.storage.record_placement(
            article_id,
            run_id,
            PlacementCandidate(
                collection="latest",
                rank=1,
                title="Latest",
                url="https://www.hkej.com/instantnews/current/article/1234/story",
                homepage_section="時事脈搏",
            ),
            "2026-04-03T00:00:00+00:00",
        )
        self.storage.record_backup(
            run_id,
            "parsed_article_json",
            "2026/04/03/run_1/article-1234.json",
            "hash",
            "2026-04-03T00:00:00+00:00",
            article_id=article_id,
        )
        self.storage.finish_run(
            run_id,
            finished_at="2026-04-03T00:10:00+00:00",
            status="success",
            article_count=1,
            placement_count=2,
        )

        latest_run = self.storage.fetch_latest_run()
        totals = self.storage.fetch_total_counts()
        run_backup_count = self.storage.fetch_latest_run_backup_count(run_id)
        recent_items = self.storage.fetch_latest_run_items(run_id, limit=5)

        self.assertIsNotNone(latest_run)
        self.assertEqual(latest_run["id"], run_id)
        self.assertEqual(totals["article_count"], 1)
        self.assertEqual(totals["backup_count"], 1)
        self.assertEqual(run_backup_count, 1)
        self.assertEqual(len(recent_items), 2)
        self.assertEqual(recent_items[0]["collection"], "head_news")
        self.assertEqual(recent_items[0]["rank"], 1)
        self.assertEqual(recent_items[1]["collection"], "latest")

    def test_query_helpers_support_date_search_and_article_lookup(self) -> None:
        seen_at = "2026-04-03T00:00:00+00:00"
        run_id = self.storage.start_run(seen_at)
        article = self._build_article("https://www.hkej.com/instantnews/current/article/1234/story", summary="macro theme")
        article.title = "Iran market update"
        article.content_text = "This article discusses macro risk and Iran."
        article_id = self.storage.upsert_article(article, seen_at)
        self.storage.record_backup(
            run_id,
            "parsed_article_json",
            "2026/04/03/run_1/article-1234.json",
            "hash",
            seen_at,
            article_id=article_id,
        )

        by_date = self.storage.fetch_articles_by_date_with_limit("2026-04-03", limit=5)
        matches = self.storage.search_articles("Iran", limit=5)
        fetched = self.storage.fetch_article_with_latest_backup(source_article_id="1234")

        self.assertEqual(len(by_date), 1)
        self.assertEqual(len(matches), 1)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["title"], "Iran market update")
        self.assertIsNotNone(fetched["latest_backup"])


if __name__ == "__main__":
    unittest.main()
