from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from daily_macro.cli import main
from daily_macro.storage import Storage


class CliTests(unittest.TestCase):
    def test_smoke_json_output(self) -> None:
        with patch(
            "daily_macro.cli.run_smoke",
            return_value={
                "head_news_count": 5,
                "latest_count": 12,
                "head_titles": ["頭條主新聞"],
                "latest_first_title": "最新新聞一",
            },
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["smoke", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["head_news_count"], 5)
        self.assertEqual(payload["latest_first_title"], "最新新聞一")

    def test_scrape_command_passes_paths(self) -> None:
        with patch(
            "daily_macro.cli.run_scrape",
            return_value={
                "run_id": 1,
                "status": "success",
                "article_count": 2,
                "placement_count": 7,
                "errors": [],
            },
        ) as mock_run_scrape:
            exit_code = main(["scrape", "--data-dir", "/tmp/data", "--db-path", "/tmp/data/news.sqlite"])

        self.assertEqual(exit_code, 0)
        mock_run_scrape.assert_called_once_with(data_dir="/tmp/data", db_path="/tmp/data/news.sqlite")

    def test_inspect_json_output(self) -> None:
        with patch(
            "daily_macro.cli.inspect_latest_run",
            return_value={
                "latest_run": {"id": 3, "status": "success", "article_count": 10, "placement_count": 10, "backup_count": 10},
                "totals": {"article_count": 15, "backup_count": 15},
                "recent_items": [{"collection": "head_news", "rank": 1, "title": "Top story", "canonical_url": "https://example.com"}],
            },
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["inspect", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["latest_run"]["id"], 3)
        self.assertEqual(payload["recent_items"][0]["title"], "Top story")

    def test_inspect_default_returns_zero_with_empty_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/news.sqlite"
            storage = Storage(db_path)
            storage.close()

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["inspect", "--db-path", db_path])

        self.assertEqual(exit_code, 0)
        self.assertIn("No scrape runs found.", buffer.getvalue())

    def test_query_date_json_output(self) -> None:
        with patch(
            "daily_macro.cli.run_query_command",
            return_value={"mode": "date", "query": "2026-04-03", "items": [{"title": "A"}]},
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["query", "date", "2026-04-03", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["mode"], "date")
        self.assertEqual(payload["items"][0]["title"], "A")

    def test_query_article_default_output(self) -> None:
        with patch(
            "daily_macro.cli.run_query_command",
            return_value={
                "mode": "article",
                "item": {
                    "title": "Article title",
                    "article_section": "時事脈搏",
                    "published_at": "2026-04-03T08:00:00+08:00",
                    "canonical_url": "https://example.com/article",
                    "source_article_id": "1234",
                    "summary_snippet": "Summary",
                    "content_text": "Body text",
                    "latest_backup": {"relative_path": "2026/04/03/run_1/article-1234.json"},
                },
            },
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["query", "article", "--id", "1234"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Article title", buffer.getvalue())
        self.assertIn("Latest backup:", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
