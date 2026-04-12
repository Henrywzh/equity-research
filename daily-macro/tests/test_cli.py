from __future__ import annotations

import io
import json
import logging
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

    def test_analyze_json_output(self) -> None:
        with patch(
            "daily_macro.cli.run_analysis",
            return_value={
                "report_date": "2026-04-03",
                "status": "success",
                "model": {
                    "provider": "groq",
                    "primary_model": "qwen/qwen3-32b",
                    "fallback_model": "llama-3.1-8b-instant",
                },
                "model_switches": [],
                "input": {"article_count": 2, "category_count": 1},
                "totals": {
                    "article_count": 2,
                    "full_text_article_count": 2,
                    "truncated_article_count": 0,
                },
                "output_path": "/tmp/report.json",
            },
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["analyze", "today", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["report_date"], "2026-04-03")
        self.assertEqual(payload["model"]["primary_model"], "qwen/qwen3-32b")

    def test_analyze_default_output(self) -> None:
        with patch(
            "daily_macro.cli.run_analysis",
            return_value={
                "report_date": "2026-04-03",
                "status": "partial",
                "model": {
                    "provider": "groq",
                    "primary_model": "qwen/qwen3-32b",
                    "fallback_model": "llama-3.1-8b-instant",
                },
                "model_switches": [{"from_model": "qwen/qwen3-32b", "to_model": "llama-3.1-8b-instant"}],
                "input": {"article_count": 3, "category_count": 2},
                "diagnostics": {
                    "rate_limit_wait_count": 1,
                    "rate_limit_wait_seconds_total": 12.5,
                    "pre_send_split_count": 2,
                    "response_413_split_count": 1,
                },
                "totals": {
                    "article_count": 3,
                    "full_text_article_count": 2,
                    "truncated_article_count": 1,
                },
                "output_path": "/tmp/report.json",
                "cached": False,
            },
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["analyze", "today"])

        self.assertEqual(exit_code, 0)
        output = buffer.getvalue()
        self.assertIn("Report date: 2026-04-03", output)
        self.assertIn("Truncated articles: 1", output)
        self.assertIn("Model switches: 1", output)
        self.assertIn("Rate-limit waits: 1 (12.5s)", output)
        self.assertIn("Batch splits: pre-send=2, after-413=1", output)

    def test_analyze_verbose_sets_analysis_logger_to_debug(self) -> None:
        logger = logging.getLogger("daily_macro.analysis")
        previous_level = logger.level
        try:
            logger.setLevel(logging.NOTSET)
            with patch(
                "daily_macro.cli.run_analysis",
                return_value={
                    "report_date": "2026-04-03",
                    "status": "success",
                    "model": {
                        "provider": "groq",
                        "primary_model": "qwen/qwen3-32b",
                        "fallback_model": "llama-3.1-8b-instant",
                    },
                    "model_switches": [],
                    "input": {"article_count": 1, "category_count": 1},
                    "diagnostics": {},
                    "totals": {
                        "article_count": 1,
                        "full_text_article_count": 1,
                        "truncated_article_count": 0,
                    },
                    "output_path": "/tmp/report.json",
                },
            ):
                exit_code = main(["analyze", "today", "--verbose"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(logger.level, logging.DEBUG)
        finally:
            logger.setLevel(previous_level)

    def test_build_site_json_output(self) -> None:
        with patch(
            "daily_macro.cli.build_site",
            return_value={
                "status": "success",
                "report_count": 3,
                "latest_report_date": "2026-04-09",
                "output_dir": "/tmp/site",
                "generated_files": ["/tmp/site/index.html"],
            },
        ) as mock_build_site:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["build-site", "--json", "--data-dir", "/tmp/data", "--output-dir", "/tmp/site"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["latest_report_date"], "2026-04-09")
        mock_build_site.assert_called_once_with(data_dir="/tmp/data", output_dir="/tmp/site")

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

    def test_notify_command_outputs_json(self) -> None:
        with patch(
            "daily_macro.cli.load_analysis_result",
            return_value={"status": "success"},
        ), patch(
            "daily_macro.cli.send_analysis_summary_email",
            return_value=(True, "Sent daily macro Gmail summary to me@example.com."),
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["notify", "--result-path", "/tmp/report.json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertTrue(payload["sent"])

    def test_test_email_command_outputs_json(self) -> None:
        with patch(
            "daily_macro.cli.send_test_email",
            return_value=(True, "Sent daily macro Gmail summary to me@example.com."),
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["test-email"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertTrue(payload["sent"])

    def test_warn_tomorrow_command_outputs_json(self) -> None:
        with patch(
            "daily_macro.release_calendar.fetch_warning_releases",
            return_value=[
                {
                    "release_id": 10,
                    "name": "Consumer Price Index",
                    "date": "2026-04-05",
                    "impact": "high",
                    "series_id": "CPIAUCSL",
                    "prior_value": "3.2%",
                    "display_unit": "%",
                    "source": "FRED",
                }
            ],
        ), patch(
            "daily_macro.cli.send_release_warning_email",
            return_value=(True, "Sent pre-release warning email to me@example.com."),
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["warn-tomorrow"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertTrue(payload["sent"])
        self.assertEqual(payload["release_count"], 1)

    def test_local_test_runs_scrape_analyze_and_notify(self) -> None:
        with patch(
            "daily_macro.cli.run_scrape",
            return_value={
                "run_id": 1,
                "status": "success",
                "article_count": 5,
                "placement_count": 12,
                "errors": [],
            },
        ) as mock_scrape, patch(
            "daily_macro.cli.run_analysis",
            return_value={
                "report_date": "2026-04-04",
                "status": "partial",
                "input": {"category_count": 3},
                "totals": {"article_count": 5},
                "output_path": "/tmp/report.json",
                "cached": False,
            },
        ) as mock_analyze, patch(
            "daily_macro.cli.send_analysis_summary_email",
            return_value=(True, "Sent daily macro Gmail summary to me@example.com."),
        ) as mock_notify:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["local-test", "--json", "--data-dir", "/tmp/data", "--db-path", "/tmp/data/news.sqlite"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["scrape"]["status"], "success")
        self.assertEqual(payload["analysis"]["status"], "partial")
        self.assertTrue(payload["notify"]["sent"])
        mock_scrape.assert_called_once_with(data_dir="/tmp/data", db_path="/tmp/data/news.sqlite")
        mock_analyze.assert_called_once_with(
            date_string=None,
            data_dir="/tmp/data",
            db_path="/tmp/data/news.sqlite",
            force=True,
        )
        mock_notify.assert_called_once()

    def test_local_test_can_skip_email(self) -> None:
        with patch(
            "daily_macro.cli.run_scrape",
            return_value={
                "run_id": 1,
                "status": "success",
                "article_count": 2,
                "placement_count": 4,
                "errors": [],
            },
        ), patch(
            "daily_macro.cli.run_analysis",
            return_value={
                "report_date": "2026-04-04",
                "status": "success",
                "input": {"category_count": 1},
                "totals": {"article_count": 2},
                "output_path": "/tmp/report.json",
                "cached": False,
            },
        ), patch(
            "daily_macro.cli.send_analysis_summary_email",
        ) as mock_notify:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["local-test", "--json", "--skip-email"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertFalse(payload["notify"]["attempted"])
        self.assertFalse(payload["notify"]["sent"])
        mock_notify.assert_not_called()

    def test_local_test_exits_non_zero_and_skips_followups_when_scrape_is_partial(self) -> None:
        with patch(
            "daily_macro.cli.run_scrape",
            return_value={
                "run_id": 1,
                "status": "partial_success",
                "article_count": 4,
                "placement_count": 10,
                "errors": ["https://example.com/a: timeout"],
            },
        ), patch(
            "daily_macro.cli.run_analysis",
        ) as mock_analyze, patch(
            "daily_macro.cli.send_analysis_summary_email",
        ) as mock_notify:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["local-test", "--json"])

        self.assertEqual(exit_code, 1)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["scrape"]["status"], "partial_success")
        self.assertEqual(payload["analysis"]["status"], "skipped")
        self.assertFalse(payload["notify"]["attempted"])
        mock_analyze.assert_not_called()
        mock_notify.assert_not_called()

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
