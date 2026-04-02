from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from daily_macro.cli import main


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


if __name__ == "__main__":
    unittest.main()
