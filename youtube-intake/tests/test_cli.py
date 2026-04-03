from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from youtube_intake.cli import main


class CliTests(unittest.TestCase):
    def test_smoke_prints_json(self) -> None:
        with patch(
            "youtube_intake.cli.run_smoke",
            return_value={"channels": {"top3pct": {"candidate_count": 2, "latest_video_id": "abc"}}},
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["smoke"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["channels"]["top3pct"]["latest_video_id"], "abc")

    def test_run_passes_override_paths(self) -> None:
        with patch(
            "youtube_intake.cli.run_sync",
            return_value={"status": "success", "channels": {}, "errors": []},
        ) as mock_run_sync:
            exit_code = main(["run", "--config-path", "/tmp/config.json", "--state-path", "/tmp/state.json"])

        self.assertEqual(exit_code, 0)
        mock_run_sync.assert_called_once_with(
            config_path="/tmp/config.json",
            state_path="/tmp/state.json",
            data_dir=None,
        )


if __name__ == "__main__":
    unittest.main()
