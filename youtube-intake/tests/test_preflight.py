from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from youtube_intake.preflight import run_preflight


class PreflightTests(unittest.TestCase):
    def test_preflight_fails_when_required_env_missing(self) -> None:
        with patch("youtube_intake.preflight.load_local_config", return_value=None), patch.dict(os.environ, {}, clear=True):
            payload = run_preflight()

        self.assertEqual(payload["status"], "failed")
        self.assertIn("GROQ_API_KEY", payload["missing_required"])
        self.assertFalse(payload["optional_env"]["YOUTUBE_INTAKE_YT_COOKIES"])

    def test_preflight_succeeds_when_required_env_present(self) -> None:
        with patch("youtube_intake.preflight.load_local_config", return_value=None), patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "groq",
                "GMAIL_SENDER": "sender@example.com",
                "GMAIL_APP_PASSWORD": "app-password",
                "GMAIL_RECIPIENT": "recipient@example.com",
                "YOUTUBE_INTAKE_YT_COOKIES": "# Netscape HTTP Cookie File",
            },
            clear=True,
        ):
            payload = run_preflight()

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["missing_required"], [])
        self.assertTrue(payload["optional_env"]["YOUTUBE_INTAKE_YT_COOKIES"])


if __name__ == "__main__":
    unittest.main()
