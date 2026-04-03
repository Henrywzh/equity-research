from __future__ import annotations

import json
import os
import tempfile
import unittest
from email import message_from_string
from pathlib import Path
from unittest.mock import patch

from youtube_intake.notifier import (
    GMAIL_APP_PASSWORD_ENV,
    GMAIL_RECIPIENT_ENV,
    GMAIL_SENDER_ENV,
    LEGACY_GMAIL_APP_PASSWORD_ENV,
    LEGACY_GMAIL_RECIPIENT_ENV,
    LEGACY_GMAIL_SENDER_ENV,
    _merge_config_file,
    load_run_result,
    send_run_summary_email,
    send_test_email,
)


class NotifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = {
            "status": "success",
            "run_started_at": "2026-04-03T01:00:00+00:00",
            "archived_count": 2,
            "bootstrap_count": 0,
            "transcript_unavailable_count": 1,
            "channels": {},
            "errors": ["meitou-news: transcript unavailable"],
            "new_items": [
                {
                    "archive_path": "/tmp/top3pct.json",
                    "channel_slug": "top3pct",
                    "channel_handle": "@top3pct",
                    "channel_name": "3% 財富覺醒",
                    "video_id": "abc",
                    "title": "First title",
                    "webpage_url": "https://www.youtube.com/watch?v=abc",
                    "published_at": "2026-04-03T00:00:00+00:00",
                    "source_kind": "video",
                    "transcript_status": "fetched",
                    "description_excerpt": "First description",
                },
                {
                    "archive_path": "/tmp/edge.json",
                    "channel_slug": "finding-your-edge",
                    "channel_handle": "@FindingYourEdge",
                    "channel_name": "Finding Your Edge",
                    "video_id": "def",
                    "title": "Second title",
                    "webpage_url": "https://www.youtube.com/watch?v=def",
                    "published_at": "2026-04-03T00:10:00+00:00",
                    "source_kind": "livestream_replay",
                    "transcript_status": "unavailable",
                    "description_excerpt": "Second description",
                },
            ],
        }

    def test_no_new_items_skips_send(self) -> None:
        sent, message = send_run_summary_email({**self.summary, "new_items": [], "archived_count": 0})
        self.assertFalse(sent)
        self.assertIn("No new items", message)

    def test_send_summary_email_sends_single_message(self) -> None:
        with patch.dict(
            "os.environ",
            {
                GMAIL_SENDER_ENV: "sender@example.com",
                GMAIL_APP_PASSWORD_ENV: "app-password",
                GMAIL_RECIPIENT_ENV: "recipient@example.com",
            },
            clear=False,
        ):
            with patch("smtplib.SMTP_SSL") as mock_smtp:
                sent, _message = send_run_summary_email(self.summary)

        self.assertTrue(sent)
        mock_smtp.assert_called_once_with("smtp.gmail.com", 465)
        server = mock_smtp.return_value.__enter__.return_value
        server.login.assert_called_once_with("sender@example.com", "app-password")
        server.sendmail.assert_called_once()

        sent_message = server.sendmail.call_args.args[2]
        parsed = message_from_string(sent_message)
        self.assertIn("2 new item(s)", parsed["Subject"])
        decoded_parts = []
        for part in parsed.walk():
            if part.get_content_maintype() == "multipart":
                continue
            payload = part.get_payload(decode=True)
            if payload is not None:
                decoded_parts.append(payload.decode(part.get_content_charset() or "utf-8"))
        body = "\n".join(decoded_parts)
        self.assertIn("First title", body)
        self.assertIn("Second title", body)
        self.assertIn("Run notes", body)

    def test_missing_credentials_raises_when_send_needed(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("youtube_intake.notifier._load_local_config", return_value=None):
                with self.assertRaises(RuntimeError):
                    send_run_summary_email(self.summary)

    def test_local_config_file_can_supply_legacy_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".config"
            config_path.write_text(
                "\n".join(
                    [
                        f"{LEGACY_GMAIL_SENDER_ENV}=sender@example.com",
                        f"{LEGACY_GMAIL_APP_PASSWORD_ENV}=app-password",
                        f"{LEGACY_GMAIL_RECIPIENT_ENV}=recipient@example.com",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {}, clear=True):
                _merge_config_file(config_path)
                self.assertEqual(os.getenv(LEGACY_GMAIL_SENDER_ENV), "sender@example.com")
                self.assertEqual(os.getenv(LEGACY_GMAIL_APP_PASSWORD_ENV), "app-password")
                self.assertEqual(os.getenv(LEGACY_GMAIL_RECIPIENT_ENV), "recipient@example.com")

    def test_load_run_result_reads_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            result_path.write_text(json.dumps(self.summary), encoding="utf-8")
            loaded = load_run_result(result_path)
        self.assertEqual(loaded["archived_count"], 2)

    def test_test_email_uses_summary_sender(self) -> None:
        with patch.dict(
            "os.environ",
            {
                GMAIL_SENDER_ENV: "sender@example.com",
                GMAIL_APP_PASSWORD_ENV: "app-password",
                GMAIL_RECIPIENT_ENV: "recipient@example.com",
            },
            clear=False,
        ):
            with patch("smtplib.SMTP_SSL") as mock_smtp:
                sent, _message = send_test_email()

        self.assertTrue(sent)
        server = mock_smtp.return_value.__enter__.return_value
        server.sendmail.assert_called_once()
