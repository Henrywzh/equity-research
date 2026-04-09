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
    load_analysis_result,
    send_analysis_summary_email,
    send_test_email,
)


class NotifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = {
            "status": "success",
            "run_started_at": "2026-04-03T01:00:00+00:00",
            "analysis_started_at": "2026-04-03T01:02:00+00:00",
            "analysis_model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "analysis_models_used": [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama-3.3-70b-versatile",
            ],
            "fallback_activated": True,
            "rate_limit_events": [{"event_type": "http_retry"}],
            "videos": [
                {
                    "video_id": "abc",
                    "channel_slug": "top3pct",
                    "channel_name": "3% 財富覺醒",
                    "title": "First title",
                    "webpage_url": "https://www.youtube.com/watch?v=abc",
                    "published_at": "2026-04-03T00:00:00+00:00",
                    "source_kind": "video",
                    "source_basis": "transcript",
                    "executive_summary": "Speaker says the selloff still looks tactical rather than structural.",
                    "notable_claims": ["Dealers are still cushioning downside."],
                    "notable_opinions": ["The host would keep buying staged pullbacks."],
                    "key_timestamps": [
                        {
                            "timestamp": "00:02:14",
                            "label": "Support thesis",
                            "snippet": "Explains why panic is not turning into a breakdown.",
                            "why_it_matters": "Sets the short-term bullish framing.",
                        }
                    ],
                    "topic_tags": [{"tag": "market structure", "score": 93}],
                    "confidence": 0.82,
                },
                {
                    "video_id": "def",
                    "channel_slug": "finding-your-edge",
                    "channel_name": "Finding Your Edge",
                    "title": "Second title",
                    "webpage_url": "https://www.youtube.com/watch?v=def",
                    "published_at": "2026-04-03T00:10:00+00:00",
                    "source_kind": "livestream_replay",
                    "source_basis": "metadata_only",
                    "executive_summary": "Description suggests the speaker expects a rebound to fade near resistance.",
                    "notable_claims": ["A second panic leg could create a better entry."],
                    "notable_opinions": ["The host appears cautious on chasing upside."],
                    "key_timestamps": [],
                    "topic_tags": [{"tag": "resistance", "score": 77}],
                    "confidence": 0.44,
                },
            ],
            "channels": {
                "top3pct": {
                    "channel_name": "3% 財富覺醒",
                    "video_count": 1,
                    "summary": "Bullish near-term framing with market-structure focus.",
                    "top_topics": ["market structure"],
                }
            },
            "run_summary": {
                "overall_day_summary": "Today’s videos leaned constructive but still warned about a later reset.",
                "cross_video_themes": ["buying fear", "resistance overhead"],
                "agreements": ["Pullbacks are still being framed as tactical opportunities."],
                "disagreements": [],
                "top_claims_worth_watching": ["Watch whether markets stall near major resistance."],
                "run_notes": ["meitou-news: transcript unavailable"],
            },
            "retryable_failure_count": 1,
            "non_retryable_failure_count": 0,
            "queued_retry_count": 1,
            "failed_items": [
                {
                    "channel_slug": "meitou-news",
                    "video_id": "retry-me",
                    "retryable": True,
                    "failure_kind": "rate_limit",
                    "next_retry_after": "2026-04-03T02:10:00+00:00",
                }
            ],
            "errors": [],
        }

    def test_no_videos_skips_send(self) -> None:
        sent, message = send_analysis_summary_email(
            {
                **self.summary,
                "videos": [],
                "retryable_failure_count": 0,
                "non_retryable_failure_count": 0,
                "queued_retry_count": 0,
                "failed_items": [],
            }
        )
        self.assertFalse(sent)
        self.assertIn("No analyzed items", message)

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
                sent, _message = send_analysis_summary_email(self.summary)

        self.assertTrue(sent)
        mock_smtp.assert_called_once_with("smtp.gmail.com", 465)
        server = mock_smtp.return_value.__enter__.return_value
        server.login.assert_called_once_with("sender@example.com", "app-password")
        server.sendmail.assert_called_once()

        sent_message = server.sendmail.call_args.args[2]
        parsed = message_from_string(sent_message)
        self.assertIn("2 video(s)", parsed["Subject"])
        decoded_parts = []
        for part in parsed.walk():
            if part.get_content_maintype() == "multipart":
                continue
            payload = part.get_payload(decode=True)
            if payload is not None:
                decoded_parts.append(payload.decode(part.get_content_charset() or "utf-8"))
        body = "\n".join(decoded_parts)
        self.assertIn("Top run summary", body)
        self.assertIn("First title", body)
        self.assertIn("Key timestamps", body)
        self.assertIn("metadata only", body)
        self.assertIn("Models used", body)
        self.assertIn("fallback model was activated", body)
        self.assertIn("queued for retry", body)
        self.assertIn("Failed videos", body)
        self.assertIn("meitou-news / retry-me", body)

    def test_retry_only_summary_still_sends(self) -> None:
        retry_summary = {
            **self.summary,
            "videos": [],
            "run_summary": {
                **self.summary["run_summary"],
                "overall_day_summary": "No videos were analyzed in this attempt, but retryable failures were queued for a later replay.",
            },
        }
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
                sent, _message = send_analysis_summary_email(retry_summary)

        self.assertTrue(sent)
        server = mock_smtp.return_value.__enter__.return_value
        sent_message = server.sendmail.call_args.args[2]
        self.assertIn("retry scheduled", sent_message)
        parsed = message_from_string(sent_message)
        decoded_parts = []
        for part in parsed.walk():
            if part.get_content_maintype() == "multipart":
                continue
            payload = part.get_payload(decode=True)
            if payload is not None:
                decoded_parts.append(payload.decode(part.get_content_charset() or "utf-8"))
        body = "\n".join(decoded_parts)
        self.assertIn("meitou-news / retry-me", body)

    def test_missing_credentials_raises_when_send_needed(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("youtube_intake.notifier.load_local_config", return_value=None):
                with self.assertRaises(RuntimeError):
                    send_analysis_summary_email(self.summary)

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

    def test_load_analysis_result_reads_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "analysis-result.json"
            result_path.write_text(json.dumps(self.summary), encoding="utf-8")
            loaded = load_analysis_result(result_path)
        self.assertEqual(len(loaded["videos"]), 2)

    def test_test_email_uses_analyst_sender(self) -> None:
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
