from __future__ import annotations

import json
import os
import tempfile
import unittest
from email import message_from_string
from pathlib import Path
from unittest.mock import patch

from daily_macro.notifier import (
    load_analysis_result,
    send_analysis_summary_email,
    send_test_email,
)


class _FakeSMTP:
    sent_messages: list[dict[str, str]] = []
    login_calls: list[tuple[str, str]] = []

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def __enter__(self) -> "_FakeSMTP":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def login(self, sender: str, password: str) -> None:
        self.__class__.login_calls.append((sender, password))

    def sendmail(self, sender: str, recipient: str, payload: str) -> None:
        self.__class__.sent_messages.append(
            {
                "sender": sender,
                "recipient": recipient,
                "payload": payload,
            }
        )


def _success_report() -> dict[str, object]:
    return {
        "report_date": "2026-04-04",
        "generated_at": "2026-04-04T07:05:00+00:00",
        "status": "success",
        "model": {
            "provider": "groq",
            "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "fallback_models": ["qwen/qwen3-32b", "llama-3.1-8b-instant"],
        },
        "model_switches": [
            {
                "from_model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "to_model": "qwen/qwen3-32b",
                "reason": "Model meta-llama/llama-4-scout-17b-16e-instruct returned 429 rate_limit_exceeded.",
            }
        ],
        "input": {"article_count": 3, "category_count": 2},
        "diagnostics": {
            "rate_limit_wait_count": 1,
            "rate_limit_wait_seconds_total": 12.0,
            "fallback_switch_count": 1,
            "pre_send_split_count": 1,
            "response_413_split_count": 0,
            "json_repair_retry_count": 0,
            "batch_count": 3,
            "failed_batch_count": 0,
        },
        "totals": {
            "article_count": 3,
            "truncated_article_count": 1,
        },
        "categories": [
            {
                "category": "國際財經",
                "article_count": 2,
                "key_developments": ["International development"],
                "named_entities": [{"name": "伊朗", "type": "country"}],
            },
            {
                "category": "時事脈搏",
                "article_count": 1,
                "key_developments": ["Pulse development"],
                "named_entities": [{"name": "Oracle", "type": "company"}],
            },
        ],
        "output_path": "/tmp/hkej-news-analysis.json",
    }


class NotifierTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSMTP.sent_messages = []
        _FakeSMTP.login_calls = []

    def test_load_analysis_result_reads_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(_success_report(), ensure_ascii=False), encoding="utf-8")
            loaded = load_analysis_result(path)

        self.assertEqual(loaded["report_date"], "2026-04-04")

    def test_send_analysis_summary_email_sends_for_success(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DAILY_MACRO_GMAIL_SENDER": "sender@example.com",
                "DAILY_MACRO_GMAIL_APP_PASSWORD": "app-password",
                "DAILY_MACRO_GMAIL_RECIPIENT": "recipient@example.com",
            },
            clear=False,
        ), patch("daily_macro.notifier._load_local_config", return_value={}), patch(
            "daily_macro.notifier.smtplib.SMTP_SSL", _FakeSMTP
        ):
            sent, message = send_analysis_summary_email(_success_report())

        self.assertTrue(sent)
        self.assertIn("Sent daily macro Gmail summary", message)
        self.assertEqual(_FakeSMTP.login_calls, [("sender@example.com", "app-password")])
        email_payload = message_from_string(_FakeSMTP.sent_messages[0]["payload"])
        self.assertIn("[DAILY MACRO]", email_payload["Subject"])
        rendered = message_from_string(_FakeSMTP.sent_messages[0]["payload"])
        decoded_parts = []
        for part in rendered.walk():
            if part.get_content_maintype() == "multipart":
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            decoded_parts.append(payload.decode(part.get_content_charset() or "utf-8"))
        rendered_text = "\n".join(decoded_parts)
        self.assertIn("International development", rendered_text)
        self.assertIn("Pulse development", rendered_text)
        self.assertIn("Model switch", rendered_text)

    def test_send_analysis_summary_email_sends_partial_with_warning(self) -> None:
        report = _success_report()
        report["status"] = "partial"
        report["errors"] = [
            {
                "type": "article",
                "target": "https://example.com/2",
                "classification": "incomplete_model_output",
                "message": "Model response omitted this article from the category batch.",
            }
        ]
        report["totals"]["failed_article_analyses"] = 1

        with patch.dict(
            os.environ,
            {
                "DAILY_MACRO_GMAIL_SENDER": "sender@example.com",
                "DAILY_MACRO_GMAIL_APP_PASSWORD": "app-password",
                "DAILY_MACRO_GMAIL_RECIPIENT": "recipient@example.com",
            },
            clear=False,
        ), patch("daily_macro.notifier._load_local_config", return_value={}), patch(
            "daily_macro.notifier.smtplib.SMTP_SSL", _FakeSMTP
        ):
            sent, message = send_analysis_summary_email(report)

        self.assertTrue(sent)
        self.assertIn("Sent daily macro Gmail summary", message)
        email_payload = message_from_string(_FakeSMTP.sent_messages[0]["payload"])
        self.assertIn("[PARTIAL]", email_payload["Subject"])

    def test_send_analysis_summary_email_skips_empty_categories(self) -> None:
        report = _success_report()
        report["categories"] = []
        report["totals"]["article_count"] = 0
        sent, message = send_analysis_summary_email(report)
        self.assertFalse(sent)
        self.assertIn("no analyzable content", message.lower())

    def test_send_analysis_summary_email_requires_credentials(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("daily_macro.notifier._load_local_config", return_value={}):
            with self.assertRaises(RuntimeError):
                send_analysis_summary_email(_success_report())

    def test_send_test_email_uses_same_delivery_path(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DAILY_MACRO_GMAIL_SENDER": "sender@example.com",
                "DAILY_MACRO_GMAIL_APP_PASSWORD": "app-password",
                "DAILY_MACRO_GMAIL_RECIPIENT": "recipient@example.com",
            },
            clear=False,
        ), patch("daily_macro.notifier._load_local_config", return_value={}), patch(
            "daily_macro.notifier.smtplib.SMTP_SSL", _FakeSMTP
        ):
            sent, message = send_test_email()

        self.assertTrue(sent)
        self.assertIn("Sent daily macro Gmail summary", message)
        self.assertEqual(len(_FakeSMTP.sent_messages), 1)


if __name__ == "__main__":
    unittest.main()
