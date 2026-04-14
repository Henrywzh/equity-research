from __future__ import annotations

import json
import os
import tempfile
import unittest
from email import message_from_string
from pathlib import Path
from unittest.mock import patch

from daily_macro.notifier import (
    _build_html_body,
    _build_html_articles,
    load_analysis_result,
    send_analysis_summary_email,
    send_release_warning_email,
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
            "synthesis_budget_exhausted_count": 0,
            "degraded_merge_count": 0,
        },
        "totals": {
            "article_count": 3,
            "truncated_article_count": 1,
        },
        "macro_release_digest": {
            "generated_at": "2026-04-04T06:58:00+00:00",
            "window_start": "2026-04-04",
            "window_end": "2026-04-10",
            "fetch_status": "success",
            "source": "FRED",
            "items": [
                {
                    "release_id": 10,
                    "name": "Consumer Price Index",
                    "date": "2026-04-04",
                    "impact": "high",
                    "series_id": "CPIAUCSL",
                    "display_unit": "%",
                    "prior_value": None,
                    "source": "FRED",
                },
                {
                    "release_id": 36,
                    "name": "Producer Price Index",
                    "date": "2026-04-05",
                    "impact": "medium",
                    "series_id": "PPIACO",
                    "display_unit": "%",
                    "prior_value": None,
                    "source": "FRED",
                },
            ],
        },
        "categories": [
            {
                "category": "國際財經",
                "article_count": 2,
                "key_developments": ["International development"],
                "named_entities": [{"name": "伊朗", "type": "country"}],
                "subgroups": [
                    {
                        "title": "Macro risk",
                        "theme_rationale": "These stories grouped around international macro risk.",
                        "article_count": 2,
                        "key_developments": ["International subgroup development"],
                        "named_entities": [{"name": "伊朗", "type": "country"}],
                        "articles": [
                            {
                                "title": "Iran market headline",
                                "canonical_url": "https://example.com/iran",
                                "published_at": "2026-04-04T07:00:00+00:00",
                                "attention_tier": "high",
                                "error": None,
                            },
                            {
                                "title": "Bank contingency update",
                                "canonical_url": "https://example.com/bank",
                                "published_at": "2026-04-04T07:05:00+00:00",
                                "attention_tier": "medium",
                                "error": None,
                            },
                        ],
                    }
                ],
                "diagnostics": {
                    "sub_batch_count": 2,
                    "split_reasons": [],
                    "models_attempted": ["qwen/qwen3-32b"],
                    "estimated_input_tokens_max": 1800,
                    "serialized_request_bytes_max": 7200,
                    "rate_limit_waits": 0,
                    "partial_article_count": 0,
                    "synthesis_wait_seconds_total": 0.0,
                    "synthesis_retry_count": 0,
                    "synthesis_retry_skipped_count": 0,
                    "synthesis_budget_exhausted": False,
                    "degraded_merge_used": False,
                    "degraded_merge_reason": "",
                    "synthesis_merge_depth_max": 0,
                    "model_switches": [],
                },
            },
            {
                "category": "時事脈搏",
                "article_count": 1,
                "key_developments": ["Pulse development"],
                "named_entities": [{"name": "Oracle", "type": "company"}],
                "subgroups": [
                    {
                        "title": "Corporate pulse",
                        "theme_rationale": "Fast company headlines.",
                        "article_count": 1,
                        "key_developments": ["Pulse subgroup development"],
                        "named_entities": [{"name": "Oracle", "type": "company"}],
                        "articles": [
                            {
                                "title": "Oracle pulse headline",
                                "canonical_url": "https://example.com/oracle",
                                "published_at": "2026-04-04T07:10:00+00:00",
                                "attention_tier": "light",
                                "error": None,
                            }
                        ],
                    }
                ],
                "diagnostics": {
                    "sub_batch_count": 1,
                    "split_reasons": [],
                    "models_attempted": ["qwen/qwen3-32b"],
                    "estimated_input_tokens_max": 900,
                    "serialized_request_bytes_max": 3500,
                    "rate_limit_waits": 0,
                    "partial_article_count": 0,
                    "synthesis_wait_seconds_total": 0.0,
                    "synthesis_retry_count": 0,
                    "synthesis_retry_skipped_count": 0,
                    "synthesis_budget_exhausted": False,
                    "degraded_merge_used": False,
                    "degraded_merge_reason": "",
                    "synthesis_merge_depth_max": 0,
                    "model_switches": [],
                },
            },
        ],
        "unresolved_articles": [],
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
        self.assertIn("https://example.com/iran", rendered_text)
        self.assertIn("Oracle pulse headline", rendered_text)
        self.assertIn("[HIGH]", rendered_text)
        self.assertIn("UPCOMING MACRO RELEASES", rendered_text)
        self.assertIn("Today (Apr 4): Consumer Price Index [HIGH]", rendered_text)
        self.assertIn("Tomorrow (Apr 5): Producer Price Index [MEDIUM]", rendered_text)

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
        report["unresolved_articles"] = [
            {
                "category": "國際財經",
                "title": "Bank contingency update",
                "canonical_url": "https://example.com/bank",
                "source_article_id": "bank-1",
                "attention_tier": "high",
                "theme": "macro",
                "must_keep": True,
                "error_classification": "incomplete_model_output",
                "error": "Model response omitted this article from the category batch.",
                "model_used": "qwen/qwen3-32b",
                "delayed_retry_attempted": True,
                "delayed_retry_model_chain": ["llama-3.1-8b-instant", "openai/gpt-oss-20b"],
                "delayed_retry_final_model": "openai/gpt-oss-20b",
                "published_at": "2026-04-04T07:05:00+00:00",
            }
        ]

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
        self.assertIn("Unresolved articles", rendered_text)
        self.assertIn("Bank contingency update", rendered_text)
        self.assertIn("delayed retry failed", rendered_text.lower())

    def test_send_analysis_summary_email_accepts_degraded_synthesis_diagnostics(self) -> None:
        report = _success_report()
        report["status"] = "partial"
        report["diagnostics"]["synthesis_budget_exhausted_count"] = 1
        report["diagnostics"]["degraded_merge_count"] = 1
        report["categories"][0]["diagnostics"]["degraded_merge_used"] = True
        report["categories"][0]["diagnostics"]["degraded_merge_reason"] = "synthesis_budget_exhausted"
        report["categories"][0]["diagnostics"]["synthesis_wait_seconds_total"] = 126.0
        report["categories"][0]["diagnostics"]["synthesis_retry_count"] = 2
        report["categories"][0]["diagnostics"]["synthesis_retry_skipped_count"] = 1
        report["categories"][0]["diagnostics"]["synthesis_budget_exhausted"] = True

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
        self.assertIn("Model switch", rendered_text)
        self.assertIn("International subgroup development", rendered_text)

    def test_build_html_body_keeps_run_notes_after_coverage_stats(self) -> None:
        report = _success_report()
        report["daily_stats"] = {
            "total_scraped": 10,
            "analyzed": 3,
            "success_rate": 30.0,
        }

        rendered = _build_html_body(report)

        self.assertIn("Upcoming Macro Releases", rendered)
        self.assertLess(rendered.index("Upcoming Macro Releases"), rendered.index("Market Coverage"))
        self.assertIn("Market Coverage:</b> 3 analyzed / 10 scraped (30.0% success)", rendered)
        self.assertIn("Run notes", rendered)
        self.assertIn("Stored analysis report: /tmp/hkej-news-analysis.json", rendered)
        self.assertIn("Articles: 3 |", rendered)

    def test_build_html_body_omits_release_digest_when_empty(self) -> None:
        report = _success_report()
        report["macro_release_digest"] = {
            "generated_at": "2026-04-04T06:58:00+00:00",
            "window_start": "2026-04-04",
            "window_end": "2026-04-10",
            "fetch_status": "failed",
            "items": [],
            "source": "FRED",
        }

        rendered = _build_html_body(report)

        self.assertNotIn("Upcoming Macro Releases", rendered)

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

    def test_send_release_warning_email_renders_prior_values(self) -> None:
        releases = [
            {
                "release_id": 10,
                "name": "Consumer Price Index",
                "date": "2026-04-05",
                "impact": "high",
                "series_id": "CPIAUCSL",
                "display_unit": "%",
                "prior_value": "3.2%",
                "source": "FRED",
            },
            {
                "release_id": "fomc_2026-04-05_statement",
                "release_key": "fomc_2026-04-05_statement",
                "name": "FOMC Statement Day",
                "date": "2026-04-05",
                "impact": "high",
                "series_id": None,
                "display_unit": "",
                "prior_value": None,
                "source": "Federal Reserve",
                "event_type": "statement_day",
                "is_sep_meeting": False,
            },
        ]

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
            sent, message = send_release_warning_email(releases)

        self.assertTrue(sent)
        self.assertIn("pre-release warning", message.lower())
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
        self.assertIn("Consumer Price Index", rendered_text)
        self.assertIn("3.2% vs prior", rendered_text)
        self.assertIn("Tomorrow (Apr 5)", rendered_text)
        self.assertIn("FOMC Statement Day", rendered_text)
        self.assertIn("statement day", rendered_text.lower())
        self.assertNotIn("n/a vs prior", rendered_text)

    def test_build_html_articles_sorts_iso_timestamps_newest_first_within_tier(self) -> None:
        html = _build_html_articles(
            [
                {
                    "title": "Older high",
                    "canonical_url": "https://example.com/older-high",
                    "published_at": "2026-04-04T07:00:00+00:00",
                    "attention_tier": "high",
                },
                {
                    "title": "Newer high",
                    "canonical_url": "https://example.com/newer-high",
                    "published_at": "2026-04-04T07:05:00+00:00",
                    "attention_tier": "high",
                },
                {
                    "title": "Medium item",
                    "canonical_url": "https://example.com/medium",
                    "published_at": "2026-04-04T07:10:00+00:00",
                    "attention_tier": "medium",
                },
            ]
        )

        self.assertLess(html.index("Newer high"), html.index("Older high"))
        self.assertLess(html.index("Older high"), html.index("Medium item"))


if __name__ == "__main__":
    unittest.main()
