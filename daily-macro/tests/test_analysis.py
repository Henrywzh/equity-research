from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

import daily_macro.analysis as analysis_module
from daily_macro.analysis import DEFAULT_INPUT_BUDGET_TOKENS, RateLimitGovernor, run_analysis, select_content_for_analysis
from daily_macro.models import ArticleDetails
from daily_macro.storage import Storage


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.request = type("Request", (), {"body": ""})()

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class _FakeGroqSession:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = responses
        self.calls = 0
        self.models_used: list[str] = []

    def post(self, url: str, json: dict, timeout: int) -> _FakeResponse:
        response = self._responses[self.calls]
        self.calls += 1
        self.models_used.append(json["model"])
        return response

    def close(self) -> None:
        return None


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data"
        self.db_path = self.data_dir / "news.sqlite"
        self.storage = Storage(self.db_path)

    def tearDown(self) -> None:
        self.storage.close()
        self.temp_dir.cleanup()

    def _insert_article(
        self,
        *,
        title: str,
        source_article_id: str,
        section: str,
        published_at: str,
        content_text: str,
        source_site: str = "hkej",
    ) -> None:
        article = ArticleDetails(
            canonical_url=f"https://example.com/{source_article_id}",
            title=title,
            source_site=source_site,
            source_article_id=source_article_id,
            article_section=section,
            published_at=published_at,
            summary_snippet="Summary",
            content_text=content_text,
            content_hash=f"hash-{source_article_id}",
        )
        self.storage.upsert_article(article, "2026-04-03T00:00:00+00:00")

    def test_select_content_uses_full_text_for_short_articles(self) -> None:
        selected = select_content_for_analysis("a" * 1200)
        self.assertFalse(selected["content_truncated"])
        self.assertEqual(selected["analysis_method"], "full_text")
        self.assertEqual(selected["original_content_length_chars"], 1200)
        self.assertEqual(selected["analyzed_content_length_chars"], 1200)

    def test_select_content_keeps_full_text_when_long_but_within_budget(self) -> None:
        selected = select_content_for_analysis("a" * 2000)
        self.assertFalse(selected["content_truncated"])
        self.assertEqual(selected["analysis_method"], "full_text")
        self.assertEqual(selected["analyzed_content_length_chars"], 2000)

    def test_select_content_truncates_when_budget_is_exceeded(self) -> None:
        selected = select_content_for_analysis("a" * 20000)
        self.assertTrue(selected["content_truncated"])
        self.assertEqual(selected["analysis_method"], "truncated_text")
        self.assertLess(selected["analyzed_content_length_chars"], selected["original_content_length_chars"])
        self.assertIsNotNone(selected["truncation_reason"])

    def test_rate_limit_governor_waits_for_reset(self) -> None:
        slept: list[float] = []
        governor = RateLimitGovernor(time_fn=lambda: 10.0, sleep_fn=lambda seconds: slept.append(seconds))
        governor.record_response(
            "qwen/qwen3-32b",
            _FakeResponse(
                status_code=200,
                headers={
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-reset-requests": "2",
                },
            ),
        )

        governor.before_request("qwen/qwen3-32b", estimated_input_tokens=100)

        self.assertEqual(slept, [2.0])

    def test_run_analysis_returns_empty_report_for_missing_day(self) -> None:
        report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)
        self.assertEqual(report["status"], "empty")
        self.assertEqual(report["totals"]["article_count"], 0)
        self.assertEqual(report["diagnostics"]["batch_count"], 0)
        report_path = self.data_dir / "analyses" / "2026-04-03" / "hkej-news-analysis.json"
        self.assertTrue(report_path.exists())

    def test_run_analysis_batches_by_category_and_preserves_article_scores(self) -> None:
        self._insert_article(
            title="Pulse one",
            source_article_id="1001",
            section="時事脈搏",
            published_at="2026-04-03T08:00:00+08:00",
            content_text="a" * 800,
        )
        self._insert_article(
            title="Pulse two",
            source_article_id="1002",
            section="時事脈搏",
            published_at="2026-04-03T07:00:00+08:00",
            content_text="b" * 18000,
        )
        self._insert_article(
            title="Global one",
            source_article_id="2001",
            section="國際財經",
            published_at="2026-04-03T09:00:00+08:00",
            content_text="c" * 900,
        )

        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "2001",
                                                    "canonical_url": "https://example.com/2001",
                                                    "novelty_score": 7,
                                                    "relevance_score": 8,
                                                    "urgency_score": 6,
                                                    "named_entities": [{"name": "Fed", "type": "institution"}],
                                                    "key_points": ["Global point"],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "key_developments": ["Global development"],
                                            "named_entities": [{"name": "Fed", "type": "institution"}],
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "1001",
                                                    "canonical_url": "https://example.com/1001",
                                                    "novelty_score": 5,
                                                    "relevance_score": 7,
                                                    "urgency_score": 6,
                                                    "named_entities": [{"name": "香港", "type": "country"}],
                                                    "key_points": ["Pulse point one"],
                                                },
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "1002",
                                                    "canonical_url": "https://example.com/1002",
                                                    "novelty_score": 6,
                                                    "relevance_score": 8,
                                                    "urgency_score": 7,
                                                    "named_entities": [{"name": "香港", "type": "country"}],
                                                    "key_points": ["Pulse point two"],
                                                },
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "key_developments": ["Pulse development"],
                                            "named_entities": [{"name": "香港", "type": "country"}],
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["input"]["category_count"], 2)
        self.assertEqual(report["totals"]["article_count"], 3)
        self.assertEqual(session.calls, 5)
        self.assertEqual(session.models_used, ["qwen/qwen3-32b"] * 5)
        self.assertEqual(report["diagnostics"]["batch_count"], 5)

        pulse = next(category for category in report["categories"] if category["category"] == "時事脈搏")
        self.assertEqual(pulse["status"], "success")
        self.assertEqual(pulse["sub_batch_count"], 2)
        self.assertTrue(any(article["content_truncated"] for article in pulse["articles"]))
        self.assertTrue(all(article["model_used"] == "qwen/qwen3-32b" for article in pulse["articles"]))
        self.assertIn("estimated_input_tokens_max", pulse["diagnostics"])
        self.assertIn("serialized_request_bytes_max", pulse["diagnostics"])

    def test_run_analysis_retries_invalid_json_with_repair_prompt(self) -> None:
        self._insert_article(
            title="Pulse one",
            source_article_id="1001",
            section="時事脈搏",
            published_at="2026-04-03T08:00:00+08:00",
            content_text="a" * 800,
        )

        session = _FakeGroqSession(
            [
                _FakeResponse(200, payload={"choices": [{"message": {"content": "not-json"}}]}),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "1001",
                                                    "canonical_url": "https://example.com/1001",
                                                    "novelty_score": 5,
                                                    "relevance_score": 7,
                                                    "urgency_score": 6,
                                                    "named_entities": [{"name": "香港", "type": "country"}],
                                                    "key_points": ["Pulse point one"],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "key_developments": ["Pulse development"],
                                            "named_entities": [{"name": "香港", "type": "country"}],
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        self.assertEqual(report["status"], "success")
        self.assertEqual(session.calls, 3)
        self.assertEqual(report["categories"][0]["articles"][0]["key_points"][0], "Pulse point one")
        self.assertEqual(report["diagnostics"]["json_repair_retry_count"], 1)

    def test_run_analysis_salvages_omitted_articles_from_missing_subset(self) -> None:
        self._insert_article(
            title="Global one",
            source_article_id="2001",
            section="國際財經",
            published_at="2026-04-03T09:00:00+08:00",
            content_text="a" * 900,
        )
        self._insert_article(
            title="Global two",
            source_article_id="2002",
            section="國際財經",
            published_at="2026-04-03T08:00:00+08:00",
            content_text="b" * 900,
        )

        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "2001",
                                                    "canonical_url": "https://example.com/2001",
                                                    "novelty_score": 7,
                                                    "relevance_score": 8,
                                                    "urgency_score": 6,
                                                    "named_entities": [],
                                                    "key_points": ["Global point one"],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "2002",
                                                    "canonical_url": "https://example.com/2002",
                                                    "novelty_score": 6,
                                                    "relevance_score": 7,
                                                    "urgency_score": 5,
                                                    "named_entities": [],
                                                    "key_points": ["Global point two"],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {"key_developments": ["Global development"], "named_entities": []},
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        category = report["categories"][0]
        self.assertEqual(report["status"], "success")
        self.assertEqual(category["status"], "success")
        self.assertEqual(len(category["articles"]), 2)
        self.assertTrue(all(not article["error"] for article in category["articles"]))
        self.assertEqual(session.calls, 3)

    def test_run_analysis_retries_smaller_batches_after_invalid_json_failure(self) -> None:
        self._insert_article(
            title="Pulse one",
            source_article_id="1001",
            section="時事脈搏",
            published_at="2026-04-03T08:00:00+08:00",
            content_text="a" * 800,
        )
        self._insert_article(
            title="Pulse two",
            source_article_id="1002",
            section="時事脈搏",
            published_at="2026-04-03T07:00:00+08:00",
            content_text="b" * 800,
        )

        session = _FakeGroqSession(
            [
                _FakeResponse(200, payload={"choices": [{"message": {"content": "not-json"}}]}),
                _FakeResponse(200, payload={"choices": [{"message": {"content": "still-not-json"}}]}),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "1001",
                                                    "canonical_url": "https://example.com/1001",
                                                    "novelty_score": 5,
                                                    "relevance_score": 7,
                                                    "urgency_score": 6,
                                                    "named_entities": [],
                                                    "key_points": ["Pulse point one"],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "1002",
                                                    "canonical_url": "https://example.com/1002",
                                                    "novelty_score": 6,
                                                    "relevance_score": 8,
                                                    "urgency_score": 7,
                                                    "named_entities": [],
                                                    "key_points": ["Pulse point two"],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {"key_developments": ["Pulse development"], "named_entities": []},
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["categories"][0]["status"], "success")
        self.assertEqual(len(report["categories"][0]["articles"]), 2)
        self.assertEqual(session.calls, 5)

    def test_run_analysis_records_pre_send_split_diagnostics(self) -> None:
        self._insert_article(
            title="Pulse one",
            source_article_id="1001",
            section="時事脈搏",
            published_at="2026-04-03T08:00:00+08:00",
            content_text="a" * 800,
        )
        self._insert_article(
            title="Pulse two",
            source_article_id="1002",
            section="時事脈搏",
            published_at="2026-04-03T07:00:00+08:00",
            content_text="b" * 800,
        )

        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "1001",
                                                    "canonical_url": "https://example.com/1001",
                                                    "novelty_score": 5,
                                                    "relevance_score": 7,
                                                    "urgency_score": 6,
                                                    "named_entities": [],
                                                    "key_points": ["Pulse point one"],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "1002",
                                                    "canonical_url": "https://example.com/1002",
                                                    "novelty_score": 6,
                                                    "relevance_score": 8,
                                                    "urgency_score": 7,
                                                    "named_entities": [],
                                                    "key_points": ["Pulse point two"],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({"key_developments": ["Pulse development"], "named_entities": []})
                                }
                            }
                        ]
                    },
                ),
            ]
        )

        original_estimator = analysis_module._estimate_batch_request_tokens

        def fake_estimator(category_name: str, batch_articles: list[dict[str, object]]) -> int:
            if len(batch_articles) > 1:
                return DEFAULT_INPUT_BUDGET_TOKENS + 1
            return original_estimator(category_name, batch_articles)

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ), patch("daily_macro.analysis._estimate_batch_request_tokens", side_effect=fake_estimator):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        self.assertEqual(report["diagnostics"]["pre_send_split_count"], 1)
        self.assertIn("pre_send_budget", report["categories"][0]["diagnostics"]["split_reasons"])

    def test_run_analysis_records_rate_limit_wait_diagnostics(self) -> None:
        self._insert_article(
            title="Pulse one",
            source_article_id="1001",
            section="時事脈搏",
            published_at="2026-04-03T08:00:00+08:00",
            content_text="a" * 800,
        )

        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "1001",
                                                    "canonical_url": "https://example.com/1001",
                                                    "novelty_score": 5,
                                                    "relevance_score": 7,
                                                    "urgency_score": 6,
                                                    "named_entities": [],
                                                    "key_points": ["Pulse point one"],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                    headers={
                        "x-ratelimit-remaining-requests": "0",
                        "x-ratelimit-reset-requests": "2",
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({"key_developments": ["Pulse development"], "named_entities": []})
                                }
                            }
                        ]
                    },
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ), patch(
            "daily_macro.analysis.RateLimitGovernor",
            side_effect=lambda: RateLimitGovernor(time_fn=lambda: 10.0, sleep_fn=lambda _seconds: None),
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        self.assertEqual(report["diagnostics"]["rate_limit_wait_count"], 1)
        self.assertEqual(report["diagnostics"]["rate_limit_wait_seconds_total"], 2.0)
        self.assertEqual(report["categories"][0]["diagnostics"]["rate_limit_waits"], 1)

    def test_run_analysis_splits_after_413_and_merges_sub_batches(self) -> None:
        for index in range(4):
            self._insert_article(
                title=f"Pulse {index}",
                source_article_id=f"10{index}",
                section="時事脈搏",
                published_at=f"2026-04-03T0{8-index}:00:00+08:00",
                content_text="a" * 600,
            )

        session = _FakeGroqSession(
            [
                _FakeResponse(413),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "100",
                                                    "canonical_url": "https://example.com/100",
                                                    "novelty_score": 5,
                                                    "relevance_score": 6,
                                                    "urgency_score": 5,
                                                    "named_entities": [],
                                                    "key_points": ["P0"],
                                                },
                                                {
                                                    "source_article_id": "101",
                                                    "canonical_url": "https://example.com/101",
                                                    "novelty_score": 5,
                                                    "relevance_score": 6,
                                                    "urgency_score": 5,
                                                    "named_entities": [],
                                                    "key_points": ["P1"],
                                                },
                                            ]
                                        }
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "102",
                                                    "canonical_url": "https://example.com/102",
                                                    "novelty_score": 5,
                                                    "relevance_score": 6,
                                                    "urgency_score": 5,
                                                    "named_entities": [],
                                                    "key_points": ["P2"],
                                                },
                                                {
                                                    "source_article_id": "103",
                                                    "canonical_url": "https://example.com/103",
                                                    "novelty_score": 5,
                                                    "relevance_score": 6,
                                                    "urgency_score": 5,
                                                    "named_entities": [],
                                                    "key_points": ["P3"],
                                                },
                                            ]
                                        }
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({"key_developments": ["Merged"], "named_entities": []})
                                }
                            }
                        ]
                    },
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        category = report["categories"][0]
        self.assertEqual(category["sub_batch_count"], 2)
        self.assertEqual(len(category["articles"]), 4)
        self.assertEqual(session.calls, 4)
        self.assertEqual(report["diagnostics"]["response_413_split_count"], 1)
        self.assertIn("response_413", category["diagnostics"]["split_reasons"])

    def test_run_analysis_switches_to_fallback_for_rest_of_run_after_qwen_429(self) -> None:
        self._insert_article(
            title="China one",
            source_article_id="3001",
            section="中國財經",
            published_at="2026-04-03T08:00:00+08:00",
            content_text="a" * 800,
        )
        self._insert_article(
            title="Global one",
            source_article_id="4001",
            section="國際財經",
            published_at="2026-04-03T09:00:00+08:00",
            content_text="b" * 900,
        )

        session = _FakeGroqSession(
            [
                _FakeResponse(429, headers={"Retry-After": "0"}),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "3001",
                                                    "canonical_url": "https://example.com/3001",
                                                    "novelty_score": 6,
                                                    "relevance_score": 8,
                                                    "urgency_score": 7,
                                                    "named_entities": [{"name": "小米", "type": "company"}],
                                                    "key_points": ["China point"],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({"key_developments": ["China development"], "named_entities": []})
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "articles": [
                                                {
                                                    "source_article_id": "4001",
                                                    "canonical_url": "https://example.com/4001",
                                                    "novelty_score": 7,
                                                    "relevance_score": 8,
                                                    "urgency_score": 6,
                                                    "named_entities": [{"name": "Fed", "type": "institution"}],
                                                    "key_points": ["Global point"],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({"key_developments": ["Global development"], "named_entities": []})
                                }
                            }
                        ]
                    },
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        self.assertEqual(report["status"], "success")
        self.assertEqual(
            session.models_used,
            [
                "qwen/qwen3-32b",
                "llama-3.1-8b-instant",
                "llama-3.1-8b-instant",
                "llama-3.1-8b-instant",
                "llama-3.1-8b-instant",
            ],
        )
        self.assertEqual(len(report["model_switches"]), 1)
        self.assertEqual(report["model_switches"][0]["to_model"], "llama-3.1-8b-instant")
        self.assertEqual(report["diagnostics"]["fallback_switch_count"], 1)
        self.assertIn("llama-3.1-8b-instant", report["categories"][0]["diagnostics"]["models_attempted"])


if __name__ == "__main__":
    unittest.main()
