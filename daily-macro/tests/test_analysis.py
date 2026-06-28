from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

import daily_macro.analysis as analysis_module
from daily_macro.analysis import (
    AnalysisRuntime,
    DEFAULT_INPUT_BUDGET_TOKENS,
    ModelConfig,
    REPORT_SCHEMA_VERSION,
    RateLimitGovernor,
    run_analysis,
    select_content_for_analysis,
)
from daily_macro.llm_client import ModelResolver, LLMTask, load_active_groq_model_ids
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

    def _report_article(
        self,
        *,
        source_article_id: str,
        title: str,
        section: str,
        published_at: str,
        success: bool,
    ) -> dict[str, object]:
        return {
            "source_article_id": source_article_id,
            "title": title,
            "canonical_url": f"https://example.com/{source_article_id}",
            "published_at": published_at,
            "section": section,
            "novelty_score": 6 if success else None,
            "relevance_score": 7 if success else None,
            "urgency_score": 5 if success else None,
            "named_entities": [],
            "key_points": ["Existing point"] if success else [],
            "content_truncated": False,
            "original_content_length_chars": 100,
            "analyzed_content_length_chars": 100,
            "original_content_token_estimate": 25,
            "analyzed_content_token_estimate": 25,
            "truncation_reason": None,
            "analysis_method": "full_text",
            "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
            "error_classification": None if success else "incomplete_model_output",
            "error": None if success else "Model response omitted this article from the category batch.",
        }

    def _write_report_file(self, date_string: str, payload: dict[str, object]) -> Path:
        report_path = self.data_dir / "analyses" / date_string / "hkej-news-analysis.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return report_path

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

    def test_heuristic_attention_router_marks_obvious_macro_story_high(self) -> None:
        routed = analysis_module._heuristic_attention_metadata(
            {
                "title": "聯儲局官員稱利率與通脹前景仍不明朗",
                "summary_snippet": "市場關注聯儲局利率路徑與通脹走勢",
                "article_section": "國際財經",
            }
        )

        self.assertEqual(routed["attention_tier"], "high")
        self.assertEqual(routed["theme"], "macro")
        self.assertTrue(routed["must_keep"])

    def test_merge_attention_results_defaults_missing_items_for_salvage(self) -> None:
        batch = [
            {
                "source_article_id": "1001",
                "canonical_url": "https://example.com/1001",
                "title": "銀行回購股份",
                "summary_snippet": "股份回購與資本配置更新",
                "article_section": "香港財經",
                "published_at": "2026-04-04T08:00:00+08:00",
            },
            {
                "source_article_id": "1002",
                "canonical_url": "https://example.com/1002",
                "title": "一般本地消息",
                "summary_snippet": "社區活動摘要",
                "article_section": "時事脈搏",
                "published_at": "2026-04-04T08:10:00+08:00",
            },
        ]

        merged, missing = analysis_module._merge_attention_results(
            batch,
            {
                "routes": [
                    {
                        "source_article_id": "1001",
                        "canonical_url": "https://example.com/1001",
                        "attention_tier": "high",
                        "theme": "stocks",
                        "reason": "Share buyback headline.",
                        "must_keep": True,
                    }
                ]
            },
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["attention_tier"], "high")
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["source_article_id"], "1002")

    def test_delayed_retry_recovers_high_attention_article_with_final_model(self) -> None:
        prepared_article = analysis_module._prepare_single_article(
            {
                "source_article_id": "9001",
                "title": "聯儲局暗示利率路徑仍偏緊",
                "canonical_url": "https://example.com/9001",
                "published_at": "2026-04-04T08:00:00+08:00",
                "article_section": "國際財經",
                "summary_snippet": "聯儲局與利率前景仍是市場焦點",
                "content_text": "macro " * 200,
            }
        )
        current_result = analysis_module._build_failed_article_result(
            article=prepared_article,
            error_message="Model response omitted this article from the category batch.",
            model_used="qwen/qwen3-32b",
            error_classification="incomplete_model_output",
        )
        groq_session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"articles": []}, ensure_ascii=False)}}]},
                )
            ]
        )
        openai_session = _FakeGroqSession(
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
                                                    "source_article_id": "9001",
                                                    "canonical_url": "https://example.com/9001",
                                                    "novelty_score": 8,
                                                    "relevance_score": 9,
                                                    "urgency_score": 8,
                                                    "named_entities": [{"name": "聯儲局", "type": "institution"}],
                                                    "key_points": ["Rates remain restrictive."],
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                )
            ]
        )
        runtime = AnalysisRuntime(
            session=groq_session,
            governor=RateLimitGovernor(sleep_fn=lambda seconds: None),
            model_chain=[
                ModelConfig("meta-llama/llama-4-scout-17b-16e-instruct"),
                ModelConfig("qwen/qwen3-32b"),
                ModelConfig("llama-3.1-8b-instant"),
            ],
            delayed_retry_final_model=ModelConfig(
                "openai/gpt-oss-20b",
                provider="openai",
                api_url="https://api.openai.com/v1/chat/completions",
                api_key_env="OPENAI_API_KEY",
            ),
            provider_sessions={"openai": openai_session},
        )

        with patch("daily_macro.analysis.time.sleep", lambda seconds: None):
            updated_results, retry_batches = analysis_module._run_delayed_retry_pass(
                runtime,
                "國際財經",
                [prepared_article],
                [current_result],
            )

        self.assertEqual(retry_batches, 1)
        self.assertIsNone(updated_results[0].get("error"))
        self.assertTrue(updated_results[0]["delayed_retry_attempted"])
        self.assertEqual(
            updated_results[0]["delayed_retry_model_chain"],
            ["llama-3.1-8b-instant", "openai/gpt-oss-20b"],
        )
        self.assertEqual(updated_results[0]["model_used"], "openai/gpt-oss-20b")

    def test_delayed_retry_skips_light_attention_articles(self) -> None:
        prepared_article = analysis_module._prepare_single_article(
            {
                "source_article_id": "9002",
                "title": "一般社區消息",
                "canonical_url": "https://example.com/9002",
                "published_at": "2026-04-04T08:00:00+08:00",
                "article_section": "時事脈搏",
                "summary_snippet": "本地一般消息",
                "content_text": "pulse " * 100,
            }
        )
        current_result = analysis_module._build_failed_article_result(
            article=prepared_article,
            error_message="Model response omitted this article from the category batch.",
            model_used="qwen/qwen3-32b",
            error_classification="incomplete_model_output",
        )
        runtime = AnalysisRuntime(
            session=_FakeGroqSession([]),
            governor=RateLimitGovernor(sleep_fn=lambda seconds: None),
            model_chain=[
                ModelConfig("meta-llama/llama-4-scout-17b-16e-instruct"),
                ModelConfig("qwen/qwen3-32b"),
                ModelConfig("llama-3.1-8b-instant"),
            ],
        )

        with patch("daily_macro.analysis.time.sleep", side_effect=AssertionError("light articles should not trigger delayed sleep")):
            updated_results, retry_batches = analysis_module._run_delayed_retry_pass(
                runtime,
                "時事脈搏",
                [prepared_article],
                [current_result],
            )

        self.assertEqual(retry_batches, 0)
        self.assertFalse(updated_results[0]["delayed_retry_attempted"])

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

    def test_select_key_caps_far_future_reset(self) -> None:
        # A far-future reset (e.g. a daily token limit) must not block for hours:
        # select_key sleeps at most max_wait_seconds and hands control back so the
        # caller can rotate / fall back / degrade.
        slept: list[float] = []
        governor = RateLimitGovernor(
            time_fn=lambda: 10.0,
            sleep_fn=lambda seconds: slept.append(seconds),
            max_wait_seconds=20.0,
        )
        governor.record_response(
            "qwen/qwen3-32b",
            _FakeResponse(
                status_code=429,
                headers={
                    "x-ratelimit-remaining-tokens": "0",
                    "x-ratelimit-reset-tokens": "100000",  # ~27h away
                },
            ),
        )

        best_key, slept_seconds = governor.select_key(
            "qwen/qwen3-32b", current_key=0, num_keys=1, estimated_input_tokens=100
        )

        self.assertEqual(best_key, 0)
        self.assertEqual(slept_seconds, 20.0)
        self.assertEqual(slept, [20.0])

    def test_model_resolver_excludes_preview_and_unavailable_models_by_default(self) -> None:
        resolver = ModelResolver(
            active_model_ids={"llama-3.1-8b-instant", "openai/gpt-oss-20b"},
            model_policy="production_only",
        )

        selected = resolver.resolve(
            LLMTask.TOP_ALERTS,
            [
                ModelConfig("meta-llama/llama-4-scout-17b-16e-instruct"),
                ModelConfig("llama-3.3-70b-versatile"),
                ModelConfig("llama-3.1-8b-instant"),
            ],
            estimated_input_tokens=500,
            requested_output_tokens=500,
        )

        self.assertEqual(selected.model.model_id, "llama-3.1-8b-instant")
        rejected = {item["model_id"]: item["reason"] for item in selected.rejections}
        self.assertEqual(rejected["meta-llama/llama-4-scout-17b-16e-instruct"], "preview_model_disallowed")
        self.assertEqual(rejected["llama-3.3-70b-versatile"], "model_not_active")

    def test_resolver_fallback_avoids_preview_when_all_rate_limited(self) -> None:
        # When every production model is rate-limited past the cap and nothing
        # scores, the resolver must fall back to an eligible production model,
        # NOT model_chain[0] which is the preview Scout model the policy forbids.
        resolver = ModelResolver(
            active_model_ids={
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama-3.1-8b-instant",
            },
            model_policy="production_only",
            max_wait_seconds=20.0,
        )
        chain = [
            ModelConfig("meta-llama/llama-4-scout-17b-16e-instruct"),  # preview, chain head
            ModelConfig("llama-3.1-8b-instant"),  # production
        ]

        selected = resolver.resolve(
            LLMTask.ARTICLE_ANALYSIS,
            chain,
            estimated_input_tokens=100,
            requested_output_tokens=100,
            rate_limit_waits={"llama-3.1-8b-instant": 9999.0},  # over the cap
        )

        self.assertEqual(selected.model.model_id, "llama-3.1-8b-instant")
        reasons = {item["model_id"]: item["reason"] for item in selected.rejections}
        self.assertEqual(
            reasons["meta-llama/llama-4-scout-17b-16e-instruct"], "preview_model_disallowed"
        )

    def test_model_resolver_prefers_different_models_for_routing_and_top_alerts(self) -> None:
        resolver = ModelResolver(
            active_model_ids={"llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"},
            model_policy="production_only",
        )
        chain = [
            ModelConfig("llama-3.1-8b-instant"),
            ModelConfig("llama-3.3-70b-versatile"),
            ModelConfig("openai/gpt-oss-20b"),
        ]

        routing = resolver.resolve(LLMTask.ROUTING, chain, estimated_input_tokens=300, requested_output_tokens=300)
        alerts = resolver.resolve(LLMTask.TOP_ALERTS, chain, estimated_input_tokens=300, requested_output_tokens=800)

        self.assertEqual(routing.model.model_id, "llama-3.1-8b-instant")
        self.assertEqual(alerts.model.model_id, "openai/gpt-oss-20b")

    def test_model_resolver_honors_task_preference_environment_override(self) -> None:
        resolver = ModelResolver(
            active_model_ids={"llama-3.1-8b-instant", "llama-3.3-70b-versatile"},
            model_policy="production_only",
        )

        with patch.dict(os.environ, {"DAILY_MACRO_MODEL_ROUTING_PREFERENCES": "llama-3.3-70b-versatile"}, clear=False):
            selected = resolver.resolve(
                LLMTask.ROUTING,
                [ModelConfig("llama-3.1-8b-instant"), ModelConfig("llama-3.3-70b-versatile")],
                estimated_input_tokens=100,
                requested_output_tokens=100,
            )

        self.assertEqual(selected.model.model_id, "llama-3.3-70b-versatile")

    def test_active_groq_model_refresh_is_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "daily_macro.llm_client._build_groq_session",
            side_effect=AssertionError("model refresh should be opt-in"),
        ):
            active_ids = load_active_groq_model_ids("test-key")

        self.assertIsNone(active_ids)

    def test_chat_completion_switches_model_instead_of_sleeping_for_long_reset(self) -> None:
        slept: list[float] = []
        governor = RateLimitGovernor(time_fn=lambda: 10.0, sleep_fn=lambda seconds: slept.append(seconds))
        governor.record_response(
            "llama-3.1-8b-instant",
            _FakeResponse(
                200,
                headers={
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-reset-requests": "120",
                },
            ),
        )
        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
                )
            ]
        )
        runtime = AnalysisRuntime(
            session=session,
            governor=governor,
            model_chain=[
                ModelConfig("llama-3.1-8b-instant"),
                ModelConfig("llama-3.3-70b-versatile"),
            ],
            resolver=ModelResolver(
                active_model_ids={"llama-3.1-8b-instant", "llama-3.3-70b-versatile"},
                model_policy="production_only",
                max_wait_seconds=30.0,
            ),
        )

        payload, model_used = analysis_module._invoke_json_with_retry(
            runtime,
            [{"role": "user", "content": "{}"}],
            10,
            analysis_module.BatchContext("test", "article_batch", "1", 1, 10, 100),
        )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(model_used, "llama-3.3-70b-versatile")
        self.assertEqual(slept, [])
        self.assertEqual(runtime.diagnostics.avoided_rate_limit_wait_count, 1)
        self.assertEqual(session.models_used, ["llama-3.3-70b-versatile"])

    def test_generate_top_alerts_uses_structured_schema_and_shared_invocation(self) -> None:
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
                                            "top_alerts": [
                                                {
                                                    "summary": "Fed repricing is the key macro risk.",
                                                    "why_it_matters": "Higher yields pressure growth equities.",
                                                    "affected_assets": ["US10Y", "HSI"],
                                                    "time_horizon": "1w",
                                                    "confidence": 0.82,
                                                    "source_article_ids": ["9001"],
                                                }
                                            ]
                                        }
                                    )
                                }
                            }
                        ]
                    },
                )
            ]
        )
        runtime = AnalysisRuntime(
            session=session,
            governor=RateLimitGovernor(sleep_fn=lambda seconds: None),
            model_chain=[ModelConfig("llama-3.1-8b-instant"), ModelConfig("openai/gpt-oss-20b")],
            resolver=ModelResolver(
                active_model_ids={"llama-3.1-8b-instant", "openai/gpt-oss-20b"},
                model_policy="production_only",
            ),
        )

        alerts = analysis_module._generate_top_alerts(
            runtime=runtime,
            market_context="US10Y +8bp",
            developments=[
                {
                    "category": "國際財經",
                    "subgroup": "Rates",
                    "text": "Fed repricing is the key macro risk.",
                    "ref_ids": ["9001"],
                }
            ],
            article_metadata={
                "9001": {
                    "title": "Fed headline",
                    "date": "2026-04-03",
                    "url": "https://example.com/9001",
                }
            },
        )

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["source_article_ids"], ["9001"])
        self.assertEqual(alerts[0]["affected_assets"], ["US10Y", "HSI"])
        self.assertEqual(alerts[0]["source_articles"][0]["url"], "https://example.com/9001")
        self.assertEqual(session.models_used, ["openai/gpt-oss-20b"])

    def test_run_analysis_returns_empty_report_for_missing_day(self) -> None:
        with patch(
            "daily_macro.release_calendar.build_release_digest",
            return_value={
                "generated_at": "2026-04-03T00:10:00+00:00",
                "window_start": "2026-04-03",
                "window_end": "2026-04-09",
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
                    }
                ],
            },
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)
        self.assertEqual(report["status"], "empty")
        self.assertEqual(report["totals"]["article_count"], 0)
        self.assertEqual(report["diagnostics"]["batch_count"], 0)
        self.assertEqual(report["incremental"]["new_articles_analyzed"], 0)
        self.assertEqual(report["macro_release_digest"]["fetch_status"], "success")
        self.assertEqual(report["macro_release_digest"]["items"][0]["name"], "Consumer Price Index")
        report_path = self.data_dir / "analyses" / "2026-04-03" / "hkej-news-analysis.json"
        self.assertTrue(report_path.exists())

    def test_run_analysis_force_reuses_fully_successful_today_report_without_api_calls(self) -> None:
        self._insert_article(
            title="Pulse one",
            source_article_id="1001",
            section="時事脈搏",
            published_at="2026-04-03T08:00:00+08:00",
            content_text="a" * 800,
        )
        self._write_report_file(
            "2026-04-03",
            {
                "report_schema_version": REPORT_SCHEMA_VERSION,
                "report_date": "2026-04-03",
                "generated_at": "2026-04-03T10:00:00+00:00",
                "source_site": "hkej",
                "status": "success",
                "model": {"provider": "groq", "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct", "fallback_models": []},
                "model_switches": [],
                "input": {"article_count": 1, "category_count": 1},
                "diagnostics": {},
                "incremental": {"reused_successful_articles": 1, "new_articles_analyzed": 0, "retried_previous_day_articles": 0, "previous_day_retry_successes": 0},
                "totals": {"article_count": 1, "successful_article_analyses": 1, "failed_article_analyses": 0, "full_text_article_count": 1, "truncated_article_count": 0, "successful_categories": 1, "partial_categories": 0, "failed_categories": 0},
                "categories": [
                    {
                        "category": "時事脈搏",
                        "article_count": 1,
                        "status": "success",
                        "key_developments": ["Existing category summary"],
                        "named_entities": [],
                        "articles": [
                            self._report_article(
                                source_article_id="1001",
                                title="Pulse one",
                                section="時事脈搏",
                                published_at="2026-04-03T08:00:00+08:00",
                                success=True,
                            )
                        ],
                        "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "sub_batch_count": 1,
                        "diagnostics": {},
                        "error": None,
                    }
                ],
                "errors": [],
            },
        )

        with patch("daily_macro.analysis.load_groq_api_key", side_effect=AssertionError("API key should not be loaded")), patch(
            "daily_macro.analysis._build_groq_session", side_effect=AssertionError("Groq session should not be built")
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path, force=True)

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["incremental"]["reused_successful_articles"], 1)
        self.assertEqual(report["incremental"]["new_articles_analyzed"], 0)
        self.assertEqual(report["totals"]["article_count"], 1)
        reused = report["categories"][0]["articles"][0]
        self.assertIn(reused["attention_tier"], {"medium", "light", "high"})
        self.assertIn("theme", reused)

    def test_run_analysis_ignores_malformed_previous_day_report(self) -> None:
        self._insert_article(
            title="Today pulse",
            source_article_id="3001",
            section="時事脈搏",
            published_at="2026-04-04T08:00:00+08:00",
            content_text="c" * 800,
        )
        broken_report_path = self.data_dir / "analyses" / "2026-04-03" / "hkej-news-analysis.json"
        broken_report_path.parent.mkdir(parents=True, exist_ok=True)
        broken_report_path.write_text('{\n  "report_schema_version": 6,\n  invalid\n}\n', encoding="utf-8")

        with patch("daily_macro.analysis._build_analysis_graph") as mock_build_graph:
            mock_build_graph.return_value.invoke.return_value = {
                "report": {
                    "report_schema_version": REPORT_SCHEMA_VERSION,
                    "report_date": "2026-04-04",
                    "generated_at": "2026-04-04T00:30:00+00:00",
                    "source_site": "hkej",
                    "status": "success",
                    "model": {"provider": "groq", "primary_model": "qwen/qwen3-32b", "fallback_models": []},
                    "model_switches": [],
                    "input": {"article_count": 1, "category_count": 1},
                    "diagnostics": {"batch_count": 0},
                    "incremental": {"reused_successful_articles": 0, "new_articles_analyzed": 1},
                    "totals": {"article_count": 1},
                    "categories": [],
                    "errors": [],
                }
            }
            report = run_analysis(date_string="2026-04-04", data_dir=self.data_dir, db_path=self.db_path, force=True)

        self.assertEqual(report["status"], "success")
        self.assertEqual(
            Path(report["output_path"]).resolve(),
            (self.data_dir / "analyses" / "2026-04-04" / "hkej-news-analysis.json").resolve(),
        )
        state = mock_build_graph.return_value.invoke.call_args.args[0]
        self.assertIsNone(state["previous_report"])
        self.assertEqual(len(state["articles"]), 1)
        self.assertEqual(state["previous_articles"], [])

    def test_run_analysis_retries_previous_day_failed_articles_when_today_is_empty(self) -> None:
        self._insert_article(
            title="Yesterday global",
            source_article_id="2001",
            section="國際財經",
            published_at="2026-04-03T09:00:00+08:00",
            content_text="b" * 800,
        )
        yesterday_path = self._write_report_file(
            "2026-04-03",
            {
                "report_schema_version": REPORT_SCHEMA_VERSION,
                "report_date": "2026-04-03",
                "generated_at": "2026-04-03T10:00:00+00:00",
                "source_site": "hkej",
                "status": "partial",
                "model": {"provider": "groq", "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct", "fallback_models": []},
                "model_switches": [],
                "input": {"article_count": 1, "category_count": 1},
                "diagnostics": {},
                "incremental": {"reused_successful_articles": 0, "new_articles_analyzed": 1, "retried_previous_day_articles": 0, "previous_day_retry_successes": 0},
                "totals": {"article_count": 1, "successful_article_analyses": 0, "failed_article_analyses": 1, "full_text_article_count": 1, "truncated_article_count": 0, "successful_categories": 0, "partial_categories": 1, "failed_categories": 0},
                "categories": [
                    {
                        "category": "國際財經",
                        "article_count": 1,
                        "status": "partial",
                        "key_developments": [],
                        "named_entities": [],
                        "articles": [self._report_article(source_article_id="2001", title="Yesterday global", section="國際財經", published_at="2026-04-03T09:00:00+08:00", success=False)],
                        "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "sub_batch_count": 1,
                        "diagnostics": {},
                        "error": "Model response omitted this article from the category batch.",
                    }
                ],
                "errors": [{"type": "article", "target": "https://example.com/2001", "message": "Model response omitted this article from the category batch.", "classification": "incomplete_model_output"}],
            },
        )

        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"articles": [{"source_article_id": "2001", "canonical_url": "https://example.com/2001", "novelty_score": 7, "relevance_score": 8, "urgency_score": 6, "named_entities": [], "key_points": ["Yesterday recovered"]}]}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Yesterday summary"], "named_entities": []}, ensure_ascii=False)}}]},
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ), patch(
            "daily_macro.analysis._should_use_llm_router",
            return_value=False,
        ), patch(
            "daily_macro.analysis.time.sleep",
            return_value=None,
        ):
            report = run_analysis(date_string="2026-04-04", data_dir=self.data_dir, db_path=self.db_path, force=True)

        self.assertEqual(report["status"], "empty")
        self.assertEqual(report["incremental"]["retried_previous_day_articles"], 1)
        self.assertEqual(report["incremental"]["previous_day_retry_successes"], 1)
        updated_yesterday = json.loads(yesterday_path.read_text(encoding="utf-8"))
        self.assertEqual(updated_yesterday["status"], "success")

    def test_run_analysis_reuses_successful_today_articles_and_only_analyzes_new_ones(self) -> None:
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
        self._write_report_file(
            "2026-04-03",
            {
                "report_schema_version": REPORT_SCHEMA_VERSION,
                "report_date": "2026-04-03",
                "generated_at": "2026-04-03T10:00:00+00:00",
                "source_site": "hkej",
                "status": "partial",
                "model": {"provider": "groq", "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct", "fallback_models": []},
                "model_switches": [],
                "input": {"article_count": 2, "category_count": 1},
                "diagnostics": {},
                "totals": {"article_count": 1, "successful_article_analyses": 1, "failed_article_analyses": 0, "full_text_article_count": 1, "truncated_article_count": 0, "successful_categories": 0, "partial_categories": 1, "failed_categories": 0},
                "categories": [
                    {
                        "category": "時事脈搏",
                        "article_count": 1,
                        "status": "partial",
                        "key_developments": ["Existing category summary"],
                        "named_entities": [],
                        "articles": [
                            self._report_article(
                                source_article_id="1001",
                                title="Pulse one",
                                section="時事脈搏",
                                published_at="2026-04-03T08:00:00+08:00",
                                success=True,
                            )
                        ],
                        "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "sub_batch_count": 1,
                        "diagnostics": {},
                        "error": None,
                    }
                ],
                "errors": [],
            },
        )

        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {"message": {"content": json.dumps({"articles": [{"source_article_id": "1002", "canonical_url": "https://example.com/1002", "novelty_score": 6, "relevance_score": 8, "urgency_score": 7, "named_entities": [], "key_points": ["Pulse point two"]}]}, ensure_ascii=False)}}
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Merged pulse development"], "named_entities": []}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Final merged pulse development"], "named_entities": []}, ensure_ascii=False)}}]},
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ), patch(
            "daily_macro.analysis._should_use_llm_router",
            return_value=False,
        ), patch(
            "daily_macro.analysis.time.sleep",
            return_value=None,
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path, force=True)

        self.assertEqual(session.calls, 3)
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["incremental"]["reused_successful_articles"], 1)
        self.assertEqual(report["incremental"]["new_articles_analyzed"], 1)
        self.assertEqual(report["totals"]["article_count"], 2)
        self.assertEqual(len(report["categories"][0]["articles"]), 2)
        reused = next(article for article in report["categories"][0]["articles"] if article["source_article_id"] == "1001")
        self.assertEqual(reused["key_points"], ["Existing point"])

    def test_run_analysis_retries_failed_today_articles_but_not_successful_ones(self) -> None:
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
        self._write_report_file(
            "2026-04-03",
            {
                "report_schema_version": REPORT_SCHEMA_VERSION,
                "report_date": "2026-04-03",
                "generated_at": "2026-04-03T10:00:00+00:00",
                "source_site": "hkej",
                "status": "partial",
                "model": {"provider": "groq", "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct", "fallback_models": []},
                "model_switches": [],
                "input": {"article_count": 2, "category_count": 1},
                "diagnostics": {},
                "totals": {"article_count": 2, "successful_article_analyses": 1, "failed_article_analyses": 1, "full_text_article_count": 2, "truncated_article_count": 0, "successful_categories": 0, "partial_categories": 1, "failed_categories": 0},
                "categories": [
                    {
                        "category": "時事脈搏",
                        "article_count": 2,
                        "status": "partial",
                        "key_developments": ["Old summary"],
                        "named_entities": [],
                        "articles": [
                            self._report_article(source_article_id="1001", title="Pulse one", section="時事脈搏", published_at="2026-04-03T08:00:00+08:00", success=True),
                            self._report_article(source_article_id="1002", title="Pulse two", section="時事脈搏", published_at="2026-04-03T07:00:00+08:00", success=False),
                        ],
                        "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "sub_batch_count": 1,
                        "diagnostics": {},
                        "error": "Model response omitted this article from the category batch.",
                    }
                ],
                "errors": [{"type": "article", "target": "https://example.com/1002", "message": "Model response omitted this article from the category batch.", "classification": "incomplete_model_output"}],
            },
        )

        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"articles": [{"source_article_id": "1002", "canonical_url": "https://example.com/1002", "novelty_score": 6, "relevance_score": 8, "urgency_score": 7, "named_entities": [], "key_points": ["Recovered point"]}]}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Recovered category"], "named_entities": []}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Merged recovered category"], "named_entities": []}, ensure_ascii=False)}}]},
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path, force=True)

        self.assertEqual(session.calls, 3)
        self.assertEqual(report["incremental"]["reused_successful_articles"], 1)
        self.assertEqual(report["incremental"]["new_articles_analyzed"], 1)
        self.assertEqual(report["status"], "success")
        retried = next(article for article in report["categories"][0]["articles"] if article["source_article_id"] == "1002")
        self.assertIsNone(retried.get("error"))
        self.assertIsNone(retried.get("error_classification"))

    def test_run_analysis_retries_previous_day_failed_articles_and_updates_previous_report(self) -> None:
        self._insert_article(
            title="Today pulse",
            source_article_id="3001",
            section="時事脈搏",
            published_at="2026-04-04T08:00:00+08:00",
            content_text="a" * 800,
        )
        self._insert_article(
            title="Yesterday global",
            source_article_id="2001",
            section="國際財經",
            published_at="2026-04-03T09:00:00+08:00",
            content_text="b" * 800,
        )
        self._write_report_file(
            "2026-04-04",
            {
                "report_schema_version": REPORT_SCHEMA_VERSION,
                "report_date": "2026-04-04",
                "generated_at": "2026-04-04T10:00:00+00:00",
                "source_site": "hkej",
                "status": "success",
                "model": {"provider": "groq", "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct", "fallback_models": []},
                "model_switches": [],
                "input": {"article_count": 1, "category_count": 1},
                "diagnostics": {},
                "totals": {"article_count": 1, "successful_article_analyses": 1, "failed_article_analyses": 0, "full_text_article_count": 1, "truncated_article_count": 0, "successful_categories": 1, "partial_categories": 0, "failed_categories": 0},
                "categories": [
                    {
                        "category": "時事脈搏",
                        "article_count": 1,
                        "status": "success",
                        "key_developments": ["Today summary"],
                        "named_entities": [],
                        "articles": [self._report_article(source_article_id="3001", title="Today pulse", section="時事脈搏", published_at="2026-04-04T08:00:00+08:00", success=True)],
                        "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "sub_batch_count": 1,
                        "diagnostics": {},
                        "error": None,
                    }
                ],
                "errors": [],
            },
        )
        yesterday_path = self._write_report_file(
            "2026-04-03",
            {
                "report_schema_version": REPORT_SCHEMA_VERSION,
                "report_date": "2026-04-03",
                "generated_at": "2026-04-03T10:00:00+00:00",
                "source_site": "hkej",
                "status": "partial",
                "model": {"provider": "groq", "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct", "fallback_models": []},
                "model_switches": [],
                "input": {"article_count": 1, "category_count": 1},
                "diagnostics": {},
                "totals": {"article_count": 1, "successful_article_analyses": 0, "failed_article_analyses": 1, "full_text_article_count": 1, "truncated_article_count": 0, "successful_categories": 0, "partial_categories": 1, "failed_categories": 0},
                "categories": [
                    {
                        "category": "國際財經",
                        "article_count": 1,
                        "status": "partial",
                        "key_developments": [],
                        "named_entities": [],
                        "articles": [self._report_article(source_article_id="2001", title="Yesterday global", section="國際財經", published_at="2026-04-03T09:00:00+08:00", success=False)],
                        "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "sub_batch_count": 1,
                        "diagnostics": {},
                        "error": "Model response omitted this article from the category batch.",
                    }
                ],
                "errors": [{"type": "article", "target": "https://example.com/2001", "message": "Model response omitted this article from the category batch.", "classification": "incomplete_model_output"}],
            },
        )

        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"articles": [{"source_article_id": "2001", "canonical_url": "https://example.com/2001", "novelty_score": 7, "relevance_score": 8, "urgency_score": 6, "named_entities": [], "key_points": ["Yesterday recovered"]}]}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Yesterday summary"], "named_entities": []}, ensure_ascii=False)}}]},
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ), patch(
            "daily_macro.analysis._should_use_llm_router",
            return_value=False,
        ), patch(
            "daily_macro.analysis.time.sleep",
            return_value=None,
        ):
            report = run_analysis(date_string="2026-04-04", data_dir=self.data_dir, db_path=self.db_path, force=True)

        self.assertEqual(session.calls, 2)
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["incremental"]["reused_successful_articles"], 1)
        self.assertEqual(report["incremental"]["new_articles_analyzed"], 0)
        self.assertEqual(report["incremental"]["retried_previous_day_articles"], 1)
        self.assertEqual(report["incremental"]["previous_day_retry_successes"], 1)
        self.assertEqual(report["totals"]["article_count"], 1)
        updated_yesterday = json.loads(yesterday_path.read_text(encoding="utf-8"))
        self.assertEqual(updated_yesterday["status"], "success")
        self.assertIsNone(updated_yesterday["categories"][0]["articles"][0].get("error"))
        self.assertFalse(updated_yesterday["categories"][0]["articles"][0].get("error_classification"))

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
        ), patch(
            "daily_macro.analysis._should_use_llm_router",
            return_value=False,
        ), patch(
            "daily_macro.analysis.RateLimitGovernor",
            side_effect=lambda: RateLimitGovernor(sleep_fn=lambda _seconds: None),
        ), patch(
            "daily_macro.analysis.time.sleep",
            return_value=None,
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["input"]["category_count"], 2)
        self.assertEqual(report["totals"]["article_count"], 3)
        self.assertEqual(session.calls, 5)
        self.assertTrue(
            set(session.models_used).issubset(
                {
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "qwen/qwen3-32b",
                    "llama-3.1-8b-instant",
                }
            )
        )
        self.assertEqual(report["diagnostics"]["batch_count"], 5)

        pulse = next(category for category in report["categories"] if category["category"] == "時事脈搏")
        self.assertEqual(pulse["status"], "success")
        self.assertEqual(pulse["sub_batch_count"], 3)
        self.assertTrue(any(article["content_truncated"] for article in pulse["articles"]))
        self.assertTrue(all(article.get("model_used") for article in pulse["articles"]))
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
        ), patch(
            "daily_macro.analysis.RateLimitGovernor",
            side_effect=lambda: RateLimitGovernor(sleep_fn=lambda _seconds: None),
        ), patch(
            "daily_macro.analysis.time.sleep",
            return_value=None,
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        self.assertEqual(report["status"], "success")
        self.assertEqual(session.calls, 3)
        self.assertFalse(report["categories"][0]["articles"][0].get("error"))
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
        ), patch(
            "daily_macro.analysis._should_use_llm_router",
            return_value=False,
        ), patch(
            "daily_macro.analysis.RateLimitGovernor",
            side_effect=lambda: RateLimitGovernor(sleep_fn=lambda _seconds: None),
        ), patch(
            "daily_macro.analysis.time.sleep",
            return_value=None,
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        category = report["categories"][0]
        self.assertEqual(report["status"], "success")
        self.assertEqual(category["status"], "success")
        self.assertEqual(len(category["articles"]), 2)
        self.assertTrue(all(not article.get("error") for article in category["articles"]))
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
        self.assertEqual(category["sub_batch_count"], 3)
        self.assertEqual(len(category["articles"]), 4)
        self.assertEqual(session.calls, 4)
        self.assertEqual(report["diagnostics"]["response_413_split_count"], 1)
        self.assertIn("response_413", category["diagnostics"]["split_reasons"])

    def test_analyze_category_resets_preferred_model_for_each_category_after_429(self) -> None:
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
            ]
        )

        runtime = AnalysisRuntime(
            session=session,
            governor=RateLimitGovernor(sleep_fn=lambda _seconds: None),
            model_chain=[
                ModelConfig("meta-llama/llama-4-scout-17b-16e-instruct"),
                ModelConfig("qwen/qwen3-32b"),
                ModelConfig("llama-3.1-8b-instant"),
            ],
            resolver=ModelResolver(model_policy="allow_preview"),
        )
        global_article = analysis_module._prepare_single_article(
            {
                "source_article_id": "4001",
                "canonical_url": "https://example.com/4001",
                "title": "Global one",
                "article_section": "國際財經",
                "published_at": "2026-04-03T09:00:00+08:00",
                "summary_snippet": None,
                "content_text": "b" * 900,
            }
        )
        china_article = analysis_module._prepare_single_article(
            {
                "source_article_id": "3001",
                "canonical_url": "https://example.com/3001",
                "title": "China one",
                "article_section": "中國財經",
                "published_at": "2026-04-03T08:00:00+08:00",
                "summary_snippet": None,
                "content_text": "a" * 800,
            }
        )

        global_report, _ = analysis_module._analyze_category(runtime, "國際財經", [global_article])
        china_report, _ = analysis_module._analyze_category(runtime, "中國財經", [china_article])

        self.assertEqual(global_report["status"], "success")
        self.assertEqual(china_report["status"], "success")
        self.assertEqual(
            session.models_used,
            [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "qwen/qwen3-32b",
                "qwen/qwen3-32b",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "meta-llama/llama-4-scout-17b-16e-instruct",
            ],
        )
        self.assertEqual(len(runtime.model_switches), 1)
        self.assertEqual(runtime.model_switches[0]["to_model"], "qwen/qwen3-32b")
        self.assertEqual(runtime.diagnostics.fallback_switch_count, 1)
        self.assertIn("qwen/qwen3-32b", global_report["diagnostics"]["models_attempted"])
        self.assertEqual(
            china_report["diagnostics"]["models_attempted"],
            ["meta-llama/llama-4-scout-17b-16e-instruct"],
        )

    def test_run_analysis_uses_local_degraded_merge_when_merge_depth_cap_is_hit(self) -> None:
        for index in range(2):
            self._insert_article(
                title=f"Market {index}",
                source_article_id=f"800{index}",
                section="港股直擊",
                published_at=f"2026-04-03T0{8-index}:00:00+08:00",
                content_text="market content " * 30,
            )

        article_results = [
            {
                "source_article_id": "8000",
                "canonical_url": "https://example.com/8000",
                "novelty_score": 6,
                "relevance_score": 7,
                "urgency_score": 6,
                "named_entities": [{"name": "HSI", "type": "index"}],
                "key_points": ["Market point 0"],
            },
            {
                "source_article_id": "8001",
                "canonical_url": "https://example.com/8001",
                "novelty_score": 6,
                "relevance_score": 7,
                "urgency_score": 6,
                "named_entities": [{"name": "Tencent", "type": "company"}],
                "key_points": ["Market point 1"],
            },
        ]

        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"articles": article_results}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Batch one development"], "named_entities": [{"name": "HSI", "type": "index"}]}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Batch two development"], "named_entities": [{"name": "Tencent", "type": "company"}]}, ensure_ascii=False)}}]},
                ),
            ]
        )

        def split_summary_batches(_runtime: object, _category_name: str, synthesis_items: list[dict[str, object]], *, model_id: str) -> list[list[dict[str, object]]]:
            if len(synthesis_items) > 1:
                return [[synthesis_items[0]], [synthesis_items[1]]]
            return [synthesis_items]

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ), patch(
            "daily_macro.analysis._should_use_llm_router",
            return_value=False,
        ), patch(
            "daily_macro.analysis._plan_synthesis_batches",
            side_effect=split_summary_batches,
        ), patch.object(
            analysis_module,
            "MAX_SYNTHESIS_MERGE_DEPTH",
            0,
        ), patch(
            "daily_macro.analysis.time.sleep",
            return_value=None,
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        category = next(category for category in report["categories"] if category["category"] == "港股直擊")
        self.assertEqual(session.calls, 3)
        self.assertEqual(category["key_developments"], ["Batch one development", "Batch two development"])
        self.assertTrue(category["diagnostics"]["degraded_merge_used"])
        self.assertEqual(category["diagnostics"]["degraded_merge_reason"], "merge_depth_cap:0")

    def test_analyze_category_switches_from_qwen_to_instant_after_second_429(self) -> None:
        session = _FakeGroqSession(
            [
                _FakeResponse(429, headers={"Retry-After": "0"}),
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
                                                    "named_entities": [],
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
            ]
        )

        runtime = AnalysisRuntime(
            session=session,
            governor=RateLimitGovernor(sleep_fn=lambda _seconds: None),
            model_chain=[
                ModelConfig("meta-llama/llama-4-scout-17b-16e-instruct"),
                ModelConfig("qwen/qwen3-32b"),
                ModelConfig("llama-3.1-8b-instant"),
            ],
            resolver=ModelResolver(model_policy="allow_preview"),
        )
        article = analysis_module._prepare_single_article(
            {
                "source_article_id": "3001",
                "canonical_url": "https://example.com/3001",
                "title": "China one",
                "article_section": "中國財經",
                "published_at": "2026-04-03T08:00:00+08:00",
                "summary_snippet": None,
                "content_text": "a" * 800,
            }
        )

        category_report, _ = analysis_module._analyze_category(runtime, "中國財經", [article])

        self.assertEqual(category_report["status"], "success")
        self.assertEqual(
            session.models_used,
            [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "qwen/qwen3-32b",
                "llama-3.1-8b-instant",
                "llama-3.1-8b-instant",
            ],
        )
        self.assertEqual(len(runtime.model_switches), 2)
        self.assertEqual(runtime.model_switches[0]["to_model"], "qwen/qwen3-32b")
        self.assertEqual(runtime.model_switches[1]["to_model"], "llama-3.1-8b-instant")
        self.assertEqual(
            category_report["diagnostics"]["models_attempted"],
            [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "qwen/qwen3-32b",
                "llama-3.1-8b-instant",
            ],
        )

    def test_run_analysis_uses_local_degraded_merge_when_synthesis_retry_budget_is_exhausted(self) -> None:
        for index in range(2):
            self._insert_article(
                title=f"Budget {index}",
                source_article_id=f"810{index}",
                section="港股直擊",
                published_at=f"2026-04-03T0{8-index}:00:00+08:00",
                content_text="budget content " * 30,
            )

        article_results = [
            {
                "source_article_id": "8100",
                "canonical_url": "https://example.com/8100",
                "novelty_score": 6,
                "relevance_score": 7,
                "urgency_score": 6,
                "named_entities": [{"name": "HSI", "type": "index"}],
                "key_points": ["Budget point 0"],
            },
            {
                "source_article_id": "8101",
                "canonical_url": "https://example.com/8101",
                "novelty_score": 6,
                "relevance_score": 7,
                "urgency_score": 6,
                "named_entities": [{"name": "HKEX", "type": "company"}],
                "key_points": ["Budget point 1"],
            },
        ]

        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"articles": article_results}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Budget batch one"], "named_entities": [{"name": "HSI", "type": "index"}]}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Budget batch two"], "named_entities": [{"name": "HKEX", "type": "company"}]}, ensure_ascii=False)}}]},
                ),
            ]
        )

        def split_summary_batches(_runtime: object, _category_name: str, synthesis_items: list[dict[str, object]], *, model_id: str) -> list[list[dict[str, object]]]:
            if len(synthesis_items) > 1:
                return [[synthesis_items[0]], [synthesis_items[1]]]
            return [synthesis_items]

        budget_checks = {"count": 0}

        def exhaust_synthesis_budget(runtime: AnalysisRuntime, category_name: str) -> None:
            budget_checks["count"] += 1
            if budget_checks["count"] < 3:
                return None
            diagnostics = runtime.get_category_diagnostics(category_name)
            diagnostics.synthesis_budget_exhausted = True
            diagnostics.synthesis_retry_count = max(diagnostics.synthesis_retry_count, 1)
            diagnostics.synthesis_retry_skipped_count += 1
            runtime.diagnostics.synthesis_budget_exhausted_count += 1
            raise analysis_module.SynthesisBudgetExceeded(
                f"Category {category_name} exhausted synthesis retry budget after 1 retries."
            )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ), patch(
            "daily_macro.analysis._should_use_llm_router",
            return_value=False,
        ), patch(
            "daily_macro.analysis._plan_synthesis_batches",
            side_effect=split_summary_batches,
        ), patch.object(
            analysis_module.AnalysisRuntime,
            "ensure_synthesis_budget",
            new=exhaust_synthesis_budget,
        ), patch(
            "daily_macro.analysis.time.sleep",
            return_value=None,
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        category = next(category for category in report["categories"] if category["category"] == "港股直擊")
        self.assertEqual(report["status"], "success")
        self.assertEqual(category["status"], "success")
        self.assertEqual(session.calls, 3)
        self.assertEqual(category["key_developments"], ["Budget batch one", "Budget batch two"])
        self.assertTrue(category["diagnostics"]["degraded_merge_used"])
        self.assertEqual(category["diagnostics"]["degraded_merge_reason"], "synthesis_budget_exhausted")
        self.assertTrue(category["diagnostics"]["synthesis_budget_exhausted"])
        self.assertGreaterEqual(category["diagnostics"]["synthesis_retry_count"], 1)

    def test_run_analysis_creates_subgroups_for_large_light_section(self) -> None:
        for index in range(7):
            self._insert_article(
                title=f"Pulse {index}",
                source_article_id=f"700{index}",
                section="時事脈搏",
                published_at=f"2026-04-03T0{index}:00:00+08:00",
                content_text="pulse content " * 40,
            )

        article_results = [
            {
                "source_article_id": f"700{index}",
                "canonical_url": f"https://example.com/700{index}",
                "novelty_score": 5 + (index % 3),
                "relevance_score": 6,
                "urgency_score": 5,
                "named_entities": [{"name": f"Entity {index}", "type": "company"}],
                "key_points": [f"Point {index}a", f"Point {index}b", f"Point {index}c"],
            }
            for index in range(7)
        ]

        session = _FakeGroqSession(
            [
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"articles": article_results}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "subgroups": [
                                                {
                                                    "title": "Regional policy pulse",
                                                    "theme_rationale": "These headlines cluster around policy-sensitive regional developments.",
                                                    "article_keys": ["7000", "7001", "7002", "7003"],
                                                },
                                                {
                                                    "title": "Corporate reaction",
                                                    "theme_rationale": "These headlines center on company-facing market reactions.",
                                                    "article_keys": ["7004", "7005", "7006"],
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
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Regional subgroup"], "named_entities": [{"name": "Entity 0", "type": "company"}]}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Corporate subgroup"], "named_entities": [{"name": "Entity 4", "type": "company"}]}, ensure_ascii=False)}}]},
                ),
                _FakeResponse(
                    200,
                    payload={"choices": [{"message": {"content": json.dumps({"key_developments": ["Pulse section overview"], "named_entities": [{"name": "Entity 0", "type": "company"}]}, ensure_ascii=False)}}]},
                ),
            ]
        )

        with patch("daily_macro.analysis.load_groq_api_key", return_value="test-key"), patch(
            "daily_macro.analysis._build_groq_session", return_value=session
        ):
            report = run_analysis(date_string="2026-04-03", data_dir=self.data_dir, db_path=self.db_path)

        category = report["categories"][0]
        self.assertEqual(category["analysis_profile"], "light")
        self.assertEqual(len(category["subgroups"]), 2)
        self.assertEqual(category["subgroups"][0]["title"], "Regional policy pulse")
        self.assertLessEqual(len(category["articles"][0]["key_points"]), 2)


if __name__ == "__main__":
    unittest.main()
