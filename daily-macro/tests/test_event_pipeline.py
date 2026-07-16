from __future__ import annotations

import unittest

import daily_macro.analysis as analysis
from daily_macro.llm_client import LLMTask, ModelResolver
from daily_macro.types import ModelConfig
from daily_macro.run_notes import build_run_notes


def _article(article_id: str, title: str, *, high: bool = False, published_at: str = "2026-07-16T09:00:00+08:00") -> dict:
    return {
        "source_article_id": article_id,
        "canonical_url": f"https://example.test/{article_id}",
        "title": title,
        "published_at": published_at,
        "theme": "stocks",
        "research_lane": "hk_china_equity",
        "attention_tier": "high" if high else "medium",
        "relevance_score": 9 if high else 6,
        "urgency_score": 8 if high else 5,
        "novelty_score": 7,
        "named_entities": [{"name": "ACME", "type": "company"}],
        "key_points": [f"{title} affects ACME earnings."],
        "error": None,
    }


class EventPacketTests(unittest.TestCase):
    def test_task_wait_budgets_allow_alert_quality_but_bound_bulk_waits(self) -> None:
        resolver = ModelResolver(active_model_ids={"openai/gpt-oss-120b"}, model_policy="production_only")
        chain = [ModelConfig("openai/gpt-oss-120b")]
        routing = resolver.resolve(
            LLMTask.ROUTING,
            chain,
            estimated_input_tokens=100,
            requested_output_tokens=100,
            rate_limit_waits={chain[0].endpoint_id: 20},
        )
        alerts = resolver.resolve(
            LLMTask.TOP_ALERTS,
            chain,
            estimated_input_tokens=100,
            requested_output_tokens=100,
            rate_limit_waits={chain[0].endpoint_id: 20},
        )
        self.assertTrue(routing.wait_exceeded)
        self.assertFalse(alerts.wait_exceeded)

    def test_clusters_only_conservative_same_event_articles(self) -> None:
        reports = [
            {
                "category": "港股新聞",
                "articles": [
                    _article("a1", "ACME cuts guidance after weak demand"),
                    _article("a2", "ACME cuts guidance as demand weakens", published_at="2026-07-16T14:00:00+08:00"),
                    _article("b1", "ACME opens a new factory in Europe", high=True),
                ],
            }
        ]
        events, review_queue = analysis._build_event_packets(reports, target_date="2026-07-16")
        self.assertEqual(len(events), 2)
        self.assertEqual(sorted(event["source_count"] for event in events), [1, 2])
        self.assertTrue(any(item["source_article_ids"] == ["b1"] for item in review_queue))

    def test_event_validation_and_critic_remove_unsupported_citations(self) -> None:
        state = {
            "category_reports": [{"category": "港股新聞", "articles": [_article("a1", "ACME guidance")]}],
            "top_alerts": [
                {
                    "summary": "Supported alert",
                    "source_article_ids": ["a1", "hallucinated"],
                    "source_articles": [],
                },
                {"summary": "Unsupported alert", "source_article_ids": ["hallucinated"]},
            ],
            "target_date": "2026-07-16",
            "critic_issues": [],
            "runtime": None,
        }
        analysis._graph_critic_outputs(state)
        self.assertEqual(len(state["top_alerts"]), 1)
        self.assertEqual(state["top_alerts"][0]["source_article_ids"], ["a1"])
        self.assertTrue(state["critic_issues"])

    def test_alert_critic_catches_unit_scale_errors(self) -> None:
        source = {
            **_article("seres", "賽力斯料上半年轉蝕逾15億人民幣"),
            "content_text": "賽力斯(09927)預計上半年虧損15億至18億元；去年同期盈利29.41億元。",
            "summary_snippet": "去年同期盈利29.41億元。",
        }
        state = {
            "articles": [source],
            "category_reports": [{"category": "重要通告", "articles": [source]}],
            "top_alerts": [
                {
                    "summary": "Seres projects a loss of 15-18 billion RMB versus 29.41 billion RMB profit last year.",
                    "why_it_matters": "The loss could pressure Chinese EV equities.",
                    "affected_assets": ["Seres", "SAIC", "XLI"],
                    "confidence": 0.8,
                    "source_article_ids": ["seres"],
                }
            ],
            "target_date": "2026-07-16",
            "market_context_string": "",
            "critic_issues": [],
            "runtime": None,
        }
        analysis._graph_critic_outputs(state)
        self.assertTrue(state["critic_issues"])
        self.assertTrue(any(issue["type"] == "numeric_fact_unsupported" for issue in state["critic_issues"]))
        self.assertEqual(state["top_alerts"][0]["critic_status"], "needs_review")
        self.assertEqual(state["top_alerts"][0]["affected_assets"], ["09927.HK"])
        self.assertEqual([detail["input"] for detail in state["top_alerts"][0]["affected_asset_details"]], ["Seres"])

    def test_alert_critic_keeps_market_context_ticker_and_drops_unverified_ticker(self) -> None:
        source = {
            **_article("oil", "Middle East escalation raises energy risk"),
            "content_text": "Iran attacks were reported. Brent crude rose 3.28%.",
        }
        state = {
            "articles": [source],
            "category_reports": [{"category": "時事脈搏", "articles": [source]}],
            "top_alerts": [
                {
                    "summary": "Middle East escalation raises oil risk.",
                    "why_it_matters": "Energy markets may remain volatile.",
                    "affected_assets": ["CL=F", "XLI"],
                    "confidence": 0.7,
                    "source_article_ids": ["oil"],
                }
            ],
            "target_date": "2026-07-16",
            "market_context_string": "CL=F: 73.75 +3.28%",
            "critic_issues": [],
            "runtime": None,
        }
        analysis._graph_critic_outputs(state)
        self.assertEqual(state["top_alerts"][0]["affected_assets"], ["CL=F"])
        self.assertTrue(any(issue["type"] == "unsupported_asset_identifier" for issue in state["critic_issues"]))

    def test_top_alert_normalization_preserves_critic_metadata(self) -> None:
        normalized = analysis._normalize_top_alerts(
            [
                {
                    "summary": "Supported alert",
                    "source_article_ids": ["a1"],
                    "critic_status": "needs_review",
                    "affected_asset_details": [{"canonical_id": "^HSI", "status": "canonical"}],
                }
            ],
            {"a1": {"title": "Source", "date": "2026-07-16", "url": "https://example.test/a1"}},
        )
        self.assertEqual(normalized[0]["critic_status"], "needs_review")
        self.assertEqual(normalized[0]["affected_asset_details"][0]["canonical_id"], "^HSI")


class RunNotesTests(unittest.TestCase):
    def test_run_notes_include_wait_and_failure_breakdowns(self) -> None:
        notes = build_run_notes(
            {
                "status": "partial",
                "totals": {"article_count": 4, "failed_article_analyses": 1, "partial_categories": 1},
                "unresolved_articles": [{"source_article_id": "a1"}],
                "errors": [{"classification": "no_eligible_endpoint"}],
                "event_pipeline": {"event_count": 2, "review_count": 1},
                "diagnostics": {
                    "wall_clock_seconds": 90,
                    "llm_request_seconds_total": 40,
                    "rate_limit_wait_seconds_total": 20,
                    "rate_limit_wait_count": 2,
                    "rate_limit_waits_by_endpoint": {"cerebras:a:gpt": {"seconds": 20, "count": 2}},
                    "failure_classifications": {"no_eligible_endpoint": 2},
                    "split_counts_by_kind": {"article_batch": 3},
                    "phase_seconds": {"analyze_today": 50},
                    "critic_checked_alert_count": 2,
                },
            }
        )
        rendered = "\n".join(notes)
        self.assertIn("Wait detail:", rendered)
        self.assertIn("Failure counts:", rendered)
        self.assertIn("Split detail:", rendered)
        self.assertIn("Event layer:", rendered)
        self.assertIn("Alert quality checks:", rendered)


if __name__ == "__main__":
    unittest.main()
