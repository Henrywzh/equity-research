from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from daily_macro.site import build_site


def _report(report_date: str, *, status: str = "success", unresolved: bool = False) -> dict[str, object]:
    unresolved_articles = []
    if unresolved:
        unresolved_articles = [
            {
                "category": "國際財經",
                "title": "Missed sanctions headline",
                "canonical_url": "https://example.com/missed",
                "source_article_id": "missed-1",
                "attention_tier": "high",
                "theme": "geopolitics",
                "must_keep": True,
                "error_classification": "incomplete_model_output",
                "error": "Model response omitted this article from the category batch.",
                "model_used": "qwen/qwen3-32b",
                "delayed_retry_attempted": True,
                "delayed_retry_model_chain": ["llama-3.1-8b-instant", "openai/gpt-oss-20b"],
                "delayed_retry_final_model": "openai/gpt-oss-20b",
                "published_at": f"{report_date}T08:30:00+08:00",
            }
        ]

    return {
        "report_date": report_date,
        "generated_at": f"{report_date}T01:05:00+00:00",
        "status": status,
        "source_site": "hkej",
        "report_schema_version": 1,
        "executive_summary": ["Macro tension rose overnight.", "Energy shipping risks stayed elevated."],
        "market_context": ["Brent held above $90.", "Regional equities were mixed."],
        "daily_stats": {"total_scraped": 5, "analyzed": 5, "success_rate": 100.0},
        "totals": {
            "article_count": 5,
            "successful_article_analyses": 4 if unresolved else 5,
            "failed_article_analyses": 1 if unresolved else 0,
            "full_text_article_count": 5,
            "truncated_article_count": 1 if unresolved else 0,
            "successful_categories": 1 if not unresolved else 0,
            "partial_categories": 1 if unresolved else 0,
            "failed_categories": 0,
        },
        "diagnostics": {
            "rate_limit_wait_count": 1 if unresolved else 0,
            "rate_limit_wait_seconds_total": 12.5 if unresolved else 0.0,
            "pre_send_split_count": 2 if unresolved else 0,
            "response_413_split_count": 0,
            "fallback_switch_count": 1 if unresolved else 0,
            "delayed_retry_candidate_count": 1 if unresolved else 0,
            "delayed_retry_attempted_count": 1 if unresolved else 0,
            "delayed_retry_recovered_count": 0,
            "delayed_retry_failed_count": 1 if unresolved else 0,
            "high_medium_unresolved_count": 1 if unresolved else 0,
            "light_unresolved_count": 0,
            "delayed_retry_skipped_final_model_count": 0,
        },
        "model": {
            "provider": "groq",
            "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "fallback_models": ["qwen/qwen3-32b", "llama-3.1-8b-instant"],
        },
        "model_switches": [],
        "categories": [
            {
                "category": "國際財經",
                "status": status,
                "analysis_profile": "standard",
                "article_count": 5,
                "key_developments": ["Shipping risk rose after the latest closure headline."],
                "named_entities": [{"name": "霍爾木茲海峽", "type": "place"}],
                "diagnostics": {
                    "sub_batch_count": 2,
                    "split_reasons": ["pre_send_budget"] if unresolved else [],
                    "partial_article_count": 1 if unresolved else 0,
                },
                "subgroups": [
                    {
                        "title": "Energy chokepoints",
                        "theme_rationale": "These stories clustered around shipping and energy risk.",
                        "article_count": 3,
                        "key_developments": ["Tanker insurance costs climbed."],
                        "named_entities": [{"name": "霍爾木茲海峽", "type": "place"}],
                        "articles": [
                            {
                                "title": "Shipping disruption headline",
                                "canonical_url": "https://example.com/shipping",
                                "published_at": f"{report_date}T08:00:00+08:00",
                                "attention_tier": "high",
                                "theme": "geopolitics",
                                "key_points": ["Shipping risk increased.", "Energy costs may rise."],
                                "error": None,
                                "error_classification": None,
                                "raw_prompt": "should never be published",
                            }
                        ],
                    }
                ],
                "articles": [
                    {
                        "title": "Shipping disruption headline",
                        "canonical_url": "https://example.com/shipping",
                        "published_at": f"{report_date}T08:00:00+08:00",
                        "attention_tier": "high",
                        "theme": "geopolitics",
                        "key_points": ["Shipping risk increased.", "Energy costs may rise."],
                        "error": None,
                        "error_classification": None,
                        "raw_prompt": "should never be published",
                    }
                ],
            }
        ],
        "unresolved_articles": unresolved_articles,
        "raw_prompt": "top-level prompt should not be published",
    }


class SiteBuildTests(unittest.TestCase):
    def test_build_site_creates_latest_archive_and_report_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            analyses = data_dir / "analyses"
            for date_string, status in [("2026-04-08", "success"), ("2026-04-09", "partial")]:
                report_dir = analyses / date_string
                report_dir.mkdir(parents=True, exist_ok=True)
                report_dir.joinpath("hkej-news-analysis.json").write_text(
                    json.dumps(_report(date_string, status=status, unresolved=(status == "partial")), ensure_ascii=False),
                    encoding="utf-8",
                )

            output_dir = Path(tmp) / "site"
            result = build_site(data_dir=data_dir, output_dir=output_dir)

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["latest_report_date"], "2026-04-09")
            self.assertTrue((output_dir / "index.html").exists())
            self.assertTrue((output_dir / "today" / "index.html").exists())
            self.assertTrue((output_dir / "archive" / "index.html").exists())
            self.assertTrue((output_dir / "reports" / "2026-04-09" / "index.html").exists())
            archive_html = (output_dir / "archive" / "index.html").read_text(encoding="utf-8")
            self.assertIn("2026-04-08", archive_html)
            self.assertIn("2026-04-09", archive_html)
            latest_html = (output_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Unresolved articles", latest_html)
            self.assertIn("https://example.com/missed", latest_html)
            self.assertIn("Energy chokepoints", latest_html)

    def test_build_site_writes_sanitized_report_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            report_dir = data_dir / "analyses" / "2026-04-09"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_dir.joinpath("hkej-news-analysis.json").write_text(
                json.dumps(_report("2026-04-09", status="partial", unresolved=True), ensure_ascii=False),
                encoding="utf-8",
            )

            output_dir = Path(tmp) / "site"
            build_site(data_dir=data_dir, output_dir=output_dir)

            report_json = (output_dir / "reports" / "2026-04-09" / "report.json").read_text(encoding="utf-8")
            self.assertIn("Missed sanctions headline", report_json)
            self.assertNotIn("raw_prompt", report_json)
            self.assertNotIn("top-level prompt should not be published", report_json)

    def test_build_site_returns_empty_without_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            output_dir = Path(tmp) / "site"

            result = build_site(data_dir=data_dir, output_dir=output_dir)

            self.assertEqual(result["status"], "empty")
            self.assertEqual(result["report_count"], 0)
            self.assertTrue(output_dir.exists())

    def test_build_site_skips_unreadable_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            bad_report_dir = data_dir / "analyses" / "2026-04-08"
            good_report_dir = data_dir / "analyses" / "2026-04-09"
            bad_report_dir.mkdir(parents=True, exist_ok=True)
            good_report_dir.mkdir(parents=True, exist_ok=True)
            bad_report_dir.joinpath("hkej-news-analysis.json").write_text("{bad json", encoding="utf-8")
            good_report_dir.joinpath("hkej-news-analysis.json").write_text(
                json.dumps(_report("2026-04-09"), ensure_ascii=False),
                encoding="utf-8",
            )

            output_dir = Path(tmp) / "site"
            result = build_site(data_dir=data_dir, output_dir=output_dir)

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["report_count"], 1)
            self.assertEqual(len(result["skipped_reports"]), 1)
            self.assertTrue((output_dir / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
