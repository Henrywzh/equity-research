from __future__ import annotations

import unittest
from pathlib import Path

from daily_macro.site_adapters import HkejAdapter


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class HkejAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = HkejAdapter()

    def test_parse_homepage_extracts_featured_and_latest(self) -> None:
        html = (FIXTURES / "homepage.html").read_text(encoding="utf-8")

        placements = self.adapter.parse_homepage(html)
        head_news = [item for item in placements if item.collection == "head_news"]
        latest = [item for item in placements if item.collection == "latest"]

        self.assertEqual(len(head_news), 5)
        self.assertEqual([item.rank for item in head_news], [1, 2, 3, 4, 5])
        self.assertEqual(head_news[0].title, "頭條主新聞")
        self.assertEqual(head_news[1].title, "側欄新聞一")
        self.assertTrue(all(item.url.startswith("https://www.hkej.com/instantnews/") for item in head_news))

        self.assertEqual(len(latest), 3)
        self.assertEqual(latest[0].title, "最新新聞一")
        self.assertEqual(latest[0].summary_snippet, "第一條摘要")
        self.assertEqual(latest[1].homepage_section, "國際財經")

    def test_parse_article_prefers_jsonld(self) -> None:
        html = (FIXTURES / "article_jsonld.html").read_text(encoding="utf-8")

        article = self.adapter.parse_article(
            html,
            "https://www.hkej.com/instantnews/current/article/4364598/example-story",
        )

        self.assertEqual(article.title, "示例 JSON-LD 新聞")
        self.assertEqual(article.article_section, "時事脈搏")
        self.assertEqual(article.published_at, "2026-04-02T19:27:00+08:00")
        self.assertEqual(article.source_article_id, "4364598")
        self.assertEqual(article.content_text, "第一段正文。\n\n第二段正文。")
        self.assertEqual(article.extraction_method, "jsonld")
        self.assertFalse(article.malformed_jsonld_recovered)

    def test_parse_article_uses_html_fallback(self) -> None:
        html = (FIXTURES / "article_fallback.html").read_text(encoding="utf-8")

        article = self.adapter.parse_article(
            html,
            "https://www.hkej.com/instantnews/current/article/5000/fallback-story",
        )

        self.assertEqual(article.title, "示例 Fallback 新聞")
        self.assertEqual(article.article_section, "時事脈搏")
        self.assertEqual(article.published_at, "2026-04-03T08:00:00+08:00")
        self.assertEqual(
            article.content_text,
            "這是第一段 fallback 正文，應該被保留下來。\n\n這是第二段 fallback 正文，也應該被保留下來。",
        )
        self.assertEqual(article.extraction_method, "fallback_html")
        self.assertFalse(article.malformed_jsonld_recovered)

    def test_parse_article_accepts_relaxed_jsonld(self) -> None:
        html = (FIXTURES / "article_jsonld_malformed.html").read_text(encoding="utf-8")

        article = self.adapter.parse_article(
            html,
            "https://www.hkej.com/instantnews/current/article/6000/malformed-story",
        )

        self.assertEqual(article.title, "示例 非法 JSON-LD 新聞")
        self.assertEqual(article.article_section, "時事脈搏")
        self.assertEqual(article.published_at, "2026-04-02T19:27:00+08:00")
        self.assertEqual(article.content_text, "第一段正文。\n\n第二段正文。")
        self.assertEqual(article.extraction_method, "jsonld")
        self.assertTrue(article.malformed_jsonld_recovered)


if __name__ == "__main__":
    unittest.main()
