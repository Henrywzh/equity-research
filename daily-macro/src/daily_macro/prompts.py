"""Prompt-building helpers for the daily-macro analysis pipeline.

Each function returns a list of chat messages (role/content dicts) ready to
be sent to the LLM via _invoke_json_with_retry / _chat_completion.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .types import (
    ATTENTION_TIER_RANK,
    _section_profile,
)

LOGGER = logging.getLogger(__name__)


def _build_attention_routing_messages(category_name: str, batch_articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a financial news triage router. Return one valid JSON object only. "
                "Route every article into an attention tier and compact theme. Do not omit any article."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Assign compact routing metadata before deeper article analysis.",
                    "required_schema": {
                        "routes": [
                            {
                                "source_article_id": "string or null",
                                "canonical_url": "string",
                                "attention_tier": "high|medium|light",
                                "theme": "short theme such as stocks|macro|geopolitics|property|general",
                                "reason": "one short sentence",
                                "must_keep": "boolean",
                            }
                        ]
                    },
                    "routing_policy": {
                        "high": [
                            "stocks, earnings, guidance, placements, buybacks, capital markets",
                            "macro, central banks, inflation, rates, growth, FX, oil",
                            "geopolitics, sanctions, tariffs, war, trade restrictions",
                        ],
                        "light": [
                            "softer pulse-style or local-interest items",
                            "stories in 時事脈搏 or 地產新聞 without obvious market-moving signals",
                        ],
                        "default": "Use medium when ambiguous.",
                    },
                    "category": category_name,
                    "articles": [
                        {
                            "source_article_id": article.get("source_article_id"),
                            "canonical_url": article.get("canonical_url"),
                            "title": article.get("title"),
                            "summary_snippet": article.get("summary_snippet"),
                            "article_section": article.get("article_section") or article.get("section"),
                            "published_at": article.get("published_at"),
                        }
                        for article in batch_articles
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]


def _batch_attention_tier(batch_articles: list[dict[str, Any]]) -> str:
    if not batch_articles:
        return "medium"
    return min(
        (
            str(article.get("attention_tier") or "medium")
            for article in batch_articles
        ),
        key=lambda tier: ATTENTION_TIER_RANK.get(tier, ATTENTION_TIER_RANK["medium"]),
    )


def _build_article_batch_messages(category_name: str, batch_articles: list[dict[str, Any]], market_context: str = "") -> list[dict[str, Any]]:
    profile = _section_profile(category_name)
    batch_tier = _batch_attention_tier(batch_articles)
    return [
        {
            "role": "system",
            "content": (
                "You are a financial news analyst. Return one valid JSON object only. "
                "Analyze every article in the batch. Use integer scores from 1 to 10. "
                "Do not omit any article. "
                f"This batch is primarily {batch_tier}-attention; keep low-attention stories concise."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Analyze one batch of articles from a fixed financial news category.",
                    "required_schema": {
                        "articles": [
                            {
                                "source_article_id": "string or null",
                                "canonical_url": "string",
                                "novelty_score": "integer 1-10",
                                "relevance_score": "integer 1-10",
                                "urgency_score": "integer 1-10",
                                "named_entities": [
                                    {"name": "entity name", "type": "person|company|country|institution|index|organization|asset|other"}
                                ],
                                "key_points": [profile.article_key_points_instruction],
                            }
                        ]
                    },
                    "rules": [
                        "Return one article result for every input article.",
                        "Match article results by source_article_id when available and also include canonical_url.",
                        "Do not include category-level summary in this response.",
                        "If an article is tagged as light attention, keep key points especially concise.",
                        "If market_context data is provided, reference specific price movements and percentage changes in your key_points when relevant to the article.",
                    ],
                    "scoring_rubric": {
                        "novelty_score": "How new or non-repetitive this development is within the current news flow.",
                        "relevance_score": "How important this article is for daily finance, equity research, or macro monitoring.",
                        "urgency_score": "How quickly a human analyst should pay attention today.",
                    },
                    "category": category_name,
                    "articles": [
                        {
                            "source_article_id": article["source_article_id"],
                            "canonical_url": article["canonical_url"],
                            "title": article["title"],
                            "published_at": article["published_at"],
                            "article_section": article["section"],
                            "summary_snippet": article["summary_snippet"],
                            "content_text": article["content_text"],
                            "content_truncated": article["content_truncated"],
                            "analysis_method": article["analysis_method"],
                            "attention_tier": article.get("attention_tier"),
                            "theme": article.get("theme"),
                            "must_keep": article.get("must_keep"),
                        }
                        for article in batch_articles
                    ],
                    "market_context": market_context,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _build_synthesis_messages(
    category_name: str,
    synthesis_items: list[dict[str, Any]],
    *,
    bullet_limit: int,
    scope_kind: str,
    scope_title: str,
    market_context: str = "",
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a financial news analyst. Return one valid JSON object only. "
                f"Summarize the {scope_kind} using only the provided article analyses or prior summary blocks."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": f"Synthesize one {scope_kind} from already-analyzed article results.",
                    "required_schema": {
                        "key_developments": [f"2 to {bullet_limit} developments — each must contain a specific detail such as a figure, causal link, named company, or concrete outcome; avoid abstract statements"],
                        "named_entities": [
                            {"name": "entity name", "type": "person|company|country|institution|index|organization|asset|other"}
                        ],
                    },
                    "rules": [
                        "If market_context data is provided, incorporate specific price movements into developments where relevant.",
                    ],
                    "category": category_name,
                    "scope_title": scope_title,
                    "inputs": synthesis_items,
                    "market_context": market_context,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _build_grouping_messages(category_name: str, grouping_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profile = _section_profile(category_name)
    return [
        {
            "role": "system",
            "content": (
                "You are a financial news analyst. Return one valid JSON object only. "
                "Assign every article to one thematic subgroup. Do not omit any article."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Create thematic subgroups inside one financial news category from compact article analyses.",
                    "required_schema": {
                        "subgroups": [
                            {
                                "title": "short subgroup title",
                                "theme_rationale": "one sentence theme explanation",
                                "article_keys": ["article key strings"],
                            }
                        ]
                    },
                    "rules": [
                        "Every input article must appear in exactly one subgroup.",
                        "Avoid many tiny subgroups; focus on 2-3 broad thematic clusters.",
                        f"Target around {profile.subgroup_target_size} article(s) per subgroup when possible.",
                        "Use the provided theme and attention tier as hints when forming subgroups.",
                    ],
                    "category": category_name,
                    "analysis_profile": profile.name,
                    "articles": grouping_items,
                },
                ensure_ascii=False,
            ),
        },
    ]
