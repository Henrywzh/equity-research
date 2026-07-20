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
                "Route every article into an attention tier and compact theme. Do not omit any article. "
                "High precision matters more than producing many high-priority items."
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
                                "market_channel": "equity|macro|rates|fx|commodity|geopolitics|property|none|multi",
                                "market_impact_score": "integer 0-5",
                                "urgency_score": "integer 0-3",
                                "novelty_score": "integer 0-2",
                                "priority_score": "integer 0-15, derived as 2*market_impact_score+urgency_score+novelty_score",
                                "reason": "one short sentence",
                                "must_keep": "boolean",
                            }
                        ]
                    },
                    "routing_policy": {
                        "high": [
                            "Use high only when the title/summary contains a concrete event or catalyst and a clear market channel: earnings/guidance/profit warning, buyback/placement/M&A/IPO/regulatory action; a central-bank/rates/inflation/jobs/GDP/fiscal decision or release; sanctions/tariffs/export controls/military escalation; or a material asset move with a stated catalyst.",
                            "A broad word such as 股價, 經濟, 利率, 油價, 樓市, stocks, economy, or oil is not enough by itself. Routine commentary, interviews, forecasts, and daily market wraps are not high without a concrete catalyst.",
                        ],
                        "medium": [
                            "Use medium for relevant company, macro, property, sector, or market context where the impact is plausible but indirect, routine, or not yet a concrete catalyst.",
                            "When evidence is incomplete or ambiguous, choose medium rather than light.",
                        ],
                        "light": [
                            "Use light only for routine/local-interest or generic commentary with no specific actor, event, market channel, or material number. Section name alone must never force light.",
                        ],
                        "scoring": "market_impact_score measures directness/materiality (0-5); urgency_score measures how quickly a human should read it today (0-3); novelty_score measures whether it is a new development rather than context (0-2).",
                        "default": "Use medium when ambiguous. Set must_keep true for high-signal items even if the tier is medium after uncertainty calibration.",
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


def _build_article_quality_review_messages(
    category_name: str,
    article: dict[str, Any],
    first_pass: dict[str, Any],
    *,
    report_date: str | None = None,
) -> list[dict[str, Any]]:
    """Build a compact, source-grounded critic request for one article."""
    return [
        {
            "role": "system",
            "content": (
                "You are a rigorous financial-news quality reviewer. Return one valid JSON object only. "
                "Check the first-pass analysis against the source, preserve the source language, and do not "
                "invent facts. Treat report_date as the reference date when interpreting relative dates."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Review one article analysis and identify material factual or usefulness problems.",
                    "required_schema": {
                        "verdict": "pass|needs_correction|needs_review",
                        "factuality_score": "integer 1-5",
                        "completeness_score": "integer 1-5",
                        "financial_usefulness_score": "integer 1-5",
                        "language_fit_score": "integer 1-5",
                        "issues": ["short, source-grounded issue"],
                        "corrections": ["short correction; empty when none"],
                        "confidence": "number 0-1",
                    },
                    "rules": [
                        "Check dates, numbers, currencies, named entities, and causal claims against the source.",
                        "Do not penalize Chinese source material for being Chinese; judge language fit to the source.",
                        "Use needs_correction when the first-pass result needs a material correction.",
                        "Use needs_review only when the source does not support a reliable decision.",
                        "Keep every issue and correction concise.",
                    ],
                    "category": category_name,
                    "report_date": report_date,
                    "article": {
                        "source_article_id": article.get("source_article_id"),
                        "canonical_url": article.get("canonical_url"),
                        "title": article.get("title"),
                        "published_at": article.get("published_at"),
                        "article_section": article.get("section") or article.get("article_section"),
                        "summary_snippet": article.get("summary_snippet"),
                        "content_text": article.get("content_text"),
                        "content_truncated": article.get("content_truncated"),
                    },
                    "first_pass_analysis": {
                        key: first_pass.get(key)
                        for key in (
                            "source_article_id",
                            "canonical_url",
                            "title",
                            "novelty_score",
                            "relevance_score",
                            "urgency_score",
                            "named_entities",
                            "key_points",
                            "attention_tier",
                            "theme",
                            "must_keep",
                        )
                        if key in first_pass
                    },
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
    lane_names = sorted({str(article.get("research_lane") or "general_research") for article in batch_articles})
    lane_guidance = {
        "macro_policy": "focus on policy transmission, rates, inflation, growth, FX, and timing",
        "hk_china_equity": "focus on company, sector, valuation, earnings, and China/HK market transmission",
        "geopolitical_risk": "focus on sanctions, trade restrictions, conflict escalation, and affected markets",
        "commodities": "focus on supply/demand, inventories, price moves, and commodity-linked assets",
        "company_specific": "focus on earnings, guidance, capital allocation, catalysts, and risks",
        "low_relevance": "keep the analysis concise and flag only concrete market relevance",
        "general_research": "separate verifiable facts from interpretation and identify the market channel",
    }
    lane_instruction = "; ".join(lane_guidance.get(lane, lane_guidance["general_research"]) for lane in lane_names)
    return [
        {
            "role": "system",
            "content": (
                "You are a financial news analyst. Return one valid JSON object only. "
                "Analyze every article in the batch. Use integer scores from 1 to 10. "
                "Do not omit any article. "
                f"This batch is primarily {batch_tier}-attention; keep low-attention stories concise. "
                f"Research-lane guidance: {lane_instruction}."
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
                        "Preserve the source's numeric scale and currency exactly; do not convert 億/億元 into billions or millions unless the conversion is explicit and verified.",
                        "Separate facts stated by the source from interpretation. Do not claim that one market move caused another without source evidence.",
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
                            "research_lane": article.get("research_lane"),
                            "must_keep": article.get("must_keep"),
                            "market_channel": article.get("market_channel"),
                            "priority_score": article.get("priority_score"),
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
                        "key_developments": [
                            {
                                "text": f"2 to {bullet_limit} developments — one sentence each, containing a specific detail such as a figure, causal link, named company, or concrete outcome; avoid abstract statements",
                                "source_article_ids": ["the source_article_id(s) of the input article(s) this development came from"],
                            }
                        ],
                        "named_entities": [
                            {"name": "entity name", "type": "person|company|country|institution|index|organization|asset|other"}
                        ],
                    },
                    "rules": [
                        "Each key_developments item is an object with `text` (a single sentence string) and `source_article_ids`.",
                        "Populate `source_article_ids` only with source_article_id values that appear in the provided inputs; do not invent ids.",
                        "If market_context data is provided, incorporate specific price movements into developments where relevant.",
                        "Preserve every source number's currency, unit, and scale. Do not turn Chinese 億/億元 values into the same numeric value in English billions.",
                        "Do not add a ticker, company, or causal relationship that is not supported by the provided article analyses or market context.",
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
