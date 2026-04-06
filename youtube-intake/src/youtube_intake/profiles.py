"""Profile-based prompt and schema registry for finance-domain video analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


NOISE_PREAMBLE = (
    "Completely ignore sponsor reads, YouTube engagement hooks, subscription reminders, "
    "lifestyle anecdotes, channel merchandise plugs, and any non-market commentary. "
    "Return JSON only."
)

COMMON_OUTPUT_FIELDS: dict[str, Any] = {
    "executive_summary": "string",
    "key_timestamps": [
        {
            "timestamp": "HH:MM:SS",
            "label": "string",
            "snippet": "string",
            "why_it_matters": "string",
        }
    ],
    "topic_tags": [{"tag": "string", "score": "0-100 integer"}],
    "tickers_mentioned": ["string"],
    "confidence": "0-1 float",
}


@dataclass(frozen=True)
class AnalysisProfile:
    name: str
    system_prompt: str
    output_schema: dict[str, Any]
    synthesis_section: str
    chunk_system_prompt: str
    consolidation_system_prompt: str
    extra_fields: tuple[str, ...] = ()

    def build_full_schema(self) -> dict[str, Any]:
        merged = dict(COMMON_OUTPUT_FIELDS)
        merged.update(self.output_schema)
        return merged


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

PROFILES: dict[str, AnalysisProfile] = {}


def _register(profile: AnalysisProfile) -> AnalysisProfile:
    PROFILES[profile.name] = profile
    return profile


_register(AnalysisProfile(
    name="technical_trading",
    system_prompt=(
        "You are a quantitative technical analyst. Analyze this video transcript for "
        "actionable trading intelligence. Extract specific tickers, directional bias, "
        "key price levels, trade setups, and timeframes. Focus on concrete data — prices, "
        "percentages, support/resistance zones, and risk/reward ratios. "
        + NOISE_PREAMBLE
    ),
    output_schema={
        "trading_signals": [
            {
                "ticker": "string",
                "direction": "bullish | bearish | neutral",
                "reasoning": "string",
                "timeframe": "string (e.g. intraday, swing, weekly)",
            }
        ],
        "key_price_levels": [
            {
                "ticker": "string",
                "level": "number or string",
                "significance": "support | resistance | target | stop_loss",
            }
        ],
    },
    synthesis_section="📈 Trading Desk",
    chunk_system_prompt=(
        "You are a quantitative technical analyst. Review this chronological transcript chunk "
        "for trading signals, price levels, and directional bias. Return JSON only."
    ),
    consolidation_system_prompt=(
        "You are a quantitative technical analyst consolidating chunk-level findings into one "
        "video analysis. Merge trading signals and key price levels. Return JSON only."
    ),
    extra_fields=("trading_signals", "key_price_levels"),
))


_register(AnalysisProfile(
    name="macroeconomics",
    system_prompt=(
        "You are a global macro research analyst. Analyze this video transcript for "
        "macroeconomic intelligence. Extract monetary policy views, inflation data points, "
        "bond market commentary, central bank actions, GDP/labor statistics, and their "
        "market implications. Prioritize concrete numbers over vague opinions. "
        + NOISE_PREAMBLE
    ),
    output_schema={
        "macro_developments": ["string"],
        "monetary_policy_views": ["string"],
        "inflation_expectations": ["string"],
    },
    synthesis_section="🏛️ Macro & Rates",
    chunk_system_prompt=(
        "You are a global macro research analyst. Review this transcript chunk for "
        "macroeconomic data, policy shifts, and market implications. Return JSON only."
    ),
    consolidation_system_prompt=(
        "You are a global macro research analyst consolidating chunk-level findings into one "
        "video analysis. Deduplicate and merge macro developments. Return JSON only."
    ),
    extra_fields=("macro_developments", "monetary_policy_views", "inflation_expectations"),
))


_register(AnalysisProfile(
    name="geopolitics",
    system_prompt=(
        "You are a geopolitical intelligence analyst specializing in defense, military, "
        "and international relations. Analyze this video transcript for geopolitical events, "
        "military developments, sanctions, policy changes, supply chain disruptions, and "
        "their impact on commodities and global markets. "
        + NOISE_PREAMBLE
    ),
    output_schema={
        "geopolitical_events": ["string"],
        "affected_commodities": ["string"],
        "supply_chain_impacts": ["string"],
        "policy_changes": ["string"],
    },
    synthesis_section="🌍 Geopolitical Watch",
    chunk_system_prompt=(
        "You are a geopolitical intelligence analyst. Review this transcript chunk for "
        "military, defense, and policy developments. Return JSON only."
    ),
    consolidation_system_prompt=(
        "You are a geopolitical intelligence analyst consolidating chunk-level findings. "
        "Merge events and impacts. Return JSON only."
    ),
    extra_fields=("geopolitical_events", "affected_commodities", "supply_chain_impacts", "policy_changes"),
))


_register(AnalysisProfile(
    name="tech_finance",
    system_prompt=(
        "You are a technology and financial markets analyst. Analyze this video transcript "
        "for AI and technology developments AND their financial market implications. "
        "Track company mentions, sector effects, revenue/valuation data, and which tickers "
        "are affected. Connect tech narratives to investable conclusions. "
        + NOISE_PREAMBLE
    ),
    output_schema={
        "tech_developments": ["string"],
        "market_implications": ["string"],
        "companies_mentioned": ["string"],
    },
    synthesis_section="💻 Tech & Markets",
    chunk_system_prompt=(
        "You are a technology and financial markets analyst. Review this transcript chunk "
        "for tech/AI developments and their market implications. Return JSON only."
    ),
    consolidation_system_prompt=(
        "You are a technology and financial markets analyst consolidating chunk-level "
        "findings. Merge tech developments and market implications. Return JSON only."
    ),
    extra_fields=("tech_developments", "market_implications", "companies_mentioned"),
))


_register(AnalysisProfile(
    name="news_research",
    system_prompt=(
        "You are an investigative news analyst. Analyze this video transcript for key facts, "
        "verifiable claims, data points, and forward-looking consequences from in-depth "
        "reporting. Distinguish confirmed facts from speculation. Focus on developments "
        "that may affect financial markets, policy, or public sentiment. "
        + NOISE_PREAMBLE
    ),
    output_schema={
        "key_facts": ["string"],
        "forward_implications": ["string"],
    },
    synthesis_section="📰 News Digest",
    chunk_system_prompt=(
        "You are an investigative news analyst. Review this transcript chunk for verifiable "
        "facts and forward-looking implications. Return JSON only."
    ),
    consolidation_system_prompt=(
        "You are an investigative news analyst consolidating chunk-level findings. "
        "Merge key facts and implications. Return JSON only."
    ),
    extra_fields=("key_facts", "forward_implications"),
))


_register(AnalysisProfile(
    name="institutional_macro",
    system_prompt=(
        "You are an institutional research analyst at a major investment bank. Analyze this "
        "video transcript for investment themes, asset allocation views, risk assessments, "
        "and market outlook. Extract specific trade recommendations, probability-weighted "
        "scenarios, and any price targets or return forecasts. "
        + NOISE_PREAMBLE
    ),
    output_schema={
        "investment_themes": ["string"],
        "asset_allocation_views": ["string"],
        "risk_assessments": ["string"],
        "market_outlook": "string",
    },
    synthesis_section="🏦 Institutional Research",
    chunk_system_prompt=(
        "You are an institutional research analyst. Review this transcript chunk for "
        "investment themes, allocation views, and market outlook. Return JSON only."
    ),
    consolidation_system_prompt=(
        "You are an institutional research analyst consolidating chunk-level findings. "
        "Merge investment themes and risk assessments. Return JSON only."
    ),
    extra_fields=("investment_themes", "asset_allocation_views", "risk_assessments", "market_outlook"),
))


DEFAULT_PROFILE_NAME = "macroeconomics"


def get_profile(name: str | None) -> AnalysisProfile:
    """Look up a profile by name, falling back to macroeconomics."""
    if name and name in PROFILES:
        return PROFILES[name]
    return PROFILES[DEFAULT_PROFILE_NAME]


# Ordered list of synthesis sections for email rendering.
SYNTHESIS_SECTION_ORDER: list[str] = [
    "📈 Trading Desk",
    "🏛️ Macro & Rates",
    "🏦 Institutional Research",
    "🌍 Geopolitical Watch",
    "💻 Tech & Markets",
    "📰 News Digest",
]
