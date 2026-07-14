# Daily Macro AI Agent Ideas

This note captures the main ideas discussed for improving `daily-macro` beyond article summarization.

## Core Direction

The better use of an AI agent is not to summarize every article. The stronger direction is to combine:

- news signals
- market moves
- macro calendar events
- nowcasts
- watchlists
- cross-day memory
- quality control

The product should behave more like a macro research assistant with memory and market awareness, not a generic news digest.

## Event-Aware Macro Brief

Use the LLM around scheduled macro events, not only around articles.

Example question:

> CPI is tomorrow, Cleveland Fed inflation nowcasts are running hot, US yields moved higher, and today’s news flow has Fed hawkishness. What should I watch?

Useful inputs:

- FRED release calendar
- FOMC calendar
- Cleveland Fed nowcasts
- GDPNow
- market snapshot
- current and prior news flow

Desired output:

- upcoming event risk map
- expected asset channels
- what would surprise markets
- key indicators to monitor
- concise pre-event briefing

## Market Reaction Explainer

Use the agent to explain whether price action matches news flow.

Core question:

> What moved today, and does today’s news explain it?

Examples:

- HSI down while China policy news is positive -> divergence worth flagging
- Brent up and Middle East headlines rising -> price action confirmed by news
- CNH weaker with no obvious FX/macroeconomic news -> unexplained move

Possible output fields:

- asset
- move
- likely explanation
- supporting article/theme IDs
- confidence
- unexplained flag

This is more useful than summarization because it connects news to market behavior.

## Theme Monitor

Themes should be living research objects, not static labels.

Bad themes:

- China
- Markets
- Stocks
- Policy

Good themes:

- China policy easing through property support
- HK property balance-sheet stress
- Fed repricing from sticky inflation
- Middle East oil and shipping risk
- AI capex and semiconductor supply chain
- Tariff escalation pressure on China exporters

A valid theme should have:

- a clear causal mechanism
- affected assets or sectors
- supporting evidence over time, or one very high-impact item
- a direction or lifecycle state

Suggested lifecycle:

```text
new -> active -> strengthening/fading -> dormant -> archived
```

Suggested theme object:

```json
{
  "theme_id": "china_policy_easing_property",
  "title": "China policy easing through property support",
  "status": "active",
  "trend": "strengthening",
  "first_seen": "2026-05-02",
  "last_seen": "2026-06-28",
  "affected_assets": ["HSI", "HSCEI", "CNH", "HK property", "brokers"],
  "core_thesis": "Policy support is shifting from verbal guidance to liquidity/property measures.",
  "evidence_article_ids": [],
  "contradicting_evidence_ids": [],
  "confidence": 0.74
}
```

Daily agent actions:

- attach today’s signal to an existing theme
- update direction and confidence
- create a new theme only when evidence is strong
- merge duplicate themes
- retire/archive stale or resolved themes

Theme size should sit between a single article and a broad regime:

- signal: one article or event
- theme: recurring investable story over days or weeks
- regime: broad macro backdrop over months

`daily-macro` should mostly track themes, not broad regimes.

## Watchlist / Ticker Impact Mapping

For thousands of tickers, do not ask the LLM to know or rank everything. That would hallucinate.

Use deterministic retrieval first, LLM reasoning second.

Pipeline:

```text
article/theme
-> retrieve candidate assets by alias, company name, sector, geography, and macro exposure
-> pass only 10-30 candidates to the LLM
-> force structured impact output
-> abstain when evidence is weak
```

Ticker knowledge base fields:

- ticker
- company name
- Chinese and English aliases
- sector
- country/listing
- revenue/geographic exposure if available
- macro sensitivities: rates, oil, CNH, property, AI, exports
- watchlist priority

Suggested impact output:

```json
{
  "ticker": "0700.HK",
  "impact": "positive|negative|mixed|none",
  "confidence": 0.0,
  "pathway": "direct_mention|sector_readthrough|macro_factor|weak_indirect",
  "evidence": "...",
  "do_not_trade_reason": "optional"
}
```

Important rule:

The LLM should be a reasoning layer over retrieved candidates, not a ticker oracle.

## Report Critic / Quality Gate

A separate critic should evaluate the report before publication.

The writer and critic should not use the same prompt/persona:

- writer: research analyst
- critic: risk/control editor

Critic checks:

- are top alerts actually the most important?
- does every alert have source IDs?
- are claims supported by articles?
- are asset implications explicit?
- are there duplicate themes?
- are ticker calls unsupported?
- are important market moves explained or flagged as unexplained?
- is confidence calibrated?
- is the report concise enough?

Suggested critic output:

```json
{
  "overall_score": 0.82,
  "publish_decision": "publish|publish_with_warnings|hold",
  "issues": [
    {
      "severity": "high",
      "section": "top_alerts",
      "problem": "Oil impact claimed but no article supports Brent linkage.",
      "suggested_fix": "Remove Brent claim or lower confidence."
    }
  ]
}
```

This helps measure how well the LLM performed, not just whether it produced text.

## Source Quality / Noise Filter

Not all HKEJ sections deserve equal LLM spend.

The system should learn which sections and article types are usually low-value and route them to:

- deterministic handling
- light extraction
- local grouping
- no premium model usage

This saves quota and improves report density.

## Signal Extraction Instead Of Article Summarization

Replace “summarize this article” with “extract the investable signal.”

Suggested signal schema:

```json
{
  "signal_type": "macro_policy|equity|geopolitics|commodity|company|noise",
  "claim": "...",
  "evidence": "...",
  "source_article_id": "...",
  "affected_assets": [],
  "confidence": 0.0
}
```

This makes downstream theme tracking, market explanation, and watchlist mapping much easier.

## Recommended Build Order

1. Signal schema
2. Theme monitor
3. Market reaction explainer
4. Watchlist impact mapper
5. Report critic / quality gate

This order builds from structured inputs toward higher-level research judgment.

## Guiding Principle

The agent should answer:

> What matters today, why does it matter, what assets are affected, and how has the story changed over time?

Not merely:

> What did the articles say?
