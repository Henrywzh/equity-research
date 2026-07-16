# Event-Centric Daily Macro Agent

## Current Graph

```text
initialize
  -> fetch_market_data
  -> route_attention
  -> analyze_today
  -> build_event_packets
  -> retry_previous_day
  -> validate_outputs
  -> update_theme_memory
  -> summarize_top_alerts
  -> critic_outputs
  -> finalize
```

The article and category report remains the compatibility surface. The event
layer is an additional evidence-oriented view built from successful article
analyses, so existing email/site consumers can continue using the legacy
fields while new consumers use `events` and `review_queue`.

## Execution Policy

`ModelResolver` chooses an endpoint for each task. It checks lifecycle policy,
active provider models, JSON support, context/output fit, daily token budget,
quota wait, task quality, and premium-model reservation. Production models are
the default; preview models require explicit policy.

Temporary quota exhaustion is different from endpoint failure:

- a reset within the task budget is waited out on the same endpoint;
- a reset beyond the task budget moves to another eligible endpoint;
- timeout, connection, deprecation, or invalid endpoint errors can cool down or
  evict the endpoint;
- if no endpoint is usable, the affected unit degrades locally and the run
  continues.

Task wait budgets can be overridden with variables such as
`DAILY_MACRO_MAX_LLM_WAIT_SECONDS_ARTICLE_ANALYSIS` and the legacy global
`DAILY_MACRO_MAX_LLM_WAIT_SECONDS`.

## Event Packets

Event clustering is intentionally conservative: articles must share category,
research lane, theme, a 72-hour window, and strong title/evidence overlap (or
shared entities plus meaningful overlap). Each packet stores source IDs, facts,
evidence, affected assets, novelty, market relevance, confidence, and a stable
event ID. High-impact single-source packets enter the review queue instead of
being treated as confirmed multi-source events.

Themes remain living objects in `theme_memory.json`. They retain related article
and event IDs, transition through open/cooling/closed states, and become closed
after `DAILY_MACRO_THEME_CLOSE_DAYS` of inactivity (seven days by default).

## Critic Gate

The current critic is deterministic and quota-free. It removes alerts without
valid source citations, strips unsupported source IDs, and records critic issues
before rendering. A future premium LLM critic can be enabled for only the small
high-impact review queue without adding a critic call to every article.

## Diagnostics

Reports now include task and endpoint usage, wait detail by endpoint, typed
failure counts, split counts by batch kind, resolver rejection/substitution
records, event counts, and critic issues. Run notes show a compact summary of
these fields, including previous-day retry work deferred to protect today's
quota.
