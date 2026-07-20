# Multi-Source News and LLM Engine Working Log

<!--
Maintenance contract:
- Hard limit: 500 lines.
- This is a compact decision log, not an append-only transcript.
- Update conclusions in place and remove superseded detail before adding more.
- Keep only current evidence, decisions, rejected options, open questions, and next steps.
- Verify with: wc -l daily-macro/notes/llm-provider-refactor.md
-->

Status: design discussion in progress

Last reviewed: 2026-07-19

Implementation approval: not yet granted

## Objective and Constraints

Build a source-agnostic financial-news analysis engine that:

- is not tied to HKEJ or to one language;
- uses legitimate free API capacity without treating extra keys in one account as extra quota;
- maintains current summary quality or improves it;
- processes 100 accepted articles in less than 30 minutes;
- preserves numbers, units, entities, provenance, and source links;
- degrades gracefully when a source or LLM provider is unavailable.

The working SLA should eventually specify whether “100 articles” means fetched
items, post-deduplication stories, or full-text-equivalent analysis units.

## Confirmed Findings

### Current LLM workload

The 2026-07-16 HKEJ run processed 113 articles in 723 seconds using 279 recorded
LLM request attempts and 458,367 total tokens. Category synthesis accounted for
220 attempts (79%), but this is a request-attempt share, not a token share. It
includes thematic grouping, recursive synthesis/merge work, retries, and failed
calls. Historical diagnostics do not preserve successful token usage by task,
so the synthesis token percentage cannot be reconstructed exactly.

The 2026-07-15 run was less stable: 123 articles, 617 attempts, 478 synthesis
attempts, 326 pre-send splits, 104 fallback switches, and 16 failed batches.
The immediate problem is variance and call amplification rather than average
speed alone.

### Chinese-aware token measurements

The current estimator counts CJK characters at approximately one token each.
Replaying the existing prompt builders against the 2026-07-16 HKEJ corpus gave:

| Prompt payload | Mean input | Maximum input |
| --- | ---: | ---: |
| 5 raw Chinese articles | 1.6K | 3.4K |
| 10 raw Chinese articles | 2.5K | 4.3K |
| 5 synthesis items | 1.7K | 2.3K |
| 10 synthesis items | 2.9K | 4.1K |

The 113 article bodies totalled about 28.9K estimated source tokens. The median
article was 196 tokens, p95 was 707, and the longest was 2,432. Fixed article
counts are still unsafe because another day or source can have much longer
articles. Packing must use estimated tokens and bytes.

Synthesis receives compact analysis results rather than full Chinese bodies.
Those results are currently mostly English, with Chinese titles retained. The
largest one-call category synthesis estimates were about 10.2K input tokens for
`國際財經` and 7.8K for `時事脈搏`; these should remain multi-batch.

### LLM free-capacity implications

- Groq GPT-OSS and Qwen free limits are organization-level. The relevant models
  currently expose 8K TPM and 200K TPD, so multiple keys in one organization do
  not add capacity. A 4.3K input plus 1.2K output reservation consumes most of a
  minute's token allowance.
- Cerebras has no durable free tier. Do not schedule it unless an active credit
  balance and current account limits have been explicitly confirmed.
- Cloudflare Workers AI is the intended bulk lane. It provides 10,000 free
  neurons/day and 300 RPM for text generation. Use Qwen3 30B for now: it is
  healthy, fast, cheap (4,625 input/30,475 output neurons per million tokens),
  and multilingual. GLM-4.7-Flash is explicit opt-in only until its serving
  latency is reliable. Compact outputs remain the largest efficiency lever.
- Google Gemini free input/output remains attractive, but exact limits are tied
  to the project and must be read from AI Studio. Keys in one project must share
  one quota scope. Free-tier content may be used to improve Google products.
- Z.AI `GLM-4.7-Flash` is currently free and is a strong Chinese candidate, but
  its effective account limits must be measured.
- OpenRouter free variants are overflow only: 20 RPM and 50 requests/day unless
  the account has purchased at least $10 of credits, which raises the daily free
  limit to 1,000. Extra keys do not multiply the account limit.

Current provider references:

- Groq: <https://console.groq.com/docs/rate-limits>
- Cerebras: <https://inference-docs.cerebras.ai/support/rate-limits>
- Gemini: <https://ai.google.dev/gemini-api/docs/rate-limits>
- Z.AI: <https://docs.z.ai/guides/overview/pricing>
- Cloudflare: <https://developers.cloudflare.com/workers-ai/platform/pricing/>
- OpenRouter: <https://openrouter.ai/docs/api/reference/limits>

Implemented 2026-07-19: `provider_registry.py` constructs direct Z.AI and
Cloudflare accounts. Cloudflare defaults to Qwen3 30B; GLM-4.7-Flash remains
explicit opt-in after HTTP 500/read-timeout probes. The scheduler atomically
shares the 10K-neuron daily budget, and the client accepts Cloudflare Qwen's
current `reasoning_content` response field. No credential values were copied.

## News Source Inventory

The sibling repository `/Users/henrywzh/Desktop/Quant/alternative-data`
contains `src/global_news_data` with four source clients and a unified Parquet
store. The configured API key names are `GUARDIAN_API_KEY`,
`MARKETAUX_API_KEY`, and `CURRENTS_API_KEY`; GDELT is keyless. No credentials
were copied or exposed.

| Source/API | Free capacity | Available content | Language/coverage | Best engine role | Current issue |
| --- | --- | --- | --- | --- | --- |
| HKEJ scraper | site-dependent | Full body | Traditional Chinese, HK finance | Primary local full-text source | Engine and report are hard-coded around HKEJ |
| Guardian Open Platform | 500 calls/day, 1 call/sec for non-commercial developer key | Full article body | English, broad international reporting | Full-text macro/geopolitical source | Query lacks recency ordering/window; licensing differs for commercial or AI-derived use |
| Marketaux | 100 requests/day, 3 articles/request | Title, description, short snippet, URL, entities and sentiment | Finance-focused, 30+ languages advertised | Ticker discovery, metadata, corroboration | No full article content; fixed symbol list is narrow |
| Currents | 1,000 requests/day, max 20 results/request on free plan | Headlines, descriptions, metadata and links | 70+ countries, 20+ languages advertised | Broad discovery and event monitoring | Current `business` latest feed is noisy; keyword search is not used |
| GDELT DOC 2.0 | Keyless; client caps at 250 records/query | Title, URL, source metadata; no body in current adapter | Global multilingual discovery | Early discovery, cross-source coverage, trend/event signal | Results may be low precision and cannot support body-level summarization alone |

Source references:

- Guardian access: <https://open-platform.theguardian.com/access/>
- Marketaux pricing: <https://www.marketaux.com/pricing>
- Marketaux content FAQ: <https://www.marketaux.com/faq>
- Currents pricing and rights: <https://currentsapi.services/en/product/price>
- GDELT DOC API: <https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/>

### Observed sibling-repository sample

The current normalized Parquet sample has 19 rows:

- Guardian: 3 English records, all with full bodies;
- Marketaux: 3 English records, all with descriptions/snippets;
- Currents: 3 English records, descriptions but no bodies;
- GDELT: 10 records, including 7 Chinese, 2 Indonesian, and 1 Arabic item,
  with titles and URLs but no summaries or bodies.

The Currents sample includes sports and entertainment despite the `business`
category. The Guardian sample includes older articles because the client sends
the query and page size but no `from-date` or `order-by=newest`. The current
store deduplicates only by `(source, article_id)`, so the same story from
different feeds remains duplicated.

### AI-boundary payload measurements

Ingestion mechanics are out of scope for the current refactor. At the AI
boundary, the available text in the sibling sample measured as follows with the
existing CJK-aware estimator:

| Source | Records | Mean available text | Mean title | Current usable depth |
| --- | ---: | ---: | ---: | --- |
| HKEJ 2026-07-16 | 113 | 256 body tokens | not separately measured | Full text |
| Guardian sample | 3 | 684 body tokens | 17 tokens | Full text |
| Marketaux sample | 3 | 77 description/snippet tokens | 19 tokens | Snippet |
| Currents sample | 3 | 52 description tokens | 19 tokens | Headline/description |
| GDELT sample | 10 | 0 body/summary tokens | 27 tokens | Discovery only |

Naively passing each source sample through the current prompts would use about
3,072 input tokens for three Guardian articles, 1,224 for three Marketaux
items, 1,334 for three Currents items, and 2,122 for ten GDELT headlines. The
last two numbers demonstrate waste: metadata-only records should be batched for
triage or attached to a corroborated story, not run through full extraction.

## Proposed Source-Agnostic Contract

Every ingestion adapter should produce a canonical `NewsCandidate` before any
LLM work:

```text
source
source_article_id
canonical_url
published_at
fetched_at
title
summary
content_text
content_level       # full_text | snippet | headline | discovery_only
language            # normalized code plus detected confidence
section_or_topics
provider_entities
provider_sentiment
rights_profile      # internal_only, retention constraints, attribution
raw_metadata
```

Source-specific sections must not become the engine's universal taxonomy.
Routing should map candidates into a shared finance taxonomy after ingestion.

## Proposed Data Flow

```text
source adapters
  -> canonical candidates
  -> deterministic freshness and finance filters
  -> URL/title/story deduplication
  -> content-depth policy
  -> language-aware factual extraction
  -> event/story clustering
  -> token-aware synthesis batches
  -> local rank/deduplicate/merge
  -> optional high-quality final synthesis and critic
```

### Content-depth policy

- `full_text`: eligible for full factual extraction.
- `snippet`: eligible for triage and limited-confidence facts; it must not imply
  facts absent from the snippet.
- `headline`: routing and corroboration only unless another source supplies the
  same story with usable content.
- `discovery_only`: event signal and link only; never summarize as a full article.

This prevents GDELT or Currents metadata from consuming the same LLM budget as a
full HKEJ or Guardian article.

### Multilingual policy

Do not translate every article before analysis; that doubles work and can alter
numbers or units. Extract structured facts in the source language, retaining raw
number/unit strings and provenance. Translate only selected final facts into the
report language.

Initial model candidates:

- Traditional Chinese/Cantonese extraction: Qwen, GLM, then Gemini benchmarked
  against GPT-OSS-120B.
- English extraction: Gemini Flash-Lite, Groq candidates, and GLM/Qwen fallback.
- High-value synthesis: strongest quota-available model that passes the replay
  benchmark, independent of the extraction model.

### Token and quota policy

Article count is not a batch limit. For each provider/model:

```text
reserved tokens = estimated input + requested maximum output
safe input cap = min(context cap, TPM-derived cap, byte cap) - output reserve
```

News API quotas and LLM quotas are separate resources. The scheduler should
first cap and deduplicate source candidates, then schedule accepted analysis
against shared organization/project quota buckets. Retry storms must not use a
different key in the same quota scope as if it were new capacity.

## Capacity Planning for 100 Items

The following treats one 100-item run per day, so submitted requests approximate
RPD demand and total tokens approximate TPD demand. More daily runs multiply
both values linearly.

Measured current-engine equivalents:

| Run basis | Submitted attempts / 100 | Successful calls / 100 | Input / 100 | Output / 100 | Total / 100 | Minimum average TPM for 30 min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HKEJ 2026-07-16 | 247 | 196 | 279K | 127K | 406K | 13.5K |
| HKEJ 2026-07-15 | 502 | 413 | 476K | 249K | 725K | 24.2K |

The July 15/16 gap shows why a single current run can fit a provider's RPD but
still fail its TPD or TPM. Groq's 1,000 RPD is not the immediate obstacle; an
8K TPM and 200K TPD model cannot carry either measured workload alone.

Refactored engineering targets, to be validated by replay:

| Future workload | Attempts / 100 | Input / 100 | Output / 100 | Total / 100 | Minimum average TPM for 30 min |
| --- | ---: | ---: | ---: | ---: | ---: |
| HKEJ or balanced mixed-depth feed | 60-100 | 100-160K | 40-60K | 140-220K | 4.7-7.3K |
| Full-text/Guardian-heavy feed | 70-110 | 140-210K | 50-70K | 190-280K | 6.3-9.3K |

These targets assume deterministic pre-filtering and story deduplication,
metadata-only triage, one factual extraction per usable article/story, compact
fact packets, and roughly 12-25 synthesis/finalization calls. They are estimates,
not observed results.

For p95 completion rather than theoretical saturation, provision at least
10-15K aggregate usable TPM and 5-8 aggregate RPM across eligible providers.
The target RPM is modest; TPM, TPD, provider availability, and model quality are
the constraints. Task-specific output caps are important because providers may
reserve `estimated input + max_completion_tokens`, even when actual output is
shorter.

Example daily multiplication:

- one refactored run: about 60-110 RPD and 140-280K TPD;
- three refactored runs: about 180-330 RPD and 420-840K TPD;
- one current run: about 247-502 RPD and 406-725K TPD.

Provider fit under the refactored target:

- Cloudflare Qwen3 30B would use roughly 1.7-2.6K neurons for the balanced
  140-220K-token target using the target input/output split, leaving useful daily
  headroom;
- Z.AI is the intended quality lane, with concurrency capped and backed off when
  observed 429s indicate account pressure;
- one Groq 8K TPM/200K TPD model is marginal for the light case and insufficient
  for a long full-text run, so Groq remains an opportunistic fast lane;
- Cerebras is disabled unless active credits and limits are confirmed;
- OpenRouter's 50 free requests/day is below the target run range, so it remains
  emergency capacity unless the account qualifies for 1,000 free requests/day;
- Gemini and Z.AI can become primary lanes only after their actual project/account
  quotas and multilingual replay quality are recorded.

## AI-Engine Design Options

### A. Layered fact-packet and story engine — recommended

Deterministically reduce candidates first, then use language/task-specific
models to create compact evidence-backed fact packets. Cluster packets into
cross-source stories/events, synthesize token-aware batches, and merge locally
before an optional final LLM pass.

Advantages: best quota efficiency, supports multilingual and mixed-depth inputs,
and makes provenance and confidence explicit. Cost: targeted restructuring of
the current category workflow and a new intermediate fact-packet contract.

### B. Extend the current category engine

Add `source`, `language`, and `content_level` to current article objects, retain
LLM routing/grouping/category synthesis, and rely on the provider scheduler for
capacity.

Advantages: lowest implementation risk and fastest compatibility. Costs: future
sources multiply cross-source duplicates and synthesis fan-out; the measured
247-502 attempts per 100 items remain plausible.

### C. One-pass large-batch multilingual analysis

Send large token-packed groups to one capable model to perform extraction,
deduplication, clustering, and summary generation together.

Advantages: potentially 15-30 calls per 100 items and simple orchestration.
Costs: large calls are difficult under 8K TPM free limits, omissions are harder
to recover, provenance is weaker, and one provider/model becomes a quality and
availability bottleneck.

Working recommendation: choose A. It reduces quota demand without relying on
large individual calls and allows headline-only feeds to contribute discovery
signals without pretending they contain full evidence.

## High-Recall First-Pass Triage

The current engine already performs a related step: deterministic attention
defaults use title, snippet, and section, followed by category-level LLM routing.
On 2026-07-16 it marked 37 of 113 HKEJ articles light, and light articles already
bypass normal article-analysis batches. A new gate therefore improves the router
and future noisy feeds, but is not a wholly new source of HKEJ savings.

Measured compact-gate cost for the same 113 articles:

| Gate | Calls at 25-40 items | Input total | Estimated compact output | Cloudflare GLM neuron estimate |
| --- | ---: | ---: | ---: | ---: |
| Title only | 3-5 | 4.5-4.7K | about 0.35K | about 38-39 |
| Title plus available snippet | 3-5 | 9.5-9.7K | about 0.35K | about 65-67 |

The additional snippets cost only about 27 Cloudflare neurons, around 0.27% of
the daily free allocation. Prefer title plus snippet/description when available;
fall back to title-only for GDELT-like discovery records.

The response should be compact ID lists rather than per-item explanations:

```text
must_analyze
review
corroboration_only
skip
```

Use deterministic must-keep rules before the model. Permit `skip` only for clear
irrelevance/duplication; uncertainty defaults to `review`. Batch 25-40 items to
limit omission blast radius and keep each request well below constrained-provider
TPM and byte limits. Do not assign verbose themes or reasons in this pass.

Expected value depends on the source. For curated HKEJ, routing-call savings are
modest because light bypass already exists. For Currents/GDELT and overlapping
multi-source feeds, the gate can prevent large numbers of noisy or metadata-only
records from reaching extraction and synthesis.

## Revised AI-Engine Refactor Plan

Z.AI and Cloudflare are now runnable lanes with shared-scope quota accounting.
The next refactor should target the analysis workflow in this order:

1. **Task-level telemetry.** Record submitted, successful, failed, and retried
   calls plus reserved/input/output tokens by task, source, language,
   content level, model, and quota scope. This closes the current synthesis-token
   measurement gap.
2. **Source-agnostic AI input.** Accept canonical candidates with language and
   content level; remove HKEJ-only filtering, filenames, section assumptions,
   and report labels from the analysis core.
3. **Pre-LLM story reduction.** Apply deterministic relevance gates, URL/title
   deduplication, and cross-source story clustering before expensive analysis.
4. **Compact fact packets.** Full text gets evidence-backed extraction; snippets
   get limited-confidence facts; headlines/discovery records only route or
   corroborate. Preserve raw Chinese numbers and units.
5. **Remove synthesis fan-out.** Replace LLM subgroup assignment with story/event
   clusters, provider-aware token packing, task-specific output limits, and local
   rank/deduplicate/merge. Use another LLM merge only for compact final packets.
6. **Language/task model routing.** Benchmark Qwen, GLM, Gemini, and GPT-OSS on
   HKEJ plus English full-text and snippet cases. Select models by measured task
   quality, not one global ranking.
7. **Deadline-aware quota scheduling.** Extend the implemented RPM/TPM/RPD/TPD
   and Cloudflare-neuron budgets with estimated finish time, quality floors, and
   a 25-minute analysis budget with five minutes reserved for finalization.

This changes the previous priority from “add more provider capacity” to
“measure and remove repeated work first.” Future sources strengthen the case:
without content-depth gating and cross-source story deduplication, every new API
would multiply token usage and repeated synthesis.

## Quality and SLA Gates

Replay a multilingual golden set containing HKEJ, Guardian, Marketaux snippets,
Currents descriptions, and GDELT discovery records. Measure:

- numeric, currency, and Chinese unit preservation;
- entity and ticker accuracy;
- unsupported causal claims;
- source/provenance correctness;
- cross-source duplicate and story-cluster precision;
- omission rate for high/medium-priority stories;
- JSON reliability, calls, tokens, rate-limit waits, and wall time.

Rollout gates:

- 100 accepted post-filter items complete within 30 minutes at p95;
- no quality regression versus the current HKEJ engine;
- at least 50% fewer LLM attempts on the HKEJ replay baseline;
- headline/discovery-only records never appear as body-supported claims;
- provider/source failure still produces a clearly marked partial report.

## Decisions and Open Questions

Confirmed:

- The engine must be source-agnostic and multilingual.
- Batching must be token-aware, not based on a fixed article count.
- API keys sharing one provider quota scope do not multiply capacity.
- Source content depth controls how much analysis is allowed.
- Cross-source deduplication must happen before expensive LLM processing.

Proposed, awaiting approval:

- Use the layered fact-packet and story engine.
- Make the AI engine own shared taxonomy, story clustering, LLM work, and reports.
- Treat GDELT and Currents as discovery-first sources.

Open:

- Final report language: English, Traditional Chinese, or configurable.
- Whether “100 articles” is measured before or after cross-source deduplication.
- Intended internal/commercial use, which affects Guardian and Currents rights.

## Next Design Steps

1. Approve the AI-boundary schema and content-depth rules independent of ingestion.
2. Add task/source/language token telemetry before changing batching behavior.
3. Define deterministic finance relevance and story deduplication.
4. Build a multilingual replay set and establish the current quality baseline.
5. Replay-test Z.AI and Cloudflare GLM lanes under real quotas; keep Qwen fallback.
6. Finalize the TPM-aware scheduler and 30-minute deadline behavior.
7. Defer ingestion ownership and integration until the AI design is stable.
8. Write an approved implementation plan only after the design is accepted.
