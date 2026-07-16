# Video News Workflow Review

Source: [YouTube video](https://www.youtube.com/watch?v=fRX1vvqr61Q), "Lightweight reproduction: how an AI morning-news production line is built" by Juya Juya (8:25, published 2025-12-09).

Local review bundle: `daily-macro/local_experiments/video-analysis/fRX1vvqr61Q/`.

## What The Video Actually Does

The workflow is a three-stage editorial pipeline rather than a free-running agent:

1. **Collect**: RSS/RSSHub, manual browser clipping to Markdown with metadata, X/community leads, comments, private messages, and groups.
2. **Filter and process**: AI removes irrelevant or low-value items, creates a short summary, extracts keywords, detects old/repeated news against the previous issue and a 72-hour window, and assigns a value score.
3. **Edit and distribute**: candidates are grouped by event; a human chooses what to report, keeps the most official/earliest/accurate sources, adds official context, proofreads the AI draft, preserves links, chooses media, and publishes text and video versions.

The important unit is the **event**, not the individual article. Multiple sources are first made visible as one candidate event. AI is then used to enrich and draft; humans decide which event survives and which evidence is authoritative.

## The Human Checkpoints Are Deliberate

The video does not claim that full automation is the goal. Human review is concentrated at the points where an error changes editorial meaning:

- select events for the briefing;
- choose the best source for each event;
- supplement official information;
- correct hallucinations, links, and media;
- proofread generated text.

This is a useful model for Daily Macro: automate breadth and repetitive structure, but expose a compact evidence-backed review queue for high-impact or low-confidence items.

## Mapping To Daily Macro

Our current graph already has a strong backbone:

`initialize -> fetch_market_data -> route_attention -> analyze_today -> retry_previous_day -> validate_outputs -> update_theme_memory -> summarize_top_alerts -> finalize`

The closest matches are:

- `route_attention`: the video's first-pass value scoring and triage;
- `analyze_today`: article-level structured enrichment and category synthesis;
- `validate_outputs`: the beginning of the hallucination and schema check;
- `update_theme_memory`: the feedback/memory loop;
- `summarize_top_alerts`: the editorial briefing layer.

The main difference is that our current core object is still mostly an article/category report, while the video treats an event and its source set as the editorial object.

## Design Changes Worth Adopting

### 1. Make event packets first-class

Create an intermediate event packet before category synthesis. It should contain:

- stable `event_id` and normalized event title;
- member `source_article_ids`;
- source URLs, publisher, publication times, and source-quality signals;
- a short neutral fact summary;
- novelty/old-news status;
- market relevance and affected assets/themes;
- contradictions and missing evidence;
- confidence and a reason for the score.

This lets the system deduplicate and compare sources before asking a model to write prose. It also gives the report critic something auditable to inspect.

### 2. Separate triage from synthesis

Use cheap deterministic rules plus a fast model for:

- relevance, novelty, source quality, event key, and attention tier;
- short summaries and entity extraction.

Reserve the strongest model and the larger request budget for:

- high-impact event synthesis;
- cross-source contradiction resolution;
- top alerts and the critic.

Do not send every raw article to the most capable model. Send event packets containing only the best evidence plus links to the full records.

### 3. Add a review queue instead of pretending certainty

A human-review queue should be generated for events with any of these signals:

- high market impact;
- conflicting sources;
- only one weak source;
- unsupported claim or citation;
- low model confidence;
- unusual or new theme.

For a fully scheduled run, the queue can be included in the email/site output. It does not need to block a usable degraded digest.

### 4. Treat the report as a structured draft

Keep facts, evidence, analysis, and prose separate. A practical order is:

`source records -> event packets -> selected evidence -> structured analysis -> alert objects -> rendered briefing`

The renderer should consume validated objects, not raw model text. This also makes text, email, site, and future video outputs different views of the same research record.

### 5. Extend theme memory around events

The video uses a recent-history check mainly to remove repeats. Daily Macro can go further: themes should be mutable collections of related events with first/last seen dates, affected assets, trend, confidence, and related article/event IDs. A new event can open a theme, extend it, split it, merge it, or close it; the system should preserve the reason for that transition.

## Video Production Ideas Relevant Later

The video generation branch is downstream of the editorial record:

- generate an event-segmented narration script;
- split narration into short TTS requests;
- measure each audio segment to build the timeline;
- render HTML visual cards in a browser and capture them;
- combine narration, cards, subtitles, transitions, music, and effects with ffmpeg.

That suggests a future Daily Macro media layer should consume the same validated event packets and alert objects. It should not ask a second model to rediscover the news from the final prose.

## Bottom Line

The most useful idea to borrow is not "add an agent." It is the separation of responsibilities: broad automated intake, event-centric candidate formation, targeted human or critic review, then multi-channel rendering. Our current LangGraph backbone can support this; the next architectural improvement should be an event/evidence layer between routing and category synthesis.
