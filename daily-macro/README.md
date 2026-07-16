# Daily Macro

`daily-macro` is the repository’s news-input collection layer for daily finance summary, equity research, and macro research workflows.

Its current objective is to scrape and store structured news data that can later feed downstream research, summarization, or analysis pipelines.

## Current Focus

The current source integration is:
- HKEJ `instantnews`

The project currently:
- scrapes the featured stories shown before `最新`
- scrapes all articles listed under paginated `最新` pages until the active listing title changes from `最新`
- normalizes article metadata and body text
- stores normalized articles in SQLite
- writes compact parsed JSON article backups
- analyzes all articles published on a target day with Groq and writes daily JSON reports
- batches article analysis by category and further splits oversized categories into smaller sub-batches when needed
- supports scheduled GitHub Actions runs

## Commands

From inside `daily-macro/`:

```bash
PYTHONPATH=src python -m daily_macro smoke --json
PYTHONPATH=src python -m daily_macro scrape
PYTHONPATH=src python -m daily_macro cleanup --retention-days 30
PYTHONPATH=src python -m daily_macro analyze today
PYTHONPATH=src python -m daily_macro analyze today --force --verbose
PYTHONPATH=src python -m daily_macro build-site --json
PYTHONPATH=src python -m daily_macro notify --result-path data/analyses/2026-04-03/hkej-news-analysis.json
PYTHONPATH=src python -m daily_macro test-email
PYTHONPATH=src python -m daily_macro inspect
PYTHONPATH=src python -m daily_macro query date 2026-04-03
PYTHONPATH=src python -m daily_macro query search "伊朗"
PYTHONPATH=src python -m daily_macro query article --id 4364598
```

What they do:
- `smoke`: validate that the homepage structure can still be parsed
- `scrape`: fetch the homepage and article pages, then persist normalized data
- `cleanup`: remove old parsed JSON backups based on retention
- `analyze today`: analyze articles published on a day and save a reusable JSON report under `data/analyses/`
  - add `--verbose` to print category/batch sizing, waits, splits, fallbacks, and retry diagnostics
- `build-site`: generate a public static site under `site/` from the saved daily analysis reports
- `notify`: send a Gmail summary for a previously generated analysis report
- `test-email`: send a Gmail connectivity check email
- `inspect`: show a quick overview of the latest scrape run and a short list of recent items
- `query date`: list stored articles for a date
- `query search`: search stored articles by keyword
- `query article`: inspect one stored article by URL or source article id

## Data Layout

Runtime data lives in:
- `data/news.sqlite`
- `data/article_backups/`
- `data/analyses/`
- `site/` (generated static reader site)

The SQLite database is the primary store for normalized article data.

The JSON backups are lightweight article-level snapshots that preserve the parsed result and parser metadata without storing full raw HTML pages.

Daily analysis reports are stored as JSON files. The analysis command builds a
quota-aware pool from the configured providers, then chooses a model per task
using context/output limits, JSON support, quality priors, live rate-limit
headers, and recent usage. Groq remains compatible with the existing
`GROQ_API_KEY` setup; Cerebras, Google AI Studio, and OpenRouter are optional.

LLM provider configuration:
- `GROQ_API_KEY`: one key or comma-separated keys. Keys default to one Groq organization quota scope.
- `CEREBRAS_API_KEY`, `CEREBRAS_API_KEY_2`, ...: replacement keys for separate Cerebras accounts. Set `DAILY_MACRO_CEREBRAS_QUOTA_SCOPE` when they share one organization/project.
- `GOOGLE_AI_STUDIO_API_KEY` (or `GEMINI_API_KEY` / `GOOGLE_API_KEY`): Google AI Studio key.
- `OPENROUTER_API_KEY`: OpenRouter key.
- `DAILY_MACRO_GROQ_MODELS`, `DAILY_MACRO_CEREBRAS_MODELS`, `DAILY_MACRO_GOOGLE_AI_MODELS`, `DAILY_MACRO_OPENROUTER_MODELS`: optional comma-separated model overrides. Groq defaults to `qwen/qwen3.6-27b`, followed by `openai/gpt-oss-120b` and `openai/gpt-oss-20b`.
- `DAILY_MACRO_MODEL_POLICY=production_only|allow_preview` and `DAILY_MACRO_MAX_LLM_WAIT_SECONDS`: selection and latency controls. Per-task wait caps such as `DAILY_MACRO_MAX_LLM_WAIT_SECONDS_ARTICLE_ANALYSIS`, plus per-task model preferences such as `DAILY_MACRO_MODEL_TOP_ALERTS_PREFERENCES`, override the defaults.
- `DAILY_MACRO_LLM_PARALLELISM`: opt-in bounded worker count for independent article batches. Defaults to `1` (sequential); start with `3` after replay validation.
- `DAILY_MACRO_PREVIOUS_RETRY_MAX_ARTICLES` and `DAILY_MACRO_PREVIOUS_RETRY_MAX_WAIT_SECONDS`: cap yesterday's salvage work so today's digest keeps priority.
- `DAILY_MACRO_MAX_CATEGORY_SYNTHESIS_WAIT_SECONDS`: cumulative per-category synthesis wait cap; defaults to 180 seconds before local merge.
- `DAILY_MACRO_THEME_CLOSE_DAYS` and `DAILY_MACRO_EVENT_PIPELINE_MODE`: control theme lifecycle expiry and event-layer mode metadata.
- `DAILY_MACRO_LLM_READ_TIMEOUT_SECONDS`, `DAILY_MACRO_LLM_DEADLINE_SECONDS`, and `DAILY_MACRO_LLM_TIMEOUT_COOLDOWN_SECONDS`: stalled-endpoint protection and failover timing controls.

Do not paste API keys into source or reports. Put them in the local `.config`
file or in CI secrets.

Gmail notification can use either daily-macro-specific env vars or generic Gmail env vars:
- `DAILY_MACRO_GMAIL_SENDER`
- `DAILY_MACRO_GMAIL_APP_PASSWORD`
- `DAILY_MACRO_GMAIL_RECIPIENT`
- or `GMAIL_SENDER`, `GMAIL_APP_PASSWORD`, `GMAIL_RECIPIENT`

GitHub Actions automation is split into two workflows:
- `Daily Macro Scraper`: scrape and persist `daily-macro/data/`
- `Daily Macro Analysis`: runs after a successful scrape, generates `data/analyses/`, and sends Gmail when the report status is `success` or `partial`
- `Daily Macro Site`: rebuilds a public GitHub Pages site from the saved reports and publishes the latest digest plus archive

GitHub Secrets for the analysis/email workflow:
- `GROQ_API_KEY`
- optional `CEREBRAS_API_KEY`, `GOOGLE_AI_STUDIO_API_KEY`, and `OPENROUTER_API_KEY`
- `GMAIL_SENDER`
- `GMAIL_APP_PASSWORD`
- `GMAIL_RECIPIENT`

## Notes

- This subproject is designed to stay simple, reusable, and easy to extend
- Articles are analyzed in category-sized batches, and oversized categories are split into smaller sub-batches before analysis
- Saved analysis reports include compact diagnostics for rate-limit waits by endpoint/task, batch splits by kind, resolver substitutions, JSON-repair retries, typed failures, and failed batches. They also include conservative evidence-backed `events`, a `review_queue`, category-level local rollups when synthesis is unavailable, and alert-quality checks for citations, numeric scales, causal wording, and asset identifiers alongside the legacy category fields.
- Long articles are analyzed in full when they fit the working request budget; otherwise the analyzed slice is truncated and explicitly flagged in the report JSON
- The default Groq path uses the approved Qwen 3.6/GPT OSS lineup; Cerebras, Google AI Studio, and OpenRouter remain provider-level fallbacks, while older Llama and Qwen 3.0/3.2 models remain only in historical report fixtures
- Analysis/email automation currently runs after both daily scrape runs, and the public site is rebuilt from the saved report JSON without re-running LLM analysis
- Future downstream workflows can read from the SQLite database, parsed JSON backups, or saved daily analysis reports
