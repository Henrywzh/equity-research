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
- `inspect`: show a quick overview of the latest scrape run and a short list of recent items
- `query date`: list stored articles for a date
- `query search`: search stored articles by keyword
- `query article`: inspect one stored article by URL or source article id

## Data Layout

Runtime data lives in:
- `data/news.sqlite`
- `data/article_backups/`
- `data/analyses/`

The SQLite database is the primary store for normalized article data.

The JSON backups are lightweight article-level snapshots that preserve the parsed result and parser metadata without storing full raw HTML pages.

Daily analysis reports are stored as JSON files. The analysis command uses `GROQ_API_KEY`, starts on `qwen/qwen3-32b`, can fall back to `llama-3.1-8b-instant` when the primary model hits rate limits, and paces requests based on Groq rate-limit headers.

## Notes

- This subproject is designed to stay simple, reusable, and easy to extend
- Articles are analyzed in category-sized batches, and oversized categories are split into smaller sub-batches before analysis
- Saved analysis reports include compact diagnostics for rate-limit waits, batch splits, fallback switches, JSON-repair retries, and failed batches
- Long articles are analyzed in full when they fit the working request budget; otherwise the analyzed slice is truncated and explicitly flagged in the report JSON
- The analysis path uses only non-deprecated Groq models
- Future downstream workflows can read from the SQLite database, parsed JSON backups, or saved daily analysis reports
