# Daily Macro

`daily-macro` is the repository’s news-input collection layer for daily finance summary, equity research, and macro research workflows.

Its current objective is to scrape and store structured news data that can later feed downstream research, summarization, or analysis pipelines.

## Current Focus

The current source integration is:
- HKEJ `instantnews`

The project currently:
- scrapes the featured stories shown before `最新`
- scrapes all articles currently listed under `最新`
- normalizes article metadata and body text
- stores normalized articles in SQLite
- writes compact parsed JSON article backups
- supports scheduled GitHub Actions runs

## Commands

From inside `daily-macro/`:

```bash
PYTHONPATH=src python -m daily_macro smoke --json
PYTHONPATH=src python -m daily_macro scrape
PYTHONPATH=src python -m daily_macro cleanup --retention-days 30
```

What they do:
- `smoke`: validate that the homepage structure can still be parsed
- `scrape`: fetch the homepage and article pages, then persist normalized data
- `cleanup`: remove old parsed JSON backups based on retention

## Data Layout

Runtime data lives in:
- `data/news.sqlite`
- `data/article_backups/`

The SQLite database is the primary store for normalized article data.

The JSON backups are lightweight article-level snapshots that preserve the parsed result and parser metadata without storing full raw HTML pages.

## Notes

- This subproject is designed to stay simple, reusable, and easy to extend
- It is intentionally focused on scraping and storage only
- Future downstream analysis workflows can read from the SQLite database or the parsed JSON backups
