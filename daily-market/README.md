# daily-market

Daily macro market price snapshot collection via Yahoo Finance and Polymarket.

Fetches prices for a configurable watchlist (indices, commodities, FX, crypto) via yfinance,
stores snapshots in SQLite, and emails a bilingual (EN/ZH) formatted summary once daily.
It also stores a curated hourly Polymarket watchlist for macro and asset prediction markets.

## Default watchlist

| Class | Symbols |
|-------|---------|
| Indices | `^HSI`, `000001.SS`, `^SPX`, `^NDX`, `^DJI`, `^N225`, `^KS11`, `^STOXX50E`, `^FTSE` |
| Commodities | `GC=F` (Gold), `CL=F` (WTI Crude) |
| FX | `USDCNH=X`, `GBPHKD=X`, `USDJPY=X`, `DX-Y.NYB` |
| Crypto | `BTC-USD` |

## Schedule

| Session | Cron (UTC) | HKT | Purpose |
|---------|-----------|-----|---------|
| Evening | `0 16 * * *` | 00:00 | Daily market brief |

## Commands

All commands: `PYTHONPATH=src python -m daily_market <command>`

| Command | Description |
|---------|-------------|
| `smoke` | Quick connectivity check (3 tickers) |
| `fetch [--session morning\|evening] [--skip-email]` | Full fetch + store + email |
| `local-test [--session] [--skip-email]` | End-to-end test locally |
| `test-email` | Send Gmail connectivity test |
| `inspect [--json]` | Show latest run + sample snapshots |
| `query date YYYY-MM-DD [--json]` | All snapshots for a date |
| `query ticker TICKER [--limit N] [--json]` | Historical snapshots for one ticker |
| `fetch-polymarket [--json]` | Fetch curated Polymarket metadata + snapshots |
| `inspect-polymarket [--json]` | Show latest Polymarket run + sample snapshots |
| `query-polymarket date YYYY-MM-DD [--json]` | Polymarket snapshots for a date |
| `query-polymarket group GROUP [--limit N] [--json]` | History for a Polymarket group |
| `query-polymarket market SLUG_OR_ID [--limit N] [--json]` | History for one Polymarket market |

## Data layout

```
data/
├── market.sqlite              # SQLite: fetch_runs + price_snapshots
├── polymarket_runs/
│   └── YYYY-MM-DD/
│       └── polymarket.json    # Curated Polymarket fetch payload
├── snapshots/
│   └── YYYY-MM-DD/
│       ├── morning.json       # Raw fetched data per run
│       └── evening.json
└── summaries/
    └── YYYY-MM-DD/
        ├── morning.json       # Formatted summary (structured + status)
        └── evening.json
```

`market.sqlite` also stores:

- `polymarket_fetch_runs`
- `polymarket_markets`
- `polymarket_snapshots`

## Credentials

Uses the **same** Gmail secrets as `daily-macro` — no new secrets needed:

```
GMAIL_SENDER        (env var or .config)
GMAIL_APP_PASSWORD  (env var or .config)
GMAIL_RECIPIENT     (env var or .config)
```

For GitHub Actions, these are already set as repo secrets. Locally, set them
in the `.config` file at the repo root (same file used by daily-macro).

## Custom symbols

Edit `config/watchlist.json` to add per-user extra symbols:

```json
{
  "users": {
    "your@email.com": {
      "extra_symbols": [
        "0700.HK",
        {"ticker": "NVDA", "asset_class": "index"}
      ]
    }
  }
}
```

Any Yahoo Finance-supported ticker is valid. No limit on custom symbols.

## Polymarket watchlist

The curated Polymarket list lives in `config/polymarket_watchlist.json`. V1 covers:

- Fed rate cut distributions
- QQQ daily up/down
- BTC daily up/down
- SPX proxy weekly thresholds plus SPX month-end thresholds
- BTC weekly and monthly thresholds
- Gold weekly/month-end thresholds
- Oil weekly/month-end thresholds

The hourly GitHub Actions workflow is `.github/workflows/daily-market-polymarket.yml`.

## Architecture notes

- **No AI in v1** — summaries are mechanical % change tables. A `// TODO: AI commentary`
  placeholder in `formatter.py` marks where LLM synthesis will be added.
- **Latest available data** — yfinance returns the most recent available bar (may be T-1
  for closed markets). The `data_timestamp` field in each snapshot records the bar date.
- **Partial success** — per-ticker failures do not abort the run. Errors are logged in the
  snapshot and the email notes which symbols failed.
- **Curated Polymarket scope** — the Polymarket collector uses a small fixed watchlist and
  deterministic slug templates for rolling daily/weekly markets. Missing or closed contracts
  are recorded as run errors rather than failing the whole hourly job.
