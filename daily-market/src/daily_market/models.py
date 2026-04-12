from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PriceSnapshot:
    ticker: str
    asset_class: str  # index | commodity | fx | crypto | custom
    fetched_at: str   # ISO-8601 timestamp of when the fetch ran
    price: float | None
    prev_close: float | None
    pct_change: float | None
    open: float | None
    high: float | None
    low: float | None
    volume: float | None  # None for FX; may be 0 for some assets
    data_timestamp: str | None  # date string of the bar (e.g. "2026-04-04")
    error: str | None = None


@dataclass
class FetchRun:
    id: str
    started_at: str
    session: str  # morning | evening
    finished_at: str | None = None
    status: str = "running"  # running | success | partial | failed
    ticker_count: int = 0
    success_count: int = 0
    error_summary: str | None = None


@dataclass
class WatchlistConfig:
    symbols: list[tuple[str, str]]  # (ticker, asset_class)

    @property
    def tickers(self) -> list[str]:
        return [t for t, _ in self.symbols]

    def asset_class_map(self) -> dict[str, str]:
        return {t: ac for t, ac in self.symbols}


@dataclass
class PolymarketWatchlistEntry:
    key: str
    group_key: str
    asset: str
    horizon: str  # daily | weekly | monthly | meeting | year_end
    title: str
    entry_type: str  # exact_market | rolling_market
    market_slug: str | None = None
    event_slug: str | None = None
    market_slug_template: str | None = None
    event_slug_template: str | None = None
    date_rule: str | None = None
    selected_strikes: list[str] = field(default_factory=list)
    source_url: str | None = None


@dataclass
class PolymarketWatchlistConfig:
    entries: list[PolymarketWatchlistEntry]


@dataclass
class PolymarketFetchRun:
    id: str
    started_at: str
    finished_at: str | None = None
    status: str = "running"  # running | success | partial | failed
    market_count: int = 0
    snapshot_count: int = 0
    error_summary: str | None = None


@dataclass
class PolymarketMarket:
    market_id: str
    event_id: str | None
    event_slug: str | None
    event_title: str | None
    market_slug: str
    question: str
    group_key: str
    asset: str
    horizon: str
    end_date: str | None
    active: bool
    closed: bool
    archived: bool
    yes_label: str | None
    no_label: str | None
    outcomes_json: str | None
    outcome_prices_json: str | None
    clob_token_ids_json: str | None
    source_url: str | None
    last_metadata_refresh_at: str


@dataclass
class PolymarketSnapshot:
    market_id: str
    market_slug: str
    group_key: str
    asset: str
    horizon: str
    fetched_at: str
    bucket_hour: str
    implied_probability: float | None
    best_bid: float | None
    best_ask: float | None
    midpoint: float | None
    spread: float | None
    last_trade_price: float | None
    volume: float | None
    volume_24h: float | None
    liquidity: float | None
    open_interest: float | None
    expiry_timestamp: str | None
    market_status: str
    error: str | None = None
