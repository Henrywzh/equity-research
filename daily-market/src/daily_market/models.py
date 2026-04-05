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
