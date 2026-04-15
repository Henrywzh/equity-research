from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


class DataEngine:
    def __init__(
        self,
        cache_dir: str = "data/cache",
        bootstrap_period: str = "5y",
        incremental_period: str = "5d",
        chunk_size: int = 15,
        stale_after_days: int = 10,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"etf_data_{bootstrap_period}.pkl"
        self.bootstrap_period = bootstrap_period
        self.incremental_period = incremental_period
        self.chunk_size = chunk_size
        self.stale_after_days = stale_after_days
        self.last_refresh_meta: dict[str, object] | None = None

    def fetch_data(self, tickers, period: str | None = None, interval: str = "1d", force_refresh: bool = False):
        """Backward-compatible wrapper around the persisted market-data refresh path."""
        prices, meta = self.refresh_market_data(tickers, interval=interval, force_bootstrap=force_refresh)
        self.last_refresh_meta = meta
        return prices

    def refresh_market_data(
        self,
        tickers,
        interval: str = "1d",
        force_bootstrap: bool = False,
    ) -> tuple[pd.DataFrame | None, dict[str, object]]:
        requested_tickers = self._normalize_tickers(tickers)
        cache = self._load_cache()
        mode_parts: list[str] = []

        if force_bootstrap:
            mode_parts.append("bootstrap")
            cache = None
        elif self._cache_is_stale(cache):
            print(
                f"Warning: Existing cache is stale or unusable at {self.cache_file}. "
                f"Rebuilding {self.bootstrap_period} history."
            )
            mode_parts.append("bootstrap")
            cache = None

        bootstrap_targets = requested_tickers if cache is None else [t for t in requested_tickers if t not in cache.columns]
        bootstrap_prices = pd.DataFrame()
        bootstrap_failed: list[str] = []
        if bootstrap_targets:
            print(
                f"Market data refresh mode: bootstrap ({self.bootstrap_period}) "
                f"for {len(bootstrap_targets)} tickers."
            )
            bootstrap_prices, bootstrap_failed = self._download_prices(
                bootstrap_targets,
                period=self.bootstrap_period,
                interval=interval,
            )
            cache = self._merge_price_frames(cache, bootstrap_prices)
            mode_parts.append("bootstrap")

        incremental_targets = requested_tickers if cache is not None else []
        incremental_prices = pd.DataFrame()
        incremental_failed: list[str] = []
        if cache is not None and requested_tickers:
            has_cached_requested = not cache.columns.intersection(requested_tickers).empty
            should_run_incremental = has_cached_requested and not (
                set(requested_tickers) == set(bootstrap_targets) and bootstrap_failed == []
            )
            if should_run_incremental:
                print(
                    f"Market data refresh mode: incremental ({self.incremental_period}) "
                    f"for {len(requested_tickers)} tickers."
                )
                incremental_prices, incremental_failed = self._download_prices(
                    requested_tickers,
                    period=self.incremental_period,
                    interval=interval,
                )
                cache = self._merge_price_frames(cache, incremental_prices)
                mode_parts.append("incremental")

        if cache is None or cache.empty:
            meta = {
                "mode": "+".join(dict.fromkeys(mode_parts)) or "none",
                "requested_tickers": requested_tickers,
                "refreshed_tickers": [],
                "fallback_tickers": requested_tickers,
                "unavailable_tickers": requested_tickers,
                "last_date": None,
            }
            self.last_refresh_meta = meta
            return None, meta

        cache = self._normalize_prices(cache)
        self._save_cache(cache)

        refreshed_tickers = sorted(
            set(bootstrap_prices.columns.tolist()).union(incremental_prices.columns.tolist())
        )
        available_tickers = [ticker for ticker in requested_tickers if ticker in cache.columns]
        fallback_tickers = sorted(set(available_tickers) - set(refreshed_tickers))
        unavailable_tickers = sorted(set(requested_tickers) - set(cache.columns))
        last_date = cache.index.max()
        last_date_str = last_date.strftime("%Y-%m-%d") if last_date is not None else None

        meta = {
            "mode": "+".join(dict.fromkeys(mode_parts)) or "cache_only",
            "requested_tickers": requested_tickers,
            "refreshed_tickers": refreshed_tickers,
            "fallback_tickers": fallback_tickers,
            "unavailable_tickers": unavailable_tickers,
            "bootstrap_failed": sorted(set(bootstrap_failed)),
            "incremental_failed": sorted(set(incremental_failed)),
            "last_date": last_date_str,
            "cache_file": str(self.cache_file),
        }
        self.last_refresh_meta = meta

        print(
            f"Market data ready from {meta['mode']} path. "
            f"Refreshed {len(refreshed_tickers)} tickers; "
            f"fallback {len(fallback_tickers)}; unavailable {len(unavailable_tickers)}; "
            f"last date {last_date_str}."
        )
        if fallback_tickers:
            print(f"Using cached history for tickers with no fresh update: {', '.join(fallback_tickers)}")
        if unavailable_tickers:
            print(f"Warning: No usable market data available for: {', '.join(unavailable_tickers)}")

        return cache, meta

    def _normalize_tickers(self, tickers) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for ticker in tickers:
            value = str(ticker).strip()
            if value and value not in seen:
                seen.add(value)
                normalized.append(value)
        return normalized

    def _chunk_tickers(self, tickers: list[str]) -> list[list[str]]:
        return [tickers[i : i + self.chunk_size] for i in range(0, len(tickers), self.chunk_size)]

    def _load_cache(self) -> pd.DataFrame | None:
        if not self.cache_file.exists():
            return None
        try:
            cache = pd.read_pickle(self.cache_file)
        except Exception as exc:
            print(f"Warning: Failed to load market cache {self.cache_file}: {exc}")
            return None
        if not isinstance(cache, pd.DataFrame):
            print(f"Warning: Market cache {self.cache_file} is not a DataFrame.")
            return None
        return self._normalize_prices(cache)

    def _save_cache(self, prices: pd.DataFrame) -> None:
        prices.to_pickle(self.cache_file)

    def _cache_is_stale(self, cache: pd.DataFrame | None) -> bool:
        if cache is None or cache.empty:
            return True
        if not isinstance(cache.index, pd.DatetimeIndex):
            return True
        last_date = cache.index.max()
        if pd.isna(last_date):
            return True
        age_days = (datetime.utcnow().date() - last_date.date()).days
        return age_days > self.stale_after_days

    def _download_prices(
        self,
        tickers: list[str],
        *,
        period: str,
        interval: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        if not tickers:
            return pd.DataFrame(), []

        merged = pd.DataFrame()
        failed: list[str] = []
        for chunk in self._chunk_tickers(tickers):
            print(f"Fetching {period} market data from YFinance for chunk: {', '.join(chunk)}")
            try:
                data = yf.download(
                    chunk,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
            except Exception as exc:
                print(f"Error fetching chunk {chunk}: {exc}")
                failed.extend(chunk)
                continue

            prices = self.get_close_prices(data)
            if prices is None or prices.empty:
                failed.extend(chunk)
                continue

            if len(chunk) == 1 and list(prices.columns) == ["Adj Close"]:
                prices.columns = [chunk[0]]
            if len(chunk) == 1 and list(prices.columns) == ["Close"]:
                prices.columns = [chunk[0]]

            successful = set(prices.columns.tolist())
            failed.extend([ticker for ticker in chunk if ticker not in successful])
            merged = self._merge_price_frames(merged, prices)

        return self._normalize_prices(merged), sorted(set(failed))

    def _merge_price_frames(
        self,
        base: pd.DataFrame | None,
        incoming: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if incoming is None or incoming.empty:
            return self._normalize_prices(base) if base is not None else pd.DataFrame()
        if base is None or base.empty:
            return self._normalize_prices(incoming)
        merged = incoming.combine_first(base)
        return self._normalize_prices(merged)

    def _normalize_prices(self, prices: pd.DataFrame | None) -> pd.DataFrame:
        if prices is None or prices.empty:
            return pd.DataFrame()

        frame = prices.copy()
        if isinstance(frame, pd.Series):
            frame = frame.to_frame()

        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index)
        if getattr(frame.index, "tz", None) is not None:
            frame.index = frame.index.tz_localize(None)

        frame = frame.loc[~frame.index.duplicated(keep="last")]
        frame = frame.sort_index()
        frame.columns = [str(col).strip() for col in frame.columns]
        frame.columns.name = None
        frame = frame.loc[:, ~pd.Index(frame.columns).duplicated(keep="last")]
        return frame

    def get_close_prices(self, data):
        """Extract adjusted close prices when available, otherwise close prices."""
        if data is None or data.empty:
            print("Warning: No data provided to get_close_prices.")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                adj_close = data["Adj Close"]
            else:
                adj_close = pd.DataFrame()

            if "Close" in data.columns.get_level_values(0):
                close = data["Close"]
            else:
                close = pd.DataFrame()

            if adj_close.empty:
                prices = close.copy()
            elif close.empty:
                prices = adj_close.copy()
            else:
                prices = adj_close.combine_first(close)
        else:
            print("Extracting from single-level Index...")
            if "Adj Close" in data.columns:
                prices = data["Adj Close"].to_frame()
            elif "Close" in data.columns:
                prices = data["Close"].to_frame()
            else:
                prices = data

        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        prices = self._normalize_prices(prices)

        print(f"Initial tickers in prices DF: {len(prices.columns)}")
        valid_prices = prices.dropna(axis=1, how="all")
        missing = set(prices.columns) - set(valid_prices.columns)
        if missing:
            print(f"Warning: Dropped {len(missing)} tickers due to all NaNs.")

        print(f"Final valid tickers: {len(valid_prices.columns)}")
        return valid_prices
