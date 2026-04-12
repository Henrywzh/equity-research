from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests

from .config import DEFAULT_REQUEST_TIMEOUT, get_polymarket_watchlist_path
from .models import (
    PolymarketFetchRun,
    PolymarketMarket,
    PolymarketSnapshot,
    PolymarketWatchlistConfig,
    PolymarketWatchlistEntry,
)

_GAMMA_BASE = "https://gamma-api.polymarket.com"


def load_polymarket_watchlist(
    config_path: str | Path | None = None,
) -> PolymarketWatchlistConfig:
    path = Path(config_path) if config_path is not None else get_polymarket_watchlist_path()
    raw = json.loads(path.read_text(encoding="utf-8"))

    entries: list[PolymarketWatchlistEntry] = []
    for group in raw.get("groups") or []:
        group_key = str(group["group_key"])
        asset = str(group["asset"])
        horizon = str(group["horizon"])
        for entry in group.get("entries") or []:
            entries.append(
                PolymarketWatchlistEntry(
                    key=str(entry["key"]),
                    group_key=group_key,
                    asset=asset,
                    horizon=str(entry.get("horizon") or horizon),
                    title=str(entry["title"]),
                    entry_type=str(entry["entry_type"]),
                    market_slug=entry.get("market_slug"),
                    event_slug=entry.get("event_slug"),
                    market_slug_template=entry.get("market_slug_template"),
                    event_slug_template=entry.get("event_slug_template"),
                    date_rule=entry.get("date_rule"),
                    selected_strikes=list(entry.get("selected_strikes") or []),
                    source_url=entry.get("source_url"),
                )
            )
    return PolymarketWatchlistConfig(entries=entries)


def fetch_polymarket_watchlist(
    watchlist: PolymarketWatchlistConfig,
    *,
    session: requests.Session | None = None,
    today: date | None = None,
) -> tuple[PolymarketFetchRun, list[PolymarketMarket], list[PolymarketSnapshot], list[str]]:
    now = datetime.now(timezone.utc)
    today = today or now.date()
    run = PolymarketFetchRun(
        id=str(uuid4()),
        started_at=now.isoformat(),
        market_count=len(watchlist.entries),
    )
    client = session or requests.Session()

    markets: list[PolymarketMarket] = []
    snapshots: list[PolymarketSnapshot] = []
    errors: list[str] = []

    for entry in watchlist.entries:
        try:
            market_payload = _resolve_market_payload(entry, client=client, today=today)
        except Exception as exc:
            errors.append(f"{entry.key}: {exc}")
            continue

        if market_payload is None:
            errors.append(f"{entry.key}: no active market matched configured slug(s)")
            continue

        market = _build_market(entry, market_payload, now.isoformat())
        snapshot = _build_snapshot(entry, market_payload, now)
        markets.append(market)
        snapshots.append(snapshot)

    run.finished_at = datetime.now(timezone.utc).isoformat()
    run.snapshot_count = len(snapshots)
    if snapshots and errors:
        run.status = "partial"
    elif snapshots:
        run.status = "success"
    else:
        run.status = "failed"
    run.error_summary = "; ".join(errors[:10]) if errors else None
    return run, markets, snapshots, errors


def _resolve_market_payload(
    entry: PolymarketWatchlistEntry,
    *,
    client: requests.Session,
    today: date,
) -> dict[str, Any] | None:
    candidate_dates = _candidate_dates(entry.date_rule, today)
    market_slugs = _candidate_market_slugs(entry, today)
    event_slugs = _candidate_event_slugs(entry, candidate_dates)
    fallback_payload: dict[str, Any] | None = None

    for idx, slug in enumerate(market_slugs):
        payload = _fetch_market_by_slug(slug, client=client)
        if payload is not None:
            if _is_preferred_market(payload):
                return payload
            fallback_payload = fallback_payload or payload
        event_slug = event_slugs[idx] if idx < len(event_slugs) else entry.event_slug
        if event_slug:
            event_payload = _fetch_event_by_slug(event_slug, client=client)
            market_payload = _find_market_in_event(event_payload, slug)
            if market_payload is not None:
                if _is_preferred_market(market_payload):
                    return market_payload
                fallback_payload = fallback_payload or market_payload
    return fallback_payload


def _candidate_market_slugs(entry: PolymarketWatchlistEntry, today: date) -> list[str]:
    if entry.entry_type == "exact_market":
        return [entry.market_slug] if entry.market_slug else []

    if entry.entry_type != "rolling_market" or not entry.market_slug_template:
        return []

    candidates: list[str] = []
    for candidate_date in _candidate_dates(entry.date_rule, today):
        values = _slug_template_values(candidate_date)
        candidates.append(entry.market_slug_template.format(**values))
    return list(dict.fromkeys(candidates))


def _candidate_event_slugs(
    entry: PolymarketWatchlistEntry,
    candidate_dates: list[date],
) -> list[str]:
    if entry.event_slug:
        return [entry.event_slug for _ in candidate_dates]
    if entry.entry_type != "rolling_market" or not entry.event_slug_template:
        return []

    candidates: list[str] = []
    for candidate_date in candidate_dates:
        values = _slug_template_values(candidate_date)
        candidates.append(entry.event_slug_template.format(**values))
    return candidates


def _candidate_dates(rule: str | None, today: date) -> list[date]:
    if rule == "next_trading_days":
        dates: list[date] = []
        cursor = today
        while len(dates) < 5:
            if cursor.weekday() < 5:
                dates.append(cursor)
            cursor += timedelta(days=1)
        return dates

    if rule == "today_plus_three":
        return [today + timedelta(days=offset) for offset in range(4)]

    if rule == "current_or_next_week_monday":
        if today.weekday() == 6:
            current = today + timedelta(days=1)
        else:
            current = today - timedelta(days=today.weekday())
        return [current, current + timedelta(days=7)]

    if rule == "current_or_next_saturday":
        days_until_saturday = (5 - today.weekday()) % 7
        current = today + timedelta(days=days_until_saturday)
        return [current, current + timedelta(days=7)]

    return [today]


def _slug_template_values(target_date: date) -> dict[str, str]:
    date_slug = f"{target_date.strftime('%B').lower()}-{target_date.day}-{target_date.year}"
    date_slug_short = f"{target_date.strftime('%B').lower()}-{target_date.day}"
    return {
        "date_slug": date_slug,
        "date_slug_short": date_slug_short,
        "year": str(target_date.year),
        "month": target_date.strftime("%B").lower(),
        "day": str(target_date.day),
    }


def _fetch_market_by_slug(slug: str, *, client: requests.Session) -> dict[str, Any] | None:
    response = client.get(
        f"{_GAMMA_BASE}/markets",
        params={"slug": slug},
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload:
        return None
    return payload[0]


def _fetch_event_by_slug(slug: str, *, client: requests.Session) -> dict[str, Any] | None:
    response = client.get(
        f"{_GAMMA_BASE}/events",
        params={"slug": slug},
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload:
        return None
    return payload[0]


def _find_market_in_event(event_payload: dict[str, Any] | None, market_slug: str) -> dict[str, Any] | None:
    if not event_payload:
        return None
    for market in event_payload.get("markets") or []:
        if market.get("slug") == market_slug:
            return market
    return None


def _is_preferred_market(payload: dict[str, Any]) -> bool:
    return bool(payload.get("active")) and not bool(payload.get("closed")) and not bool(payload.get("archived"))


def _build_market(
    entry: PolymarketWatchlistEntry,
    payload: dict[str, Any],
    fetched_at: str,
) -> PolymarketMarket:
    events = payload.get("events") or []
    event = events[0] if events else {}
    outcomes = payload.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            outcomes = None
    yes_label = outcomes[0] if isinstance(outcomes, list) and outcomes else None
    no_label = outcomes[1] if isinstance(outcomes, list) and len(outcomes) > 1 else None

    return PolymarketMarket(
        market_id=str(payload["id"]),
        event_id=str(event["id"]) if event.get("id") is not None else None,
        event_slug=event.get("slug") or entry.event_slug,
        event_title=event.get("title"),
        market_slug=str(payload["slug"]),
        question=str(payload.get("question") or entry.title),
        group_key=entry.group_key,
        asset=entry.asset,
        horizon=entry.horizon,
        end_date=payload.get("endDate") or payload.get("endDateIso"),
        active=bool(payload.get("active")),
        closed=bool(payload.get("closed")),
        archived=bool(payload.get("archived")),
        yes_label=yes_label,
        no_label=no_label,
        outcomes_json=_json_dumps(outcomes),
        outcome_prices_json=_json_dumps(payload.get("outcomePrices")),
        clob_token_ids_json=_json_dumps(payload.get("clobTokenIds")),
        source_url=entry.source_url or _build_market_url(event.get("slug") or entry.event_slug, payload.get("slug")),
        last_metadata_refresh_at=fetched_at,
    )


def _build_snapshot(
    entry: PolymarketWatchlistEntry,
    payload: dict[str, Any],
    now: datetime,
) -> PolymarketSnapshot:
    implied_probability = _extract_implied_probability(payload)
    best_bid = _safe_float(payload.get("bestBid"))
    best_ask = _safe_float(payload.get("bestAsk"))
    midpoint = None
    if best_bid is not None and best_ask is not None:
        midpoint = (best_bid + best_ask) / 2
    spread = None
    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid

    status = "closed" if payload.get("closed") else "active" if payload.get("active") else "inactive"
    if payload.get("archived"):
        status = "archived"

    return PolymarketSnapshot(
        market_id=str(payload["id"]),
        market_slug=str(payload["slug"]),
        group_key=entry.group_key,
        asset=entry.asset,
        horizon=entry.horizon,
        fetched_at=now.isoformat(),
        bucket_hour=now.replace(minute=0, second=0, microsecond=0).isoformat(),
        implied_probability=implied_probability,
        best_bid=best_bid,
        best_ask=best_ask,
        midpoint=midpoint,
        spread=spread,
        last_trade_price=_safe_float(payload.get("lastTradePrice")),
        volume=_safe_float(payload.get("volume")),
        volume_24h=_safe_float(payload.get("volume24hr")),
        liquidity=_safe_float(payload.get("liquidity")) or _safe_float(payload.get("liquidityClob")),
        open_interest=_safe_float(payload.get("openInterest")),
        expiry_timestamp=payload.get("endDate") or payload.get("endDateIso"),
        market_status=status,
        error=None,
    )


def _extract_implied_probability(payload: dict[str, Any]) -> float | None:
    outcome_prices = payload.get("outcomePrices")
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except json.JSONDecodeError:
            outcome_prices = None
    if isinstance(outcome_prices, list) and outcome_prices:
        return _safe_float(outcome_prices[0])

    best_bid = _safe_float(payload.get("bestBid"))
    best_ask = _safe_float(payload.get("bestAsk"))
    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2
    return _safe_float(payload.get("lastTradePrice"))


def _build_market_url(event_slug: str | None, market_slug: str | None) -> str | None:
    if not event_slug:
        return None
    if market_slug:
        return f"https://polymarket.com/event/{event_slug}/{market_slug}"
    return f"https://polymarket.com/event/{event_slug}"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def serialise_polymarket_run_payload(
    run: PolymarketFetchRun,
    markets: list[PolymarketMarket],
    snapshots: list[PolymarketSnapshot],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "run": asdict(run),
        "markets": [asdict(m) for m in markets],
        "snapshots": [asdict(s) for s in snapshots],
        "errors": errors,
    }
