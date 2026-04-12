from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from .config import get_project_root

FRED_API_URL = "https://api.stlouisfed.org/fred"
FRED_API_KEY_ENV = "FRED_API_KEY"
WATCHLIST_PATH = Path("config") / "release_watchlist.json"
IMPACT_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True, slots=True)
class ReleaseWatchItem:
    release_id: int
    name: str
    impact: str
    series_id: str
    series_units: str
    display_unit: str
    display_decimals: int
    display_divisor: float
    source: str


def build_release_digest(
    *,
    start_date: str | date,
    days_ahead: int = 7,
    require_api_key: bool = False,
) -> dict[str, Any]:
    start = _coerce_date(start_date)
    end = start + timedelta(days=max(days_ahead - 1, 0))
    api_key = _get_fred_api_key(required=require_api_key)
    if not api_key:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "fetch_status": "skipped_missing_api_key",
            "items": [],
            "source": "FRED",
        }

    try:
        watchlist = load_release_watchlist()
        releases = fetch_upcoming_releases(start=start, end=end, api_key=api_key, watchlist=watchlist)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "fetch_status": "success",
            "items": releases,
            "source": "FRED",
        }
    except Exception as exc:
        if require_api_key:
            raise
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "fetch_status": "failed",
            "items": [],
            "source": "FRED",
            "error_message": str(exc),
        }


def fetch_warning_releases(
    *,
    today: str | date | None = None,
    require_api_key: bool = True,
) -> list[dict[str, Any]]:
    base_date = _coerce_date(today) if today is not None else datetime.now(timezone.utc).date()
    digest = build_release_digest(start_date=base_date, days_ahead=2, require_api_key=require_api_key)
    tomorrow = (base_date + timedelta(days=1)).isoformat()
    selected = [
        item
        for item in digest.get("items") or []
        if item.get("date") == tomorrow and str(item.get("impact") or "").lower() in {"high", "medium"}
    ]
    return enrich_releases_with_prior_values(selected, require_api_key=require_api_key)


def load_release_watchlist() -> dict[int, ReleaseWatchItem]:
    path = get_project_root() / WATCHLIST_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(item["release_id"]): ReleaseWatchItem(
            release_id=int(item["release_id"]),
            name=str(item["name"]),
            impact=str(item["impact"]).lower(),
            series_id=str(item["series_id"]),
            series_units=str(item.get("series_units") or "lin"),
            display_unit=str(item.get("display_unit") or ""),
            display_decimals=int(item.get("display_decimals", 1)),
            display_divisor=float(item.get("display_divisor", 1.0)),
            source=str(item.get("source") or "FRED"),
        )
        for item in payload
    }


def fetch_upcoming_releases(
    *,
    start: date,
    end: date,
    api_key: str,
    watchlist: dict[int, ReleaseWatchItem] | None = None,
) -> list[dict[str, Any]]:
    watch_items = watchlist or load_release_watchlist()
    response = requests.get(
        f"{FRED_API_URL}/releases/dates",
        params={
            "api_key": api_key,
            "file_type": "json",
            "realtime_start": start.isoformat(),
            "realtime_end": end.isoformat(),
            "order_by": "release_date",
            "sort_order": "asc",
            "include_release_dates_with_no_data": "true",
        },
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()

    releases: list[dict[str, Any]] = []
    for item in payload.get("release_dates", []):
        release_id = int(item["release_id"])
        watch_item = watch_items.get(release_id)
        if watch_item is None:
            continue
        if watch_item.impact not in {"high", "medium"}:
            continue
        releases.append(
            {
                "release_id": watch_item.release_id,
                "name": watch_item.name,
                "date": str(item["date"]),
                "impact": watch_item.impact,
                "series_id": watch_item.series_id,
                "display_unit": watch_item.display_unit,
                "prior_value": None,
                "source": watch_item.source,
            }
        )

    releases.sort(key=lambda item: (str(item["date"]), IMPACT_RANK.get(str(item["impact"]), 99), str(item["name"])))
    return releases


def enrich_releases_with_prior_values(
    releases: list[dict[str, Any]],
    *,
    require_api_key: bool = True,
) -> list[dict[str, Any]]:
    if not releases:
        return []

    api_key = _get_fred_api_key(required=require_api_key)
    if not api_key:
        return releases

    watchlist = load_release_watchlist()
    enriched: list[dict[str, Any]] = []
    for release in releases:
        enriched_item = dict(release)
        watch_item = watchlist.get(int(release["release_id"]))
        if watch_item is None:
            enriched.append(enriched_item)
            continue
        try:
            enriched_item["prior_value"] = fetch_prior_value(watch_item, api_key=api_key)
        except Exception:
            enriched_item["prior_value"] = None
        enriched.append(enriched_item)
    return enriched


def fetch_prior_value(item: ReleaseWatchItem, *, api_key: str) -> str | None:
    response = requests.get(
        f"{FRED_API_URL}/series/observations",
        params={
            "api_key": api_key,
            "series_id": item.series_id,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
            "units": item.series_units,
        },
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    observations = payload.get("observations") or []
    if not observations:
        return None
    raw_value = str(observations[0].get("value") or "").strip()
    if not raw_value or raw_value == ".":
        return None
    return format_prior_value(raw_value, item)


def format_prior_value(raw_value: str, item: ReleaseWatchItem) -> str | None:
    try:
        numeric = float(raw_value) / max(item.display_divisor, 1.0)
    except (TypeError, ValueError):
        return None
    formatted = f"{numeric:.{item.display_decimals}f}"
    if item.display_decimals == 0:
        formatted = formatted.split(".", 1)[0]
    return f"{formatted}{item.display_unit}"


def summarize_release_intensity(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    high_count = sum(1 for item in items if str(item.get("impact") or "").lower() == "high")
    medium_count = sum(1 for item in items if str(item.get("impact") or "").lower() == "medium")
    if high_count >= 3:
        return f"{high_count} high-impact releases in the next 7 days. Treat directional calls as tentative."
    if high_count >= 1:
        return f"{high_count} high-impact and {medium_count} medium-impact releases in the next 7 days."
    return f"{medium_count} medium-impact releases in the next 7 days."


def _get_fred_api_key(*, required: bool) -> str:
    config = _load_local_config()
    value = (os.environ.get(FRED_API_KEY_ENV) or config.get(FRED_API_KEY_ENV) or "").strip()
    if required and not value:
        raise RuntimeError(
            f"FRED API key not set. Expected {FRED_API_KEY_ENV} in the environment or local .config."
        )
    return value


def _load_local_config() -> dict[str, str]:
    for path in _candidate_config_paths():
        if path.exists():
            return _parse_simple_env_file(path)
    return {}


def _candidate_config_paths() -> list[Path]:
    project_root = get_project_root()
    return [project_root / ".config", project_root.parent / ".config"]


def _parse_simple_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _coerce_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
