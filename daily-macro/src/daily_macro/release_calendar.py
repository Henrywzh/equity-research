from __future__ import annotations

import json
import os
import re
from calendar import month_abbr
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from .config import get_project_root

FRED_API_URL = "https://api.stlouisfed.org/fred"
FRED_API_KEY_ENV = "FRED_API_KEY"
WATCHLIST_PATH = Path("config") / "release_watchlist.json"
FED_FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
IMPACT_RANK = {"high": 0, "medium": 1, "low": 2}
EVENT_TYPE_RANK = {"statement_day": 0, "meeting_day_1": 1}
MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


@dataclass(frozen=True)
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
    api_key = _get_fred_api_key(required=False)

    items: list[dict[str, Any]] = []
    errors: list[str] = []

    if api_key:
        try:
            watchlist = load_release_watchlist()
            items.extend(fetch_upcoming_releases(start=start, end=end, api_key=api_key, watchlist=watchlist))
        except Exception as exc:
            if require_api_key:
                raise
            errors.append(f"fred:{exc}")
    else:
        if require_api_key:
            raise RuntimeError(
                f"FRED API key not set. Expected {FRED_API_KEY_ENV} in the environment or local .config."
            )
        errors.append("fred:missing_api_key")

    try:
        items.extend(fetch_fomc_events(start=start, end=end))
    except Exception as exc:
        errors.append(f"fomc:{exc}")

    items.sort(key=_release_sort_key)
    fetch_status = _digest_fetch_status(items=items, errors=errors)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "fetch_status": fetch_status,
        "items": items,
        "source": "mixed",
    }
    if errors:
        payload["error_messages"] = list(errors)
    return payload


def fetch_warning_releases(
    *,
    today: str | date | None = None,
    require_api_key: bool = True,
) -> list[dict[str, Any]]:
    base_date = _coerce_date(today) if today is not None else datetime.now(timezone.utc).date()
    digest = build_release_digest(start_date=base_date, days_ahead=2, require_api_key=False)
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
                "release_key": f"fred_{watch_item.release_id}_{item['date']}",
                "name": watch_item.name,
                "date": str(item["date"]),
                "impact": watch_item.impact,
                "series_id": watch_item.series_id,
                "display_unit": watch_item.display_unit,
                "prior_value": None,
                "source": watch_item.source,
            }
        )
    return releases


def fetch_fomc_events(*, start: date, end: date) -> list[dict[str, Any]]:
    response = requests.get(FED_FOMC_URL, timeout=10)
    response.raise_for_status()
    parsed = parse_fomc_events_from_html(response.text, years=range(start.year, end.year + 1))
    return [item for item in parsed if start <= _coerce_date(item["date"]) <= end]


def parse_fomc_events_from_html(html: str, *, years: range | list[int]) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
    events: list[dict[str, Any]] = []
    for year in years:
        events.extend(_parse_fomc_year_lines(lines, year))
    return events


def enrich_releases_with_prior_values(
    releases: list[dict[str, Any]],
    *,
    require_api_key: bool = True,
) -> list[dict[str, Any]]:
    if not releases:
        return []

    api_key = _get_fred_api_key(required=False)
    enriched: list[dict[str, Any]] = []
    watchlist = load_release_watchlist()

    for release in releases:
        enriched_item = dict(release)
        if str(release.get("source") or "").lower() != "fred":
            enriched.append(enriched_item)
            continue
        release_id = release.get("release_id")
        if not isinstance(release_id, int):
            enriched.append(enriched_item)
            continue
        if not api_key:
            if require_api_key:
                raise RuntimeError(
                    f"FRED API key not set. Expected {FRED_API_KEY_ENV} in the environment or local .config."
                )
            enriched.append(enriched_item)
            continue
        watch_item = watchlist.get(release_id)
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


def format_calendar_label(date_value: str, reference_date: date | None) -> str:
    try:
        release_date = date.fromisoformat(date_value[:10])
    except ValueError:
        return date_value or "Unknown"
    absolute = f"{month_abbr[release_date.month]} {release_date.day}"
    if reference_date is not None:
        if release_date == reference_date:
            return f"Today ({absolute})"
        if release_date == reference_date + timedelta(days=1):
            return f"Tomorrow ({absolute})"
    return absolute


def _parse_fomc_year_lines(lines: list[str], year: int) -> list[dict[str, Any]]:
    header = f"{year} FOMC Meetings"
    try:
        start_index = lines.index(header)
    except ValueError:
        return []

    end_index = len(lines)
    for idx in range(start_index + 1, len(lines)):
        if re.fullmatch(r"\d{4} FOMC Meetings", lines[idx]):
            end_index = idx
            break

    events: list[dict[str, Any]] = []
    current_month_header: str | None = None
    current_sep = False
    for line in lines[start_index + 1 : end_index]:
        if line in {"Statement:", "Minutes:", "Press Conference", "Implementation Note"}:
            continue
        normalized = line.lower().strip()
        if normalized in MONTH_NAME_TO_NUMBER or "/" in line:
            current_month_header = line
            current_sep = False
            continue
        match = re.fullmatch(r"(\d{1,2})-(\d{1,2})(\*)?", line)
        if not match or current_month_header is None:
            continue
        current_sep = bool(match.group(3))
        day_one = int(match.group(1))
        day_two = int(match.group(2))
        month_one, month_two = _parse_month_header(current_month_header)
        date_one = date(year, month_one, day_one)
        second_year = year + 1 if month_two < month_one else year
        date_two = date(second_year, month_two, day_two)
        sep_suffix = " (SEP)" if current_sep else ""
        events.extend(
            [
                {
                    "release_id": f"fomc_{date_one.isoformat()}_day1",
                    "release_key": f"fomc_{date_one.isoformat()}_day1",
                    "name": f"FOMC Meeting (Day 1){sep_suffix}",
                    "date": date_one.isoformat(),
                    "impact": "high",
                    "series_id": None,
                    "display_unit": "",
                    "prior_value": None,
                    "source": "Federal Reserve",
                    "event_type": "meeting_day_1",
                    "is_sep_meeting": current_sep,
                },
                {
                    "release_id": f"fomc_{date_two.isoformat()}_statement",
                    "release_key": f"fomc_{date_two.isoformat()}_statement",
                    "name": f"FOMC Statement Day{sep_suffix}",
                    "date": date_two.isoformat(),
                    "impact": "high",
                    "series_id": None,
                    "display_unit": "",
                    "prior_value": None,
                    "source": "Federal Reserve",
                    "event_type": "statement_day",
                    "is_sep_meeting": current_sep,
                },
            ]
        )
        current_month_header = None
    return events


def _parse_month_header(header: str) -> tuple[int, int]:
    parts = [part.strip().lower() for part in header.split("/") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid FOMC month header: {header}")
    first = MONTH_NAME_TO_NUMBER[parts[0]]
    second = MONTH_NAME_TO_NUMBER[parts[-1]]
    return first, second


def _digest_fetch_status(*, items: list[dict[str, Any]], errors: list[str]) -> str:
    if not errors:
        return "success"
    if items:
        return "partial"
    if errors == ["fred:missing_api_key"]:
        return "skipped_missing_api_key"
    return "failed"


def _release_sort_key(item: dict[str, Any]) -> tuple[str, int, int, str]:
    impact = IMPACT_RANK.get(str(item.get("impact") or "").lower(), 99)
    event_type = EVENT_TYPE_RANK.get(str(item.get("event_type") or ""), 99)
    release_key = str(item.get("release_key") or item.get("release_id") or item.get("name") or "")
    return (str(item.get("date") or ""), impact, event_type, release_key)


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
