from __future__ import annotations

import os
import re
from datetime import datetime

import requests

from .http import build_session
from .nowcast_storage import NowcastStorage


# FRED details re-used from existing logic
FRED_API_URL = "https://api.stlouisfed.org/fred"

# Base registry with 4 indicator types. We will expand these with suffixes.
INDICATOR_TYPES = [
    {"base_id": "cleveland_cpi", "name": "Headline CPI", "desc": "Cleveland Fed headline CPI nowcast"},
    {"base_id": "cleveland_core_cpi", "name": "Core CPI", "desc": "Cleveland Fed core CPI nowcast"},
    {"base_id": "cleveland_pce", "name": "Headline PCE", "desc": "Cleveland Fed headline PCE nowcast"},
    {"base_id": "cleveland_core_pce", "name": "Core PCE", "desc": "Cleveland Fed core PCE nowcast"},
]

HORIZONS = [
    {"suffix": "_mom", "label": "(MoM)", "unit": "percent"},
    {"suffix": "_q", "label": "(Quarterly Ann.)", "unit": "percent"},
    {"suffix": "_yoy", "label": "(YoY)", "unit": "percent"},
]

SERIES_REGISTRY = []
for ind in INDICATOR_TYPES:
    for hor in HORIZONS:
        SERIES_REGISTRY.append({
            "series_id": f"{ind['base_id']}{hor['suffix']}",
            "source": "cleveland_fed",
            "name": f"{ind['name']} {hor['label']}",
            "unit": hor['unit'],
            "description": f"{ind['desc']} - {hor['label']}",
        })

# Add GDPNow
SERIES_REGISTRY.append({
    "series_id": "atlanta_gdpnow",
    "source": "fred",
    "name": "GDPNow",
    "unit": "percent",
    "description": "Atlanta Fed GDPNow real GDP growth estimate (Annualized)",
})


def fetch_cleveland_fed_nowcasts(url_type: str = "month") -> list[dict]:
    """
    Fetch nowcasts from Cleveland Fed JSON sources.
    url_type: 'month', 'quarter', or 'year'
    """
    url = f"https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_{url_type}.json"
    suffix = f"_{url_type}"
    if url_type == "month":
        suffix = "_mom"
    elif url_type == "quarter":
        suffix = "_q"
    elif url_type == "year":
        suffix = "_yoy"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    charts = response.json()

    observations = []
    mapping = {
        "CPI Inflation": "cleveland_cpi",
        "Core CPI": "cleveland_core_cpi",
        "Core CPI Inflation": "cleveland_core_cpi",
        "PCE Inflation": "cleveland_pce",
        "Core PCE": "cleveland_core_pce",
        "Core PCE Inflation": "cleveland_core_pce",
    }

    # The JSON is a list of chart objects, one per target period.
    # We look at all returned periods to ensure we have historical context.
    for chart_obj in charts:
        target_period = chart_obj.get("chart", {}).get("subcaption")
        if not target_period:
            continue

        as_of_raw = chart_obj.get("chart", {}).get("_comment", "")
        # Format "2026-04-14 00:00" -> "2026-04-14"
        as_of_date = (
            as_of_raw.split(" ")[0] if as_of_raw else datetime.now().strftime("%Y-%m-%d")
        )

        for dataset in chart_obj.get("dataset", []):
            series_name = dataset.get("seriesname")
            base_id = mapping.get(series_name)
            if not base_id:
                continue
            
            series_id = f"{base_id}{suffix}"

            # The 'data' array contains the revision history for this target period.
            # We want the most recent non-empty value.
            data = dataset.get("data", [])
            latest_val = None
            for item in reversed(data or []):
                val_text = str(item.get("value", "")).strip()
                if val_text and val_text != "":
                    try:
                        latest_val = float(val_text)
                        break
                    except ValueError:
                        continue

            if latest_val is not None:
                observations.append(
                    {
                        "series_id": series_id,
                        "target_period": target_period,
                        "value": latest_val,
                        "as_of_date": as_of_date,
                    }
                )

    return observations


def fetch_gdpnow_from_fred(api_key: str) -> list[dict]:
    session = build_session()
    try:
        response = session.get(
            f"{FRED_API_URL}/series/observations",
            params={
                "api_key": api_key,
                "series_id": "GDPNOW",
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            },
            timeout=10,
        )
        response.raise_for_status()
    finally:
        session.close()
    payload = response.json()
    obs_list = payload.get("observations", [])
    if not obs_list:
        return []

    obs = obs_list[0]
    # Infer quarter
    now = datetime.now()
    quarter = (now.month - 1) // 3 + 1
    target_period = f"{now.year}:Q{quarter}"

    try:
        val = float(obs["value"])
    except (ValueError, TypeError):
        return []

    return [
        {
            "series_id": "atlanta_gdpnow",
            "target_period": target_period,
            "value": val,
            "as_of_date": obs["date"],
        }
    ]


def _is_transient_http_error(exc: Exception) -> bool:
    if not isinstance(exc, requests.HTTPError):
        return False
    response = exc.response
    if response is None:
        return False
    return response.status_code in {429, 500, 502, 503, 504}


def _is_valid_fred_api_key(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9]{32}", value))


def refresh_nowcasts(data_dir: str | None = None) -> dict:
    from .config import get_data_dir

    db_path = get_data_dir(data_dir) / "macro.sqlite"
    storage = NowcastStorage(db_path)

    results = {
        "cleveland": {"count": 0, "status": "success", "error": []},
        "gdpnow": {"count": 0, "status": "success", "error": None},
    }

    # Seed series
    for s in SERIES_REGISTRY:
        storage.ensure_series(
            series_id=s["series_id"],
            source=s["source"],
            name=s["name"],
            unit=s["unit"],
            description=s["description"],
        )

    # 1. Cleveland (Month, Quarter, Year)
    total_cleveland = 0
    for h in ["month", "quarter", "year"]:
        try:
            obs_list = fetch_cleveland_fed_nowcasts(h)
            for obs in obs_list:
                storage.upsert_observation(**obs)
            total_cleveland += len(obs_list)
        except Exception as e:
            results["cleveland"]["status"] = "partial_failure"
            results["cleveland"]["error"].append(f"{h}: {e}")
    
    results["cleveland"]["count"] = total_cleveland

    # 2. GDPNow
    from .release_calendar import _get_fred_api_key

    api_key = _get_fred_api_key(required=False)
    if api_key:
        if not _is_valid_fred_api_key(api_key):
            results["gdpnow"]["status"] = "failed"
            results["gdpnow"]["error"] = (
                "Invalid FRED_API_KEY format. Expected a 32 character lower-case alpha-numeric string."
            )
            storage.close()
            return results
        try:
            gdp_obs = fetch_gdpnow_from_fred(api_key)
            for obs in gdp_obs:
                storage.upsert_observation(**obs)
            results["gdpnow"]["count"] = len(gdp_obs)
        except Exception as e:
            results["gdpnow"]["status"] = "partial_failure" if _is_transient_http_error(e) else "failed"
            results["gdpnow"]["error"] = str(e)
    else:
        results["gdpnow"]["status"] = "skipped"
        results["gdpnow"]["error"] = "Missing FRED_API_KEY"

    storage.close()
    return results
