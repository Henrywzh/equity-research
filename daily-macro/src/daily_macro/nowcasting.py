from __future__ import annotations

import os
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from .nowcast_storage import NowcastStorage


# FRED details re-used from existing logic
FRED_API_URL = "https://api.stlouisfed.org/fred"

SERIES_REGISTRY = [
    {
        "series_id": "cleveland_cpi",
        "source": "cleveland_fed",
        "name": "Headline CPI Nowcast",
        "unit": "percent",
        "description": "Cleveland Fed headline CPI nowcast",
    },
    {
        "series_id": "cleveland_core_cpi",
        "source": "cleveland_fed",
        "name": "Core CPI Nowcast",
        "unit": "percent",
        "description": "Cleveland Fed core CPI nowcast",
    },
    {
        "series_id": "cleveland_pce",
        "source": "cleveland_fed",
        "name": "Headline PCE Nowcast",
        "unit": "percent",
        "description": "Cleveland Fed headline PCE nowcast",
    },
    {
        "series_id": "cleveland_core_pce",
        "source": "cleveland_fed",
        "name": "Core PCE Nowcast",
        "unit": "percent",
        "description": "Cleveland Fed core PCE nowcast",
    },
    {
        "series_id": "atlanta_gdpnow",
        "source": "fred",
        "name": "GDPNow",
        "unit": "percent",
        "description": "Atlanta Fed GDPNow real GDP growth estimate",
    },
]


def fetch_cleveland_fed_nowcasts() -> list[dict]:
    url = "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json"
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

    # The JSON is a list of chart objects, one per target month.
    # We look at the last 2-3 months to get the current and upcoming forecasts.
    for chart_obj in charts[-3:]:
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
            series_id = mapping.get(series_name)
            if not series_id:
                continue

            # The 'data' array contains the revision history for this target month.
            # We want the most recent non-empty value.
            data = dataset.get("data", [])
            latest_val = None
            for item in reversed(data):
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
    response = requests.get(
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
    payload = response.json()
    obs_list = payload.get("observations", [])
    if not obs_list:
        return []

    obs = obs_list[0]
    # Infer quarter
    now = datetime.now()
    quarter = (now.month - 1) // 3 + 1
    target_period = f"{now.year}-Q{quarter}"

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


def refresh_nowcasts(data_dir: str | None = None) -> dict:
    from .config import get_data_dir

    db_path = get_data_dir(data_dir) / "macro.sqlite"
    storage = NowcastStorage(db_path)

    results = {
        "cleveland": {"count": 0, "status": "success", "error": None},
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

    # 1. Cleveland
    try:
        cleveland_obs = fetch_cleveland_fed_nowcasts()
        for obs in cleveland_obs:
            storage.upsert_observation(**obs)
        results["cleveland"]["count"] = len(cleveland_obs)
    except Exception as e:
        results["cleveland"]["status"] = "failed"
        results["cleveland"]["error"] = str(e)

    # 2. GDPNow
    api_key = os.environ.get("FRED_API_KEY")
    if api_key:
        try:
            gdp_obs = fetch_gdpnow_from_fred(api_key)
            for obs in gdp_obs:
                storage.upsert_observation(**obs)
            results["gdpnow"]["count"] = len(gdp_obs)
        except Exception as e:
            results["gdpnow"]["status"] = "failed"
            results["gdpnow"]["error"] = str(e)
    else:
        results["gdpnow"]["status"] = "skipped"
        results["gdpnow"]["error"] = "Missing FRED_API_KEY"

    storage.close()
    return results
