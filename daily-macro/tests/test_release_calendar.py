from __future__ import annotations

from unittest.mock import patch

import requests

from daily_macro.release_calendar import (
    build_release_digest,
    enrich_releases_with_prior_values,
    fetch_warning_releases,
    format_prior_value,
    load_release_watchlist,
)


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_build_release_digest_skips_when_api_key_missing():
    with patch.dict("os.environ", {}, clear=True), patch(
        "daily_macro.release_calendar._load_local_config", return_value={}
    ):
        digest = build_release_digest(start_date="2026-04-04", days_ahead=7, require_api_key=False)

    assert digest["fetch_status"] == "skipped_missing_api_key"
    assert digest["items"] == []


def test_build_release_digest_filters_and_sorts_watchlist_items():
    with patch.dict("os.environ", {"FRED_API_KEY": "fred-key"}, clear=True), patch(
        "daily_macro.release_calendar.requests.get",
        return_value=_FakeResponse(
            {
                "release_dates": [
                    {"release_id": 36, "date": "2026-04-05"},
                    {"release_id": 9999, "date": "2026-04-04"},
                    {"release_id": 10, "date": "2026-04-04"},
                ]
            }
        ),
    ):
        digest = build_release_digest(start_date="2026-04-04", days_ahead=7, require_api_key=True)

    assert digest["fetch_status"] == "success"
    assert [item["release_id"] for item in digest["items"]] == [10, 36]
    assert digest["items"][0]["name"] == "Consumer Price Index"


def test_fetch_warning_releases_selects_tomorrow_and_enriches_prior_values():
    with patch(
        "daily_macro.release_calendar.build_release_digest",
        return_value={
            "generated_at": "2026-04-04T00:00:00+00:00",
            "window_start": "2026-04-04",
            "window_end": "2026-04-05",
            "fetch_status": "success",
            "source": "FRED",
            "items": [
                {
                    "release_id": 10,
                    "name": "Consumer Price Index",
                    "date": "2026-04-05",
                    "impact": "high",
                    "series_id": "CPIAUCSL",
                    "display_unit": "%",
                    "prior_value": None,
                    "source": "FRED",
                },
                {
                    "release_id": 36,
                    "name": "Producer Price Index",
                    "date": "2026-04-04",
                    "impact": "medium",
                    "series_id": "PPIACO",
                    "display_unit": "%",
                    "prior_value": None,
                    "source": "FRED",
                },
            ],
        },
    ), patch(
        "daily_macro.release_calendar.enrich_releases_with_prior_values",
        side_effect=lambda releases, require_api_key=True: [dict(releases[0], prior_value="3.2%")],
    ):
        releases = fetch_warning_releases(today="2026-04-04", require_api_key=False)

    assert len(releases) == 1
    assert releases[0]["date"] == "2026-04-05"
    assert releases[0]["prior_value"] == "3.2%"


def test_enrich_releases_with_prior_values_degrades_to_none_on_fetch_failure():
    releases = [
        {
            "release_id": 10,
            "name": "Consumer Price Index",
            "date": "2026-04-05",
            "impact": "high",
            "series_id": "CPIAUCSL",
            "display_unit": "%",
            "prior_value": None,
            "source": "FRED",
        }
    ]

    with patch.dict("os.environ", {"FRED_API_KEY": "fred-key"}, clear=True), patch(
        "daily_macro.release_calendar.fetch_prior_value",
        side_effect=RuntimeError("FRED unavailable"),
    ):
        enriched = enrich_releases_with_prior_values(releases, require_api_key=True)

    assert enriched[0]["prior_value"] is None


def test_format_prior_value_applies_divisor_and_suffix():
    item = load_release_watchlist()[50]

    assert format_prior_value("245.0", item) == "245k"
