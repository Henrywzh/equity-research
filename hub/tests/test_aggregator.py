from __future__ import annotations

from unittest.mock import patch

from hub.aggregator import get_latest_fred


def test_get_latest_fred_keeps_empty_digest_empty() -> None:
    with patch(
        "daily_macro.release_calendar.build_release_digest",
        return_value={
            "items": [],
            "generated_at": "2026-04-14T00:00:00+00:00",
            "fetch_status": "skipped_missing_api_key",
        },
    ):
        data = get_latest_fred()

    assert data["items"] == []
    assert data["status"] == "skipped_missing_api_key"


def test_get_latest_fred_filters_legacy_fomc_press_release_noise() -> None:
    with patch(
        "daily_macro.release_calendar.build_release_digest",
        return_value={
            "items": [
                {
                    "release_id": 101,
                    "name": "FOMC Press Release",
                    "date": "2026-04-14",
                    "impact": "medium",
                    "source": "FRED",
                },
                {
                    "release_id": "fomc_2026-04-15_statement",
                    "name": "FOMC Statement Day",
                    "date": "2026-04-15",
                    "impact": "high",
                    "source": "Federal Reserve",
                },
            ],
            "generated_at": "2026-04-14T00:00:00+00:00",
            "fetch_status": "success",
        },
    ):
        data = get_latest_fred()

    assert [item["name"] for item in data["items"]] == ["FOMC Statement Day"]
