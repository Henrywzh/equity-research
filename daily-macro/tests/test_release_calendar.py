from __future__ import annotations

from datetime import date
from unittest.mock import patch

import requests

from daily_macro.release_calendar import (
    _get_fred_api_key,
    build_release_digest,
    enrich_releases_with_prior_values,
    fetch_warning_releases,
    format_calendar_label,
    format_prior_value,
    load_release_watchlist,
    parse_fomc_events_from_html,
)


FOMC_HTML = """
<html>
  <body>
    <h4>2026 FOMC Meetings</h4>
    <p>January</p>
    <p>27-28</p>
    <p>Statement:</p>
    <p>March</p>
    <p>17-18*</p>
    <p>Statement:</p>
    <h4>2027 FOMC Meetings</h4>
    <p>January</p>
    <p>26-27</p>
  </body>
</html>
"""


def test_build_release_digest_without_fred_key_keeps_fomc_items():
    with patch.dict("os.environ", {}, clear=True), patch(
        "daily_macro.release_calendar._load_local_config",
        return_value={},
    ), patch(
        "daily_macro.release_calendar.fetch_fomc_events",
        return_value=[
            {
                "release_id": "fomc_2026-04-08_statement",
                "release_key": "fomc_2026-04-08_statement",
                "name": "FOMC Statement Day",
                "date": "2026-04-08",
                "impact": "high",
                "series_id": None,
                "display_unit": "",
                "prior_value": None,
                "source": "Federal Reserve",
                "event_type": "statement_day",
                "is_sep_meeting": False,
            }
        ],
    ):
        digest = build_release_digest(start_date="2026-04-04", days_ahead=7, require_api_key=False)

    assert digest["fetch_status"] == "partial"
    assert digest["items"][0]["source"] == "Federal Reserve"


def test_build_release_digest_merges_fred_and_fomc_items():
    with patch.dict("os.environ", {"FRED_API_KEY": "fred-key"}, clear=True), patch(
        "daily_macro.release_calendar.fetch_upcoming_releases",
        return_value=[
            {
                "release_id": 10,
                "release_key": "fred_10_2026-04-04",
                "name": "Consumer Price Index",
                "date": "2026-04-04",
                "impact": "high",
                "series_id": "CPIAUCSL",
                "display_unit": "%",
                "prior_value": None,
                "source": "FRED",
            }
        ],
    ), patch(
        "daily_macro.release_calendar.fetch_fomc_events",
        return_value=[
            {
                "release_id": "fomc_2026-04-05_day1",
                "release_key": "fomc_2026-04-05_day1",
                "name": "FOMC Meeting (Day 1)",
                "date": "2026-04-05",
                "impact": "high",
                "series_id": None,
                "display_unit": "",
                "prior_value": None,
                "source": "Federal Reserve",
                "event_type": "meeting_day_1",
                "is_sep_meeting": False,
            }
        ],
    ):
        digest = build_release_digest(start_date="2026-04-04", days_ahead=7, require_api_key=True)

    assert digest["fetch_status"] == "success"
    assert [item["name"] for item in digest["items"]] == ["Consumer Price Index", "FOMC Meeting (Day 1)"]


def test_get_fred_api_key_strips_surrounding_quotes():
    with patch.dict("os.environ", {"FRED_API_KEY": '"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"'}, clear=True):
        assert _get_fred_api_key(required=False) == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_parse_fomc_events_from_html_builds_both_days_and_sep_flag():
    events = parse_fomc_events_from_html(FOMC_HTML, years=[2026])

    assert [event["name"] for event in events] == [
        "FOMC Meeting (Day 1)",
        "FOMC Statement Day",
        "FOMC Meeting (Day 1) (SEP)",
        "FOMC Statement Day (SEP)",
    ]
    assert events[1]["event_type"] == "statement_day"
    assert events[2]["is_sep_meeting"] is True
    assert events[3]["date"] == "2026-03-18"


def test_fetch_warning_releases_selects_fomc_statement_day():
    with patch(
        "daily_macro.release_calendar.build_release_digest",
        return_value={
            "generated_at": "2026-04-04T00:00:00+00:00",
            "window_start": "2026-04-04",
            "window_end": "2026-04-05",
            "fetch_status": "success",
            "source": "mixed",
            "items": [
                {
                    "release_id": "fomc_2026-04-05_statement",
                    "release_key": "fomc_2026-04-05_statement",
                    "name": "FOMC Statement Day",
                    "date": "2026-04-05",
                    "impact": "high",
                    "series_id": None,
                    "display_unit": "",
                    "prior_value": None,
                    "source": "Federal Reserve",
                    "event_type": "statement_day",
                    "is_sep_meeting": False,
                },
                {
                    "release_id": 36,
                    "release_key": "fred_36_2026-04-04",
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
    ):
        releases = fetch_warning_releases(today="2026-04-04", require_api_key=False)

    assert len(releases) == 1
    assert releases[0]["name"] == "FOMC Statement Day"
    assert releases[0]["prior_value"] is None


def test_enrich_releases_with_prior_values_skips_non_fred_items():
    releases = [
        {
            "release_id": "fomc_2026-04-05_day1",
            "release_key": "fomc_2026-04-05_day1",
            "name": "FOMC Meeting (Day 1)",
            "date": "2026-04-05",
            "impact": "high",
            "series_id": None,
            "display_unit": "",
            "prior_value": None,
            "source": "Federal Reserve",
            "event_type": "meeting_day_1",
            "is_sep_meeting": False,
        }
    ]

    enriched = enrich_releases_with_prior_values(releases, require_api_key=True)

    assert enriched[0]["prior_value"] is None
    assert enriched[0]["source"] == "Federal Reserve"


def test_format_prior_value_applies_divisor_and_suffix():
    item = load_release_watchlist()[50]

    assert format_prior_value("245.0", item) == "245k"


def test_format_calendar_label_includes_month_for_all_events():
    reference = date(2026, 4, 14)

    assert format_calendar_label("2026-04-14", reference) == "Today (Apr 14)"
    assert format_calendar_label("2026-04-15", reference) == "Tomorrow (Apr 15)"
    assert format_calendar_label("2026-04-16", reference) == "Apr 16"
