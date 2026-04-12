from __future__ import annotations

from datetime import date
from pathlib import Path

from daily_market.models import PolymarketWatchlistConfig, PolymarketWatchlistEntry
from daily_market.polymarket import fetch_polymarket_watchlist, load_polymarket_watchlist


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _FakeSession:
    def __init__(self, payload_by_slug: dict[str, list[dict]]):
        self.payload_by_slug = payload_by_slug
        self.requested_slugs: list[str] = []

    def get(self, url, params=None, timeout=None):
        slug = (params or {}).get("slug")
        self.requested_slugs.append(slug)
        return _FakeResponse(self.payload_by_slug.get(slug, []))


def _market_payload(
    *,
    market_id: str,
    slug: str,
    question: str,
    event_slug: str,
    active: bool = True,
    closed: bool = False,
):
    return {
        "id": market_id,
        "slug": slug,
        "question": question,
        "endDate": "2026-06-30T00:00:00Z",
        "active": active,
        "closed": closed,
        "archived": False,
        "bestBid": "0.45",
        "bestAsk": "0.55",
        "lastTradePrice": "0.51",
        "volume": "12345.6",
        "volume24hr": "3456.7",
        "liquidity": "2345.6",
        "openInterest": "456.7",
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["0.52", "0.48"],
        "clobTokenIds": ["1", "2"],
        "events": [{"id": "event-1", "slug": event_slug, "title": "Sample event"}],
    }


def test_load_polymarket_watchlist_includes_required_daily_entries():
    cfg_path = (
        Path(__file__).resolve().parents[1] / "config" / "polymarket_watchlist.json"
    )
    watchlist = load_polymarket_watchlist(cfg_path)
    groups = {(entry.group_key, entry.key) for entry in watchlist.entries}
    assert ("qqq_daily", "qqq_up_down_daily") in groups
    assert ("btc_daily", "btc_up_down_daily") in groups


def test_fetch_polymarket_resolves_exact_and_rolling_entries():
    watchlist = PolymarketWatchlistConfig(
        entries=[
            PolymarketWatchlistEntry(
                key="fed_cuts_1",
                group_key="fed_rates",
                asset="fed",
                horizon="year_end",
                title="1 Fed cut",
                entry_type="exact_market",
                market_slug="will-1-fed-rate-cut-happen-in-2026",
            ),
            PolymarketWatchlistEntry(
                key="qqq_daily",
                group_key="qqq_daily",
                asset="qqq",
                horizon="daily",
                title="QQQ daily up or down",
                entry_type="rolling_market",
                market_slug_template="qqq-up-or-down-on-{date_slug}",
                date_rule="next_trading_days",
            ),
        ]
    )
    session = _FakeSession(
        {
            "will-1-fed-rate-cut-happen-in-2026": [
                _market_payload(
                    market_id="616903",
                    slug="will-1-fed-rate-cut-happen-in-2026",
                    question="Will 1 Fed rate cut happen in 2026?",
                    event_slug="how-many-fed-rate-cuts-in-2026",
                )
            ],
            "qqq-up-or-down-on-april-13-2026": [
                _market_payload(
                    market_id="1936551",
                    slug="qqq-up-or-down-on-april-13-2026",
                    question="QQQ (QQQ) Up or Down on April 13?",
                    event_slug="qqq-up-or-down-on-april-13-2026",
                )
            ],
        }
    )

    run, markets, snapshots, errors = fetch_polymarket_watchlist(
        watchlist,
        session=session,
        today=date(2026, 4, 12),
    )

    assert run.status == "success"
    assert run.snapshot_count == 2
    assert not errors
    assert {market.market_slug for market in markets} == {
        "will-1-fed-rate-cut-happen-in-2026",
        "qqq-up-or-down-on-april-13-2026",
    }
    assert "qqq-up-or-down-on-april-13-2026" in session.requested_slugs
    assert snapshots[0].implied_probability is not None


def test_fetch_polymarket_partial_failure_does_not_abort_run():
    watchlist = PolymarketWatchlistConfig(
        entries=[
            PolymarketWatchlistEntry(
                key="btc_daily",
                group_key="btc_daily",
                asset="btc",
                horizon="daily",
                title="BTC daily up or down",
                entry_type="rolling_market",
                market_slug_template="bitcoin-up-or-down-on-{date_slug}",
                date_rule="today_plus_three",
            ),
            PolymarketWatchlistEntry(
                key="missing_market",
                group_key="oil_thresholds",
                asset="oil",
                horizon="monthly",
                title="Missing market",
                entry_type="exact_market",
                market_slug="missing-market-slug",
            ),
        ]
    )
    session = _FakeSession(
        {
            "bitcoin-up-or-down-on-april-13-2026": [
                _market_payload(
                    market_id="1947717",
                    slug="bitcoin-up-or-down-on-april-13-2026",
                    question="Bitcoin Up or Down on April 13?",
                    event_slug="bitcoin-up-or-down-on-april-13-2026",
                )
            ]
        }
    )

    run, markets, snapshots, errors = fetch_polymarket_watchlist(
        watchlist,
        session=session,
        today=date(2026, 4, 12),
    )

    assert run.status == "partial"
    assert len(markets) == 1
    assert len(snapshots) == 1
    assert len(errors) == 1
    assert "missing_market" in errors[0]


def test_fetch_polymarket_prefers_active_rolling_contract():
    watchlist = PolymarketWatchlistConfig(
        entries=[
            PolymarketWatchlistEntry(
                key="btc_daily",
                group_key="btc_daily",
                asset="btc",
                horizon="daily",
                title="BTC daily up or down",
                entry_type="rolling_market",
                market_slug_template="bitcoin-up-or-down-on-{date_slug}",
                date_rule="today_plus_three",
            )
        ]
    )
    session = _FakeSession(
        {
            "bitcoin-up-or-down-on-april-12-2026": [
                _market_payload(
                    market_id="1938105",
                    slug="bitcoin-up-or-down-on-april-12-2026",
                    question="Bitcoin Up or Down on April 12?",
                    event_slug="bitcoin-up-or-down-on-april-12-2026",
                    active=True,
                    closed=True,
                )
            ],
            "bitcoin-up-or-down-on-april-13-2026": [
                _market_payload(
                    market_id="1947717",
                    slug="bitcoin-up-or-down-on-april-13-2026",
                    question="Bitcoin Up or Down on April 13?",
                    event_slug="bitcoin-up-or-down-on-april-13-2026",
                    active=True,
                    closed=False,
                )
            ],
        }
    )

    _, markets, snapshots, errors = fetch_polymarket_watchlist(
        watchlist,
        session=session,
        today=date(2026, 4, 12),
    )

    assert not errors
    assert markets[0].market_slug == "bitcoin-up-or-down-on-april-13-2026"
    assert snapshots[0].market_status == "active"
