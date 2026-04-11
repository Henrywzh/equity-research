"""
news_fetcher.py — Multi-Source News Fetcher for Hormuz Geo-Political Context
=============================================================================
Fetches the latest headlines related to the Strait of Hormuz or Iran maritime
activity. Uses a 4-attempt cascade across two independent providers so a
Google RSS cache cold-miss never silently drops relevant context.

Fetch cascade (stops at first non-empty result):
  1. Google News RSS  when:1d     (primary)
  2. Google News RSS  when:1d     (retry after 3s — CDN cache cold-miss)
  3. GDELT DOC 2.0    timespan=24h  ← different provider entirely (no API key)
  4. Google News RSS  when:2d     (broadened fallback)

GDELT (gdeltproject.org) is an academic-grade geopolitical news monitor that
indexes thousands of global sources. It requires no API key and is purpose-built
for this kind of keyword-driven geo-political intelligence gathering.

Usage:
  from news_fetcher import get_latest_news
  news_str = get_latest_news()          # top 5 headlines as a formatted string
  news_str = get_latest_news(limit=3)   # top 3

Standalone test:
  python news_fetcher.py
"""

import json
import time
import urllib.parse
import urllib.request
import feedparser

# ------------------------------------------------------------------
# Google News RSS URLs
# ------------------------------------------------------------------
RSS_URL = (
    "https://news.google.com/rss/search"
    "?q=%22Strait+of+Hormuz%22+OR+%22Iran+maritime%22+when:1d"
    "&hl=en-US&gl=US&ceid=US:en"
)

# Fallback query (broader scope, 2-day window)
RSS_URL_FALLBACK = (
    "https://news.google.com/rss/search"
    "?q=%22Strait+of+Hormuz%22+OR+%22Iran%22+when:2d"
    "&hl=en-US&gl=US&ceid=US:en"
)

# ------------------------------------------------------------------
# GDELT DOC 2.0 API
# ------------------------------------------------------------------
GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = '(Hormuz OR "Strait of Hormuz" OR "Iran maritime" OR "Iran navy")'

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _rss_entries(url: str, limit: int) -> list[dict]:
    """Parse a Google News RSS URL and return up to `limit` normalised entries."""
    feed = feedparser.parse(url)
    out  = []
    for entry in feed.entries[:limit]:
        out.append({
            "title":  entry.get("title", "No title"),
            "pub":    entry.get("published", "Unknown date"),
            "source": "Google News",
        })
    return out


def _fetch_gdelt(hours_back: int = 24, limit: int = 5) -> list[dict]:
    """
    Query the GDELT DOC 2.0 API for Hormuz / Iran maritime articles.
    Returns a list of normalised {title, pub, source} dicts.
    Never raises — returns [] on any error.
    """
    params = urllib.parse.urlencode({
        "query":      GDELT_QUERY,
        "mode":       "artlist",
        "maxrecords": limit,
        "timespan":   f"{hours_back}h",
        "format":     "json",
    })
    url = f"{GDELT_API}?{params}"
    try:
        req  = urllib.request.Request(url, headers={"User-Agent": "Maritime-OSINT/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        articles = data.get("articles") or []
        out = []
        for a in articles[:limit]:
            # GDELT seendate format: "20260317T160000Z"
            seen = a.get("seendate", "")
            if len(seen) >= 15:
                pub = f"{seen[0:4]}-{seen[4:6]}-{seen[6:8]}T{seen[9:11]}:{seen[11:13]}:{seen[13:15]}Z"
            else:
                pub = seen or "Unknown date"
            domain = a.get("domain", "")
            title  = a.get("title", "No title")
            source = f"GDELT / {domain}" if domain else "GDELT"
            out.append({"title": title, "pub": pub, "source": source})
        return out
    except Exception:
        return []


def _format_entries(entries: list[dict]) -> str:
    """Format a list of {title, pub, source} dicts into a numbered headline string."""
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(f"{i}. [{e['pub']}] {e['title']}  — {e['source']}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_latest_news(limit: int = 5) -> str:
    """
    Fetch top N Hormuz / Iran-maritime headlines using a 4-attempt cascade.

    Cascade (stops at first non-empty result):
      1. Google RSS  when:1d    (primary)
      2. Google RSS  when:1d    (retry after 3s — CDN cold-miss)
      3. GDELT       24h        (independent provider — different index)
      4. Google RSS  when:2d    (broadened fallback)

    Returns a formatted string ready for LLM prompt injection, e.g.:
        1. [2026-03-17T16:01:00Z] Iran targets UAE energy infrastructure — GDELT / cnbc.com
        2. [Tue, 17 Mar 2026 12:48:25 GMT] Strait of Hormuz: What have...  — Google News

    Never raises — returns a graceful fallback string on any error.
    """
    try:
        # ── Attempt 1: Google RSS when:1d ──────────────────────────
        entries = _rss_entries(RSS_URL, limit)
        if entries:
            return _format_entries(entries)

        # ── Attempt 2: Retry Google RSS (CDN cache cold-miss) ──────
        time.sleep(3)
        entries = _rss_entries(RSS_URL, limit)
        if entries:
            return _format_entries(entries)

        # ── Attempt 3: GDELT DOC 2.0 (independent provider) ───────
        entries = _fetch_gdelt(hours_back=24, limit=limit)
        if entries:
            return _format_entries(entries)

        # ── Attempt 4: Google RSS when:2d (broadened) ──────────────
        entries = _rss_entries(RSS_URL_FALLBACK, limit)
        if entries:
            return _format_entries(entries)

        return "No recent Hormuz / Iran maritime news found in the last 24h."

    except Exception as e:
        return f"[News fetch unavailable: {e}]"


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Hormuz / Iran Maritime — Multi-Source News Test")
    print("=" * 60)

    # Show which source is serving results right now
    print("\n[Attempt 1] Google RSS when:1d ...")
    e1 = _rss_entries(RSS_URL, 5)
    if e1:
        print(f"  ✓ {len(e1)} headline(s) from Google RSS")
    else:
        print("  ✗ Empty — Google RSS cold miss")

    print("\n[Attempt 3] GDELT DOC 2.0 (24h) ...")
    e3 = _fetch_gdelt(hours_back=24, limit=5)
    if e3:
        print(f"  ✓ {len(e3)} headline(s) from GDELT")
    else:
        print("  ✗ Empty — GDELT returned nothing")

    print("\n[Full cascade result]")
    result = get_latest_news(limit=5)
    print(result)
    print("\n" + "=" * 60)

    lines = [l for l in result.splitlines() if l.strip()]
    if lines and not lines[0].startswith("[News fetch") and not lines[0].startswith("No recent"):
        print(f"✓  {len(lines)} headline(s) fetched successfully.")
    else:
        print("⚠  No headlines returned — check network or RSS/GDELT endpoints.")
