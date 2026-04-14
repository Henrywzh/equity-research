"""
FOMC Document Intelligence – fetch and store primary Fed documents.

Documents covered per meeting:
    - Statement (monetary policy decision text)
    - Implementation Note (operational rate parameters)
    - Minutes (~7-10k words, released ~3 weeks after meeting)
    - SEP Projection Table (dot plot meetings only, 4x/year)

All URLs are deterministic from the meeting end date (YYYYMMDD). We do not
rely on link scraping – just the calendar to get meeting dates, which the
existing release_calendar module already provides.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.federalreserve.gov"

# URL templates keyed by document_type
_URL_TEMPLATES: dict[str, str] = {
    "statement": BASE_URL + "/newsevents/pressreleases/monetary{date}a.htm",
    "implementation_note": BASE_URL + "/newsevents/pressreleases/monetary{date}a1.htm",
    "minutes": BASE_URL + "/monetarypolicy/fomcminutes{date}.htm",
    "sep": BASE_URL + "/monetarypolicy/fomcprojtabl{date}.htm",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fomc_documents (
    meeting_date     TEXT NOT NULL,
    document_type    TEXT NOT NULL,
    url              TEXT NOT NULL,
    raw_text         TEXT,
    fetched_at       TEXT NOT NULL,
    PRIMARY KEY (meeting_date, document_type)
);

CREATE TABLE IF NOT EXISTS fomc_sep_projections (
    meeting_date TEXT NOT NULL,
    variable     TEXT NOT NULL,
    horizon      TEXT NOT NULL,
    median       REAL,
    fetched_at   TEXT NOT NULL,
    PRIMARY KEY (meeting_date, variable, horizon)
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Meeting date helpers
# ---------------------------------------------------------------------------

def _meeting_date_to_code(meeting_date: str) -> str:
    """'2026-03-18' -> '20260318'"""
    return meeting_date.replace("-", "")


def get_fomc_meeting_dates(years: list[int]) -> list[str]:
    """
    Return sorted list of FOMC meeting end-dates for the requested years
    in 'YYYY-MM-DD' format, scraped from the Fed's official calendar page.
    """
    from .release_calendar import fetch_fomc_events
    from datetime import date

    all_dates = []
    for year in years:
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        try:
            events = fetch_fomc_events(start=start, end=end)
            for ev in events:
                if ev.get("event_type") == "statement_day":
                    all_dates.append(str(ev["date"]))
        except Exception as exc:
            logger.warning("Could not fetch FOMC events for %d: %s", year, exc)

    return sorted(set(all_dates))


# ---------------------------------------------------------------------------
# Clean-text extraction
# ---------------------------------------------------------------------------

def _extract_clean_text(html: str) -> str:
    """Strip nav chrome and return the article body text."""
    soup = BeautifulSoup(html, "html.parser")

    # Fed pages use <div id="article"> or <div class="col-xs-12 col-sm-8 col-md-8">
    article = (
        soup.find("div", id="article")
        or soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
        or soup.find("div", attrs={"class": lambda c: c and "article" in c.lower()})
    )

    if article:
        # Remove nav, header, footer, share widgets within the article div
        for tag in article.find_all(["nav", "header", "footer", "script", "style", "noscript"]):
            tag.decompose()
        text = article.get_text(separator="\n", strip=True)
    else:
        # Fallback: strip everything except readable paragraphs
        for tag in soup.find_all(["nav", "header", "footer", "script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join(ln for ln in lines if ln)
    return cleaned


# ---------------------------------------------------------------------------
# Document fetchers
# ---------------------------------------------------------------------------

def _fetch_document(doc_type: str, meeting_date: str, timeout: int = 20) -> str | None:
    """
    Fetch a single Fed document. Returns clean text or None if unavailable
    (e.g. minutes not yet released, SEP not applicable for this meeting).
    """
    code = _meeting_date_to_code(meeting_date)
    url = _URL_TEMPLATES[doc_type].format(date=code)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        if resp.status_code == 404:
            logger.debug("Not available (404): %s", url)
            return None
        resp.raise_for_status()
        return _extract_clean_text(resp.text)
    except requests.HTTPError as exc:
        logger.debug("HTTP error fetching %s: %s", url, exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected error fetching %s %s: %s", doc_type, meeting_date, exc)
        return None


# ---------------------------------------------------------------------------
# SEP table parser
# ---------------------------------------------------------------------------

_SEP_VARIABLE_MAP = {
    "change in real gdp": "gdp",
    "gdp": "gdp",
    "unemployment rate": "unemployment",
    "pce inflation": "pce",
    "core pce inflation": "core_pce",
    "federal funds rate": "fed_funds",
}

_HORIZON_PATTERN = re.compile(r"\b(20\d{2}|longer.?run)\b", re.IGNORECASE)


def _parse_sep_projections(html: str, meeting_date: str) -> list[dict[str, Any]]:
    """
    Extract median projections from an SEP table HTML page.

    The Fed's Table 1 has rows like:
        Variable | 2026 median | 2027 median | ... | Longer run median
    We locate these by scanning headers and numeric rows.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Look for Table 1 – "Economic projections"
    table = None
    for tbl in soup.find_all("table"):
        caption = tbl.find("caption")
        if caption and "projection" in caption.get_text().lower():
            table = tbl
            break

    projections: list[dict[str, Any]] = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    if table is None:
        # Fallback: parse the structured text (Figure 1 section)
        text = _extract_clean_text(html)
        projections = _parse_sep_from_text(text, meeting_date, fetched_at)
        return projections

    rows = table.find_all("tr")
    horizons: list[str] = []

    for row in rows:
        cells = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
        if not cells:
            continue

        # Header row: contains year columns
        years_found = [_HORIZON_PATTERN.search(c) for c in cells]
        if any(years_found) and not any(c.replace(".", "").replace("-", "").isdigit() for c in cells[1:] if c):
            horizons = []
            for c in cells[1:]:
                m = _HORIZON_PATTERN.search(c)
                if m:
                    h = m.group(1)
                    horizons.append("longer_run" if "longer" in h.lower() else h)
            continue

        if not horizons:
            continue

        # Data row: first cell is variable name
        variable_raw = cells[0].lower().strip()
        variable = None
        for key, val in _SEP_VARIABLE_MAP.items():
            if key in variable_raw:
                variable = val
                break
        if variable is None:
            continue

        # Second cell group should be "Median" values
        numeric_cells = []
        for c in cells[1:]:
            c_clean = c.replace("–", "").replace("−", "-").strip()
            try:
                numeric_cells.append(float(c_clean))
            except ValueError:
                numeric_cells.append(None)

        for i, horizon in enumerate(horizons):
            if i < len(numeric_cells):
                projections.append({
                    "meeting_date": meeting_date,
                    "variable": variable,
                    "horizon": horizon,
                    "median": numeric_cells[i],
                    "fetched_at": fetched_at,
                })

    return projections


def _parse_sep_from_text(text: str, meeting_date: str, fetched_at: str) -> list[dict[str, Any]]:
    """
    Lightweight fallback: look for known variable header lines followed by
    percent values. The Fed's SEP HTML consistently renders Figure 1 in this
    textual order: GDP header → values, Unemployment → values, PCE → values, etc.
    """
    projections: list[dict[str, Any]] = []
    lines = text.splitlines()

    # Map header keywords to variable names
    section_map = {
        "change in real gdp": "gdp",
        "unemployment rate": "unemployment",
        "pce inflation": "pce",
        "core pce inflation": "core_pce",
        "federal funds rate": "fed_funds",
    }

    # Standard horizons for current + 2 forward years + longer run
    current_year = datetime.now().year
    default_horizons = [str(current_year), str(current_year + 1), str(current_year + 2), "longer_run"]

    i = 0
    while i < len(lines):
        line_lower = lines[i].lower().strip()
        matched_var = None
        for kw, var in section_map.items():
            if kw in line_lower:
                matched_var = var
                break

        if matched_var:
            # Collect following numeric lines
            nums: list[float] = []
            j = i + 1
            while j < len(lines) and len(nums) < 4:
                try:
                    nums.append(float(lines[j].strip()))
                except ValueError:
                    if lines[j].strip():
                        break
                j += 1

            for idx, horizon in enumerate(default_horizons):
                if idx < len(nums):
                    projections.append({
                        "meeting_date": meeting_date,
                        "variable": matched_var,
                        "horizon": horizon,
                        "median": nums[idx],
                        "fetched_at": fetched_at,
                    })
            i = j
        else:
            i += 1

    return projections


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _store_document(
    conn: sqlite3.Connection,
    meeting_date: str,
    doc_type: str,
    url: str,
    raw_text: str | None,
) -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO fomc_documents (meeting_date, document_type, url, raw_text, fetched_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(meeting_date, document_type) DO UPDATE SET
            raw_text = excluded.raw_text,
            fetched_at = excluded.fetched_at
        """,
        (meeting_date, doc_type, url, raw_text, fetched_at),
    )
    conn.commit()


def _store_sep_projections(
    conn: sqlite3.Connection,
    projections: list[dict[str, Any]],
) -> None:
    for p in projections:
        conn.execute(
            """
            INSERT INTO fomc_sep_projections
                (meeting_date, variable, horizon, median, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(meeting_date, variable, horizon) DO UPDATE SET
                median = excluded.median,
                fetched_at = excluded.fetched_at
            """,
            (p["meeting_date"], p["variable"], p["horizon"], p["median"], p["fetched_at"]),
        )
    conn.commit()


def _already_stored(conn: sqlite3.Connection, meeting_date: str, doc_type: str) -> bool:
    """Return True if we already have (non-null) text for this document."""
    row = conn.execute(
        "SELECT raw_text FROM fomc_documents WHERE meeting_date=? AND document_type=?",
        (meeting_date, doc_type),
    ).fetchone()
    return row is not None and row[0] is not None


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def refresh_fomc_documents(
    data_dir: str | None = None,
    start_year: int = 2020,
    force: bool = False,
) -> dict[str, Any]:
    """
    Fetch and store FOMC documents for all meetings from `start_year` through
    the current year. Skips documents that are already stored unless
    `force=True`. Handles 404s gracefully (minutes not yet released, SEP
    not applicable).

    Returns a summary dict with counts and any errors encountered.
    """
    from .config import get_data_dir

    db_path = get_data_dir(data_dir) / "macro.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)

    current_year = datetime.now().year
    years = list(range(start_year, current_year + 1))
    meeting_dates = get_fomc_meeting_dates(years)

    logger.info("Found %d FOMC meeting dates from %d–%d", len(meeting_dates), start_year, current_year)

    summary: dict[str, Any] = {
        "meetings_found": len(meeting_dates),
        "documents_fetched": 0,
        "documents_skipped": 0,
        "sep_projections_stored": 0,
        "errors": [],
    }

    doc_types = ["statement", "implementation_note", "minutes", "sep"]

    for meeting_date in meeting_dates:
        code = _meeting_date_to_code(meeting_date)

        for doc_type in doc_types:
            url = _URL_TEMPLATES[doc_type].format(date=code)

            # Skip if already stored and not forcing
            if not force and _already_stored(conn, meeting_date, doc_type):
                summary["documents_skipped"] += 1
                continue

            logger.info("Fetching %s for %s …", doc_type, meeting_date)
            raw_text = _fetch_document(doc_type, meeting_date)

            # Store even if raw_text is None (404) so we don't retry every run
            # Only store None if it's a future meeting or known-missing doc
            if raw_text is not None:
                _store_document(conn, meeting_date, doc_type, url, raw_text)
                summary["documents_fetched"] += 1

                # Parse SEP projections if this is a dot-plot meeting
                if doc_type == "sep":
                    try:
                        code_inner = _meeting_date_to_code(meeting_date)
                        resp = requests.get(
                            _URL_TEMPLATES["sep"].format(date=code_inner),
                            headers=_HEADERS,
                            timeout=20,
                        )
                        if resp.status_code == 200:
                            projections = _parse_sep_projections(resp.text, meeting_date)
                            _store_sep_projections(conn, projections)
                            summary["sep_projections_stored"] += len(projections)
                    except Exception as exc:
                        err = f"SEP parse error for {meeting_date}: {exc}"
                        logger.warning(err)
                        summary["errors"].append(err)
            else:
                # Mark as attempted (store a sentinel) so we skip next cycle
                # Don't store None text – just log and move on
                logger.debug("No content for %s %s (likely 404)", doc_type, meeting_date)

    conn.close()
    return summary
