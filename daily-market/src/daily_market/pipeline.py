from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_data_dir, get_db_path, get_snapshots_dir, get_summaries_dir, get_watchlist_path
from .fetcher import fetch_snapshots
from .formatter import format_summary
from .models import FetchRun
from .notifier import send_market_summary
from .storage import Storage
from .watchlist import load_watchlist

_SMOKE_TICKERS = ["^HSI", "^SPX", "BTC-USD"]


def run_smoke() -> dict[str, Any]:
    """Fetch a small set of tickers to verify yfinance connectivity."""
    from .fetcher import fetch_snapshots as _fetch

    print(f"Smoke test: fetching {_SMOKE_TICKERS} ...")
    snapshots = _fetch(_SMOKE_TICKERS, {"^HSI": "index", "^SPX": "index", "BTC-USD": "crypto"})

    results = []
    all_ok = True
    for s in snapshots:
        if s.error:
            print(f"  FAIL  {s.ticker}: {s.error}")
            all_ok = False
        else:
            print(f"  OK    {s.ticker}  price={s.price}  pct={s.pct_change:.2f}%  as_of={s.data_timestamp}")
        results.append({"ticker": s.ticker, "ok": s.error is None, "error": s.error})

    status = "success" if all_ok else "partial"
    return {"status": status, "results": results}


def run_fetch(
    session: str,
    *,
    skip_email: bool = False,
    watchlist_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Full pipeline: fetch → store → format → email.

    Returns a JSON-serialisable result dict including output_path.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(uuid.uuid4())

    # Resolve paths
    data_dir_path = get_data_dir(data_dir)
    db_path_resolved = get_db_path(db_path, data_dir)
    snapshots_dir = get_snapshots_dir(data_dir)
    summaries_dir = get_summaries_dir(data_dir)
    wl_path = watchlist_path or get_watchlist_path()

    # Load watchlist
    watchlist = load_watchlist(wl_path)
    asset_class_map = watchlist.asset_class_map()

    run = FetchRun(
        id=run_id,
        started_at=started_at,
        session=session,
        ticker_count=len(watchlist.tickers),
    )

    storage = Storage(db_path_resolved)
    storage.record_run(run)

    # Fetch
    snapshots = fetch_snapshots(watchlist.tickers, asset_class_map)
    success_count = sum(1 for s in snapshots if s.error is None)
    failed_tickers = [s.ticker for s in snapshots if s.error]

    # Persist raw snapshot
    snapshot_path = storage.write_snapshot_json(run, snapshots, snapshots_dir)

    # Format
    date_str = started_at[:10]
    summary = format_summary(snapshots, session=session, date_str=date_str, fetch_time_utc=started_at)

    # Persist formatted summary
    summary_path = storage.write_summary_json(
        summary["structured"], summaries_dir, date_str, session
    )

    # Update run in DB
    finished_at = datetime.now(timezone.utc).isoformat()
    status = summary["status"]
    error_summary: str | None = None
    if failed_tickers:
        error_summary = f"Failed tickers ({len(failed_tickers)}): {', '.join(failed_tickers)}"

    storage.insert_snapshots(run_id, snapshots)
    storage.update_run(
        run_id,
        finished_at=finished_at,
        status=status,
        success_count=success_count,
        error_summary=error_summary,
    )
    storage.close()

    # Email
    email_sent = False
    email_message = "Email skipped (--skip-email)."
    if not skip_email:
        try:
            email_sent, email_message = send_market_summary(summary, session=session, date_str=date_str)
        except Exception as exc:
            email_message = f"Email failed: {exc}"

    result = {
        "run_id": run_id,
        "session": session,
        "date": date_str,
        "status": status,
        "ticker_count": len(watchlist.tickers),
        "success_count": success_count,
        "failed_tickers": failed_tickers,
        "snapshot_path": str(snapshot_path),
        "summary_path": str(summary_path),
        "email_sent": email_sent,
        "email_message": email_message,
    }

    return result
