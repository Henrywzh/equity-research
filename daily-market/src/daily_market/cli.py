from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="daily_market",
        description="Daily market price snapshot and summary pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # smoke
    sub.add_parser("smoke", help="Quick connectivity check (fetches 3 tickers).")

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch all watchlist symbols, store, and email.")
    p_fetch.add_argument(
        "--session",
        choices=["morning", "evening"],
        default=_auto_session(),
        help="Summary session (default: auto-detected from current UTC hour).",
    )
    p_fetch.add_argument("--skip-email", action="store_true", help="Skip sending email.")
    p_fetch.add_argument("--data-dir", help="Override data directory path.")
    p_fetch.add_argument("--db-path", help="Override SQLite database path.")
    p_fetch.add_argument("--watchlist", help="Override watchlist.json path.")
    p_fetch.add_argument("--json", action="store_true", dest="output_json", help="Print result as JSON.")

    # local-test
    p_local = sub.add_parser("local-test", help="End-to-end test: fetch → format → (optional) email.")
    p_local.add_argument("--session", choices=["morning", "evening"], default=_auto_session())
    p_local.add_argument("--skip-email", action="store_true")
    p_local.add_argument("--data-dir")
    p_local.add_argument("--db-path")
    p_local.add_argument("--watchlist")
    p_local.add_argument("--json", action="store_true", dest="output_json")

    # test-email
    sub.add_parser("test-email", help="Send a test email to verify Gmail credentials.")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Show latest fetch run and sample snapshots.")
    p_inspect.add_argument("--json", action="store_true", dest="output_json")
    p_inspect.add_argument("--data-dir")
    p_inspect.add_argument("--db-path")

    # query
    p_query = sub.add_parser("query", help="Query stored data.")
    query_sub = p_query.add_subparsers(dest="query_command", required=True)

    p_qdate = query_sub.add_parser("date", help="All snapshots for a given date.")
    p_qdate.add_argument("date", help="YYYY-MM-DD")
    p_qdate.add_argument("--json", action="store_true", dest="output_json")
    p_qdate.add_argument("--data-dir")
    p_qdate.add_argument("--db-path")

    p_qticker = query_sub.add_parser("ticker", help="History for a single ticker.")
    p_qticker.add_argument("ticker", help="Yahoo Finance ticker symbol.")
    p_qticker.add_argument("--limit", type=int, default=20)
    p_qticker.add_argument("--json", action="store_true", dest="output_json")
    p_qticker.add_argument("--data-dir")
    p_qticker.add_argument("--db-path")

    args = parser.parse_args()

    if args.command == "smoke":
        _cmd_smoke()

    elif args.command in ("fetch", "local-test"):
        _cmd_fetch(args)

    elif args.command == "test-email":
        _cmd_test_email()

    elif args.command == "inspect":
        _cmd_inspect(args)

    elif args.command == "query":
        if args.query_command == "date":
            _cmd_query_date(args)
        elif args.query_command == "ticker":
            _cmd_query_ticker(args)


# -------------------------------------------------------------------
# Command implementations
# -------------------------------------------------------------------

def _cmd_smoke() -> None:
    from .pipeline import run_smoke
    result = run_smoke()
    if result["status"] != "success":
        sys.exit(1)


def _cmd_fetch(args: argparse.Namespace) -> None:
    from .pipeline import run_fetch
    result = run_fetch(
        session=args.session,
        skip_email=args.skip_email,
        watchlist_path=getattr(args, "watchlist", None),
        data_dir=getattr(args, "data_dir", None),
        db_path=getattr(args, "db_path", None),
    )
    if getattr(args, "output_json", False):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_fetch_result(result)
    if result["status"] == "failed":
        sys.exit(1)


def _cmd_test_email() -> None:
    from .notifier import send_test_email
    ok, msg = send_test_email()
    print(msg)
    if not ok:
        sys.exit(1)


def _cmd_inspect(args: argparse.Namespace) -> None:
    from .config import get_db_path
    from .storage import Storage
    storage = Storage(get_db_path(getattr(args, "db_path", None), getattr(args, "data_dir", None)))
    run = storage.fetch_latest_run()
    if not run:
        print("No runs found.")
        storage.close()
        return
    snapshots = storage.fetch_snapshots_by_run(run["id"])
    storage.close()

    if getattr(args, "output_json", False):
        print(json.dumps({"run": run, "snapshots": snapshots}, indent=2, default=str))
    else:
        print(f"Latest run: {run['id']}")
        print(f"  Session:  {run['session']}")
        print(f"  Started:  {run['started_at']}")
        print(f"  Status:   {run['status']}")
        print(f"  Tickers:  {run['success_count']}/{run['ticker_count']}")
        if run.get("error_summary"):
            print(f"  Errors:   {run['error_summary']}")
        print(f"\n  Snapshots ({len(snapshots)}):")
        for s in snapshots[:10]:
            pct = f"{s['pct_change']:+.2f}%" if s["pct_change"] is not None else "N/A"
            price = f"{s['price']}" if s["price"] is not None else "ERROR"
            print(f"    {s['ticker']:<14} {price:<14} {pct}")
        if len(snapshots) > 10:
            print(f"    ... ({len(snapshots) - 10} more)")


def _cmd_query_date(args: argparse.Namespace) -> None:
    from .config import get_db_path
    from .storage import Storage
    storage = Storage(get_db_path(getattr(args, "db_path", None), getattr(args, "data_dir", None)))
    rows = storage.fetch_snapshots_by_date(args.date)
    storage.close()

    if getattr(args, "output_json", False):
        print(json.dumps(rows, indent=2, default=str))
    else:
        print(f"Snapshots for {args.date} ({len(rows)} rows):")
        for r in rows:
            pct = f"{r['pct_change']:+.2f}%" if r["pct_change"] is not None else "N/A"
            print(f"  {r['ticker']:<14} {r['price']!s:<14} {pct}  [{r['asset_class']}]")


def _cmd_query_ticker(args: argparse.Namespace) -> None:
    from .config import get_db_path
    from .storage import Storage
    storage = Storage(get_db_path(getattr(args, "db_path", None), getattr(args, "data_dir", None)))
    rows = storage.fetch_ticker_history(args.ticker, limit=args.limit)
    storage.close()

    if getattr(args, "output_json", False):
        print(json.dumps(rows, indent=2, default=str))
    else:
        print(f"History for {args.ticker} ({len(rows)} rows, newest first):")
        for r in rows:
            pct = f"{r['pct_change']:+.2f}%" if r["pct_change"] is not None else "N/A"
            print(f"  {r['fetched_at'][:19]}  {r['price']!s:<14} {pct}  as_of={r['data_timestamp']}")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _auto_session() -> str:
    hour = datetime.now(timezone.utc).hour
    # 20:00–03:59 UTC → morning brief (covers Asia open recap of US/EU close)
    # 08:00–15:59 UTC → evening brief (covers Asia close)
    if hour >= 20 or hour < 4:
        return "morning"
    return "evening"


def _print_fetch_result(result: dict) -> None:
    print(f"[{result['status'].upper()}] {result['session']} | {result['date']}")
    print(f"  Symbols: {result['success_count']}/{result['ticker_count']}")
    if result["failed_tickers"]:
        print(f"  Failed:  {', '.join(result['failed_tickers'])}")
    print(f"  Snapshot: {result['snapshot_path']}")
    print(f"  Summary:  {result['summary_path']}")
    print(f"  Email:    {result['email_message']}")
