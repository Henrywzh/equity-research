from __future__ import annotations

import argparse
import json
import logging

from .config import DEFAULT_RETENTION_DAYS, get_db_path
from .pipeline import cleanup_old_snapshots, run_scrape, run_smoke
from .storage import Storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily Macro scraping utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_parser = subparsers.add_parser("scrape", help="Scrape HKEJ instant news and persist data.")
    scrape_parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    scrape_parser.add_argument("--db-path", default=None, help="Override the SQLite database path.")

    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old raw HTML snapshots.")
    cleanup_parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    cleanup_parser.add_argument("--db-path", default=None, help="Override the SQLite database path.")
    cleanup_parser.add_argument(
        "--retention-days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help="Number of days of raw HTML snapshots to retain.",
    )

    smoke_parser = subparsers.add_parser("smoke", help="Validate homepage parsing without writing data.")
    smoke_parser.add_argument("--json", action="store_true", help="Print the result as JSON.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect the latest scrape run.")
    inspect_parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    inspect_parser.add_argument("--limit", type=int, default=5, help="Number of recent items to show.")
    inspect_parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    inspect_parser.add_argument("--db-path", default=None, help="Override the SQLite database path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scrape":
        result = run_scrape(data_dir=args.data_dir, db_path=args.db_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] != "failed" else 1

    if args.command == "cleanup":
        result = cleanup_old_snapshots(
            retention_days=args.retention_days,
            data_dir=args.data_dir,
            db_path=args.db_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "smoke":
        result = run_smoke()
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"Head news: {result['head_news_count']}")
            print(f"Latest: {result['latest_count']}")
            print(f"Top hero: {result['head_titles'][0]}")
            print(f"Latest first: {result['latest_first_title']}")
        return 0

    if args.command == "inspect":
        result = inspect_latest_run(
            data_dir=args.data_dir,
            db_path=args.db_path,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_inspect_result(result)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def inspect_latest_run(
    *,
    data_dir: str | None = None,
    db_path: str | None = None,
    limit: int = 5,
) -> dict[str, object]:
    resolved_db_path = get_db_path(db_path, data_dir)
    storage = Storage(resolved_db_path)
    try:
        latest_run = storage.fetch_latest_run()
        totals = storage.fetch_total_counts()
        if latest_run is None:
            return {
                "latest_run": None,
                "totals": totals,
                "recent_items": [],
            }

        latest_run["backup_count"] = storage.fetch_latest_run_backup_count(latest_run["id"])
        recent_items = storage.fetch_latest_run_items(latest_run["id"], limit=limit)
        return {
            "latest_run": latest_run,
            "totals": totals,
            "recent_items": recent_items,
        }
    finally:
        storage.close()


def _print_inspect_result(result: dict[str, object]) -> None:
    latest_run = result["latest_run"]
    totals = result["totals"]
    recent_items = result["recent_items"]

    if latest_run is None:
        print("No scrape runs found.")
        print(f"Total articles: {totals['article_count']}")
        print(f"Total backups: {totals['backup_count']}")
        return

    print(f"Latest run: {latest_run['id']}")
    print(f"Status: {latest_run['status']}")
    print(f"Started: {latest_run['started_at']}")
    print(f"Finished: {latest_run['finished_at'] or 'N/A'}")
    print(f"Articles in run: {latest_run['article_count']}")
    print(f"Placements in run: {latest_run['placement_count']}")
    print(f"Backups in run: {latest_run['backup_count']}")
    print(f"Total articles: {totals['article_count']}")
    print(f"Total backups: {totals['backup_count']}")
    print("Recent items:")
    for item in recent_items:
        print(f"[{item['collection']} #{item['rank']}] {item['title']}")
