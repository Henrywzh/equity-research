from __future__ import annotations

import argparse
import json
import logging

from .config import DEFAULT_RETENTION_DAYS
from .pipeline import cleanup_old_snapshots, run_scrape, run_smoke


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

    parser.error(f"Unknown command: {args.command}")
    return 2
