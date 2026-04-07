from __future__ import annotations

import argparse
import json
import logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TruthSocial monitor — @realDonaldTrump post scraper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("scrape", help="One-shot fetch of new posts and save to SQLite.")

    daemon_parser = subparsers.add_parser("daemon", help="Start the persistent polling daemon.")
    daemon_parser.add_argument(
        "--interval", type=int, default=60,
        help="Seconds between polls (default: 60).",
    )
    daemon_parser.add_argument(
        "--alert-cooldown", dest="alert_cooldown", type=int, default=900,
        help="Minimum seconds between alert emails (default: 900 = 15 min).",
    )

    subparsers.add_parser("test-email", help="Send a test Gmail alert email.")

    status_parser = subparsers.add_parser("status", help="Show recent posts and run history.")
    status_parser.add_argument(
        "--posts", type=int, default=10,
        help="Number of recent posts to show (default: 10).",
    )
    status_parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of recent scrape runs to show (default: 5).",
    )
    status_parser.add_argument("--json", action="store_true", help="Print output as JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scrape":
        from .pipeline import run_once
        result = run_once()
        print(json.dumps(
            {
                "status": result.status,
                "new_posts_count": result.new_posts_count,
                "error": result.error,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0 if result.status != "error" else 1

    if args.command == "daemon":
        from .daemon import run_daemon
        run_daemon(poll_interval=args.interval, alert_cooldown=args.alert_cooldown)
        return 0

    if args.command == "test-email":
        from .alerter import send_test_email
        from .config import get_gmail_credentials
        sender, app_password, recipient = get_gmail_credentials()
        if not sender or not app_password or not recipient:
            print("Gmail credentials not set. Add GMAIL_SENDER, GMAIL_APP_PASSWORD, GMAIL_RECIPIENT to .config.")
            return 1
        ok, message = send_test_email(sender=sender, app_password=app_password, recipient=recipient)
        print(json.dumps({"sent": ok, "message": message}, ensure_ascii=False, indent=2))
        return 0 if ok else 1

    if args.command == "status":
        from .config import get_db_path
        from .storage import Storage
        storage = Storage(get_db_path())
        try:
            posts = storage.fetch_recent_posts(limit=args.posts)
            runs = storage.fetch_recent_runs(limit=args.runs)
            total = storage.fetch_post_count()
        finally:
            storage.close()

        result = {"total_posts": total, "recent_posts": posts, "recent_runs": runs}
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_status(result)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _print_status(result: dict) -> None:
    print(f"Total posts in DB: {result['total_posts']}\n")

    print(f"Recent posts (last {len(result['recent_posts'])}):")
    print("-" * 60)
    for post in result["recent_posts"]:
        content_preview = (post["content"] or "")[:120].replace("\n", " ")
        if len(post["content"] or "") > 120:
            content_preview += "..."
        print(f"[{post['created_at'][:19]}] {content_preview}")
    print()

    print(f"Recent scrape runs (last {len(result['recent_runs'])}):")
    print("-" * 60)
    for run in result["recent_runs"]:
        err = f" | error: {run['error_summary'][:60]}" if run.get("error_summary") else ""
        print(
            f"Run #{run['id']} | {run['status']} | "
            f"+{run['new_posts_count']} posts | "
            f"started {(run['started_at'] or '')[:19]}{err}"
        )
