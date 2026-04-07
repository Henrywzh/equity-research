from __future__ import annotations

import argparse
import json
import logging

from .analysis import run_analysis
from .config import DEFAULT_RETENTION_DAYS, get_db_path
from .notifier import load_analysis_result, send_analysis_summary_email, send_test_email
from .pipeline import cleanup_old_snapshots, run_scrape, run_smoke
from .storage import Storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily Macro scraping utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_parser = subparsers.add_parser("scrape", help="Scrape HKEJ instant news and persist data.")
    scrape_parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    scrape_parser.add_argument("--db-path", default=None, help="Override the SQLite database path.")

    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old parsed JSON backups.")
    cleanup_parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    cleanup_parser.add_argument("--db-path", default=None, help="Override the SQLite database path.")
    cleanup_parser.add_argument(
        "--retention-days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help="Number of days of parsed JSON backups to retain.",
    )

    analyze_parser = subparsers.add_parser("analyze", help="Analyze stored daily articles with Groq.")
    analyze_parser.add_argument("target", choices=["today"], help="Analyze the requested daily article set.")
    analyze_parser.add_argument("--date", default=None, help="Override the target date in YYYY-MM-DD format.")
    analyze_parser.add_argument("--json", action="store_true", help="Print the generated report as JSON.")
    analyze_parser.add_argument("--force", action="store_true", help="Regenerate the report even if it already exists.")
    analyze_parser.add_argument("--verbose", action="store_true", help="Enable detailed analysis diagnostics logging.")
    analyze_parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    analyze_parser.add_argument("--db-path", default=None, help="Override the SQLite database path.")

    local_test_parser = subparsers.add_parser(
        "local-test",
        help="Run a local scrape -> analyze -> Gmail test against live news.",
    )
    local_test_parser.add_argument("--date", default=None, help="Override the analysis date in YYYY-MM-DD format.")
    local_test_parser.add_argument("--json", action="store_true", help="Print the end-to-end result as JSON.")
    local_test_parser.add_argument("--verbose", action="store_true", help="Enable detailed analysis diagnostics logging.")
    local_test_parser.add_argument("--skip-email", action="store_true", help="Run scrape and analysis only, without Gmail.")
    local_test_parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    local_test_parser.add_argument("--db-path", default=None, help="Override the SQLite database path.")

    notify_parser = subparsers.add_parser("notify", help="Send a Gmail summary for a prior analysis result.")
    notify_parser.add_argument("--result-path", required=True, help="Path to the JSON analysis result.")
    notify_parser.add_argument("--preview", action="store_true", help="Print the HTML summary to stdout instead of sending email.")

    subparsers.add_parser("test-email", help="Send a Gmail connectivity test email.")

    smoke_parser = subparsers.add_parser("smoke", help="Validate homepage parsing without writing data.")
    smoke_parser.add_argument("--json", action="store_true", help="Print the result as JSON.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect the latest scrape run.")
    inspect_parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    inspect_parser.add_argument("--limit", type=int, default=5, help="Number of recent items to show.")
    inspect_parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    inspect_parser.add_argument("--db-path", default=None, help="Override the SQLite database path.")

    query_parser = subparsers.add_parser("query", help="Query stored article data.")
    query_subparsers = query_parser.add_subparsers(dest="query_command", required=True)

    query_date_parser = query_subparsers.add_parser("date", help="List articles for a date.")
    query_date_parser.add_argument("date", help="Date in YYYY-MM-DD format.")
    query_date_parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    query_date_parser.add_argument("--limit", type=int, default=10, help="Number of items to show.")
    query_date_parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    query_date_parser.add_argument("--db-path", default=None, help="Override the SQLite database path.")

    query_search_parser = query_subparsers.add_parser("search", help="Search articles by keyword.")
    query_search_parser.add_argument("keyword", help="Keyword to search for.")
    query_search_parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    query_search_parser.add_argument("--limit", type=int, default=10, help="Number of items to show.")
    query_search_parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    query_search_parser.add_argument("--db-path", default=None, help="Override the SQLite database path.")

    query_article_parser = query_subparsers.add_parser("article", help="Show one article by URL or source id.")
    query_article_parser.add_argument("--url", default=None, help="Canonical article URL.")
    query_article_parser.add_argument("--id", dest="source_article_id", default=None, help="Source article id.")
    query_article_parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    query_article_parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    query_article_parser.add_argument("--db-path", default=None, help="Override the SQLite database path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"analyze", "local-test"} and args.verbose:
        logging.getLogger("daily_macro.analysis").setLevel(logging.DEBUG)

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

    if args.command == "analyze":
        result = run_analysis(
            date_string=args.date,
            data_dir=args.data_dir,
            db_path=args.db_path,
            force=args.force,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_analysis_result(result)
        return 0 if result["status"] != "failed" else 1

    if args.command == "local-test":
        result = run_local_test(
            date_string=args.date,
            data_dir=args.data_dir,
            db_path=args.db_path,
            send_email=not args.skip_email,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_local_test_result(result)
        return 0 if result["scrape"]["status"] == "success" and result["analysis"]["status"] != "failed" else 1

    if args.command == "notify":
        result_data = load_analysis_result(args.result_path)
        if args.preview:
            from .notifier import _build_html_body
            print(_build_html_body(result_data))
            return 0
        sent, message = send_analysis_summary_email(result_data)
        print(json.dumps({"sent": sent, "message": message}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "test-email":
        sent, message = send_test_email()
        print(json.dumps({"sent": sent, "message": message}, ensure_ascii=False, indent=2))
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

    if args.command == "query":
        result = run_query_command(args)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            if args.query_command == "article":
                _print_query_article_result(result)
            else:
                _print_query_list_result(result)
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


def run_query_command(args: argparse.Namespace) -> dict[str, object]:
    resolved_db_path = get_db_path(args.db_path, args.data_dir)
    storage = Storage(resolved_db_path)
    try:
        if args.query_command == "date":
            items = storage.fetch_articles_by_date_with_limit(args.date, limit=args.limit)
            return {"mode": "date", "query": args.date, "items": items}

        if args.query_command == "search":
            items = storage.search_articles(args.keyword, limit=args.limit)
            return {"mode": "search", "query": args.keyword, "items": items}

        if args.query_command == "article":
            if bool(args.url) == bool(args.source_article_id):
                raise SystemExit("Provide exactly one of --url or --id.")
            article = storage.fetch_article_with_latest_backup(
                url=args.url,
                source_article_id=args.source_article_id,
            )
            return {"mode": "article", "item": article}

        raise SystemExit(f"Unknown query command: {args.query_command}")
    finally:
        storage.close()


def run_local_test(
    *,
    date_string: str | None = None,
    data_dir: str | None = None,
    db_path: str | None = None,
    send_email: bool = True,
) -> dict[str, object]:
    scrape_result = run_scrape(data_dir=data_dir, db_path=db_path)
    if scrape_result.get("status") != "success":
        return {
            "scrape": scrape_result,
            "analysis": {
                "report_date": date_string,
                "status": "skipped",
                "output_path": None,
                "article_count": 0,
                "category_count": 0,
                "cached": False,
            },
            "notify": {
                "attempted": False,
                "sent": False,
                "message": "Skipped analysis and Gmail because scrape did not complete successfully.",
            },
        }
    analysis_result = run_analysis(
        date_string=date_string,
        data_dir=data_dir,
        db_path=db_path,
        force=True,
    )
    notify_result: dict[str, object]
    if send_email:
        sent, message = send_analysis_summary_email(analysis_result)
        notify_result = {"attempted": True, "sent": sent, "message": message}
    else:
        notify_result = {
            "attempted": False,
            "sent": False,
            "message": "Skipped Gmail because --skip-email was provided.",
        }

    return {
        "scrape": scrape_result,
        "analysis": {
            "report_date": analysis_result.get("report_date"),
            "status": analysis_result.get("status"),
            "output_path": analysis_result.get("output_path"),
            "article_count": (analysis_result.get("totals") or {}).get("article_count"),
            "category_count": (analysis_result.get("input") or {}).get("category_count"),
            "cached": analysis_result.get("cached", False),
        },
        "notify": notify_result,
    }


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


def _print_query_list_result(result: dict[str, object]) -> None:
    items = result["items"]
    if not items:
        print("No matching articles found.")
        return

    for item in items:
        published = item.get("published_at") or item.get("first_seen_at") or "N/A"
        section = item.get("article_section") or "N/A"
        print(f"[{published}] [{section}] {item['title']}")


def _print_query_article_result(result: dict[str, object]) -> None:
    item = result["item"]
    if item is None:
        print("Article not found.")
        return

    print(item["title"])
    print(f"Section: {item.get('article_section') or 'N/A'}")
    print(f"Published: {item.get('published_at') or 'N/A'}")
    print(f"URL: {item['canonical_url']}")
    print(f"Source article id: {item.get('source_article_id') or 'N/A'}")
    if item.get("summary_snippet"):
        print(f"Summary: {item['summary_snippet']}")

    body = item.get("content_text") or ""
    preview = body[:500]
    if len(body) > 500:
        preview += "..."
    print("Body preview:")
    print(preview)
    latest_backup = item.get("latest_backup")
    if latest_backup:
        print(f"Latest backup: {latest_backup['relative_path']}")


def _print_analysis_result(result: dict[str, object]) -> None:
    print(f"Report date: {result['report_date']}")
    print(f"Status: {result['status']}")
    print(f"Primary model: {result['model']['provider']}/{result['model']['primary_model']}")
    fallback_models = result["model"].get("fallback_models") or []
    if fallback_models:
        print(
            "Fallback models: "
            + ", ".join(f"{result['model']['provider']}/{model_id}" for model_id in fallback_models)
        )
    else:
        print(f"Fallback model: {result['model']['provider']}/{result['model']['fallback_model']}")
    print(f"Articles analyzed: {result['totals']['article_count']}")
    print(f"Categories: {result['input']['category_count']}")
    print(f"Full-text articles: {result['totals']['full_text_article_count']}")
    print(f"Truncated articles: {result['totals']['truncated_article_count']}")
    print(f"Model switches: {len(result.get('model_switches', []))}")
    diagnostics = result.get("diagnostics") or {}
    if diagnostics:
        print(
            f"Rate-limit waits: {diagnostics.get('rate_limit_wait_count', 0)} "
            f"({diagnostics.get('rate_limit_wait_seconds_total', 0)}s)"
        )
        print(
            "Batch splits: "
            f"pre-send={diagnostics.get('pre_send_split_count', 0)}, "
            f"after-413={diagnostics.get('response_413_split_count', 0)}"
        )
    if result.get("cached"):
        print("Used cached report: yes")
    print(f"Output: {result['output_path']}")


def _print_local_test_result(result: dict[str, object]) -> None:
    scrape = result["scrape"]
    analysis = result["analysis"]
    notify = result["notify"]
    print("Local test completed")
    print(f"Scrape status: {scrape['status']}")
    print(f"Scraped articles: {scrape['article_count']}")
    print(f"Analysis date: {analysis['report_date']}")
    print(f"Analysis status: {analysis['status']}")
    print(f"Analyzed articles: {analysis['article_count']}")
    print(f"Categories: {analysis['category_count']}")
    print(f"Report output: {analysis['output_path']}")
    print(f"Gmail attempted: {'yes' if notify['attempted'] else 'no'}")
    print(f"Gmail sent: {'yes' if notify['sent'] else 'no'}")
    print(f"Gmail result: {notify['message']}")
