from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .notifier import load_run_result, send_run_summary_email, send_test_email
from .pipeline import run_smoke, run_sync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YouTube intake utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Archive new videos and livestream replays.")
    run_parser.add_argument("--config-path", default=None, help="Override the channel config path.")
    run_parser.add_argument("--state-path", default=None, help="Override the state manifest path.")
    run_parser.add_argument("--data-dir", default=None, help="Override the archive data directory.")
    run_parser.add_argument("--result-path", default=None, help="Optional path to write the JSON run result.")

    smoke_parser = subparsers.add_parser("smoke", help="Validate the watched channels and fetch their latest IDs.")
    smoke_parser.add_argument("--config-path", default=None, help="Override the channel config path.")

    notify_parser = subparsers.add_parser("notify", help="Send a Gmail summary for a prior run result.")
    notify_parser.add_argument("--result-path", required=True, help="Path to the JSON run result.")

    subparsers.add_parser("test-email", help="Send a Gmail connectivity test email.")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            payload = run_sync(
                config_path=args.config_path,
                state_path=args.state_path,
                data_dir=args.data_dir,
            )
            if args.result_path:
                result_path = Path(args.result_path).expanduser().resolve()
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.command == "smoke":
            payload = run_smoke(config_path=args.config_path)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.command == "notify":
            sent, message = send_run_summary_email(load_run_result(args.result_path))
            print(json.dumps({"sent": sent, "message": message}, ensure_ascii=False, indent=2))
            return 0

        if args.command == "test-email":
            sent, message = send_test_email()
            print(json.dumps({"sent": sent, "message": message}, ensure_ascii=False, indent=2))
            return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2
