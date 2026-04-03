from __future__ import annotations

import argparse
import json
import logging

from .pipeline import run_smoke, run_sync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YouTube intake utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Archive new videos and livestream replays.")
    run_parser.add_argument("--config-path", default=None, help="Override the channel config path.")
    run_parser.add_argument("--state-path", default=None, help="Override the state manifest path.")
    run_parser.add_argument("--data-dir", default=None, help="Override the archive data directory.")

    smoke_parser = subparsers.add_parser("smoke", help="Validate the watched channels and fetch their latest IDs.")
    smoke_parser.add_argument("--config-path", default=None, help="Override the channel config path.")

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
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.command == "smoke":
            payload = run_smoke(config_path=args.config_path)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2
