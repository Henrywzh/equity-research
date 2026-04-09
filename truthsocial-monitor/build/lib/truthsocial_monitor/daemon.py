"""
daemon.py — Persistent polling loop for the TruthSocial monitor.

Intended to run as a systemd service on the Oracle Cloud VM.
Tees stdout/stderr to logs/daemon.log so journald and the file both capture output.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .alerter import check_and_alert
from .config import get_gmail_credentials, get_logs_dir, get_state_dir
from .pipeline import run_once

logger = logging.getLogger(__name__)


class _Tee:
    """Write to both the original stream and a log file simultaneously."""

    def __init__(self, stream: object, fpath: Path) -> None:
        self._stream = stream
        fpath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(fpath, "a", buffering=1, encoding="utf-8")  # line-buffered

    def write(self, data: str) -> None:
        self._stream.write(data)  # type: ignore[union-attr]
        self._file.write(data)

    def flush(self) -> None:
        self._stream.flush()  # type: ignore[union-attr]
        self._file.flush()

    def fileno(self) -> int:
        return self._stream.fileno()  # type: ignore[union-attr]


def run_daemon(poll_interval: int = 60, alert_cooldown: int = 900) -> None:
    """Poll TruthSocial every poll_interval seconds. Send Gmail alert on keyword hits."""
    logs_dir = get_logs_dir()
    log_path = logs_dir / "daemon.log"
    sys.stdout = _Tee(sys.stdout, log_path)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.stderr, log_path)  # type: ignore[assignment]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
    )

    def _handle_sigterm(signum: int, frame: object) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n[daemon] Received SIGTERM at {ts} — shutting down.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[daemon] Started at {ts}")
    print(f"[daemon] Poll interval : {poll_interval}s")
    print(f"[daemon] Alert cooldown: {alert_cooldown}s")
    print(f"[daemon] Log file      : {log_path.resolve()}")
    print(f"[daemon] Press Ctrl-C to stop\n")

    state_dir = get_state_dir()
    sender, app_password, recipient = get_gmail_credentials()

    if not sender or not app_password or not recipient:
        print("[daemon] WARNING: Gmail credentials not set — alerts will be logged but not emailed.")

    try:
        while True:
            try:
                result = run_once()

                if result.status == "error":
                    logger.error("Pipeline error: %s", result.error)

                elif result.status == "ok" and result.new_posts:
                    if sender and app_password and recipient:
                        check_and_alert(
                            result.new_posts,
                            state_dir=state_dir,
                            sender=sender,
                            app_password=app_password,
                            recipient=recipient,
                            cooldown_secs=alert_cooldown,
                        )
                    else:
                        categories = []
                        for post in result.new_posts:
                            from .alerter import _scan_post
                            hits = _scan_post(post)
                            categories.extend(cat for _, cat in hits)
                        if categories:
                            logger.warning(
                                "Keyword hits detected (%s) but Gmail not configured — no email sent.",
                                ", ".join(sorted(set(categories))),
                            )

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.error("Unhandled exception in poll loop: %s", exc, exc_info=True)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n[daemon] Stopped by user at {ts}")
