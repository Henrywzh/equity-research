"""
run.py — Entry point for the Hormuz maritime monitor.

Usage:
  python run.py                                     # default: 15-min, LLM on 1+ ship, delta ≥1
  python run.py --interval 300                      # check every 5 minutes
  python run.py --llm-threshold 2                   # only call LLM if 2+ ships detected
  python run.py --delta-threshold 2                 # only call LLM if count changed by 2+
  python run.py --interval 600 --llm-threshold 1 --delta-threshold 1
"""
import sys
import os
import signal
import argparse
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hormuz maritime monitor")
parser.add_argument(
    "--interval", type=int, default=900,
    help="Seconds between checks (default: 900 = 15 min)"
)
parser.add_argument(
    "--llm-threshold", type=int, default=1,
    dest="llm_threshold",
    help="Min ships in zone before calling LLM API (default: 1)"
)
parser.add_argument(
    "--delta-threshold", type=int, default=1,
    dest="delta_threshold",
    help="Min change in ship count vs last reading to trigger LLM (default: 1)"
)
parser.add_argument(
    "--alert-cooldown", type=int, default=5,
    dest="alert_cooldown",
    help="Minutes between alert emails (default: 5; 0 = no cooldown)"
)
parser.add_argument(
    "--digest-hour", type=int, default=10,
    dest="digest_hour",
    help="UTC hour to send daily digest email (default: 10 = 10:00 UTC)"
)
args = parser.parse_args()

# ── Tee stdout/stderr to a log file ──────────────────────────────────────────
os.makedirs(os.path.join(_HERE, "logs"), exist_ok=True)
LOG_PATH = os.path.join(_HERE, "logs", "monitor.log")


class _Tee:
    """Write to both the original stream and a log file simultaneously."""
    def __init__(self, stream, fpath):
        self._stream = stream
        self._file   = open(fpath, "a", buffering=1)  # line-buffered

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def fileno(self):  # required by some stdlib internals
        return self._stream.fileno()


sys.stdout = _Tee(sys.stdout, LOG_PATH)
sys.stderr = _Tee(sys.stderr, LOG_PATH)

# ── Graceful SIGTERM (e.g. `kill <pid>`) ─────────────────────────────────────
def _handle_sigterm(signum, frame):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n[run.py] Received SIGTERM at {ts} — shutting down.")
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)

# ── Start ─────────────────────────────────────────────────────────────────────
from marine_traffic_monitor import run_monitor  # noqa: E402 (import after path/log setup)

ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
mins, secs = divmod(args.interval, 60)
interval_str = f"{mins}m {secs}s" if secs else f"{mins}m"

print(f"[run.py] ▶  Hormuz monitor started at {ts}")
print(f"[run.py]    Interval        : {args.interval}s ({interval_str})")
print(f"[run.py]    LLM threshold   : {args.llm_threshold} ship(s) in zone")
print(f"[run.py]    Delta threshold : ±{args.delta_threshold} ship(s) change vs last reading")
print(f"[run.py]    Alert cooldown  : {args.alert_cooldown} min between alert emails")
print(f"[run.py]    Daily digest    : {args.digest_hour:02d}:00 UTC")
print(f"[run.py]    Log file        : {os.path.abspath(LOG_PATH)}")
print(f"[run.py]    Press Ctrl-C to stop\n")

try:
    run_monitor(
        check_interval=args.interval,
        iteration=5,
        llm_threshold=args.llm_threshold,
        delta_threshold=args.delta_threshold,
        alert_cooldown=args.alert_cooldown,
        digest_hour=args.digest_hour,
    )
except KeyboardInterrupt:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n[run.py] ■  Stopped by user at {ts}")
