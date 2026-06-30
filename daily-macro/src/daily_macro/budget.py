"""Per-day model token/request budget accounting with cross-run persistence.

Groq imposes daily limits (TPD/RPD) per ``(model, API key)`` that reset at UTC
midnight. The per-minute rate-limit headers we observe at request time vanish
between runs, so a second run on the same day would start blind and could hammer
a premium model whose daily budget the morning run already drained — eating 429s
and capped waits on exactly the high-value tasks the premium models exist for.

:class:`DailyBudgetLedger` records actual token/request usage per
``(UTC-date, model_id, key_index)`` and persists it to disk so budget survives
across same-day runs. The resolver consults remaining daily budget to skip an
exhausted model and to reserve scarce premium capacity for high-value tasks.

Persistence mirrors the best-effort, atomic disk-cache pattern in
``model_registry`` — a corrupt or missing file simply yields an empty ledger.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_project_root

LOGGER = logging.getLogger(__name__)

_LEDGER_FILENAME = "model_budget_usage.json"

# Sentinel returned by remaining_*(...) when the declared limit is unknown: the
# model is treated as having unlimited daily budget (no gating).
UNLIMITED = float("inf")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ledger_path(data_dir: str | Path | None) -> Path:
    base = Path(data_dir) if data_dir else (get_project_root() / "daily-macro" / "data")
    return base / _LEDGER_FILENAME


@dataclasses.dataclass
class DailyBudgetLedger:
    """Tracks per-``(model, key)`` token/request usage for the current UTC day."""

    path: Path | None = None
    date: str = dataclasses.field(default_factory=_today)
    # model_id -> key_index(str) -> {"tokens": int, "requests": int}
    usage: dict[str, dict[str, dict[str, int]]] = dataclasses.field(default_factory=dict)
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False, compare=False)

    @classmethod
    def load(cls, data_dir: str | Path | None) -> "DailyBudgetLedger":
        """Load today's usage from disk, pruning any other date. Never raises."""
        path = _ledger_path(data_dir)
        today = _today()
        usage: dict[str, dict[str, dict[str, int]]] = {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            day = raw.get(today) if isinstance(raw, dict) else None
            if isinstance(day, dict):
                usage = day
        except Exception:  # noqa: BLE001 - missing/corrupt ledger is fine
            usage = {}
        return cls(path=path, date=today, usage=usage)

    def _bucket(self, model_id: str, key_index: int) -> dict[str, int]:
        return self.usage.setdefault(model_id, {}).setdefault(
            str(key_index), {"tokens": 0, "requests": 0}
        )

    def record(self, model_id: str, key_index: int, tokens: int, requests: int = 1) -> None:
        """Add ``tokens``/``requests`` to today's bucket for this model+key."""
        with self._lock:
            bucket = self._bucket(model_id, key_index)
            bucket["tokens"] += max(0, int(tokens))
            bucket["requests"] += max(0, int(requests))

    def used_tokens(self, model_id: str, key_index: int) -> int:
        with self._lock:
            return self._bucket(model_id, key_index)["tokens"]

    def remaining_tokens(self, model_id: str, key_index: int, declared_tpd: int | None) -> float:
        """Declared TPD minus tokens used today (floored at 0); UNLIMITED if unknown."""
        if not declared_tpd:
            return UNLIMITED
        with self._lock:
            return max(0, int(declared_tpd) - self._bucket(model_id, key_index)["tokens"])

    def remaining_requests(self, model_id: str, key_index: int, declared_rpd: int | None) -> float:
        """Declared RPD minus requests made today (floored at 0); UNLIMITED if unknown."""
        if not declared_rpd:
            return UNLIMITED
        with self._lock:
            return max(0, int(declared_rpd) - self._bucket(model_id, key_index)["requests"])

    def flush(self) -> None:
        """Atomically persist today's usage to disk. Best-effort (never raises)."""
        if self.path is None:
            return
        with self._lock:
            payload: dict[str, Any] = {self.date: self.usage}
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                os.replace(tmp, self.path)
            except Exception as exc:  # noqa: BLE001 - persistence is best-effort
                LOGGER.warning("Could not write budget ledger: %s", exc)
