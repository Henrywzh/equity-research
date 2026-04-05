from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import WatchlistConfig

# Canonical asset class ordering for display
ASSET_CLASS_ORDER = ["index", "commodity", "fx", "crypto", "custom"]

_CLASS_KEYS: dict[str, str] = {
    "indices": "index",
    "commodities": "commodity",
    "fx": "fx",
    "crypto": "crypto",
}


def load_watchlist(
    config_path: str | Path | None = None,
    user_email: str | None = None,
) -> WatchlistConfig:
    """Load and merge default + per-user extra symbols from watchlist.json."""
    if config_path is None:
        from .config import get_watchlist_path
        config_path = get_watchlist_path()

    raw = _read_config(Path(config_path))
    symbols: list[tuple[str, str]] = []
    seen: set[str] = set()

    default_section = raw.get("default") or {}
    for section_key, asset_class in _CLASS_KEYS.items():
        for ticker in default_section.get(section_key) or []:
            ticker = str(ticker).strip()
            if ticker and ticker not in seen:
                symbols.append((ticker, asset_class))
                seen.add(ticker)

    if user_email:
        users = raw.get("users") or {}
        user_cfg: dict[str, Any] = users.get(user_email) or {}
        for entry in user_cfg.get("extra_symbols") or []:
            if isinstance(entry, str):
                ticker, asset_class = entry.strip(), "custom"
            elif isinstance(entry, dict):
                ticker = str(entry.get("ticker") or "").strip()
                asset_class = str(entry.get("asset_class") or "custom").strip()
            else:
                continue
            if ticker and ticker not in seen:
                symbols.append((ticker, asset_class))
                seen.add(ticker)

    return WatchlistConfig(symbols=symbols)


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Watchlist config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
