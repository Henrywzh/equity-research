from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REQUEST_TIMEOUT = 30


def get_project_root() -> Path:
    env_root = os.environ.get("DAILY_MARKET_HOME")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists():
        return cwd

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent

    return cwd


def get_data_dir(custom_data_dir: str | Path | None = None) -> Path:
    if custom_data_dir is not None:
        return Path(custom_data_dir).expanduser().resolve()
    return get_project_root() / "data"


def get_db_path(
    custom_db_path: str | Path | None = None,
    custom_data_dir: str | Path | None = None,
) -> Path:
    if custom_db_path is not None:
        return Path(custom_db_path).expanduser().resolve()
    return get_data_dir(custom_data_dir) / "market.sqlite"


def get_snapshots_dir(custom_data_dir: str | Path | None = None) -> Path:
    return get_data_dir(custom_data_dir) / "snapshots"


def get_summaries_dir(custom_data_dir: str | Path | None = None) -> Path:
    return get_data_dir(custom_data_dir) / "summaries"


def get_polymarket_runs_dir(custom_data_dir: str | Path | None = None) -> Path:
    return get_data_dir(custom_data_dir) / "polymarket_runs"


def get_config_dir() -> Path:
    return get_project_root() / "config"


def get_watchlist_path() -> Path:
    return get_config_dir() / "watchlist.json"


def get_polymarket_watchlist_path() -> Path:
    return get_config_dir() / "polymarket_watchlist.json"
