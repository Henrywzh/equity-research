from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_RETENTION_DAYS = 30
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_SOURCE_SITE = "hkej"
DEFAULT_HOMEPAGE_URL = "https://www.hkej.com/instantnews"


def get_project_root() -> Path:
    env_root = os.environ.get("DAILY_MACRO_HOME")
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
    return get_data_dir(custom_data_dir) / "news.sqlite"


def get_raw_html_dir(custom_data_dir: str | Path | None = None) -> Path:
    return get_data_dir(custom_data_dir) / "raw_html"


def get_article_backup_dir(custom_data_dir: str | Path | None = None) -> Path:
    return get_data_dir(custom_data_dir) / "article_backups"


def get_analysis_dir(custom_data_dir: str | Path | None = None) -> Path:
    return get_data_dir(custom_data_dir) / "analyses"
