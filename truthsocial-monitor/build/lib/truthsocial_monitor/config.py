from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_local_config() -> None:
    for path in (REPO_ROOT / ".config", PROJECT_ROOT / ".config"):
        _merge_config_file(path)


def read_env(primary: str, legacy: str | None = None) -> str:
    value = os.getenv(primary, "").strip()
    if value:
        return value
    if legacy:
        return os.getenv(legacy, "").strip()
    return ""


def _merge_config_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        os.environ.setdefault(key, value)


def get_db_path() -> Path:
    return PROJECT_ROOT / "data" / "trump_posts.db"


def get_state_dir() -> Path:
    return PROJECT_ROOT / "state"


def get_logs_dir() -> Path:
    return PROJECT_ROOT / "logs"


def get_truthsocial_credentials() -> tuple[str, str]:
    load_local_config()
    username = read_env("TRUTHSOCIAL_USERNAME")
    password = read_env("TRUTHSOCIAL_PASSWORD")
    return username, password


def get_gmail_credentials() -> tuple[str, str, str]:
    load_local_config()
    sender = read_env("GMAIL_SENDER")
    app_password = read_env("GMAIL_APP_PASSWORD")
    recipient = read_env("GMAIL_RECIPIENT")
    return sender, app_password, recipient
