from __future__ import annotations

import os
from pathlib import Path

DEFAULT_RECENT_LIMIT = 80
DEFAULT_REQUEST_TIMEOUT = 30
PREFERRED_TRANSCRIPT_LANGUAGES = ("zh-Hant", "zh-Hans", "zh", "en")


def get_project_root() -> Path:
    env_root = os.environ.get("YOUTUBE_INTAKE_HOME")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists():
        return cwd

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent

    return cwd


def get_config_path(custom_config_path: str | Path | None = None) -> Path:
    if custom_config_path is not None:
        return Path(custom_config_path).expanduser().resolve()
    return get_project_root() / "config" / "channels.json"


def get_state_path(custom_state_path: str | Path | None = None) -> Path:
    if custom_state_path is not None:
        return Path(custom_state_path).expanduser().resolve()
    return get_project_root() / "state" / "channels.json"


def get_data_dir(custom_data_dir: str | Path | None = None) -> Path:
    if custom_data_dir is not None:
        return Path(custom_data_dir).expanduser().resolve()
    return get_project_root() / "data"
