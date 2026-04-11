from __future__ import annotations

from typing import Any

from .runtime_env import load_local_config, read_env, read_env_list


GROQ_API_KEY_ENV = "GROQ_API_KEY"
GMAIL_SENDER_ENV = "GMAIL_SENDER"
GMAIL_APP_PASSWORD_ENV = "GMAIL_APP_PASSWORD"
GMAIL_RECIPIENT_ENV = "GMAIL_RECIPIENT"
YT_COOKIES_ENV = "YOUTUBE_INTAKE_YT_COOKIES"

REQUIRED_ENV_VARS = (
    GROQ_API_KEY_ENV,
    GMAIL_SENDER_ENV,
    GMAIL_APP_PASSWORD_ENV,
    GMAIL_RECIPIENT_ENV,
)


def run_preflight() -> dict[str, Any]:
    load_local_config()
    groq_keys = read_env_list(GROQ_API_KEY_ENV)
    missing_required = [name for name in REQUIRED_ENV_VARS if not read_env(name)]
    if not groq_keys and GROQ_API_KEY_ENV not in missing_required:
        missing_required.append(GROQ_API_KEY_ENV)
    yt_cookies_available = bool(read_env(YT_COOKIES_ENV))

    payload: dict[str, Any] = {
        "status": "success" if not missing_required else "failed",
        "required_env": {
            name: (bool(groq_keys) if name == GROQ_API_KEY_ENV else bool(read_env(name)))
            for name in REQUIRED_ENV_VARS
        },
        "missing_required": missing_required,
        "groq_key_count": len(groq_keys),
        "optional_env": {
            YT_COOKIES_ENV: yt_cookies_available,
        },
        "notes": [],
    }
    if missing_required:
        payload["notes"].append(
            "Missing required secrets. Configure GitHub Actions secrets before relying on scheduled runs."
        )
    if not yt_cookies_available:
        payload["notes"].append(
            "Optional YouTube cookies are not configured. No-caption videos will remain metadata-only if audio download is bot-blocked."
        )
    return payload
