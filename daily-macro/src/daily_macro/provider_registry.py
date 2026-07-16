"""Provider/account configuration for the daily-macro LLM pool.

This module contains no provider calls. It only turns environment/config values
into safe endpoint descriptors, keeping credentials out of source and report
data. One ``ProviderAccount`` represents one credential and one quota boundary.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from .config import get_project_root
from .types import (
    CEREBRAS_CHAT_COMPLETIONS_URL,
    GOOGLE_AI_STUDIO_CHAT_COMPLETIONS_URL,
    GROQ_CHAT_COMPLETIONS_URL,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    ProviderAccount,
)


def _config_paths() -> list[Path]:
    root = get_project_root()
    return [root / ".config", root.parent / ".config"]


def _config_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in _config_paths():
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                values[key.strip()] = value.strip().strip("'\"")
        except OSError:
            continue
    return values


def config_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value.strip()
    return _config_values().get(name) or None


def _split(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _keys_for(prefix: str, *, aliases: tuple[str, ...] = ()) -> list[str]:
    """Load comma-separated keys plus conventional numbered key variables."""
    values: list[str] = []
    for name in (prefix, *aliases):
        values.extend(_split(config_value(name)))
    for index in range(2, 10):
        value = config_value(f"{prefix}_{index}")
        if value:
            values.extend(_split(value))
    # Preserve order but avoid accidentally using the same credential twice.
    return list(dict.fromkeys(values))


def _model_override_env(provider: str) -> list[str] | None:
    env_name = {
        "groq": "DAILY_MACRO_GROQ_MODELS",
        "cerebras": "DAILY_MACRO_CEREBRAS_MODELS",
        "google_ai_studio": "DAILY_MACRO_GOOGLE_AI_MODELS",
        "openrouter": "DAILY_MACRO_OPENROUTER_MODELS",
    }.get(provider)
    if not env_name:
        return None
    values = _split(config_value(env_name))
    return values or None


def load_provider_accounts() -> list[ProviderAccount]:
    """Return configured accounts without requiring any provider API call.

    Groq keys intentionally default to one organization quota scope because
    Groq publishes limits at organization level. Cerebras defaults to one scope
    per configured account; set ``DAILY_MACRO_CEREBRAS_QUOTA_SCOPE`` when keys
    belong to the same Cerebras organization/project.
    """
    accounts: list[ProviderAccount] = []

    groq_keys = _keys_for("GROQ_API_KEY")
    groq_scope = config_value("DAILY_MACRO_GROQ_QUOTA_SCOPE") or "groq:organization"
    for index, key in enumerate(groq_keys, start=1):
        accounts.append(
            ProviderAccount(
                account_id=f"groq_{index}",
                provider="groq",
                api_key_env="GROQ_API_KEY",
                base_url=GROQ_CHAT_COMPLETIONS_URL,
                quota_scope=groq_scope,
                api_key=key,
            )
        )

    cerebras_keys = _keys_for("CEREBRAS_API_KEY", aliases=("CEREBRAS_API_KEYS", "DAILY_MACRO_CEREBRAS_API_KEYS"))
    cerebras_scope = config_value("DAILY_MACRO_CEREBRAS_QUOTA_SCOPE")
    for index, key in enumerate(cerebras_keys, start=1):
        account_id = f"cerebras_{index}"
        accounts.append(
            ProviderAccount(
                account_id=account_id,
                provider="cerebras",
                api_key_env="CEREBRAS_API_KEY",
                base_url=CEREBRAS_CHAT_COMPLETIONS_URL,
                quota_scope=cerebras_scope or f"cerebras:{account_id}",
                api_key=key,
            )
        )

    google_keys = _keys_for(
        "GOOGLE_AI_STUDIO_API_KEY",
        aliases=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    for index, key in enumerate(google_keys, start=1):
        accounts.append(
            ProviderAccount(
                account_id=f"google_ai_studio_{index}",
                provider="google_ai_studio",
                api_key_env="GOOGLE_AI_STUDIO_API_KEY",
                base_url=GOOGLE_AI_STUDIO_CHAT_COMPLETIONS_URL,
                quota_scope=f"google-ai-studio:project-{index}",
                api_key=key,
            )
        )

    openrouter_keys = _keys_for("OPENROUTER_API_KEY")
    for index, key in enumerate(openrouter_keys, start=1):
        accounts.append(
            ProviderAccount(
                account_id=f"openrouter_{index}",
                provider="openrouter",
                api_key_env="OPENROUTER_API_KEY",
                base_url=OPENROUTER_CHAT_COMPLETIONS_URL,
                quota_scope=f"openrouter:account-{index}",
                api_key=key,
            )
        )

    return [account for account in accounts if account.enabled]


def provider_model_ids(provider: str) -> list[str]:
    configured = _model_override_env(provider)
    if provider == "groq":
        approved = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
        ]
        if configured is not None:
            return [model_id for model_id in configured if model_id in approved]
        return approved
    if configured:
        return configured
    defaults = {
        "cerebras": ["gpt-oss-120b", "gemma-4-31b", "zai-glm-4.7"],
        "google_ai_studio": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
        "openrouter": ["openai/gpt-oss-20b:free", "google/gemma-4-31b-it:free"],
    }
    return list(defaults.get(provider, []))


def account_without_secret(account: ProviderAccount) -> ProviderAccount:
    """Return a descriptor suitable for diagnostics or tests."""
    return replace(account, api_key=None)
