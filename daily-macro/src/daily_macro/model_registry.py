"""Dynamic model pool construction.

Builds the candidate model pool from live provider catalogs intersected with
explicit provider allowlists, filtered to chat/JSON-capable models, with a disk
cache (TTL + last-known-good). Deprecated or unapproved live models never enter
the runnable pool.

This makes model add/removal a non-event: a removed model simply isn't in the
live list (so it drops out), and a newly released model is profiled via the
catalog heuristics and used without a code change.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

from .config import get_project_root
from .model_catalog import ModelCapability, get_capability
from .provider_registry import provider_model_ids
from .types import (
    DEFAULT_PROVIDER,
    GROQ_CHAT_COMPLETIONS_URL,
    ModelConfig,
    ProviderAccount,
)

LOGGER = logging.getLogger(__name__)

# Groq's approved primary model. GPT OSS 120B and 20B are ordered fallbacks,
# supplied by provider_model_ids().
FLOOR_MODEL_ID = "qwen/qwen3.6-27b"

_GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
_CACHE_FILENAME = "model_catalog_cache.json"
_CACHE_TTL_SECONDS = 24 * 3600


@dataclasses.dataclass
class ModelPool:
    """Result of building the candidate pool."""

    models: list[ModelConfig]
    capabilities: dict[str, ModelCapability]
    active_ids: set[str]
    accounts: list[ProviderAccount] = dataclasses.field(default_factory=list, repr=False)


def _refresh_enabled() -> bool:
    """Refresh is ON by default; only an explicit falsey value disables it."""
    raw = str(os.environ.get("DAILY_MACRO_REFRESH_MODEL_CATALOG") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def load_groq_models(api_key: str, *, timeout: int = 10) -> list[dict[str, Any]] | None:
    """Fetch live Groq model metadata. Returns None if disabled or on failure."""
    if not _refresh_enabled():
        return None
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})
    try:
        response = session.get(_GROQ_MODELS_URL, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - resilience: fall back to cache/catalog
        LOGGER.warning("Could not fetch Groq model list: %s", exc)
        return None
    finally:
        session.close()

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    models: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id or item.get("active") is False:
            continue
        models.append(
            {
                "id": model_id,
                "context_window": item.get("context_window"),
                "max_completion_tokens": item.get("max_completion_tokens"),
            }
        )
    return models or None


def _cache_path(data_dir: str | Path | None) -> Path:
    base = Path(data_dir) if data_dir else (get_project_root() / "daily-macro" / "data")
    return base / _CACHE_FILENAME


def _write_cache(path: Path, models: list[dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": time.time(), "models": models}), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - cache is best-effort
        LOGGER.warning("Could not write model catalog cache: %s", exc)


def _read_cache(path: Path) -> list[dict[str, Any]] | None:
    """Return cached models (last-known-good, even if past TTL)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        models = raw.get("models") if isinstance(raw, dict) else None
        age = time.time() - float(raw.get("fetched_at", 0)) if isinstance(raw, dict) else None
        if isinstance(models, list) and models:
            if age is not None and age > _CACHE_TTL_SECONDS:
                LOGGER.info("Using stale model catalog cache (age %.0fh).", age / 3600)
            return models
    except Exception:  # noqa: BLE001 - missing/corrupt cache is fine
        return None
    return None


def _merged_capability(model_id: str, live_meta: dict[str, Any], *, provider: str = DEFAULT_PROVIDER) -> ModelCapability:
    """Capability for a model, with live context/output metadata layered on top."""
    cap = get_capability(model_id, provider=provider)
    overrides: dict[str, Any] = {}
    live_ctx = live_meta.get("context_window")
    if isinstance(live_ctx, int) and live_ctx > 0:
        overrides["context_window"] = live_ctx
    live_out = live_meta.get("max_completion_tokens")
    if isinstance(live_out, int) and live_out > 0:
        overrides["max_output_tokens"] = live_out
        # Never request more output than the model can produce.
        overrides["max_completion_tokens"] = min(cap.max_completion_tokens, live_out)
    return dataclasses.replace(cap, **overrides) if overrides else cap


def load_provider_models(account: ProviderAccount, *, timeout: int = 10) -> list[dict[str, Any]] | None:
    """Best-effort model discovery for an OpenAI-compatible provider."""
    if not _refresh_enabled() or not account.api_key:
        return None
    url = account.base_url.rsplit("/chat/completions", 1)[0] + "/models"
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {account.api_key}"})
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - static catalog remains usable
        LOGGER.info("Could not refresh %s model list for %s: %s", account.provider, account.account_id, exc)
        return None
    finally:
        session.close()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    models: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id or item.get("active") is False:
            continue
        models.append(
            {
                "id": model_id,
                "context_window": item.get("context_window") or item.get("context_length"),
                "max_completion_tokens": item.get("max_completion_tokens") or item.get("max_output_tokens"),
            }
        )
    return models or None


def _provider_cache_path(data_dir: str | Path | None, provider: str, account_id: str) -> Path:
    base = Path(data_dir) if data_dir else (get_project_root() / "daily-macro" / "data")
    safe_name = f"model_catalog_{provider}_{account_id}.json".replace("/", "_")
    return base / safe_name


def _accounts_for_pool(api_keys: list[str] | None, provider_accounts: list[ProviderAccount] | None) -> list[ProviderAccount]:
    if provider_accounts is not None:
        return list(provider_accounts)
    keys = list(api_keys or [])
    return [
        ProviderAccount(
            account_id=f"groq_{index}",
            provider=DEFAULT_PROVIDER,
            api_key_env="GROQ_API_KEY",
            base_url=GROQ_CHAT_COMPLETIONS_URL,
            quota_scope="groq:organization",
            api_key=key,
        )
        for index, key in enumerate(keys or [None], start=1)
    ]


def build_model_pool(
    api_keys: list[str] | None = None,
    *,
    data_dir: str | Path | None = None,
    provider_accounts: list[ProviderAccount] | None = None,
) -> ModelPool:
    """Build a multi-provider pool with live discovery and static fallbacks."""
    accounts = _accounts_for_pool(api_keys, provider_accounts)
    capabilities: dict[str, ModelCapability] = {}
    models: list[ModelConfig] = []
    active_ids: set[str] = set()

    # Groq credentials share one model list; key rotation remains a credential
    # concern in AnalysisRuntime and does not duplicate every model six times.
    seen_account_models: set[tuple[str, str]] = set()
    for account in accounts:
        if account.provider == DEFAULT_PROVIDER and any(m.provider == DEFAULT_PROVIDER for m in models):
            continue
        cache_path = _cache_path(data_dir) if account.provider == DEFAULT_PROVIDER else _provider_cache_path(data_dir, account.provider, account.account_id)
        live = load_groq_models(account.api_key or "") if account.provider == DEFAULT_PROVIDER else load_provider_models(account)
        if live:
            _write_cache(cache_path, live)
        else:
            live = _read_cache(cache_path) if account.provider == DEFAULT_PROVIDER else None

        live_meta = {entry["id"]: entry for entry in (live or [])}
        if live and account.provider == DEFAULT_PROVIDER:
            # Groq's live catalog is broad and changes during the current
            # retirement cycle. Never turn every discovered chat model into a
            # runnable candidate; only use the explicit transition allowlist.
            allowed_ids = provider_model_ids(account.provider)
            candidate_ids = [model_id for model_id in allowed_ids if model_id in live_meta]
        elif live:
            # Third-party model catalogs can contain hundreds of unrelated or
            # short-lived models. Keep non-Groq providers on the explicit
            # configured/default allowlist; operators can opt in to another
            # model with DAILY_MACRO_*_MODELS.
            allowed_ids = provider_model_ids(account.provider)
            candidate_ids = [model_id for model_id in allowed_ids if model_id in live_meta]
        elif account.provider == DEFAULT_PROVIDER:
            # A failed refresh must not widen the pool back to the historical
            # catalog. Use the same explicit Groq allowlist as the live path.
            candidate_ids = provider_model_ids(account.provider)
        else:
            candidate_ids = provider_model_ids(account.provider)

        for model_id in candidate_ids:
            if (account.account_id, model_id) in seen_account_models:
                continue
            seen_account_models.add((account.account_id, model_id))
            cap = _merged_capability(model_id, live_meta.get(model_id, {}), provider=account.provider)
            if cap.kind != "chat" or not cap.supports_json:
                continue
            capability_key = model_id if account.provider == DEFAULT_PROVIDER else f"{account.provider}:{model_id}"
            capabilities[capability_key] = cap
            model = ModelConfig(
                model_id=model_id,
                provider=account.provider,
                max_completion_tokens=cap.max_completion_tokens,
                api_url=account.base_url,
                api_key_env=account.api_key_env,
                account_id=account.account_id,
                quota_scope=account.quota_scope,
                api_key=account.api_key,
            )
            models.append(model)
            capabilities[model.endpoint_id] = cap
            active_ids.update({model_id, model.endpoint_id})

    # Do not synthesize a missing Groq model here. If live discovery says a
    # model is gone, re-adding it defeats deprecation handling; if discovery
    # is unavailable, the explicit Groq allowlist above is already used.

    def _score(model: ModelConfig) -> float:
        key = model.model_id if model.provider == DEFAULT_PROVIDER else f"{model.provider}:{model.model_id}"
        scores = capabilities.get(key, get_capability(model.model_id, model.provider)).task_scores
        return sum(scores.values()) / len(scores) if scores else 0.5

    # Qwen anchors the Groq chain. Preserve the explicit Groq fallback order;
    # average task score alone would otherwise place 20B ahead of 120B.
    floors = [m for m in models if m.model_id == FLOOR_MODEL_ID and m.provider == DEFAULT_PROVIDER]
    groq_order = {model_id: index for index, model_id in enumerate(provider_model_ids(DEFAULT_PROVIDER))}
    rest = sorted(
        [m for m in models if m not in floors],
        key=lambda model: (
            0,
            groq_order.get(model.model_id, len(groq_order)),
        ) if model.provider == DEFAULT_PROVIDER else (
            1,
            -_score(model),
        ),
    )
    ordered = [*floors, *rest]
    LOGGER.info(
        "Built model pool (%d models, providers=%s): %s",
        len(ordered),
        ",".join(sorted({model.provider for model in ordered})) or "none",
        ", ".join(f"{model.provider}/{model.account_id}/{model.model_id}" for model in ordered),
    )
    return ModelPool(models=ordered, capabilities=capabilities, active_ids=active_ids, accounts=accounts)
