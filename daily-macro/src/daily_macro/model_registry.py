"""Dynamic model pool construction.

Builds the candidate model pool from the live Groq ``/models`` list intersected
with the local catalog, filtered to chat/JSON-capable models, with a disk cache
(TTL + last-known-good) and a guaranteed floor model so the pool is never empty.

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
from .model_catalog import MODEL_CATALOG, ModelCapability, get_capability
from .types import DEFAULT_PROVIDER, ModelConfig

LOGGER = logging.getLogger(__name__)

# A stable, high-throughput production model kept as the guaranteed fallback so
# the pool (and the resolver's last resort) is never empty.
FLOOR_MODEL_ID = "llama-3.1-8b-instant"

_GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
_CACHE_FILENAME = "model_catalog_cache.json"
_CACHE_TTL_SECONDS = 24 * 3600


@dataclasses.dataclass
class ModelPool:
    """Result of building the candidate pool."""

    models: list[ModelConfig]
    capabilities: dict[str, ModelCapability]
    active_ids: set[str]


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


def _merged_capability(model_id: str, live_meta: dict[str, Any]) -> ModelCapability:
    """Capability for a model, with live context/output metadata layered on top."""
    cap = get_capability(model_id)
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


def build_model_pool(api_keys: list[str], *, data_dir: str | Path | None = None) -> ModelPool:
    """Build the candidate pool: live ∩ catalog, chat-only, floor-guaranteed."""
    cache_path = _cache_path(data_dir)
    live = load_groq_models(api_keys[0]) if api_keys else None
    if live:
        _write_cache(cache_path, live)
    else:
        live = _read_cache(cache_path)

    live_meta = {entry["id"]: entry for entry in (live or [])}
    candidate_ids = list(live_meta.keys()) if live else list(MODEL_CATALOG.keys())

    capabilities: dict[str, ModelCapability] = {}
    for model_id in candidate_ids:
        cap = _merged_capability(model_id, live_meta.get(model_id, {}))
        if cap.kind != "chat" or not cap.supports_json:
            continue  # exclude audio / guard / tts / embedding / agentic models
        capabilities[model_id] = cap

    # Guarantee the floor model is always present and callable.
    if FLOOR_MODEL_ID not in capabilities:
        capabilities[FLOOR_MODEL_ID] = _merged_capability(FLOOR_MODEL_ID, live_meta.get(FLOOR_MODEL_ID, {}))

    def _mean_score(model_id: str) -> float:
        scores = capabilities[model_id].task_scores
        return sum(scores.values()) / len(scores) if scores else 0.5

    # Floor model first (anchors the resolver's preferred-bonus + last-resort
    # fallback on a safe, high-limit model); the rest by descending quality.
    rest = sorted((m for m in capabilities if m != FLOOR_MODEL_ID), key=_mean_score, reverse=True)
    ordered_ids = [FLOOR_MODEL_ID, *rest]

    models = [
        ModelConfig(
            model_id=model_id,
            provider=DEFAULT_PROVIDER,
            max_completion_tokens=capabilities[model_id].max_completion_tokens,
        )
        for model_id in ordered_ids
    ]
    LOGGER.info(
        "Built model pool (%d models, source=%s): %s",
        len(models),
        "live" if live else "catalog/cache",
        ", ".join(m.model_id for m in models),
    )
    return ModelPool(models=models, capabilities=capabilities, active_ids=set(capabilities.keys()))
