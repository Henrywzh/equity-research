from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .types import DEFAULT_OUTPUT_TOKENS

LOGGER = logging.getLogger(__name__)

_OVERRIDES_PATH = Path(__file__).with_name("model_overrides.json")

TASK_NAMES = (
    "routing",
    "article_analysis",
    "category_synthesis",
    "top_alerts",
    "json_repair",
    "critic",
)

DEFAULT_TASK_SCORES = {name: 0.5 for name in TASK_NAMES}


@dataclass(frozen=True)
class ModelCapability:
    model_id: str
    provider: str = "groq"
    lifecycle: str = "production"
    kind: str = "chat"
    context_window: int = 8192
    max_output_tokens: int = 1200
    max_completion_tokens: int = DEFAULT_OUTPUT_TOKENS
    supports_json: bool = True
    task_scores: dict[str, float] = field(default_factory=dict)
    limits: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Capability classification / heuristics (for models without a curated entry)
# ---------------------------------------------------------------------------

# Keyword → non-chat kind. Anything not matched is treated as a general chat
# model. Order matters only for readability; matches are independent.
_KIND_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("asr", ("whisper",)),
    ("tts", ("orpheus", "playai", "-tts", "dia-")),
    ("guard", ("guard", "safeguard", "moderation")),
    ("embed", ("embed", "embedding")),
    ("agentic", ("compound",)),
    ("specialized", ("allam",)),
)

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b")


def infer_kind(model_id: str) -> str:
    """Classify a model by id so non-chat models can be filtered out.

    Returns ``"chat"`` for anything that looks like a general-purpose chat
    completion model.
    """
    mid = model_id.lower()
    for kind, keywords in _KIND_KEYWORDS:
        if any(keyword in mid for keyword in keywords):
            return kind
    return "chat"


def _infer_size_b(model_id: str) -> float | None:
    matches = _SIZE_RE.findall(model_id.lower())
    if not matches:
        return None
    return max(float(m) for m in matches)


def heuristic_scores(model_id: str) -> dict[str, float]:
    """Provisional per-task quality scores for an uncatalogued model.

    Inferred from the parameter-size token in the id (bigger ≈ stronger). This
    is only a starting prior; Step 2 reliability calibration refines it from
    observed behavior.
    """
    size = _infer_size_b(model_id)
    if size is None:
        base = 0.5
    elif size >= 100:
        base = 0.85
    elif size >= 60:
        base = 0.78
    elif size >= 25:
        base = 0.68
    elif size >= 12:
        base = 0.60
    else:
        base = 0.50
    return {name: base for name in TASK_NAMES}


# ---------------------------------------------------------------------------
# Catalog loading (curated overrides file with an in-code safety net)
# ---------------------------------------------------------------------------


def _builtin_catalog() -> dict[str, ModelCapability]:
    """Fallback catalog used only if model_overrides.json is missing/malformed."""
    common = dict(context_window=131_072, max_output_tokens=8192, supports_json=True)
    return {
        "meta-llama/llama-4-scout-17b-16e-instruct": ModelCapability(
            model_id="meta-llama/llama-4-scout-17b-16e-instruct",
            lifecycle="preview",
            max_completion_tokens=1536,
            task_scores={"routing": 0.45, "article_analysis": 0.70, "category_synthesis": 0.70, "top_alerts": 0.65, "json_repair": 0.45, "critic": 0.65},
            **common,
        ),
        "qwen/qwen3-32b": ModelCapability(
            model_id="qwen/qwen3-32b",
            lifecycle="preview",
            max_completion_tokens=1536,
            task_scores={"routing": 0.60, "article_analysis": 0.72, "category_synthesis": 0.74, "top_alerts": 0.70, "json_repair": 0.65, "critic": 0.70},
            **common,
        ),
        "llama-3.1-8b-instant": ModelCapability(
            model_id="llama-3.1-8b-instant",
            lifecycle="production",
            max_completion_tokens=1200,
            task_scores={"routing": 0.95, "article_analysis": 0.55, "category_synthesis": 0.40, "top_alerts": 0.30, "json_repair": 0.95, "critic": 0.35},
            **common,
        ),
        "llama-3.3-70b-versatile": ModelCapability(
            model_id="llama-3.3-70b-versatile",
            lifecycle="production",
            max_completion_tokens=2048,
            task_scores={"routing": 0.65, "article_analysis": 0.82, "category_synthesis": 0.86, "top_alerts": 0.84, "json_repair": 0.70, "critic": 0.84},
            **common,
        ),
        "openai/gpt-oss-20b": ModelCapability(
            model_id="openai/gpt-oss-20b",
            lifecycle="production",
            max_completion_tokens=2048,
            task_scores={"routing": 0.70, "article_analysis": 0.86, "category_synthesis": 0.88, "top_alerts": 0.92, "json_repair": 0.74, "critic": 0.90},
            **common,
        ),
        "openai/gpt-oss-120b": ModelCapability(
            model_id="openai/gpt-oss-120b",
            lifecycle="production",
            max_completion_tokens=2048,
            task_scores={"routing": 0.55, "article_analysis": 0.90, "category_synthesis": 0.96, "top_alerts": 0.98, "json_repair": 0.55, "critic": 0.96},
            **common,
        ),
    }


def _capability_from_spec(model_id: str, spec: dict) -> ModelCapability:
    provider = str(spec.get("provider", "groq"))
    actual_model_id = str(spec.get("model_id") or model_id)
    if ":" in model_id and model_id.startswith(("cerebras:", "google_ai_studio:", "openrouter:")):
        actual_model_id = model_id.split(":", 1)[1]
    return ModelCapability(
        model_id=actual_model_id,
        provider=provider,
        lifecycle=str(spec.get("lifecycle", "production")),
        kind=str(spec.get("kind", "chat")),
        context_window=int(spec.get("context_window", 8192)),
        max_output_tokens=int(spec.get("max_output_tokens", 1200)),
        max_completion_tokens=int(spec.get("max_completion_tokens", DEFAULT_OUTPUT_TOKENS)),
        supports_json=bool(spec.get("supports_json", True)),
        task_scores=dict(spec.get("task_scores") or {}),
        limits={k: int(v) for k, v in (spec.get("limits") or {}).items()},
    )


def _load_catalog() -> dict[str, ModelCapability]:
    try:
        raw = json.loads(_OVERRIDES_PATH.read_text(encoding="utf-8"))
        models = raw.get("models") if isinstance(raw, dict) else None
        if not isinstance(models, dict):
            raise ValueError("model_overrides.json missing a 'models' object")
        catalog = {
            model_id: _capability_from_spec(model_id, spec)
            for model_id, spec in models.items()
            if isinstance(spec, dict)
        }
        if not catalog:
            raise ValueError("model_overrides.json parsed to an empty catalog")
        return catalog
    except Exception as exc:  # noqa: BLE001 - resilience: never break on bad config
        LOGGER.warning("Using builtin model catalog (could not load %s: %s)", _OVERRIDES_PATH.name, exc)
        return _builtin_catalog()


MODEL_CATALOG: dict[str, ModelCapability] = _load_catalog()


def get_capability(model_id: str, provider: str = "groq") -> ModelCapability:
    """Return the capability for a model: curated entry, else heuristic.

    Never raises — an unknown model gets an inferred kind + provisional scores so
    a newly released Groq model is usable without a code change.
    """
    capability = MODEL_CATALOG.get(f"{provider}:{model_id}") or MODEL_CATALOG.get(model_id)
    if capability is not None:
        return capability
    return ModelCapability(
        model_id=model_id,
        provider=provider,
        kind=infer_kind(model_id),
        task_scores=heuristic_scores(model_id),
    )
