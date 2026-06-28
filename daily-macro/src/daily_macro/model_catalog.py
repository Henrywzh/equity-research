from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCapability:
    model_id: str
    provider: str = "groq"
    lifecycle: str = "production"
    context_window: int = 8192
    max_output_tokens: int = 1200
    supports_json: bool = True
    task_scores: dict[str, float] = field(default_factory=dict)


DEFAULT_TASK_SCORES = {
    "routing": 0.5,
    "article_analysis": 0.5,
    "category_synthesis": 0.5,
    "top_alerts": 0.5,
    "json_repair": 0.5,
    "critic": 0.5,
}


MODEL_CATALOG: dict[str, ModelCapability] = {
    "meta-llama/llama-4-scout-17b-16e-instruct": ModelCapability(
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
        lifecycle="preview",
        context_window=131_072,
        max_output_tokens=8192,
        task_scores={
            "routing": 0.45,
            "article_analysis": 0.70,
            "category_synthesis": 0.70,
            "top_alerts": 0.65,
            "json_repair": 0.45,
            "critic": 0.65,
        },
    ),
    "qwen/qwen3-32b": ModelCapability(
        model_id="qwen/qwen3-32b",
        lifecycle="preview",
        context_window=131_072,
        max_output_tokens=8192,
        task_scores={
            "routing": 0.60,
            "article_analysis": 0.72,
            "category_synthesis": 0.74,
            "top_alerts": 0.70,
            "json_repair": 0.65,
            "critic": 0.70,
        },
    ),
    "llama-3.1-8b-instant": ModelCapability(
        model_id="llama-3.1-8b-instant",
        lifecycle="production",
        context_window=131_072,
        max_output_tokens=8192,
        task_scores={
            "routing": 0.95,
            "article_analysis": 0.55,
            "category_synthesis": 0.40,
            "top_alerts": 0.30,
            "json_repair": 0.95,
            "critic": 0.35,
        },
    ),
    "llama-3.3-70b-versatile": ModelCapability(
        model_id="llama-3.3-70b-versatile",
        lifecycle="production",
        context_window=131_072,
        max_output_tokens=8192,
        task_scores={
            "routing": 0.65,
            "article_analysis": 0.82,
            "category_synthesis": 0.86,
            "top_alerts": 0.84,
            "json_repair": 0.70,
            "critic": 0.84,
        },
    ),
    "openai/gpt-oss-20b": ModelCapability(
        model_id="openai/gpt-oss-20b",
        lifecycle="production",
        context_window=131_072,
        max_output_tokens=8192,
        task_scores={
            "routing": 0.70,
            "article_analysis": 0.86,
            "category_synthesis": 0.88,
            "top_alerts": 0.92,
            "json_repair": 0.74,
            "critic": 0.90,
        },
    ),
    "openai/gpt-oss-120b": ModelCapability(
        model_id="openai/gpt-oss-120b",
        lifecycle="production",
        context_window=131_072,
        max_output_tokens=8192,
        task_scores={
            "routing": 0.55,
            "article_analysis": 0.90,
            "category_synthesis": 0.96,
            "top_alerts": 0.98,
            "json_repair": 0.55,
            "critic": 0.96,
        },
    ),
}


def get_capability(model_id: str, provider: str = "groq") -> ModelCapability:
    capability = MODEL_CATALOG.get(model_id)
    if capability is not None:
        return capability
    return ModelCapability(
        model_id=model_id,
        provider=provider,
        task_scores=dict(DEFAULT_TASK_SCORES),
    )
