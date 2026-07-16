"""Provider/account pool tests without contacting any real API."""

from __future__ import annotations

from daily_macro.llm_client import LLMTask, ModelResolver, RateLimitGovernor
from daily_macro.model_registry import build_model_pool
from daily_macro.provider_registry import load_provider_accounts, provider_model_ids
from daily_macro.types import ModelConfig, ProviderAccount


def test_provider_accounts_keep_explicit_quota_boundaries(monkeypatch):
    monkeypatch.setattr("daily_macro.provider_registry._config_values", lambda: {})
    monkeypatch.setenv("GROQ_API_KEY", "groq-one,groq-two")
    monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-one")
    monkeypatch.setenv("CEREBRAS_API_KEY_2", "cerebras-two")
    monkeypatch.setenv("CEREBRAS_API_KEY_4", "cerebras-four")
    monkeypatch.setenv("CEREBRAS_API_KEY_5", "cerebras-five")
    monkeypatch.setenv("DAILY_MACRO_CEREBRAS_QUOTA_SCOPE", "cerebras:shared-org")
    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEY", "google-one")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-one")

    accounts = load_provider_accounts()
    groq = [account for account in accounts if account.provider == "groq"]
    cerebras = [account for account in accounts if account.provider == "cerebras"]

    assert len(groq) == 2
    assert {account.quota_scope for account in groq} == {"groq:organization"}
    assert len(cerebras) == 4
    assert {account.quota_scope for account in cerebras} == {"cerebras:shared-org"}
    assert {account.provider for account in accounts} == {
        "groq",
        "cerebras",
        "google_ai_studio",
        "openrouter",
    }


def test_groq_allowlist_prefers_production_gpt_oss_then_preview_qwen(monkeypatch):
    monkeypatch.setattr("daily_macro.provider_registry._config_values", lambda: {})
    monkeypatch.delenv("DAILY_MACRO_GROQ_MODELS", raising=False)
    assert provider_model_ids("groq") == [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
    ]


def test_groq_model_override_can_select_qwen_primary(monkeypatch):
    monkeypatch.setattr("daily_macro.provider_registry._config_values", lambda: {})
    monkeypatch.setenv("DAILY_MACRO_GROQ_MODELS", "qwen/qwen3.6-27b,openai/gpt-oss-120b")
    assert provider_model_ids("groq") == ["qwen/qwen3.6-27b", "openai/gpt-oss-120b"]


def test_groq_model_override_cannot_reintroduce_unapproved_models(monkeypatch):
    monkeypatch.setattr("daily_macro.provider_registry._config_values", lambda: {})
    monkeypatch.setenv("DAILY_MACRO_GROQ_MODELS", "qwen/qwen3-32b,llama-3.1-8b-instant")
    assert provider_model_ids("groq") == []


def test_multi_provider_pool_namespaces_models(monkeypatch, tmp_path):
    monkeypatch.setattr("daily_macro.model_registry.load_groq_models", lambda *args, **kwargs: None)
    monkeypatch.setattr("daily_macro.model_registry.load_provider_models", lambda *args, **kwargs: None)
    accounts = [
        ProviderAccount("groq_1", "groq", "GROQ_API_KEY", "https://groq.test", api_key="x"),
        ProviderAccount("cerebras_1", "cerebras", "CEREBRAS_API_KEY", "https://cerebras.test", api_key="y"),
        ProviderAccount("cerebras_2", "cerebras", "CEREBRAS_API_KEY_2", "https://cerebras.test", api_key="y2"),
        ProviderAccount("google_1", "google_ai_studio", "GOOGLE_AI_STUDIO_API_KEY", "https://google.test", api_key="z"),
    ]

    pool = build_model_pool(["x"], data_dir=tmp_path, provider_accounts=accounts)
    providers = {model.provider for model in pool.models}
    assert providers == {"groq", "cerebras", "google_ai_studio"}
    assert any(model.endpoint_id == "cerebras:cerebras_1:gpt-oss-120b" for model in pool.models)
    assert sum(model.provider == "cerebras" and model.model_id == "gpt-oss-120b" for model in pool.models) == 2
    assert any(model.endpoint_id == "cerebras:cerebras_2:gpt-oss-120b" for model in pool.models)
    assert any(model.endpoint_id == "google_ai_studio:google_1:gemini-2.5-flash-lite" for model in pool.models)
    assert all(":" in endpoint_id for endpoint_id in pool.active_ids if endpoint_id.startswith("cerebras:"))


def test_non_groq_live_catalog_is_allowlisted(monkeypatch, tmp_path):
    monkeypatch.setattr("daily_macro.model_registry.load_groq_models", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "daily_macro.model_registry.load_provider_models",
        lambda *args, **kwargs: [
            {"id": "openai/gpt-oss-20b:free", "context_window": 131072},
            {"id": "unreviewed/provider-model-999b", "context_window": 999999},
        ],
    )
    account = ProviderAccount("openrouter_1", "openrouter", "OPENROUTER_API_KEY", "https://router.test", api_key="x")
    pool = build_model_pool(data_dir=tmp_path, provider_accounts=[account])
    assert any(model.model_id == "openai/gpt-oss-20b:free" for model in pool.models)
    assert all(model.model_id != "unreviewed/provider-model-999b" for model in pool.models)


def test_resolver_rejects_cerebras_glm_when_context_does_not_fit():
    resolver = ModelResolver(model_policy="allow_preview")
    glm = ModelConfig(
        "zai-glm-4.7",
        provider="cerebras",
        account_id="cerebras_1",
        quota_scope="cerebras:cerebras_1",
    )
    selection = resolver.resolve(
        LLMTask.CATEGORY_SYNTHESIS,
        [glm],
        estimated_input_tokens=8000,
        requested_output_tokens=1200,
    )
    assert any(item["reason"] == "context_window_exceeded" for item in selection.rejections)


def test_quota_scope_shares_rate_limit_state_across_credentials():
    governor = RateLimitGovernor(time_fn=lambda: 10.0, model_limits={"cerebras:org:gpt-oss-120b": {"tpm": 30000}})
    response = type(
        "Response",
        (),
        {
            "status_code": 200,
            "headers": {
                "x-ratelimit-remaining-tokens-minute": "1234",
                "x-ratelimit-reset-tokens-minute": "12s",
            },
        },
    )()
    governor.record_response("gpt-oss-120b", response, key_index=0, quota_scope="cerebras:org")

    first = governor._state("gpt-oss-120b", 0, "cerebras:org")
    second = governor._state("gpt-oss-120b", 1, "cerebras:org")
    assert first is second
    assert second.remaining_tokens == 1234


def test_governor_reserves_declared_rpm_and_tpm_before_headers():
    governor = RateLimitGovernor(model_limits={"cerebras:org:gpt-oss-120b": {"rpm": 5, "tpm": 30000}})
    governor.reserve_request("gpt-oss-120b", 1200, quota_scope="cerebras:org")
    state = governor._state("gpt-oss-120b", 0, "cerebras:org")
    assert state.remaining_requests == 4
    assert state.remaining_tokens == 28800
