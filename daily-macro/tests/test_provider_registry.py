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
    monkeypatch.setenv("ZAI_API_KEY", "zai-one")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cloudflare-one")
    cloudflare_account_id = "a" * 32
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", cloudflare_account_id)

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
        "zai",
        "cloudflare",
    }

    cloudflare = next(account for account in accounts if account.provider == "cloudflare")
    assert cloudflare.base_url == (
        f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/ai/v1/chat/completions"
    )
    assert cloudflare.quota_scope == f"cloudflare:{cloudflare_account_id}"


def test_groq_allowlist_prefers_qwen_then_production_gpt_oss(monkeypatch):
    monkeypatch.setattr("daily_macro.provider_registry._config_values", lambda: {})
    monkeypatch.delenv("DAILY_MACRO_GROQ_MODELS", raising=False)
    assert provider_model_ids("groq") == [
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]


def test_groq_model_override_can_select_qwen_primary(monkeypatch):
    monkeypatch.setattr("daily_macro.provider_registry._config_values", lambda: {})
    monkeypatch.setenv("DAILY_MACRO_GROQ_MODELS", "qwen/qwen3.6-27b,openai/gpt-oss-120b")
    assert provider_model_ids("groq") == ["qwen/qwen3.6-27b", "openai/gpt-oss-120b"]


def test_groq_model_override_cannot_reintroduce_unapproved_models(monkeypatch):
    monkeypatch.setattr("daily_macro.provider_registry._config_values", lambda: {})
    monkeypatch.setenv("DAILY_MACRO_GROQ_MODELS", "qwen/qwen3-32b,llama-3.1-8b-instant")
    assert provider_model_ids("groq") == []


def test_cloudflare_defaults_to_qwen_and_gemma(monkeypatch):
    monkeypatch.setattr("daily_macro.provider_registry._config_values", lambda: {})
    monkeypatch.delenv("DAILY_MACRO_CLOUDFLARE_MODELS", raising=False)
    assert provider_model_ids("cloudflare") == [
        "@cf/qwen/qwen3-30b-a3b-fp8",
        "@cf/google/gemma-4-26b-a4b-it",
    ]


def test_cloudflare_requires_account_id(monkeypatch):
    monkeypatch.setattr("daily_macro.provider_registry._config_values", lambda: {})
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cloudflare-one")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    assert all(account.provider != "cloudflare" for account in load_provider_accounts())


def test_cloudflare_pairs_comma_separated_tokens_and_account_ids(monkeypatch):
    monkeypatch.setattr("daily_macro.provider_registry._config_values", lambda: {})
    account_one = "a" * 32
    account_two = "b" * 32
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token-one,token-two")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", f"{account_one},{account_two}")

    accounts = [account for account in load_provider_accounts() if account.provider == "cloudflare"]

    assert [account.account_id for account in accounts] == ["cloudflare_1", "cloudflare_2"]
    assert [account.api_key for account in accounts] == ["token-one", "token-two"]
    assert [account.base_url for account in accounts] == [
        f"https://api.cloudflare.com/client/v4/accounts/{account_one}/ai/v1/chat/completions",
        f"https://api.cloudflare.com/client/v4/accounts/{account_two}/ai/v1/chat/completions",
    ]
    assert [account.quota_scope for account in accounts] == [
        f"cloudflare:{account_one}",
        f"cloudflare:{account_two}",
    ]


def test_multi_provider_pool_namespaces_models(monkeypatch, tmp_path):
    monkeypatch.setattr("daily_macro.model_registry.load_groq_models", lambda *args, **kwargs: None)
    monkeypatch.setattr("daily_macro.model_registry.load_provider_models", lambda *args, **kwargs: None)
    accounts = [
        ProviderAccount("groq_1", "groq", "GROQ_API_KEY", "https://groq.test", api_key="x"),
        ProviderAccount("cerebras_1", "cerebras", "CEREBRAS_API_KEY", "https://cerebras.test", api_key="y"),
        ProviderAccount("cerebras_2", "cerebras", "CEREBRAS_API_KEY_2", "https://cerebras.test", api_key="y2"),
        ProviderAccount("google_1", "google_ai_studio", "GOOGLE_AI_STUDIO_API_KEY", "https://google.test", api_key="z"),
        ProviderAccount("zai_1", "zai", "ZAI_API_KEY", "https://zai.test", api_key="za"),
        ProviderAccount("cloudflare_1", "cloudflare", "CLOUDFLARE_API_TOKEN", "https://cf.test", api_key="cf"),
    ]

    pool = build_model_pool(["x"], data_dir=tmp_path, provider_accounts=accounts)
    providers = {model.provider for model in pool.models}
    assert providers == {"groq", "cerebras", "google_ai_studio", "zai", "cloudflare"}
    assert any(model.endpoint_id == "cerebras:cerebras_1:gpt-oss-120b" for model in pool.models)
    assert sum(model.provider == "cerebras" and model.model_id == "gpt-oss-120b" for model in pool.models) == 2
    assert any(model.endpoint_id == "cerebras:cerebras_2:gpt-oss-120b" for model in pool.models)
    assert any(model.endpoint_id == "google_ai_studio:google_1:gemini-2.5-flash-lite" for model in pool.models)
    assert any(model.endpoint_id == "zai:zai_1:glm-4.7-flash" for model in pool.models)
    assert pool.models[0].provider == "zai"
    cloudflare_models = [model.model_id for model in pool.models if model.provider == "cloudflare"]
    assert cloudflare_models == [
        "@cf/qwen/qwen3-30b-a3b-fp8",
        "@cf/google/gemma-4-26b-a4b-it",
    ]
    resolver = ModelResolver(
        active_model_ids=pool.active_ids,
        capabilities=pool.capabilities,
        model_policy="production_only",
    )
    bulk_selection = resolver.resolve(
        LLMTask.ARTICLE_ANALYSIS,
        pool.models,
        estimated_input_tokens=1000,
        requested_output_tokens=500,
        preferred_model_id=pool.models[0].endpoint_id,
    )
    assert bulk_selection.model.provider == "cloudflare"
    assert bulk_selection.model.model_id == "@cf/qwen/qwen3-30b-a3b-fp8"
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
