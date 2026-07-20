"""Tests for dynamic model pool construction and task-aware routing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from daily_macro import model_registry as mr
from daily_macro.llm_client import AnalysisRuntime, LLMTask, ModelResolver, RateLimitGovernor
from daily_macro.model_catalog import get_capability, heuristic_scores, infer_kind
from daily_macro.model_registry import FLOOR_MODEL_ID, build_model_pool
from daily_macro.types import ModelConfig

_LIVE = [
    {"id": "llama-3.1-8b-instant", "context_window": 131072, "max_completion_tokens": 8192},
    {"id": "qwen/qwen3.6-27b", "context_window": 131072, "max_completion_tokens": 8192},
    {"id": "openai/gpt-oss-120b", "context_window": 131072, "max_completion_tokens": 8192},
    {"id": "openai/gpt-oss-20b", "context_window": 131072, "max_completion_tokens": 8192},
    {"id": "whisper-large-v3", "context_window": None, "max_completion_tokens": None},
    {"id": "meta-llama/llama-prompt-guard-2-86m", "context_window": 512, "max_completion_tokens": 512},
    {"id": "canopylabs/orpheus-v1-english", "context_window": None, "max_completion_tokens": None},
    {"id": "some-brand-new-90b", "context_window": 200000, "max_completion_tokens": 16384},
]


@pytest.fixture
def tmp_data_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _patch_live(monkeypatch, value):
    monkeypatch.setattr(mr, "load_groq_models", lambda *a, **k: value)


# --------------------------------------------------------------------------- #
# Capability classification / heuristics
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "model_id,kind",
    [
        ("whisper-large-v3", "asr"),
        ("canopylabs/orpheus-v1-english", "tts"),
        ("meta-llama/llama-prompt-guard-2-86m", "guard"),
        ("openai/gpt-oss-safeguard-20b", "guard"),
        ("groq/compound", "agentic"),
        ("allam-2-7b", "specialized"),
        ("openai/gpt-oss-120b", "chat"),
        ("some-brand-new-90b", "chat"),
    ],
)
def test_infer_kind(model_id, kind):
    assert infer_kind(model_id) == kind


def test_heuristic_scores_scale_with_size():
    assert heuristic_scores("foo-120b")["article_analysis"] > heuristic_scores("foo-8b")["article_analysis"]
    assert heuristic_scores("no-size-here")["routing"] == 0.5


# --------------------------------------------------------------------------- #
# Pool construction
# --------------------------------------------------------------------------- #

def test_pool_from_catalog_when_refresh_disabled(monkeypatch, tmp_data_dir):
    _patch_live(monkeypatch, None)
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    ids = [m.model_id for m in pool.models]
    assert ids[0] == FLOOR_MODEL_ID  # floor anchors the pool
    assert FLOOR_MODEL_ID in pool.active_ids
    # only chat models survive
    assert all(pool.capabilities[m].kind == "chat" for m in pool.active_ids)


def test_pool_filters_noncat_models_and_keeps_new(monkeypatch, tmp_data_dir):
    _patch_live(monkeypatch, _LIVE)
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    # audio / guard / tts excluded
    for junk in ("whisper-large-v3", "meta-llama/llama-prompt-guard-2-86m", "canopylabs/orpheus-v1-english"):
        assert junk not in pool.active_ids
    # Groq is restricted to the approved transition models, even when the
    # live catalog exposes unrelated or newly released chat models.
    assert "some-brand-new-90b" not in pool.active_ids
    assert "llama-3.1-8b-instant" not in pool.active_ids
    assert set(model.model_id for model in pool.models) == {
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }
    assert [model.model_id for model in pool.models] == [
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]
    assert pool.models[0].model_id == FLOOR_MODEL_ID


def test_removed_model_drops_from_pool(monkeypatch, tmp_data_dir):
    # Live list without gpt-oss-120b -> it must not appear in the pool.
    live = [m for m in _LIVE if m["id"] != "openai/gpt-oss-120b"]
    _patch_live(monkeypatch, live)
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    assert "openai/gpt-oss-120b" not in pool.active_ids


def test_removed_floor_model_is_not_reintroduced(monkeypatch, tmp_data_dir):
    # Live list omits the primary model entirely. A removed model must not be
    # synthesized back into the pool by the floor logic.
    live = [{"id": "openai/gpt-oss-120b", "context_window": 131072, "max_completion_tokens": 8192}]
    _patch_live(monkeypatch, live)
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    assert FLOOR_MODEL_ID not in pool.active_ids
    assert [model.model_id for model in pool.models] == ["openai/gpt-oss-120b"]


def test_cache_used_on_fetch_failure(monkeypatch, tmp_data_dir):
    # First build with a live list writes the cache.
    _patch_live(monkeypatch, _LIVE)
    build_model_pool(["k"], data_dir=tmp_data_dir)
    assert (Path(tmp_data_dir) / "model_catalog_cache.json").exists()
    # Next build with fetch failing must reuse the cache, but still apply the
    # approved Groq allowlist.
    _patch_live(monkeypatch, None)
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    assert set(model.model_id for model in pool.models) == {
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }


# --------------------------------------------------------------------------- #
# Task-aware routing over the pool
# --------------------------------------------------------------------------- #

def test_resolver_routes_synthesis_to_premium_and_routing_to_floor(monkeypatch, tmp_data_dir):
    _patch_live(monkeypatch, None)  # catalog pool (approved transition set)
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    resolver = ModelResolver(
        active_model_ids=pool.active_ids,
        capabilities=pool.capabilities,
        model_policy="production_only",
    )

    def pick(task):
        return resolver.resolve(
            task,
            pool.models,
            estimated_input_tokens=500,
            requested_output_tokens=500,
            preferred_model_id=FLOOR_MODEL_ID,  # mirrors current_model = chain[0]
        ).model.model_id

    # Production-only policy excludes preview Qwen, so high-value tasks use
    # stable GPT OSS 120B and bulk tasks use the lighter production fallback.
    assert pick(LLMTask.CATEGORY_SYNTHESIS) == "openai/gpt-oss-120b"
    assert pick(LLMTask.TOP_ALERTS) == "openai/gpt-oss-120b"
    assert pick(LLMTask.ROUTING) == "openai/gpt-oss-20b"
    assert pick(LLMTask.ARTICLE_ANALYSIS) == "openai/gpt-oss-120b"
    # Preview models are never selected under production_only, and retired
    # Groq models are not present in this pool at all.
    assert pick(LLMTask.CATEGORY_SYNTHESIS) != "meta-llama/llama-4-scout-17b-16e-instruct"
    assert not {model.model_id for model in pool.models} & {
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
    }


def test_allow_preview_policy_selects_qwen_with_explicit_preference(monkeypatch, tmp_data_dir):
    _patch_live(monkeypatch, None)
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    resolver = ModelResolver(
        active_model_ids=pool.active_ids,
        capabilities=pool.capabilities,
        model_policy="allow_preview",
    )
    selection = resolver.resolve(
        LLMTask.TOP_ALERTS,
        pool.models,
        estimated_input_tokens=500,
        requested_output_tokens=500,
        preferred_model_id="qwen/qwen3.6-27b",
    )
    assert selection.model.model_id == "qwen/qwen3.6-27b"


def test_default_policy_uses_qwen_but_strict_policy_excludes_it(monkeypatch, tmp_data_dir):
    _patch_live(monkeypatch, None)
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    default_resolver = ModelResolver(active_model_ids=pool.active_ids, capabilities=pool.capabilities)
    strict_resolver = ModelResolver(
        active_model_ids=pool.active_ids,
        capabilities=pool.capabilities,
        model_policy="production_only",
    )
    kwargs = {
        "estimated_input_tokens": 500,
        "requested_output_tokens": 500,
        "preferred_model_id": "qwen/qwen3.6-27b",
    }
    assert default_resolver.resolve(LLMTask.ARTICLE_ANALYSIS, pool.models, **kwargs).model.model_id == "qwen/qwen3.6-27b"
    strict_selection = strict_resolver.resolve(LLMTask.ARTICLE_ANALYSIS, pool.models, **kwargs)
    assert strict_selection.model.model_id != "qwen/qwen3.6-27b"
    assert any(item["reason"] == "preview_model_disallowed" for item in strict_selection.rejections)


def test_bulk_task_falls_back_to_premium_when_nothing_else():
    # If a bulk task's only candidates are premium models, the reservation must
    # not strand it — it falls back to a premium model rather than failing.
    from daily_macro.types import ModelConfig as MC

    resolver = ModelResolver(
        active_model_ids={"openai/gpt-oss-120b", "openai/gpt-oss-20b"},
        model_policy="production_only",
    )
    chosen = resolver.resolve(
        LLMTask.ARTICLE_ANALYSIS,
        [MC("openai/gpt-oss-120b"), MC("openai/gpt-oss-20b")],
        estimated_input_tokens=300,
        requested_output_tokens=300,
    ).model.model_id
    assert chosen in {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}


# --------------------------------------------------------------------------- #
# Daily-budget gate + reservation
# --------------------------------------------------------------------------- #

def test_resolver_skips_budget_exhausted_model(monkeypatch, tmp_data_dir):
    _patch_live(monkeypatch, None)  # full catalog pool
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    resolver = ModelResolver(
        active_model_ids=pool.active_ids,
        capabilities=pool.capabilities,
        model_policy="production_only",
    )
    # gpt-oss-120b is the synthesis winner, but its remaining daily budget can't
    # cover this call -> it must be skipped, not selected.
    selection = resolver.resolve(
        LLMTask.CATEGORY_SYNTHESIS,
        pool.models,
        estimated_input_tokens=500,
        requested_output_tokens=500,
        preferred_model_id=FLOOR_MODEL_ID,
        budget_remaining={"openai/gpt-oss-120b": 100},  # < 1000 needed
    )
    assert selection.model.model_id != "openai/gpt-oss-120b"
    assert any(r["reason"] == "daily_budget_exhausted" for r in selection.rejections)


def test_resolver_compares_metered_budget_in_provider_units():
    cloudflare = ModelConfig(
        "@cf/zai-org/glm-4.7-flash",
        provider="cloudflare",
        account_id="cloudflare_1",
        quota_scope="cloudflare:account-123",
    )
    groq = ModelConfig("llama-3.1-8b-instant")
    resolver = ModelResolver(model_policy="production_only")
    selection = resolver.resolve(
        LLMTask.ARTICLE_ANALYSIS,
        [cloudflare, groq],
        estimated_input_tokens=10_000,
        requested_output_tokens=1_000,
        budget_remaining={cloudflare.endpoint_id: 80},
        budget_required={cloudflare.endpoint_id: 92},
    )
    assert selection.model == groq
    assert any(item["reason"] == "daily_budget_exhausted" for item in selection.rejections)


def test_resolver_reservation_relaxes_with_ample_budget():
    caps = {
        "openai/gpt-oss-120b": get_capability("openai/gpt-oss-120b"),
        "llama-3.1-8b-instant": get_capability("llama-3.1-8b-instant"),
    }
    resolver = ModelResolver(
        active_model_ids=set(caps),
        capabilities=caps,
        model_policy="production_only",
    )
    chain = [ModelConfig("llama-3.1-8b-instant"), ModelConfig("openai/gpt-oss-120b")]

    def pick(premium_remaining):
        return resolver.resolve(
            LLMTask.ARTICLE_ANALYSIS,
            chain,
            estimated_input_tokens=300,
            requested_output_tokens=300,
            budget_remaining={"openai/gpt-oss-120b": premium_remaining, "llama-3.1-8b-instant": 500000},
        ).model.model_id

    # Ample premium daily budget (above the reserve floor): bulk may use premium.
    assert pick(200000) == "openai/gpt-oss-120b"
    # Premium budget below the reserve floor: reservation defers bulk to the floor.
    assert pick(1000) == "llama-3.1-8b-instant"


# --------------------------------------------------------------------------- #
# Runtime decommission eviction
# --------------------------------------------------------------------------- #

def _runtime(model_ids):
    return AnalysisRuntime(
        governor=RateLimitGovernor(),
        model_chain=[ModelConfig(m) for m in model_ids],
        groq_api_keys=["k"],
        resolver=ModelResolver(active_model_ids=set(model_ids)),
    )


def test_evict_model_removes_and_reindexes():
    rt = _runtime(["a", "b", "c"])
    rt.current_model_index = 2  # pointing at "c"
    assert rt.evict_model("a") is True
    assert [m.model_id for m in rt.model_chain] == ["b", "c"]
    assert rt.current_model_index == 1  # still "c", reindexed
    assert "a" not in rt.resolver.active_model_ids


def test_evict_model_keeps_last_model():
    rt = _runtime(["only"])
    assert rt.evict_model("only") is False
    assert [m.model_id for m in rt.model_chain] == ["only"]


def test_evict_unknown_model_is_noop():
    rt = _runtime(["a", "b"])
    assert rt.evict_model("missing") is False
    assert len(rt.model_chain) == 2
