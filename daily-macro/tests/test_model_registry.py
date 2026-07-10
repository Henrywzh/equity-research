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
    # brand-new unknown model included, with live context merged in
    assert "some-brand-new-90b" in pool.active_ids
    assert pool.capabilities["some-brand-new-90b"].context_window == 200000
    assert pool.models[0].model_id == FLOOR_MODEL_ID


def test_removed_model_drops_from_pool(monkeypatch, tmp_data_dir):
    # Live list without gpt-oss-120b -> it must not appear in the pool.
    live = [m for m in _LIVE if m["id"] != "openai/gpt-oss-120b"]
    _patch_live(monkeypatch, live)
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    assert "openai/gpt-oss-120b" not in pool.active_ids


def test_floor_model_always_present(monkeypatch, tmp_data_dir):
    # Live list omits the floor model entirely.
    live = [{"id": "openai/gpt-oss-120b", "context_window": 131072, "max_completion_tokens": 8192}]
    _patch_live(monkeypatch, live)
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    assert FLOOR_MODEL_ID in pool.active_ids
    assert pool.models[0].model_id == FLOOR_MODEL_ID


def test_cache_used_on_fetch_failure(monkeypatch, tmp_data_dir):
    # First build with a live list writes the cache.
    _patch_live(monkeypatch, _LIVE)
    build_model_pool(["k"], data_dir=tmp_data_dir)
    assert (Path(tmp_data_dir) / "model_catalog_cache.json").exists()
    # Next build with fetch failing must reuse the cached list (incl. new model).
    _patch_live(monkeypatch, None)
    pool = build_model_pool(["k"], data_dir=tmp_data_dir)
    assert "some-brand-new-90b" in pool.active_ids


# --------------------------------------------------------------------------- #
# Task-aware routing over the pool
# --------------------------------------------------------------------------- #

def test_resolver_routes_synthesis_to_premium_and_routing_to_floor(monkeypatch, tmp_data_dir):
    _patch_live(monkeypatch, None)  # catalog pool (full set)
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

    # High-value, low-volume tasks go to the premium model (highest task_score).
    assert pick(LLMTask.CATEGORY_SYNTHESIS) == "qwen/qwen3.6-27b"
    assert pick(LLMTask.TOP_ALERTS) == "qwen/qwen3.6-27b"
    # High-volume cheap tasks stay on the floor model (conserves premium budget).
    assert pick(LLMTask.ROUTING) == FLOOR_MODEL_ID
    assert pick(LLMTask.ARTICLE_ANALYSIS) == FLOOR_MODEL_ID
    # Preview models are never selected under production_only.
    assert pick(LLMTask.CATEGORY_SYNTHESIS) != "meta-llama/llama-4-scout-17b-16e-instruct"


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
