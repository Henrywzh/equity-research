from __future__ import annotations

import threading
import time

import daily_macro.analysis as analysis_module
from daily_macro.llm_client import AnalysisRuntime, ModelResolver, RateLimitGovernor
from daily_macro.types import ModelConfig


def _runtime() -> AnalysisRuntime:
    model = ModelConfig("openai/gpt-oss-20b", provider="openai")
    return AnalysisRuntime(
        governor=RateLimitGovernor(),
        model_chain=[model],
        resolver=ModelResolver(active_model_ids={model.model_id}),
    )


def test_parallel_article_batches_preserve_order_and_merge_diagnostics(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setenv("DAILY_MACRO_LLM_PARALLELISM", "3")
    seen_threads: set[str] = set()
    seen_lock = threading.Lock()

    def fake_process(worker, category_name, batch, batch_label):
        with seen_lock:
            seen_threads.add(threading.current_thread().name)
        # Make completion order differ from input order.
        time.sleep(0.03 * (4 - int(batch_label)))
        article = dict(batch[0])
        return [{"source_article_id": article["source_article_id"]}], [], 1

    monkeypatch.setattr(analysis_module, "_process_batch_recursive", fake_process)
    batches = [[{"source_article_id": str(index)}] for index in range(1, 4)]

    results, errors, count = analysis_module._run_article_batches(runtime, "國際財經", batches)

    assert [item["source_article_id"] for item in results] == ["1", "2", "3"]
    assert errors == []
    assert count == 3
    assert len(seen_threads) >= 2
    assert runtime.diagnostics.parallel_batch_count == 3
    assert runtime.diagnostics.parallel_worker_count == 3


def test_governor_select_and_reserve_uses_separate_keys_for_parallel_burst() -> None:
    governor = RateLimitGovernor(
        model_limits={"model": {"rpm": 1, "tpm": 2000}},
        sleep_fn=lambda _seconds: None,
    )

    selected = [
        governor.select_and_reserve("model", 0, 2, estimated_input_tokens=10)[0]
        for _ in range(2)
    ]

    assert selected == [0, 1]
