"""Tests for the daily token/request budget ledger and its persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from daily_macro import budget as budget_mod
from daily_macro.budget import UNLIMITED, DailyBudgetLedger, _ledger_path


@pytest.fixture
def tmp_data_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_record_accumulates_per_model_and_key(tmp_data_dir):
    ledger = DailyBudgetLedger.load(tmp_data_dir)
    ledger.record("m", 0, 100)
    ledger.record("m", 0, 50, requests=2)
    ledger.record("m", 1, 10)
    assert ledger.used_tokens("m", 0) == 150
    assert ledger.used_tokens("m", 1) == 10


def test_remaining_tokens_is_declared_minus_used(tmp_data_dir):
    ledger = DailyBudgetLedger.load(tmp_data_dir)
    ledger.record("m", 0, 100)
    assert ledger.remaining_tokens("m", 0, 200) == 100
    # Floors at zero, never negative.
    ledger.record("m", 0, 500)
    assert ledger.remaining_tokens("m", 0, 200) == 0
    # Unknown declared limit -> unlimited (no gating).
    assert ledger.remaining_tokens("m", 0, None) == UNLIMITED
    assert ledger.remaining_tokens("m", 0, 0) == UNLIMITED


def test_remaining_requests(tmp_data_dir):
    ledger = DailyBudgetLedger.load(tmp_data_dir)
    ledger.record("m", 0, 10, requests=3)
    assert ledger.remaining_requests("m", 0, 5) == 2
    assert ledger.remaining_requests("m", 0, None) == UNLIMITED


def test_persists_and_reloads_across_runs(tmp_data_dir):
    # Simulate a morning run that burns budget and flushes.
    first = DailyBudgetLedger.load(tmp_data_dir)
    first.record("openai/gpt-oss-120b", 0, 41230)
    first.flush()
    assert _ledger_path(tmp_data_dir).exists()
    # A second run the same day must see the already-spent budget.
    second = DailyBudgetLedger.load(tmp_data_dir)
    assert second.used_tokens("openai/gpt-oss-120b", 0) == 41230
    assert second.remaining_tokens("openai/gpt-oss-120b", 0, 200000) == 200000 - 41230


def test_stale_date_is_pruned_on_load(tmp_data_dir, monkeypatch):
    path = _ledger_path(tmp_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"1999-01-01": {"m": {"0": {"tokens": 999, "requests": 9}}}}), encoding="utf-8")
    ledger = DailyBudgetLedger.load(tmp_data_dir)
    # Yesterday's usage must not count toward today.
    assert ledger.used_tokens("m", 0) == 0
    # After a flush, the stale date is gone and only today remains.
    ledger.record("m", 0, 5)
    ledger.flush()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "1999-01-01" not in on_disk
    assert list(on_disk.keys()) == [ledger.date]


def test_corrupt_and_missing_file_yield_empty_ledger(tmp_data_dir):
    # Missing file.
    ledger = DailyBudgetLedger.load(tmp_data_dir)
    assert ledger.used_tokens("m", 0) == 0
    # Corrupt file.
    path = _ledger_path(tmp_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    ledger = DailyBudgetLedger.load(tmp_data_dir)
    assert ledger.used_tokens("m", 0) == 0


def test_flush_without_path_is_noop():
    # The default in-memory ledger (used in tests / legacy constructions) must
    # not raise when flushed.
    DailyBudgetLedger().flush()


def test_shared_compute_reservation_is_atomic_across_models(tmp_data_dir):
    ledger = DailyBudgetLedger.load(tmp_data_dir)
    scope = "cloudflare:account-123"
    assert ledger.try_reserve_compute_units(scope, 7000, 10000)
    assert not ledger.try_reserve_compute_units(scope, 4000, 10000)
    assert ledger.remaining_compute_units(scope, 10000) == 3000

    # Settle the pessimistic reservation to actual usage, then another model on
    # the same Cloudflare account can consume the released capacity.
    ledger.settle_compute_units(scope, reserved=7000, actual=2500)
    assert ledger.used_compute_units(scope) == 2500
    assert ledger.try_reserve_compute_units(scope, 7000, 10000)


def test_shared_compute_usage_persists_across_runs(tmp_data_dir):
    scope = "cloudflare:account-123"
    first = DailyBudgetLedger.load(tmp_data_dir)
    assert first.try_reserve_compute_units(scope, 1234, 10000)
    first.flush()
    second = DailyBudgetLedger.load(tmp_data_dir)
    assert second.used_compute_units(scope) == 1234
