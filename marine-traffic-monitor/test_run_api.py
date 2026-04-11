import asyncio
import csv
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyst
import run_api


def test_no_image_abstain_payload_matches_evidence_schema():
    evidence = analyst._run_evidence_analyst(
        image_path="",
        reported_count=1,
        csv_path=str(HERE / "data" / "hormuz_traffic_log.csv"),
        model_key="llama_4_scout",
        current_state="NORMAL",
        news="",
    )

    is_valid, reason = analyst._validate_evidence(evidence)
    assert is_valid, reason
    assert evidence["abstain"] is True
    assert evidence["hypotheses"] == []


def test_run_api_uses_context_models_without_image(monkeypatch, tmp_path):
    csv_path = tmp_path / "hormuz_traffic_log.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Timestamp", "Detected_Ships", "Status_Note"])
        writer.writerow(["2026-04-11_00-00-00", 0, "Active Blockade / Clear Zone"])

    captured: dict[str, object] = {}

    async def fake_snapshot(api_key, duration=60):
        return [{"name": "Test Vessel", "type": "Cargo", "speed": 12.3}]

    def fake_consensus_check(**kwargs):
        captured.update(kwargs)
        return {"analyst_briefing": "ok"}

    monkeypatch.setattr(run_api, "CSV_FILENAME", str(csv_path))
    monkeypatch.setattr(run_api, "get_ais_snapshot", fake_snapshot)
    monkeypatch.setattr(run_api, "run_consensus_check", fake_consensus_check)
    monkeypatch.setenv("AIS_STREAM_API_KEY", "test-key")

    asyncio.run(run_api.main())

    assert captured["image_path"] == ""
    assert captured["model_a"] == "llama_3_3_70b"
    assert captured["model_b"] == "llama_3_1_8b"
