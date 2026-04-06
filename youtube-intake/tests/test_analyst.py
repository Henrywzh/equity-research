from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import requests

from youtube_intake.analyst import (
    FALLBACK_ANALYSIS_MODEL_ID,
    GROQ_CHAT_COMPLETIONS_URL,
    ModelLimits,
    PRIMARY_ANALYSIS_MODEL_ID,
    GroqAnalystClient,
    SlidingWindowRateLimiter,
    _normalize_video_analysis,
    analyze_run,
)
from youtube_intake.storage import write_json_document


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeTransport:
    def __init__(self, responses: list[FakeResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def __call__(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("No more fake responses were configured.")
        return self.responses.pop(0)


class PartialFailureClient:
    model_id = PRIMARY_ANALYSIS_MODEL_ID
    analysis_models_used = [PRIMARY_ANALYSIS_MODEL_ID]
    fallback_activated = False
    rate_limit_events: list[dict] = []
    run_notes = ["Second video failed after fallback retries."]

    def analyze_video(self, archive: dict) -> dict:
        if archive["video"]["video_id"] == "def":
            raise RuntimeError("fallback exhausted")
        return {
            "video_id": archive["video"]["video_id"],
            "channel_slug": archive["channel"]["slug"],
            "channel_name": archive["channel"]["channel_name"],
            "title": archive["video"]["title"],
            "webpage_url": archive["video"]["webpage_url"],
            "published_at": archive["published_at"],
            "source_kind": archive["source_kind"],
            "transcript_status": archive["transcript_status"],
            "source_basis": archive["analysis_input_basis"],
            "profile": archive.get("channel", {}).get("profile", "macroeconomics"),
            "synthesis_section": "🏦 Institutional Research",
            "executive_summary": "Recovered first video summary.",
            "tickers_mentioned": [],
            "profile_data": {},
            "key_timestamps": [],
            "topic_tags": [{"tag": "macro", "score": 80}],
            "confidence": 0.7,
            "analysis_model": PRIMARY_ANALYSIS_MODEL_ID,
            "analysis_attempts": 1,
            "analysis_mode": "single_pass",
        }

    def synthesize_run(self, *, run_result: dict, video_analyses: list[dict]) -> dict:
        return {
            "channels": {
                video_analyses[0]["channel_slug"]: {
                    "channel_name": video_analyses[0]["channel_name"],
                    "video_count": 1,
                    "summary": "Channel summary.",
                    "top_topics": ["macro"],
                }
            },
            "run_summary": {
                "overall_day_summary": "Partial run summary.",
                "cross_video_themes": [],
                "agreements": [],
                "disagreements": [],
                "top_claims_worth_watching": [],
                "crowded_trades": [],
                "contrarian_flags": [],
                "run_notes": [],
            },
            "summary_analysis_model": PRIMARY_ANALYSIS_MODEL_ID,
            "summary_analysis_attempts": 1,
        }


class AnalystTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data"
        self.run_result_path = self.root / "run-result.json"
        self.analysis_result_path = self.root / "analysis-result.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_rate_limiter_waits_for_projected_token_limit(self) -> None:
        clock = FakeClock(start=100.0)

        limiter = SlidingWindowRateLimiter(
            model_limits={
                PRIMARY_ANALYSIS_MODEL_ID: ModelLimits(
                    rpm=30,
                    tpm=100,
                    safe_input_tokens=80,
                    reserve_output_tokens=10,
                    chunk_input_tokens=60,
                ),
            },
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )
        limiter.register_request(PRIMARY_ANALYSIS_MODEL_ID, 80)
        limiter.wait_for_capacity(PRIMARY_ANALYSIS_MODEL_ID, 30)

        self.assertEqual(len(clock.sleeps), 1)
        self.assertGreater(clock.sleeps[0], 0)

    def test_retry_after_header_is_honored(self) -> None:
        clock = FakeClock()
        transport = FakeTransport(
            responses=[
                FakeResponse(status_code=429, headers={"retry-after": "3"}),
                _success_response({"executive_summary": "ok", "tickers_mentioned": [], "key_timestamps": [], "topic_tags": [], "confidence": 0.7}),
            ]
        )
        client = GroqAnalystClient(
            api_key="test-key",
            sleep_fn=clock.sleep,
            time_fn=clock.time,
            jitter_fn=lambda: 0.0,
            transport=transport,
        )

        result = client._chat_json(system_prompt="system", user_payload={"task": "video_analysis", "video": {"id": "abc"}})

        self.assertEqual(result.attempts, 2)
        self.assertEqual(clock.sleeps, [3.0])
        self.assertEqual(transport.calls[0]["url"], GROQ_CHAT_COMPLETIONS_URL)

    def test_primary_failures_activate_fallback_for_remaining_calls(self) -> None:
        clock = FakeClock()
        transport = FakeTransport(
            responses=[
                FakeResponse(status_code=503),
                FakeResponse(status_code=503),
                FakeResponse(status_code=503),
                _success_response({"executive_summary": "fallback ok", "tickers_mentioned": [], "key_timestamps": [], "topic_tags": [], "confidence": 0.7}),
                _success_response({"executive_summary": "second ok", "tickers_mentioned": [], "key_timestamps": [], "topic_tags": [], "confidence": 0.7}),
            ]
        )
        client = GroqAnalystClient(
            api_key="test-key",
            sleep_fn=clock.sleep,
            time_fn=clock.time,
            jitter_fn=lambda: 0.0,
            transport=transport,
        )

        first = client._chat_json(system_prompt="system", user_payload={"task": "video_analysis", "video": {"id": "abc"}})
        second = client._chat_json(system_prompt="system", user_payload={"task": "video_analysis", "video": {"id": "def"}})

        self.assertTrue(client.fallback_activated)
        self.assertEqual(first.model_id, FALLBACK_ANALYSIS_MODEL_ID)
        self.assertEqual(second.model_id, FALLBACK_ANALYSIS_MODEL_ID)
        self.assertIn(FALLBACK_ANALYSIS_MODEL_ID, client.analysis_models_used)
        self.assertTrue(any(FALLBACK_ANALYSIS_MODEL_ID == call["json"]["model"] for call in transport.calls))

    def test_oversized_transcript_uses_chunked_mode(self) -> None:
        clock = FakeClock()
        transport = FakeTransport(
            responses=[
                _success_response(
                    {
                        "executive_summary": "chunk one",
                        "tickers_mentioned": ["SPX"],
                        "key_timestamps": [{"timestamp": "00:00:10", "label": "Point", "snippet": "s1", "why_it_matters": "w1"}],
                        "topic_tags": [{"tag": "macro", "score": 70}],
                    }
                ),
                _success_response(
                    {
                        "executive_summary": "chunk two",
                        "tickers_mentioned": ["QQQ"],
                        "key_timestamps": [{"timestamp": "00:05:00", "label": "Point", "snippet": "s2", "why_it_matters": "w2"}],
                        "topic_tags": [{"tag": "earnings", "score": 72}],
                    }
                ),
                _success_response(
                    {
                        "executive_summary": "final",
                        "tickers_mentioned": ["SPX", "QQQ"],
                        "key_timestamps": [{"timestamp": "00:05:00", "label": "Final", "snippet": "sf", "why_it_matters": "wf"}],
                        "topic_tags": [{"tag": "macro", "score": 85}],
                        "confidence": 0.76,
                    }
                ),
            ]
        )
        client = GroqAnalystClient(
            api_key="test-key",
            sleep_fn=clock.sleep,
            time_fn=clock.time,
            jitter_fn=lambda: 0.0,
            transport=transport,
        )
        archive = _archive_payload("top3pct", "3% 財富覺醒", "abc", segment_count=120, segment_words=120)

        result = client.analyze_video(archive)

        self.assertEqual(result["analysis_mode"], "chunked")
        self.assertEqual(result["analysis_model"], PRIMARY_ANALYSIS_MODEL_ID)
        tasks = [call["json"]["messages"][1]["content"] for call in transport.calls]
        self.assertTrue(any("video_chunk_analysis" in content for content in tasks))
        self.assertTrue(any("video_chunk_consolidation" in content for content in tasks))

    def test_invalid_hh_mm_ss_timestamp_is_dropped_for_short_video(self) -> None:
        archive = _archive_payload("top3pct", "3% 財富覺醒", "abc", duration_seconds=1500)
        result = _normalize_video_analysis(
            {
                "executive_summary": "Summary.",
                "notable_claims": [],
                "notable_opinions": [],
                "key_timestamps": [
                    {
                        "timestamp": "20:00:00",
                        "label": "Bad timestamp",
                        "snippet": "no transcript match here",
                        "why_it_matters": "Should be rejected.",
                    }
                ],
                "topic_tags": [],
                "confidence": 0.7,
            },
            archive,
        )

        self.assertEqual(result["key_timestamps"], [])
        self.assertTrue(result["analysis_notes"])

    def test_mm_ss_timestamp_is_normalized_to_hh_mm_ss(self) -> None:
        archive = _archive_payload("top3pct", "3% 財富覺醒", "abc", duration_seconds=1500)
        result = _normalize_video_analysis(
            {
                "executive_summary": "Summary.",
                "notable_claims": [],
                "notable_opinions": [],
                "key_timestamps": [
                    {
                        "timestamp": "20:00",
                        "label": "Mid-video",
                        "snippet": "segment segment",
                        "why_it_matters": "Valid mm:ss input.",
                    }
                ],
                "topic_tags": [],
                "confidence": 0.7,
            },
            archive,
        )

        self.assertEqual(result["key_timestamps"][0]["timestamp"], "00:20:00")

    def test_start_seconds_is_preferred_over_conflicting_timestamp_string(self) -> None:
        archive = _archive_payload("top3pct", "3% 財富覺醒", "abc", duration_seconds=1500)
        result = _normalize_video_analysis(
            {
                "executive_summary": "Summary.",
                "notable_claims": [],
                "notable_opinions": [],
                "key_timestamps": [
                    {
                        "start_seconds": 75,
                        "timestamp": "20:00",
                        "label": "Authoritative start_seconds",
                        "snippet": "segment segment",
                        "why_it_matters": "Should use the numeric cue.",
                    }
                ],
                "topic_tags": [],
                "confidence": 0.7,
            },
            archive,
        )

        self.assertEqual(result["key_timestamps"][0]["timestamp"], "00:01:15")

    def test_snippet_only_timestamp_resolves_to_matching_transcript_segment(self) -> None:
        archive = _archive_payload(
            "top3pct",
            "3% 財富覺醒",
            "abc",
            transcript_segments=[
                {"start_seconds": 10.0, "duration_seconds": 5.0, "text": "opening overview"},
                {"start_seconds": 558.0, "duration_seconds": 5.0, "text": "market makers hedging behavior can create support for the market"},
                {"start_seconds": 1200.0, "duration_seconds": 5.0, "text": "closing thoughts"},
            ],
            duration_seconds=1500,
        )
        result = _normalize_video_analysis(
            {
                "executive_summary": "Summary.",
                "notable_claims": [],
                "notable_opinions": [],
                "key_timestamps": [
                    {
                        "label": "Support thesis",
                        "snippet": "hedging behavior can create support for the market",
                        "why_it_matters": "Should anchor to transcript.",
                    }
                ],
                "topic_tags": [],
                "confidence": 0.7,
            },
            archive,
        )

        self.assertEqual(result["key_timestamps"][0]["timestamp"], "00:09:18")

    def test_metadata_only_videos_return_no_key_timestamps(self) -> None:
        archive = _archive_payload("top3pct", "3% 財富覺醒", "abc", metadata_only=True)
        result = _normalize_video_analysis(
            {
                "executive_summary": "Summary.",
                "notable_claims": [],
                "notable_opinions": [],
                "key_timestamps": [{"timestamp": "00:01:00", "label": "Ignored", "snippet": "x", "why_it_matters": "y"}],
                "topic_tags": [],
                "confidence": 0.7,
            },
            archive,
        )

        self.assertEqual(result["key_timestamps"], [])

    def test_groq_stt_transcripts_add_analysis_note(self) -> None:
        archive = _archive_payload(
            "top3pct",
            "3% 財富覺醒",
            "abc",
            transcript_source="groq_whisper_large_v3_turbo",
        )
        result = _normalize_video_analysis(
            {
                "executive_summary": "Summary.",
                "notable_claims": [],
                "notable_opinions": [],
                "key_timestamps": [],
                "topic_tags": [],
                "confidence": 0.7,
            },
            archive,
        )

        self.assertTrue(any("Groq STT fallback" in note for note in result["analysis_notes"]))

    def test_metadata_only_duration_skip_adds_analysis_note(self) -> None:
        archive = _archive_payload(
            "top3pct",
            "3% 財富覺醒",
            "abc",
            metadata_only=True,
            error="Groq STT skipped due to duration limit (4200s > 3600s).",
        )
        result = _normalize_video_analysis(
            {
                "executive_summary": "Summary.",
                "notable_claims": [],
                "notable_opinions": [],
                "key_timestamps": [],
                "topic_tags": [],
                "confidence": 0.7,
            },
            archive,
        )

        self.assertIn("Groq STT skipped due to duration limit", result["analysis_notes"][0])

    def test_invalid_out_of_range_timestamp_entries_are_dropped(self) -> None:
        archive = _archive_payload("top3pct", "3% 財富覺醒", "abc", duration_seconds=300)
        result = _normalize_video_analysis(
            {
                "executive_summary": "Summary.",
                "notable_claims": [],
                "notable_opinions": [],
                "key_timestamps": [
                    {"timestamp": "00:06:00", "label": "Too long", "snippet": "not in transcript", "why_it_matters": "Bad"},
                    {"timestamp": "-1:00", "label": "Negative", "snippet": "still not in transcript", "why_it_matters": "Bad"},
                ],
                "topic_tags": [],
                "confidence": 0.7,
            },
            archive,
        )

        self.assertEqual(result["key_timestamps"], [])

    def test_valid_hh_mm_ss_timestamp_survives_unchanged(self) -> None:
        archive = _archive_payload("top3pct", "3% 財富覺醒", "abc", duration_seconds=1500)
        result = _normalize_video_analysis(
            {
                "executive_summary": "Summary.",
                "notable_claims": [],
                "notable_opinions": [],
                "key_timestamps": [
                    {
                        "timestamp": "00:09:18",
                        "label": "Valid timestamp",
                        "snippet": "segment segment",
                        "why_it_matters": "Should survive.",
                    }
                ],
                "topic_tags": [],
                "confidence": 0.7,
            },
            archive,
        )

        self.assertEqual(result["key_timestamps"][0]["timestamp"], "00:09:18")

    def test_partial_success_when_fallback_also_fails_preserves_completed_videos(self) -> None:
        archive_one = write_json_document(
            self.data_dir / "youtube" / "top3pct" / "videos" / "abc.json",
            _archive_payload("top3pct", "3% 財富覺醒", "abc"),
        )
        archive_two = write_json_document(
            self.data_dir / "youtube" / "meitou-news" / "videos" / "def.json",
            _archive_payload("meitou-news", "美投君", "def"),
        )
        write_json_document(
            self.run_result_path,
            {
                "status": "success",
                "run_started_at": "2026-04-03T01:00:00+00:00",
                "new_items": [
                    {"archive_path": str(archive_one), "channel_slug": "top3pct", "video_id": "abc"},
                    {"archive_path": str(archive_two), "channel_slug": "meitou-news", "video_id": "def"},
                ],
                "errors": [],
            },
        )

        result = analyze_run(
            result_path=self.run_result_path,
            analysis_result_path=self.analysis_result_path,
            data_dir=self.data_dir,
            client=PartialFailureClient(),
        )

        self.assertEqual(result["status"], "partial_success")
        self.assertEqual(len(result["videos"]), 1)
        self.assertIn("fallback exhausted", " ".join(result["errors"]))
        self.assertIn("Second video failed after fallback retries.", result["run_summary"]["run_notes"])

    def test_analyze_run_records_models_used_and_run_summary(self) -> None:
        clock = FakeClock()
        archive_one = write_json_document(
            self.data_dir / "youtube" / "top3pct" / "videos" / "abc.json",
            _archive_payload("top3pct", "3% 財富覺醒", "abc"),
        )
        write_json_document(
            self.run_result_path,
            {
                "status": "success",
                "run_started_at": "2026-04-03T01:00:00+00:00",
                "new_items": [{"archive_path": str(archive_one), "channel_slug": "top3pct", "video_id": "abc"}],
                "errors": [],
            },
        )
        transport = FakeTransport(
            responses=[
                _success_response(
                    {
                        "executive_summary": "Video summary.",
                        "tickers_mentioned": ["SPX"],
                        "macro_developments": ["Fed held rates steady."],
                        "key_timestamps": [{"timestamp": "00:01:23", "label": "Key turn", "snippet": "Snippet", "why_it_matters": "Why"}],
                        "topic_tags": [{"tag": "macro", "score": 90}],
                        "confidence": 0.8,
                    }
                ),
                _success_response(
                    {
                        "overall_day_summary": "Run-level summary.",
                        "channel_summaries": [{"channel_slug": "top3pct", "summary": "Channel summary.", "top_topics": ["macro"]}],
                        "cross_video_themes": ["macro"],
                        "agreements": ["Agreement"],
                        "disagreements": [],
                        "top_claims_worth_watching": ["Claim to watch"],
                        "crowded_trades": [],
                        "contrarian_flags": [],
                    }
                ),
            ]
        )
        client = GroqAnalystClient(
            api_key="test-key",
            sleep_fn=clock.sleep,
            time_fn=clock.time,
            jitter_fn=lambda: 0.0,
            transport=transport,
        )

        result = analyze_run(
            result_path=self.run_result_path,
            analysis_result_path=self.analysis_result_path,
            data_dir=self.data_dir,
            client=client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["analysis_models_used"], [PRIMARY_ANALYSIS_MODEL_ID])
        self.assertFalse(result["fallback_activated"])
        self.assertEqual(result["summary_analysis_model"], PRIMARY_ANALYSIS_MODEL_ID)
        self.assertEqual(result["videos"][0]["analysis_mode"], "single_pass")

    def test_analyze_run_noop_with_no_new_items(self) -> None:
        write_json_document(
            self.run_result_path,
            {
                "status": "success",
                "run_started_at": "2026-04-03T01:00:00+00:00",
                "new_items": [],
                "errors": [],
            },
        )

        result = analyze_run(
            result_path=self.run_result_path,
            analysis_result_path=self.analysis_result_path,
            data_dir=self.data_dir,
            client=PartialFailureClient(),
        )

        self.assertEqual(result["status"], "noop")
        self.assertEqual(result["videos"], [])


def _success_response(payload: dict) -> FakeResponse:
    return FakeResponse(
        status_code=200,
        payload={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                }
            ]
        },
        headers={
            "x-ratelimit-remaining-requests": "999",
            "x-ratelimit-remaining-tokens": "999999",
            "x-ratelimit-reset-requests": "1s",
            "x-ratelimit-reset-tokens": "1s",
        },
    )


def _archive_payload(
    channel_slug: str,
    channel_name: str,
    video_id: str,
    *,
    metadata_only: bool = False,
    segment_count: int = 2,
    segment_words: int = 2,
    duration_seconds: int = 600,
    transcript_segments: list[dict[str, object]] | None = None,
    transcript_source: str | None = None,
    error: str | None = None,
    profile: str = "macroeconomics",
) -> dict:
    normalized_segments: list[dict[str, object]] = []
    if transcript_segments is not None:
        normalized_segments = list(transcript_segments)
    elif not metadata_only:
        text = " ".join(["segment"] * segment_words)
        normalized_segments = [
            {"start_seconds": float(index * 10), "duration_seconds": 5.0, "text": text}
            for index in range(segment_count)
        ]

    return {
        "archived_at": "2026-04-03T01:00:00+00:00",
        "source_kind": "video",
        "analysis_input_basis": "metadata_only" if metadata_only else "transcript",
        "channel": {
            "slug": channel_slug,
            "handle": f"@{channel_slug}",
            "channel_id": f"{channel_slug}-id",
            "channel_name": channel_name,
            "profile": profile,
        },
        "video": {
            "video_id": video_id,
            "title": f"Title {video_id}",
            "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
            "duration_seconds": duration_seconds,
            "view_count": 1234,
            "live_status": "not_live",
            "was_live": False,
            "is_live": False,
            "media_type": "video",
            "source_tab": "videos",
            "thumbnail_url": None,
        },
        "published_at": "2026-04-03T00:00:00+00:00",
        "description": f"Description for {video_id}",
        "transcript_status": "unavailable" if metadata_only else "fetched",
        "transcript_language": None if metadata_only else "en",
        "transcript_text": None if metadata_only else "\n".join(str(segment["text"]) for segment in normalized_segments),
        "transcript_segments": normalized_segments,
        "transcript_source": transcript_source if transcript_source is not None else (None if metadata_only else "youtube_transcript_api"),
        "error": error,
    }


if __name__ == "__main__":
    unittest.main()


class FakeClock:
    def __init__(self, *, start: float = 0.0) -> None:
        self.current = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.current += seconds
