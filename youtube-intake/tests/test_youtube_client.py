from __future__ import annotations

import unittest
from pathlib import Path

from youtube_intake.models import TranscriptPayload, VideoMetadata
from youtube_intake.youtube_client import (
    FALLBACK_STT_MODEL_ID,
    MAX_STT_AUDIO_BYTES,
    PRIMARY_STT_MODEL_ID,
    YoutubeClient,
    _select_audio_format_id,
)


class FakeTranscriptApi:
    def list(self, video_id: str):
        raise RuntimeError(f"no transcript for {video_id}")


class StubYoutubeClient(YoutubeClient):
    def __init__(
        self,
        *,
        caption_fallback: TranscriptPayload,
        stt_payloads: dict[str, dict] | None = None,
        stt_errors: dict[str, Exception] | None = None,
        groq_api_key: str = "test-key",
    ) -> None:
        super().__init__(groq_api_key=groq_api_key, sleep_fn=lambda _: None, jitter_fn=lambda: 0.0)
        self.caption_fallback = caption_fallback
        self.stt_payloads = stt_payloads or {}
        self.stt_errors = stt_errors or {}
        self.transcript_api = FakeTranscriptApi()
        self.download_calls: list[str] = []
        self.model_calls: list[str] = []

    def _fetch_caption_fallback(self, video, *, transcript_api_error=None):
        return self.caption_fallback

    def _download_audio(self, video: VideoMetadata, *, temp_dir: Path) -> Path:
        self.download_calls.append(video.video_id)
        path = temp_dir / f"{video.video_id}.mp3"
        path.write_bytes(b"fake audio")
        return path

    def _request_groq_transcription(self, audio_path: Path, *, model_id: str):
        self.model_calls.append(model_id)
        if model_id in self.stt_errors:
            raise self.stt_errors[model_id]
        if model_id not in self.stt_payloads:
            raise AssertionError(f"No STT payload configured for {model_id}")
        return self.stt_payloads[model_id]


class YoutubeClientTests(unittest.TestCase):
    def test_audio_format_selection_prefers_best_quality_within_size_limit(self) -> None:
        format_id = _select_audio_format_id(
            [
                {"format_id": "140", "vcodec": "none", "acodec": "mp4a.40.2", "ext": "m4a", "abr": 129.0, "filesize": 27_245_881},
                {"format_id": "139", "vcodec": "none", "acodec": "mp4a.40.5", "ext": "m4a", "abr": 48.0, "filesize": 10_266_613},
                {"format_id": "249", "vcodec": "none", "acodec": "opus", "ext": "webm", "abr": 48.0, "filesize": 10_158_965},
            ],
            max_bytes=MAX_STT_AUDIO_BYTES,
        )

        self.assertEqual(format_id, "139")

    def test_audio_format_selection_falls_back_to_smallest_audio_when_all_known_sizes_exceed_limit(self) -> None:
        format_id = _select_audio_format_id(
            [
                {"format_id": "251", "vcodec": "none", "acodec": "opus", "ext": "webm", "abr": 124.0, "filesize": 26_270_728},
                {"format_id": "140", "vcodec": "none", "acodec": "mp4a.40.2", "ext": "m4a", "abr": 129.0, "filesize": 27_245_881},
            ],
            max_bytes=10_000_000,
        )

        self.assertEqual(format_id, "251")

    def test_no_transcript_video_uses_turbo_stt_when_available(self) -> None:
        client = StubYoutubeClient(
            caption_fallback=_unavailable_transcript("no captions"),
            stt_payloads={
                PRIMARY_STT_MODEL_ID: _stt_payload("Turbo transcript", start=12.0),
            },
        )

        transcript = client.fetch_transcript(_video("abc", duration_seconds=1800))

        self.assertEqual(transcript.status, "fetched")
        self.assertEqual(transcript.source, "groq_whisper_large_v3_turbo")
        self.assertEqual(transcript.segments[0].start_seconds, 12.0)
        self.assertEqual(client.model_calls, [PRIMARY_STT_MODEL_ID])

    def test_stt_falls_back_to_second_model_after_turbo_failure(self) -> None:
        client = StubYoutubeClient(
            caption_fallback=_unavailable_transcript("no captions"),
            stt_errors={PRIMARY_STT_MODEL_ID: RuntimeError("turbo failed")},
            stt_payloads={
                FALLBACK_STT_MODEL_ID: _stt_payload("Fallback transcript", start=22.0),
            },
        )

        transcript = client.fetch_transcript(_video("abc", duration_seconds=1800))

        self.assertEqual(transcript.status, "fetched")
        self.assertEqual(transcript.source, "groq_whisper_large_v3")
        self.assertEqual(client.model_calls, [PRIMARY_STT_MODEL_ID, FALLBACK_STT_MODEL_ID])

    def test_both_stt_models_failing_returns_metadata_only_transcript_payload(self) -> None:
        client = StubYoutubeClient(
            caption_fallback=_unavailable_transcript("no captions"),
            stt_errors={
                PRIMARY_STT_MODEL_ID: RuntimeError("turbo failed"),
                FALLBACK_STT_MODEL_ID: RuntimeError("fallback failed"),
            },
        )

        transcript = client.fetch_transcript(_video("abc", duration_seconds=1800))

        self.assertEqual(transcript.status, "unavailable")
        self.assertIn(PRIMARY_STT_MODEL_ID, transcript.error or "")
        self.assertIn(FALLBACK_STT_MODEL_ID, transcript.error or "")

    def test_long_videos_skip_stt_and_keep_metadata_only(self) -> None:
        client = StubYoutubeClient(
            caption_fallback=_unavailable_transcript("no captions"),
        )

        transcript = client.fetch_transcript(_video("abc", duration_seconds=4200))

        self.assertEqual(transcript.status, "unavailable")
        self.assertEqual(client.download_calls, [])
        self.assertIn("duration limit", transcript.error or "")


def _video(video_id: str, *, duration_seconds: int) -> VideoMetadata:
    return VideoMetadata(
        video_id=video_id,
        title=f"Title {video_id}",
        channel_id="chan-1",
        channel_name="Channel",
        channel_handle="@channel",
        webpage_url=f"https://www.youtube.com/watch?v={video_id}",
        description="description",
        published_at="2026-04-03T00:00:00+00:00",
        published_timestamp=1,
        duration_seconds=duration_seconds,
        view_count=100,
        live_status="not_live",
        was_live=False,
        is_live=False,
        media_type="video",
        thumbnail_url=None,
        source_tab="videos",
        subtitles={},
        automatic_captions={},
    )


def _unavailable_transcript(error: str) -> TranscriptPayload:
    return TranscriptPayload(
        status="unavailable",
        language=None,
        text=None,
        source=None,
        segments=[],
        error=error,
    )


def _stt_payload(text: str, *, start: float) -> dict:
    return {
        "language": "zh",
        "text": text,
        "segments": [
            {
                "id": 0,
                "start": start,
                "end": start + 5.0,
                "text": text,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
