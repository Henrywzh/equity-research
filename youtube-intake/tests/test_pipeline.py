from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from youtube_intake.models import ChannelState, FlatPlaylistEntry, TranscriptPayload, VideoMetadata
from youtube_intake.pipeline import run_sync
from youtube_intake.storage import save_state


class FakeYoutubeClient:
    def __init__(
        self,
        *,
        candidates_by_slug: dict[str, list[FlatPlaylistEntry]],
        metadata_by_id: dict[str, VideoMetadata],
        transcript_by_id: dict[str, TranscriptPayload] | None = None,
        failing_channels: set[str] | None = None,
    ) -> None:
        self.candidates_by_slug = candidates_by_slug
        self.metadata_by_id = metadata_by_id
        self.transcript_by_id = transcript_by_id or {}
        self.failing_channels = failing_channels or set()
        self.transcript_calls: list[str] = []

    def list_recent_candidates(self, channel, *, stop_id=None):
        if channel.slug in self.failing_channels:
            raise RuntimeError(f"boom: {channel.slug}")
        candidates = self.candidates_by_slug.get(channel.slug, [])
        if stop_id is None:
            return candidates
        trimmed = []
        for candidate in candidates:
            if candidate.video_id == stop_id:
                break
            trimmed.append(candidate)
        return trimmed

    def fetch_video_metadata(self, candidate: FlatPlaylistEntry) -> VideoMetadata:
        return self.metadata_by_id[candidate.video_id]

    def fetch_transcript(self, video: VideoMetadata) -> TranscriptPayload:
        self.transcript_calls.append(video.video_id)
        return self.transcript_by_id.get(
            video.video_id,
            TranscriptPayload(status="fetched", language="en", text=f"Transcript for {video.video_id}", source="fake"),
        )


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config_path = self.root / "config.json"
        self.state_path = self.root / "state.json"
        self.data_dir = self.root / "data"
        self.config_path.write_text(
            json.dumps(
                [
                    {
                        "slug": "top3pct",
                        "handle": "@top3pct",
                        "channel_id": "chan-1",
                        "videos_url": "https://example.com/top3pct/videos",
                        "streams_url": "https://example.com/top3pct/streams",
                        "enabled": True,
                    },
                    {
                        "slug": "meitou-news",
                        "handle": "@MeiTouNews",
                        "channel_id": "chan-2",
                        "videos_url": "https://example.com/meitou-news/videos",
                        "streams_url": "https://example.com/meitou-news/streams",
                        "enabled": True,
                    },
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        self.state_path.write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_bootstrap_archives_only_newest_item_per_channel(self) -> None:
        client = FakeYoutubeClient(
            candidates_by_slug={
                "top3pct": [_candidate("old-video", source_tab="videos"), _candidate("new-video", source_tab="streams")],
                "meitou-news": [_candidate("news-new"), _candidate("news-old")],
            },
            metadata_by_id={
                "old-video": _video("old-video", published_timestamp=100, source_tab="videos"),
                "new-video": _video("new-video", published_timestamp=200, source_tab="streams", was_live=True),
                "news-old": _video("news-old", published_timestamp=300, source_tab="videos"),
                "news-new": _video("news-new", published_timestamp=400, source_tab="videos"),
            },
        )

        result = run_sync(
            config_path=self.config_path,
            state_path=self.state_path,
            data_dir=self.data_dir,
            client=client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["archived_count"], 2)
        self.assertEqual(result["bootstrap_count"], 2)
        self.assertTrue((self.data_dir / "youtube" / "top3pct" / "videos" / "new-video.json").exists())
        self.assertFalse((self.data_dir / "youtube" / "top3pct" / "videos" / "old-video.json").exists())
        self.assertTrue((self.data_dir / "youtube" / "meitou-news" / "videos" / "news-new.json").exists())

        state_payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state_payload["top3pct"]["last_processed_video_id"], "new-video")
        self.assertTrue(state_payload["top3pct"]["bootstrap_completed"])

    def test_backfill_processes_items_oldest_to_newest(self) -> None:
        save_state(
            self.state_path,
            {
                "top3pct": ChannelState(
                    last_processed_video_id="baseline-video",
                    last_processed_published_at="1970-01-01T00:01:40+00:00",
                    last_successful_run_at=None,
                    bootstrap_completed=True,
                ),
                "meitou-news": ChannelState(),
            },
        )

        client = FakeYoutubeClient(
            candidates_by_slug={
                "top3pct": [_candidate("newest-video"), _candidate("older-video"), _candidate("baseline-video")],
                "meitou-news": [],
            },
            metadata_by_id={
                "baseline-video": _video("baseline-video", published_timestamp=100, source_tab="videos"),
                "older-video": _video("older-video", published_timestamp=200, source_tab="videos"),
                "newest-video": _video("newest-video", published_timestamp=300, source_tab="streams", was_live=True),
            },
            transcript_by_id={
                "older-video": TranscriptPayload(
                    status="unavailable",
                    language=None,
                    text=None,
                    source=None,
                    error="no captions",
                )
            },
        )

        result = run_sync(
            config_path=self.config_path,
            state_path=self.state_path,
            data_dir=self.data_dir,
            client=client,
        )

        self.assertEqual(result["channels"]["top3pct"]["archived_count"], 2)
        self.assertEqual(result["channels"]["top3pct"]["transcript_unavailable_count"], 1)
        self.assertEqual(client.transcript_calls, ["older-video", "newest-video"])

        state_payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state_payload["top3pct"]["last_processed_video_id"], "newest-video")

    def test_partial_channel_failure_does_not_block_other_channels(self) -> None:
        client = FakeYoutubeClient(
            candidates_by_slug={
                "top3pct": [_candidate("top3pct-video")],
                "meitou-news": [],
            },
            metadata_by_id={
                "top3pct-video": _video("top3pct-video", published_timestamp=200, source_tab="videos"),
            },
            failing_channels={"meitou-news"},
        )

        result = run_sync(
            config_path=self.config_path,
            state_path=self.state_path,
            data_dir=self.data_dir,
            client=client,
        )

        self.assertEqual(result["status"], "partial_success")
        self.assertEqual(result["channels"]["top3pct"]["archived_count"], 1)
        self.assertEqual(len(result["channels"]["meitou-news"]["errors"]), 1)
        self.assertTrue((self.data_dir / "youtube" / "top3pct" / "videos" / "top3pct-video.json").exists())


def _candidate(video_id: str, *, source_tab: str = "videos") -> FlatPlaylistEntry:
    return FlatPlaylistEntry(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        title=video_id,
        source_tab=source_tab,
        position=1,
    )


def _video(
    video_id: str,
    *,
    published_timestamp: int,
    source_tab: str,
    was_live: bool = False,
) -> VideoMetadata:
    return VideoMetadata(
        video_id=video_id,
        title=video_id,
        channel_id="chan-1",
        channel_name="Channel",
        channel_handle="@handle",
        webpage_url=f"https://www.youtube.com/watch?v={video_id}",
        description="description",
        published_at=f"1970-01-01T00:{published_timestamp // 60:02d}:{published_timestamp % 60:02d}+00:00",
        published_timestamp=published_timestamp,
        duration_seconds=1200,
        view_count=100,
        live_status="was_live" if was_live else None,
        was_live=was_live,
        is_live=False,
        media_type="livestream" if was_live else None,
        thumbnail_url=None,
        source_tab=source_tab,
        subtitles={},
        automatic_captions={},
    )


if __name__ == "__main__":
    unittest.main()
