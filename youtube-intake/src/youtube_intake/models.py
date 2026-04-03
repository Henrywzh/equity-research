from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChannelTarget:
    slug: str
    handle: str
    channel_id: str
    videos_url: str
    streams_url: str
    enabled: bool = True


@dataclass(slots=True)
class ChannelState:
    last_processed_video_id: str | None = None
    last_processed_published_at: str | None = None
    last_successful_run_at: str | None = None
    bootstrap_completed: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ChannelState":
        payload = payload or {}
        return cls(
            last_processed_video_id=payload.get("last_processed_video_id"),
            last_processed_published_at=payload.get("last_processed_published_at"),
            last_successful_run_at=payload.get("last_successful_run_at"),
            bootstrap_completed=bool(payload.get("bootstrap_completed", False)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "last_processed_video_id": self.last_processed_video_id,
            "last_processed_published_at": self.last_processed_published_at,
            "last_successful_run_at": self.last_successful_run_at,
            "bootstrap_completed": self.bootstrap_completed,
        }


@dataclass(slots=True)
class FlatPlaylistEntry:
    video_id: str
    url: str
    title: str | None
    source_tab: str
    position: int


@dataclass(slots=True)
class VideoMetadata:
    video_id: str
    title: str
    channel_id: str | None
    channel_name: str | None
    channel_handle: str | None
    webpage_url: str
    description: str | None
    published_at: str | None
    published_timestamp: int | None
    duration_seconds: int | None
    view_count: int | None
    live_status: str | None
    was_live: bool
    is_live: bool
    media_type: str | None
    thumbnail_url: str | None
    source_tab: str
    subtitles: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    automatic_captions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def source_kind(self) -> str:
        return "livestream_replay" if self.was_live or self.live_status == "was_live" else "video"


@dataclass(slots=True)
class TranscriptPayload:
    status: str
    language: str | None
    text: str | None
    source: str | None
    segments: list["TranscriptSegment"] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class ArchivedItemSummary:
    archive_path: str
    channel_slug: str
    channel_handle: str
    channel_name: str | None
    video_id: str
    title: str
    webpage_url: str
    published_at: str | None
    source_kind: str
    transcript_status: str
    description_excerpt: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "archive_path": self.archive_path,
            "channel_slug": self.channel_slug,
            "channel_handle": self.channel_handle,
            "channel_name": self.channel_name,
            "video_id": self.video_id,
            "title": self.title,
            "webpage_url": self.webpage_url,
            "published_at": self.published_at,
            "source_kind": self.source_kind,
            "transcript_status": self.transcript_status,
            "description_excerpt": self.description_excerpt,
        }


@dataclass(slots=True)
class TranscriptSegment:
    start_seconds: float
    duration_seconds: float | None
    text: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "start_seconds": round(self.start_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3) if self.duration_seconds is not None else None,
            "text": self.text,
        }
