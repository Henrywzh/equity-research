from __future__ import annotations

import json
from pathlib import Path

from .models import ChannelState, ChannelTarget, TranscriptPayload, VideoMetadata


def load_channel_targets(config_path: str | Path) -> list[ChannelTarget]:
    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    channels: list[ChannelTarget] = []
    for item in payload:
        channels.append(
            ChannelTarget(
                slug=item["slug"],
                handle=item["handle"],
                channel_id=item["channel_id"],
                videos_url=item["videos_url"],
                streams_url=item["streams_url"],
                enabled=bool(item.get("enabled", True)),
                profile=str(item.get("profile", "macroeconomics")),
                title_filter=item.get("title_filter"),
            )
        )
    return channels


def load_state(state_path: str | Path) -> dict[str, ChannelState]:
    path = Path(state_path).expanduser().resolve()
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    payload = json.loads(raw)
    return {slug: ChannelState.from_mapping(item) for slug, item in payload.items()}


def save_state(state_path: str | Path, state_by_slug: dict[str, ChannelState]) -> None:
    path = Path(state_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        slug: state.to_mapping()
        for slug, state in sorted(state_by_slug.items(), key=lambda item: item[0])
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_archive(
    *,
    data_dir: str | Path,
    channel: ChannelTarget,
    video: VideoMetadata,
    transcript: TranscriptPayload,
    archived_at: str,
) -> Path:
    root = Path(data_dir).expanduser().resolve()
    archive_dir = root / "youtube" / channel.slug / "videos"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{video.video_id}.json"

    payload = {
        "archived_at": archived_at,
        "source_kind": video.source_kind,
        "analysis_input_basis": "transcript" if transcript.segments or transcript.text else "metadata_only",
        "channel": {
            "slug": channel.slug,
            "handle": channel.handle,
            "channel_id": channel.channel_id,
            "channel_name": video.channel_name,
            "profile": channel.profile,
        },
        "video": {
            "video_id": video.video_id,
            "title": video.title,
            "webpage_url": video.webpage_url,
            "duration_seconds": video.duration_seconds,
            "view_count": video.view_count,
            "live_status": video.live_status,
            "was_live": video.was_live,
            "is_live": video.is_live,
            "media_type": video.media_type,
            "source_tab": video.source_tab,
            "thumbnail_url": video.thumbnail_url,
        },
        "published_at": video.published_at,
        "description": video.description,
        "transcript_status": transcript.status,
        "transcript_language": transcript.language,
        "transcript_text": transcript.text,
        "transcript_segments": [segment.to_mapping() for segment in transcript.segments],
        "transcript_source": transcript.source,
        "error": transcript.error,
    }
    write_json_document(archive_path, payload)
    return archive_path


def load_json_document(path: str | Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    return json.loads(resolved.read_text(encoding="utf-8"))


def write_json_document(path: str | Path, payload: dict) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resolved
