from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
import re

from .config import get_config_path, get_data_dir, get_state_path
from .models import ArchivedItemSummary, ChannelState, ChannelTarget, TranscriptPayload, VideoMetadata
from .storage import load_channel_targets, load_state, save_state, write_archive

if TYPE_CHECKING:
    from .youtube_client import YoutubeClient


def run_sync(
    *,
    config_path: str | Path | None = None,
    state_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    client: YoutubeClient | None = None,
) -> dict[str, object]:
    if client is None:
        from .youtube_client import YoutubeClient

    resolved_config_path = get_config_path(config_path)
    resolved_state_path = get_state_path(state_path)
    resolved_data_dir = get_data_dir(data_dir)

    channels = [channel for channel in load_channel_targets(resolved_config_path) if channel.enabled]
    if not channels:
        raise ValueError("No enabled channels were configured.")

    client = client or YoutubeClient()
    state_by_slug = load_state(resolved_state_path)
    run_started_at = utc_now()

    summary: dict[str, object] = {
        "status": "success",
        "run_started_at": run_started_at,
        "discovery_source": getattr(client, "discovery_source", "unknown"),
        "yt_cookies_available": bool(getattr(client, "yt_cookies_available", False)),
        "archived_count": 0,
        "bootstrap_count": 0,
        "transcript_unavailable_count": 0,
        "analysis_result_path": None,
        "analysis_artifact_dir": None,
        "channels": {},
        "errors": [],
        "run_notes": [],
        "new_items": [],
    }

    for channel in channels:
        result = {
            "archived_count": 0,
            "bootstrap_count": 0,
            "noop": False,
            "transcript_unavailable_count": 0,
            "errors": [],
            "notes": [],
        }
        state = state_by_slug.get(channel.slug, ChannelState())

        try:
            candidates = client.list_recent_candidates(channel, stop_id=state.last_processed_video_id)
            candidates_to_resolve = candidates
            if not state.bootstrap_completed:
                candidates_to_resolve = _select_bootstrap_candidates(candidates)

            videos = [_load_video_metadata(client, candidate) for candidate in candidates_to_resolve]
            selected = _select_videos_to_archive(videos, state)

            if channel.title_filter:
                before_count = len(selected)
                selected = [v for v in selected if _matches_content_filter(v, channel)]
                skipped = before_count - len(selected)
                if skipped:
                    note = f"{channel.slug}: skipped {skipped} video(s) not matching content_filter '{channel.title_filter}'."
                    result["notes"].append(note)
                    summary["run_notes"].append(note)

            if not selected:
                result["noop"] = True
                state.last_successful_run_at = run_started_at
                state_by_slug[channel.slug] = state
                save_state(resolved_state_path, state_by_slug)
                summary["channels"][channel.slug] = result
                continue

            bootstrap_mode = not state.bootstrap_completed
            for video in selected:
                transcript = client.fetch_transcript(video)
                archived_at = utc_now()
                archive_path = write_archive(
                    data_dir=resolved_data_dir,
                    channel=channel,
                    video=video,
                    transcript=transcript,
                    archived_at=archived_at,
                )
                state = ChannelState(
                    last_processed_video_id=video.video_id,
                    last_processed_published_at=video.published_at,
                    last_successful_run_at=run_started_at,
                    bootstrap_completed=True,
                )
                state_by_slug[channel.slug] = state
                save_state(resolved_state_path, state_by_slug)

                result["archived_count"] += 1
                if transcript.status != "fetched":
                    result["transcript_unavailable_count"] += 1
                    note = _build_transcript_note(channel_slug=channel.slug, video_id=video.video_id, transcript=transcript)
                    if note:
                        result["notes"].append(note)
                        summary["run_notes"].append(note)
                if bootstrap_mode:
                    result["bootstrap_count"] = 1
                summary["new_items"].append(
                    _build_new_item_summary(
                        archive_path=archive_path,
                        channel=channel,
                        video=video,
                        transcript=transcript,
                    ).to_mapping()
                )

            summary["archived_count"] = int(summary["archived_count"]) + int(result["archived_count"])
            summary["bootstrap_count"] = int(summary["bootstrap_count"]) + int(result["bootstrap_count"])
            summary["transcript_unavailable_count"] = int(summary["transcript_unavailable_count"]) + int(
                result["transcript_unavailable_count"]
            )
        except Exception as exc:  # pragma: no cover - exercised with fake clients in tests
            result["errors"].append(str(exc))
            summary["errors"].append(f"{channel.slug}: {exc}")
            summary["status"] = "partial_success"

        summary["channels"][channel.slug] = result

    summary["run_notes"] = _dedupe_preserve_order(summary["run_notes"])
    return summary


def run_smoke(
    *,
    config_path: str | Path | None = None,
    client: YoutubeClient | None = None,
) -> dict[str, object]:
    if client is None:
        from .youtube_client import YoutubeClient

    resolved_config_path = get_config_path(config_path)
    channels = [channel for channel in load_channel_targets(resolved_config_path) if channel.enabled]
    if not channels:
        raise ValueError("No enabled channels were configured.")

    client = client or YoutubeClient()
    payload: dict[str, object] = {"channels": {}}
    for channel in channels:
        candidates = client.list_recent_candidates(channel, stop_id=None)
        latest_video = None
        if candidates:
            latest_video = client.fetch_video_metadata(candidates[0])
        payload["channels"][channel.slug] = {
            "candidate_count": len(candidates),
            "latest_video_id": latest_video.video_id if latest_video else None,
            "latest_title": latest_video.title if latest_video else None,
            "latest_source_kind": latest_video.source_kind if latest_video else None,
        }
    return payload


def _build_transcript_note(*, channel_slug: str, video_id: str, transcript: TranscriptPayload) -> str | None:
    if transcript.status == "fetched":
        return None
    error = (transcript.error or "").strip()
    if not error:
        return None
    return f"{channel_slug}/{video_id}: {error}"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_video_metadata(client: YoutubeClient, candidate: object) -> VideoMetadata:
    return client.fetch_video_metadata(candidate)


def _select_videos_to_archive(videos: list[VideoMetadata], state: ChannelState) -> list[VideoMetadata]:
    eligible = [video for video in videos if _should_include_video(video)]
    if not eligible:
        return []

    eligible.sort(key=_video_sort_key)

    if not state.bootstrap_completed:
        return [eligible[-1]]

    state_timestamp = _parse_iso_timestamp(state.last_processed_published_at)
    unseen: list[VideoMetadata] = []
    for video in eligible:
        if state.last_processed_video_id and video.video_id == state.last_processed_video_id:
            continue
        if state_timestamp is None or video.published_timestamp is None:
            unseen.append(video)
            continue
        if video.published_timestamp > state_timestamp:
            unseen.append(video)
    return unseen


def _select_bootstrap_candidates(candidates: list[object]) -> list[object]:
    selected: list[object] = []
    seen_tabs: set[str] = set()
    for candidate in candidates:
        source_tab = getattr(candidate, "source_tab", None)
        if source_tab in seen_tabs:
            continue
        seen_tabs.add(source_tab)
        selected.append(candidate)
    return selected


def _should_include_video(video: VideoMetadata) -> bool:
    if video.is_live:
        return False

    if video.live_status in {"is_live", "is_upcoming", "post_live"}:
        return False

    if video.was_live or video.live_status == "was_live":
        return True

    return True


def _video_sort_key(video: VideoMetadata) -> tuple[int, str]:
    return (video.published_timestamp or 0, video.video_id)


def _parse_iso_timestamp(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value).timestamp())
    except ValueError:
        return None


def _matches_content_filter(video: VideoMetadata, channel: ChannelTarget) -> bool:
    if channel.title_filter is None:
        return True
    
    # Check both title and description for the filter pattern
    content = f"{video.title or ''}\n{video.description or ''}"
    return bool(re.search(channel.title_filter, content))


def _build_new_item_summary(
    *,
    archive_path: Path,
    channel: object,
    video: VideoMetadata,
    transcript: TranscriptPayload,
) -> ArchivedItemSummary:
    description = (video.description or "").strip()
    excerpt = description[:280].strip()
    if excerpt and len(description) > len(excerpt):
        excerpt = f"{excerpt}..."
    return ArchivedItemSummary(
        archive_path=str(archive_path),
        channel_slug=getattr(channel, "slug"),
        channel_handle=getattr(channel, "handle"),
        channel_name=video.channel_name,
        video_id=video.video_id,
        title=video.title,
        webpage_url=video.webpage_url,
        published_at=video.published_at,
        source_kind=video.source_kind,
        transcript_status=transcript.status,
        description_excerpt=excerpt,
    )
