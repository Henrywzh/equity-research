from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Iterable

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .config import DEFAULT_RECENT_LIMIT, DEFAULT_REQUEST_TIMEOUT, PREFERRED_TRANSCRIPT_LANGUAGES
from .models import ChannelTarget, FlatPlaylistEntry, TranscriptPayload, VideoMetadata


class YoutubeClient:
    def __init__(
        self,
        *,
        recent_limit: int = DEFAULT_RECENT_LIMIT,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        transcript_languages: Iterable[str] = PREFERRED_TRANSCRIPT_LANGUAGES,
    ) -> None:
        self.recent_limit = recent_limit
        self.request_timeout = request_timeout
        self.transcript_languages = tuple(transcript_languages)
        self.transcript_api = YouTubeTranscriptApi()

    def list_recent_candidates(
        self,
        channel: ChannelTarget,
        *,
        stop_id: str | None = None,
    ) -> list[FlatPlaylistEntry]:
        candidates: dict[str, FlatPlaylistEntry] = {}
        for source_tab, url in (("videos", channel.videos_url), ("streams", channel.streams_url)):
            entries = self._list_tab_entries(url)
            for index, entry in enumerate(entries, start=1):
                video_id = str(entry.get("id") or "").strip()
                if not video_id:
                    continue
                if stop_id and video_id == stop_id:
                    break
                if video_id in candidates:
                    continue
                candidates[video_id] = FlatPlaylistEntry(
                    video_id=video_id,
                    url=entry.get("url") or f"https://www.youtube.com/watch?v={video_id}",
                    title=entry.get("title"),
                    source_tab=source_tab,
                    position=index,
                )
        return list(candidates.values())

    def fetch_video_metadata(self, candidate: FlatPlaylistEntry) -> VideoMetadata:
        info = self._extract_info(candidate.url)
        published_at, published_timestamp = _resolve_published_at(info)
        thumbnails = info.get("thumbnails") or []
        thumbnail_url = thumbnails[-1].get("url") if thumbnails else None

        return VideoMetadata(
            video_id=str(info.get("id") or candidate.video_id),
            title=info.get("title") or candidate.title or candidate.video_id,
            channel_id=info.get("channel_id"),
            channel_name=info.get("channel") or info.get("uploader"),
            channel_handle=info.get("uploader_id"),
            webpage_url=info.get("webpage_url") or candidate.url,
            description=info.get("description"),
            published_at=published_at,
            published_timestamp=published_timestamp,
            duration_seconds=_coerce_int(info.get("duration")),
            view_count=_coerce_int(info.get("view_count")),
            live_status=info.get("live_status"),
            was_live=bool(info.get("was_live")),
            is_live=bool(info.get("is_live")),
            media_type=info.get("media_type"),
            thumbnail_url=thumbnail_url,
            source_tab=candidate.source_tab,
            subtitles=info.get("subtitles") or {},
            automatic_captions=info.get("automatic_captions") or {},
        )

    def fetch_transcript(self, video: VideoMetadata) -> TranscriptPayload:
        transcript_api_error: str | None = None

        try:
            transcript_list = self.transcript_api.list(video.video_id)
            transcript = self._pick_transcript(transcript_list)
            fetched = transcript.fetch()
            transcript_text = _join_transcript_lines(item["text"] for item in fetched.to_raw_data())
            if transcript_text:
                return TranscriptPayload(
                    status="fetched",
                    language=transcript.language_code,
                    text=transcript_text,
                    source="youtube_transcript_api",
                )
        except Exception as exc:  # pragma: no cover - network/provider variability
            transcript_api_error = str(exc)

        fallback = self._fetch_caption_fallback(video, transcript_api_error=transcript_api_error)
        return fallback

    def _fetch_caption_fallback(
        self,
        video: VideoMetadata,
        *,
        transcript_api_error: str | None = None,
    ) -> TranscriptPayload:
        last_caption_error: str | None = None

        for source_name, tracks in (
            ("subtitles", video.subtitles),
            ("automatic_captions", video.automatic_captions),
        ):
            for language in self._ordered_languages(tracks):
                for item in self._ordered_caption_formats(tracks.get(language, [])):
                    ext = item.get("ext")
                    url = item.get("url")
                    if not url or ext not in {"vtt", "srt", "ttml"}:
                        continue
                    try:
                        response = requests.get(url, timeout=self.request_timeout)
                        response.raise_for_status()
                        transcript_text = _parse_caption_text(response.text, ext)
                        if transcript_text:
                            return TranscriptPayload(
                                status="fetched",
                                language=language,
                                text=transcript_text,
                                source=f"yt_dlp_{source_name}_{ext}",
                            )
                    except Exception as exc:  # pragma: no cover - network/provider variability
                        last_caption_error = str(exc)

        errors = [value for value in (transcript_api_error, last_caption_error) if value]
        return TranscriptPayload(
            status="unavailable",
            language=None,
            text=None,
            source=None,
            error="; ".join(errors) if errors else None,
        )

    def _pick_transcript(self, transcript_list: Any) -> Any:
        try:
            return transcript_list.find_manually_created_transcript(list(self.transcript_languages))
        except Exception:
            pass

        try:
            return transcript_list.find_generated_transcript(list(self.transcript_languages))
        except Exception:
            pass

        try:
            return transcript_list.find_transcript(list(self.transcript_languages))
        except Exception:
            pass

        for transcript in transcript_list:
            return transcript
        raise RuntimeError("No transcript tracks were available.")

    def _ordered_languages(self, tracks: dict[str, list[dict[str, Any]]]) -> list[str]:
        preferred = [language for language in self.transcript_languages if language in tracks]
        remaining = sorted(language for language in tracks if language not in preferred)
        return preferred + remaining

    def _ordered_caption_formats(self, formats: list[dict[str, Any]]) -> list[dict[str, Any]]:
        priorities = {"vtt": 0, "srt": 1, "ttml": 2}
        return sorted(
            formats,
            key=lambda item: priorities.get(str(item.get("ext")), 99),
        )

    def _list_tab_entries(self, url: str) -> list[dict[str, Any]]:
        options = {
            "quiet": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
            "playlistend": self.recent_limit,
            "lazy_playlist": False,
            "ignoreerrors": True,
        }
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
        except DownloadError as exc:
            if _is_missing_channel_tab_error(exc):
                return []
            raise

        if not isinstance(info, dict):
            return []

        entries = info.get("entries") or []
        return [entry for entry in entries if isinstance(entry, dict)]

    def _extract_info(self, url: str) -> dict[str, Any]:
        options = {
            "quiet": True,
            "skip_download": True,
            "noplaylist": True,
            "ignoreerrors": False,
        }
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        if not isinstance(info, dict):
            raise RuntimeError(f"Unable to extract metadata for {url}")
        return info


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_published_at(info: dict[str, Any]) -> tuple[str | None, int | None]:
    timestamp = _coerce_int(info.get("release_timestamp")) or _coerce_int(info.get("timestamp"))
    if timestamp is not None:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(), timestamp

    date_value = info.get("release_date") or info.get("upload_date")
    if isinstance(date_value, str) and len(date_value) == 8 and date_value.isdigit():
        dt = datetime.strptime(date_value, "%Y%m%d").replace(tzinfo=timezone.utc)
        return dt.isoformat(), int(dt.timestamp())

    return None, None


def _join_transcript_lines(lines: Iterable[str]) -> str | None:
    cleaned = [line.strip() for line in lines if line and line.strip()]
    if not cleaned:
        return None
    return "\n".join(cleaned)


def _parse_caption_text(payload: str, ext: str) -> str | None:
    if ext == "vtt":
        return _parse_vtt(payload)
    if ext == "srt":
        return _parse_srt(payload)
    if ext == "ttml":
        return _parse_ttml(payload)
    return None


def _parse_vtt(payload: str) -> str | None:
    lines: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if line.startswith("NOTE"):
            continue
        if "-->" in line:
            continue
        lines.append(_strip_tags(line))
    return _join_transcript_lines(lines)


def _parse_srt(payload: str) -> str | None:
    timecode_pattern = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}")
    lines: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        if timecode_pattern.match(line):
            continue
        lines.append(_strip_tags(line))
    return _join_transcript_lines(lines)


def _parse_ttml(payload: str) -> str | None:
    root = ET.fromstring(payload)
    lines: list[str] = []
    for element in root.iter():
        if element.tag.endswith("p"):
            text = _strip_tags(" ".join(part.strip() for part in element.itertext() if part.strip()))
            if text:
                lines.append(text)
    return _join_transcript_lines(lines)


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value).strip()


def _is_missing_channel_tab_error(exc: DownloadError) -> bool:
    message = str(exc).lower()
    return "does not have a streams tab" in message or "does not have a videos tab" in message
