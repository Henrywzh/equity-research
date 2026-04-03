from __future__ import annotations

import random
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .config import DEFAULT_RECENT_LIMIT, DEFAULT_REQUEST_TIMEOUT, PREFERRED_TRANSCRIPT_LANGUAGES
from .models import ChannelTarget, FlatPlaylistEntry, TranscriptPayload, TranscriptSegment, VideoMetadata
from .runtime_env import load_local_config, read_env


GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_AUDIO_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
PRIMARY_STT_MODEL_ID = "whisper-large-v3-turbo"
FALLBACK_STT_MODEL_ID = "whisper-large-v3"
MAX_STT_VIDEO_DURATION_SECONDS = 3600
DEFAULT_STT_TIMEOUT = 600
MAX_RETRIES = 3
MAX_STT_AUDIO_BYTES = 24 * 1024 * 1024
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
TRANSIENT_EXCEPTION_TYPES = (requests.Timeout, requests.ConnectionError)
DEFAULT_BACKOFF_SECONDS = (2.0, 5.0, 10.0)


class YoutubeClient:
    def __init__(
        self,
        *,
        recent_limit: int = DEFAULT_RECENT_LIMIT,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        transcript_languages: Iterable[str] = PREFERRED_TRANSCRIPT_LANGUAGES,
        groq_api_key: str | None = None,
        stt_timeout: int = DEFAULT_STT_TIMEOUT,
        sleep_fn: Any | None = None,
        jitter_fn: Any | None = None,
        transcription_transport: Any | None = None,
    ) -> None:
        load_local_config()
        self.recent_limit = recent_limit
        self.request_timeout = request_timeout
        self.transcript_languages = tuple(transcript_languages)
        self.transcript_api = YouTubeTranscriptApi()
        self.groq_api_key = (groq_api_key or read_env(GROQ_API_KEY_ENV)).strip()
        self.stt_timeout = stt_timeout
        self.sleep_fn = sleep_fn or time.sleep
        self.jitter_fn = jitter_fn or (lambda: random.uniform(0.0, 0.75))
        self.transcription_transport = transcription_transport or requests.post

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
            segments = _normalize_transcript_api_segments(fetched.to_raw_data())
            transcript_text = _segments_to_text(segments)
            if transcript_text:
                return TranscriptPayload(
                    status="fetched",
                    language=transcript.language_code,
                    text=transcript_text,
                    source="youtube_transcript_api",
                    segments=segments,
                )
        except Exception as exc:  # pragma: no cover - network/provider variability
            transcript_api_error = str(exc)

        fallback = self._fetch_caption_fallback(video, transcript_api_error=transcript_api_error)
        if fallback.status == "fetched":
            return fallback

        return self._fetch_stt_fallback(video, prior_error=fallback.error)

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
                        segments = _parse_caption_segments(response.text, ext)
                        transcript_text = _segments_to_text(segments)
                        if transcript_text:
                            return TranscriptPayload(
                                status="fetched",
                                language=language,
                                text=transcript_text,
                                source=f"yt_dlp_{source_name}_{ext}",
                                segments=segments,
                            )
                    except Exception as exc:  # pragma: no cover - network/provider variability
                        last_caption_error = str(exc)

        errors = [value for value in (transcript_api_error, last_caption_error) if value]
        return TranscriptPayload(
            status="unavailable",
            language=None,
            text=None,
            source=None,
            segments=[],
            error="; ".join(errors) if errors else None,
        )

    def _fetch_stt_fallback(
        self,
        video: VideoMetadata,
        *,
        prior_error: str | None = None,
    ) -> TranscriptPayload:
        errors = [value for value in (prior_error,) if value]

        if video.duration_seconds is not None and video.duration_seconds > MAX_STT_VIDEO_DURATION_SECONDS:
            errors.append(
                "Groq STT skipped due to duration limit "
                f"({video.duration_seconds}s > {MAX_STT_VIDEO_DURATION_SECONDS}s)."
            )
            return _build_unavailable_transcript(errors)

        if not self.groq_api_key:
            errors.append("Groq STT skipped because GROQ_API_KEY is not configured.")
            return _build_unavailable_transcript(errors)

        with tempfile.TemporaryDirectory(prefix="youtube-intake-audio-") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            try:
                audio_path = self._download_audio(video, temp_dir=temp_dir)
            except Exception as exc:  # pragma: no cover - yt-dlp/provider variability
                errors.append(f"Groq STT audio download failed: {exc}")
                return _build_unavailable_transcript(errors)

            transcript, stt_errors = self._transcribe_audio_with_fallback(audio_path)
            if transcript is not None:
                return transcript
            errors.extend(stt_errors)
            return _build_unavailable_transcript(errors)

    def _download_audio(self, video: VideoMetadata, *, temp_dir: Path) -> Path:
        info = self._extract_info(video.webpage_url)
        format_id = _select_audio_format_id(info.get("formats") or [], max_bytes=MAX_STT_AUDIO_BYTES)
        outtmpl = str(temp_dir / "%(id)s.%(ext)s")
        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "ignoreerrors": False,
            "format": (
                format_id
                or "worstaudio[ext=m4a]/worstaudio[ext=webm]/worstaudio/"
                "bestaudio[filesize<24500000]/bestaudio[filesize_approx<24500000]/bestaudio/best"
            ),
            "outtmpl": outtmpl,
        }
        with YoutubeDL(options) as ydl:
            downloaded_info = ydl.extract_info(video.webpage_url, download=True)
            if not isinstance(downloaded_info, dict):
                raise RuntimeError(f"Unable to download audio for {video.webpage_url}")
            prepared = Path(ydl.prepare_filename(downloaded_info))
            if prepared.exists():
                return prepared

        candidates = sorted(path for path in temp_dir.iterdir() if path.is_file())
        if not candidates:
            raise RuntimeError(f"No audio file was downloaded for {video.video_id}")
        return candidates[0]

    def _transcribe_audio_with_fallback(self, audio_path: Path) -> tuple[TranscriptPayload | None, list[str]]:
        errors: list[str] = []
        for model_id in (PRIMARY_STT_MODEL_ID, FALLBACK_STT_MODEL_ID):
            try:
                payload = self._request_groq_transcription(audio_path, model_id=model_id)
            except Exception as exc:  # pragma: no cover - provider variability
                errors.append(f"{model_id}: {exc}")
                continue

            transcript = _build_transcript_from_stt_payload(payload, model_id=model_id)
            if transcript is not None:
                return transcript, errors
            errors.append(f"{model_id}: returned unusable transcription output.")
        return None, errors

    def _request_groq_transcription(self, audio_path: Path, *, model_id: str) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt_index in range(MAX_RETRIES):
            try:
                with audio_path.open("rb") as audio_file:
                    response = self.transcription_transport(
                        GROQ_AUDIO_TRANSCRIPTIONS_URL,
                        headers={"Authorization": f"Bearer {self.groq_api_key}"},
                        data=[
                            ("model", model_id),
                            ("response_format", "verbose_json"),
                            ("temperature", "0"),
                            ("timestamp_granularities[]", "segment"),
                        ],
                        files={
                            "file": (
                                audio_path.name,
                                audio_file,
                                _guess_audio_mime_type(audio_path),
                            )
                        },
                        timeout=self.stt_timeout,
                    )
            except TRANSIENT_EXCEPTION_TYPES as exc:
                last_error = exc
                if attempt_index == MAX_RETRIES - 1:
                    break
                self.sleep_fn(_compute_retry_delay(None, attempt_index, jitter_fn=self.jitter_fn))
                continue
            except requests.RequestException as exc:
                raise RuntimeError(str(exc)) from exc

            if response.status_code in TRANSIENT_STATUS_CODES:
                last_error = RuntimeError(f"{model_id} returned HTTP {response.status_code}")
                if attempt_index == MAX_RETRIES - 1:
                    break
                self.sleep_fn(_compute_retry_delay(response, attempt_index, jitter_fn=self.jitter_fn))
                continue

            try:
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(f"{model_id}: {exc}") from exc

            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"{model_id}: transcription response was not a JSON object.")
            return payload

        raise RuntimeError(str(last_error) if last_error else f"{model_id}: transcription request failed.")

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
    cleaned = [_normalize_segment_text(line) for line in lines if line and _normalize_segment_text(line)]
    if not cleaned:
        return None
    return "\n".join(cleaned)


def _segments_to_text(segments: list[TranscriptSegment]) -> str | None:
    return _join_transcript_lines(segment.text for segment in segments)


def _parse_caption_segments(payload: str, ext: str) -> list[TranscriptSegment]:
    if ext == "vtt":
        return _parse_vtt(payload)
    if ext == "srt":
        return _parse_srt(payload)
    if ext == "ttml":
        return _parse_ttml(payload)
    return []


def _parse_vtt(payload: str) -> list[TranscriptSegment]:
    blocks = re.split(r"\n\s*\n", payload.replace("\r\n", "\n"))
    segments: list[TranscriptSegment] = []
    for block in blocks:
        lines = [line.strip("\ufeff ").strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith("WEBVTT") or lines[0].startswith("NOTE"):
            continue
        if lines[0].startswith("Kind:") or lines[0].startswith("Language:"):
            continue

        timecode_index = 0
        if "-->" not in lines[0]:
            if len(lines) < 2 or "-->" not in lines[1]:
                continue
            timecode_index = 1

        start_seconds, duration_seconds = _parse_vtt_timecode(lines[timecode_index])
        if start_seconds is None:
            continue
        text = _join_transcript_lines(lines[timecode_index + 1 :])
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                text=text,
            )
        )
    return segments


def _parse_srt(payload: str) -> list[TranscriptSegment]:
    blocks = re.split(r"\n\s*\n", payload.replace("\r\n", "\n"))
    segments: list[TranscriptSegment] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].isdigit():
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_seconds, duration_seconds = _parse_srt_timecode(lines[0])
        if start_seconds is None:
            continue
        text = _join_transcript_lines(lines[1:])
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                text=text,
            )
        )
    return segments


def _parse_ttml(payload: str) -> list[TranscriptSegment]:
    root = ET.fromstring(payload)
    segments: list[TranscriptSegment] = []
    for element in root.iter():
        if element.tag.endswith("p"):
            text = _join_transcript_lines(part.strip() for part in element.itertext() if part.strip())
            if not text:
                continue
            start_seconds = _parse_ttml_clock_value(element.attrib.get("begin"))
            end_seconds = _parse_ttml_clock_value(element.attrib.get("end"))
            duration_seconds = _parse_ttml_clock_value(element.attrib.get("dur"))
            if duration_seconds is None and start_seconds is not None and end_seconds is not None:
                duration_seconds = max(end_seconds - start_seconds, 0.0)
            segments.append(
                TranscriptSegment(
                    start_seconds=start_seconds or 0.0,
                    duration_seconds=duration_seconds,
                    text=text,
                )
            )
    return segments


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value).strip()


def _normalize_segment_text(value: str) -> str:
    return re.sub(r"\s+", " ", _strip_tags(value)).strip()


def _normalize_transcript_api_segments(items: Iterable[dict[str, Any]]) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for item in items:
        text = _normalize_segment_text(str(item.get("text") or ""))
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_seconds=float(item.get("start") or 0.0),
                duration_seconds=_coerce_float(item.get("duration")),
                text=text,
            )
        )
    return segments


def _parse_vtt_timecode(line: str) -> tuple[float | None, float | None]:
    start_raw, _, end_raw = line.partition("-->")
    start_seconds = _parse_timecode_value(start_raw.strip().split(" ", 1)[0].replace(",", "."))
    end_seconds = _parse_timecode_value(end_raw.strip().split(" ", 1)[0].replace(",", "."))
    if start_seconds is None:
        return None, None
    duration_seconds = None
    if end_seconds is not None:
        duration_seconds = max(end_seconds - start_seconds, 0.0)
    return start_seconds, duration_seconds


def _parse_srt_timecode(line: str) -> tuple[float | None, float | None]:
    start_raw, _, end_raw = line.partition("-->")
    start_seconds = _parse_timecode_value(start_raw.strip().replace(",", "."))
    end_seconds = _parse_timecode_value(end_raw.strip().replace(",", "."))
    if start_seconds is None:
        return None, None
    duration_seconds = None
    if end_seconds is not None:
        duration_seconds = max(end_seconds - start_seconds, 0.0)
    return start_seconds, duration_seconds


def _parse_timecode_value(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip()
    parts = cleaned.split(":")
    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return float(cleaned)
    except ValueError:
        return None


def _parse_ttml_clock_value(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.endswith("ms"):
        return _coerce_float(cleaned[:-2]) / 1000 if _coerce_float(cleaned[:-2]) is not None else None
    if cleaned.endswith("s"):
        return _coerce_float(cleaned[:-1])
    if cleaned.endswith("m"):
        base = _coerce_float(cleaned[:-1])
        return base * 60 if base is not None else None
    if cleaned.endswith("h"):
        base = _coerce_float(cleaned[:-1])
        return base * 3600 if base is not None else None
    return _parse_timecode_value(cleaned.replace(",", "."))


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_missing_channel_tab_error(exc: DownloadError) -> bool:
    message = str(exc).lower()
    return "does not have a streams tab" in message or "does not have a videos tab" in message


def _build_unavailable_transcript(errors: Iterable[str]) -> TranscriptPayload:
    cleaned = [str(error).strip() for error in errors if str(error).strip()]
    return TranscriptPayload(
        status="unavailable",
        language=None,
        text=None,
        source=None,
        segments=[],
        error="; ".join(cleaned) if cleaned else None,
    )


def _build_transcript_from_stt_payload(payload: dict[str, Any], *, model_id: str) -> TranscriptPayload | None:
    segments = _normalize_groq_stt_segments(payload.get("segments") or [])
    transcript_text = _normalize_segment_text(str(payload.get("text") or "")) or _segments_to_text(segments)
    if not transcript_text or not segments:
        return None
    return TranscriptPayload(
        status="fetched",
        language=(str(payload.get("language") or "").strip() or None),
        text=transcript_text,
        source=f"groq_{model_id.replace('-', '_')}",
        segments=segments,
    )


def _normalize_groq_stt_segments(items: Iterable[dict[str, Any]]) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = _normalize_segment_text(str(item.get("text") or ""))
        if not text:
            continue
        start_seconds = _coerce_float(item.get("start"))
        if start_seconds is None:
            continue
        end_seconds = _coerce_float(item.get("end"))
        duration_seconds = None
        if end_seconds is not None:
            duration_seconds = max(end_seconds - start_seconds, 0.0)
        segments.append(
            TranscriptSegment(
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                text=text,
            )
        )
    return segments


def _compute_retry_delay(
    response: requests.Response | None,
    attempt_index: int,
    *,
    jitter_fn: Any,
) -> float:
    retry_after = _parse_retry_after_seconds(response.headers.get("retry-after")) if response is not None else None
    if retry_after is not None:
        return retry_after
    base_delay = DEFAULT_BACKOFF_SECONDS[min(attempt_index, len(DEFAULT_BACKOFF_SECONDS) - 1)]
    return base_delay + float(jitter_fn())


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _guess_audio_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".m4a":
        return "audio/mp4"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".wav":
        return "audio/wav"
    if ext == ".ogg":
        return "audio/ogg"
    if ext == ".webm":
        return "audio/webm"
    return "application/octet-stream"


def _select_audio_format_id(formats: Iterable[dict[str, Any]], *, max_bytes: int) -> str | None:
    audio_formats: list[dict[str, Any]] = []
    for item in formats:
        if not isinstance(item, dict):
            continue
        if item.get("vcodec") != "none":
            continue
        if item.get("acodec") == "none":
            continue
        if str(item.get("ext") or "").lower() == "mhtml":
            continue
        audio_formats.append(item)

    if not audio_formats:
        return None

    within_limit = [item for item in audio_formats if (_audio_format_size(item) or 0) > 0 and (_audio_format_size(item) or 0) <= max_bytes]
    if within_limit:
        best = max(
            within_limit,
            key=lambda item: (
                _coerce_float(item.get("abr")) or _coerce_float(item.get("tbr")) or 0.0,
                -_audio_ext_priority(item),
                -float(_audio_format_size(item) or max_bytes),
            ),
        )
        return str(best.get("format_id") or "").strip() or None

    known_size = [item for item in audio_formats if _audio_format_size(item) is not None]
    if known_size:
        smallest = min(
            known_size,
            key=lambda item: (
                _audio_format_size(item) or float("inf"),
                _audio_ext_priority(item),
                -(_coerce_float(item.get("abr")) or _coerce_float(item.get("tbr")) or 0.0),
            ),
        )
        return str(smallest.get("format_id") or "").strip() or None

    fallback = min(
        audio_formats,
        key=lambda item: (
            _audio_ext_priority(item),
            _coerce_float(item.get("abr")) or _coerce_float(item.get("tbr")) or float("inf"),
        ),
    )
    return str(fallback.get("format_id") or "").strip() or None


def _audio_format_size(item: dict[str, Any]) -> int | None:
    return _coerce_int(item.get("filesize")) or _coerce_int(item.get("filesize_approx"))


def _audio_ext_priority(item: dict[str, Any]) -> int:
    ext = str(item.get("ext") or "").lower()
    if ext == "m4a":
        return 0
    if ext == "mp3":
        return 1
    if ext == "webm":
        return 2
    return 9
