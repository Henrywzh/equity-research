from __future__ import annotations

import dataclasses
import logging
import os
import random
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# --- Environment Setup (Must happen before yt_dlp import for capability discovery) ---
def _bootstrap_environment() -> None:
    """Find essential directories and ensure they are in PATH."""
    search_dirs = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    current_path = os.environ.get("PATH", "")
    paths_to_add = [d for d in search_dirs if d not in current_path and os.path.isdir(d)]
    if paths_to_add:
        os.environ["PATH"] = os.pathsep.join(paths_to_add + [current_path])

_bootstrap_environment()
# -----------------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .config import DEFAULT_RECENT_LIMIT, DEFAULT_REQUEST_TIMEOUT, PREFERRED_TRANSCRIPT_LANGUAGES
from .models import ChannelTarget, FlatPlaylistEntry, TranscriptPayload, TranscriptSegment, VideoMetadata
from .runtime_env import load_local_config, read_env
from .storage import load_json_document


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_HTTP_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
}
GROQ_API_KEY_ENV = "GROQ_API_KEY"
YT_COOKIES_ENV = "YOUTUBE_INTAKE_YT_COOKIES"
GROQ_AUDIO_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
PRIMARY_STT_MODEL_ID = "whisper-large-v3-turbo"
FALLBACK_STT_MODEL_ID = "whisper-large-v3"
MAX_STT_PROCESS_DURATION_SECONDS = 50 * 60  # 3000 — audio cut to this limit; note added
DEFAULT_STT_TIMEOUT = 600
MAX_RETRIES = 3
MAX_STT_AUDIO_BYTES = 24 * 1024 * 1024
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
TRANSIENT_EXCEPTION_TYPES = (requests.Timeout, requests.ConnectionError)
DEFAULT_BACKOFF_SECONDS = (2.0, 5.0, 10.0)
YOUTUBE_FEED_URL_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015", "media": "http://search.yahoo.com/mrss/"}


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
        http_get: Any | None = None,
        yt_cookies: str | None = None,
    ) -> None:
        load_local_config()
        self.recent_limit = recent_limit
        self.request_timeout = request_timeout
        self.transcript_languages = tuple(transcript_languages)
        self.transcript_api = YouTubeTranscriptApi()
        
        # Handle multiple comma-separated API keys for rotation
        key_source = groq_api_key or read_env(GROQ_API_KEY_ENV)
        self.groq_api_keys = [k.strip() for k in key_source.split(",") if k.strip()]
        self._key_index = 0
        
        if yt_cookies is None:
            yt_cookies = read_env(YT_COOKIES_ENV)
        self.yt_cookies = yt_cookies.strip()
        self.stt_timeout = stt_timeout
        self.sleep_fn = sleep_fn or time.sleep
        self.jitter_fn = jitter_fn or (lambda: random.uniform(0.0, 0.75))
        self.transcription_transport = transcription_transport or requests.post
        self.http_get = http_get or requests.get
        self.discovery_source = "youtube_rss"
        # Specifically locate ffmpeg for our internal truncation logic
        import shutil
        found_ffmpeg = shutil.which("ffmpeg")
        self._ffmpeg_path = Path(found_ffmpeg) if found_ffmpeg else None
        
        # Per-(model_id, key_index) rate-limit state for proactive STT slot selection.
        self._stt_rate_states: dict[tuple[str, int], dict] = {}

    @property
    def groq_api_key(self) -> str:
        """Returns the currently active API key (rotatable)."""
        if not self.groq_api_keys:
            return ""
        return self.groq_api_keys[self._key_index % len(self.groq_api_keys)]

    def _rotate_key(self) -> None:
        """Rotates to the next API key."""
        if self.groq_api_keys:
            self._key_index = (self._key_index + 1) % len(self.groq_api_keys)

    def list_recent_candidates(
        self,
        channel: ChannelTarget,
        *,
        stop_id: str | None = None,
    ) -> list[FlatPlaylistEntry]:
        candidates: list[FlatPlaylistEntry] = []
        for index, entry in enumerate(self._fetch_feed_entries(channel), start=1):
            if stop_id and entry.video_id == stop_id:
                break
            entry.position = index
            candidates.append(entry)
            if len(candidates) >= self.recent_limit:
                break
        return candidates

    def fetch_video_metadata(self, candidate: FlatPlaylistEntry) -> VideoMetadata:
        return VideoMetadata(
            video_id=candidate.video_id,
            title=candidate.title or candidate.video_id,
            channel_id=candidate.channel_id,
            channel_name=candidate.channel_name,
            channel_handle=None,
            webpage_url=candidate.url,
            description=candidate.description,
            published_at=candidate.published_at,
            published_timestamp=candidate.published_timestamp,
            duration_seconds=None,
            view_count=None,
            live_status=None,
            was_live=False,
            is_live=False,
            media_type="video",
            thumbnail_url=candidate.thumbnail_url,
            source_tab=candidate.source_tab,
            subtitles={},
            automatic_captions={},
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
                        response = self.http_get(url, timeout=self.request_timeout)
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

        # removed mandatory cookie check - fallback is now allowed without cookies if they are missing

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

            # Truncate audio to the first MAX_STT_PROCESS_DURATION_SECONDS if video is longer.
            duration = video.duration_seconds
            truncated = False
            truncation_note: str | None = None
            if duration is not None and duration > MAX_STT_PROCESS_DURATION_SECONDS:
                try:
                    audio_path = self._truncate_audio(audio_path, MAX_STT_PROCESS_DURATION_SECONDS)
                    truncated = True
                    truncation_note = (
                        f"[Note: Video is {duration}s ({duration // 60}m). "
                        f"Only the first {MAX_STT_PROCESS_DURATION_SECONDS // 60} minutes were transcribed.]"
                    )
                    LOGGER.info(
                        "Video %s is %ds; truncating audio to first %ds for STT.",
                        video.video_id, duration, MAX_STT_PROCESS_DURATION_SECONDS,
                    )
                except Exception as exc:  # pragma: no cover
                    errors.append(f"Audio truncation failed, skipping STT: {exc}")
                    return _build_unavailable_transcript(errors)

            # Final safety size check after potential local truncation (Groq limit is 25MB)
            size = audio_path.stat().st_size
            if size > MAX_STT_AUDIO_BYTES:
                errors.append(
                    f"Groq STT audio failed: Transcoded file {size/1024/1024:.1f}MB exceeds Groq 25MB limit "
                    f"for {video.video_id}"
                )
                return _build_unavailable_transcript(errors)

            transcript, stt_errors = self._transcribe_audio_with_fallback(audio_path)
            if transcript is not None:
                if truncation_note:
                    transcript = dataclasses.replace(transcript, truncation_note=truncation_note)
                return transcript
            errors.extend(stt_errors)
            return _build_unavailable_transcript(errors)

    def _download_audio(self, video: VideoMetadata, *, temp_dir: Path) -> Path:
        with self._temporary_cookie_file(temp_dir) as cookiefile:
            info = self._extract_info(video.webpage_url, cookiefile=cookiefile)
            duration_seconds = _coerce_int(info.get("duration"))
            if duration_seconds is not None:
                video.duration_seconds = duration_seconds
            # If video is long, optimization: only download the first section we need.
            section = []
            if duration_seconds is not None and duration_seconds > MAX_STT_PROCESS_DURATION_SECONDS:
                section = [f"*0-{MAX_STT_PROCESS_DURATION_SECONDS}"]
                LOGGER.info(
                    "Video %s is long (%ds); downloading first %ds only.",
                    video.video_id, duration_seconds, MAX_STT_PROCESS_DURATION_SECONDS
                )
            max_duration = MAX_STT_PROCESS_DURATION_SECONDS if duration_seconds is not None and duration_seconds > MAX_STT_PROCESS_DURATION_SECONDS else None
            format_id = _select_audio_format_id(
                info.get("formats") or [], 
                max_bytes=MAX_STT_AUDIO_BYTES,
                max_duration=max_duration
            )
            outtmpl = str(temp_dir / "%(id)s.%(ext)s")
            options = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "ignoreerrors": False,
                "user_agent": DEFAULT_USER_AGENT,
                "http_headers": DEFAULT_HTTP_HEADERS,
                "format": (
                    format_id
                    or "worstaudio[ext=m4a]/worstaudio[ext=webm]/worstaudio/"
                    "bestaudio[filesize<24500000]/bestaudio[filesize_approx<24500000]/bestaudio/best"
                ),
                "outtmpl": outtmpl,
                "download_sections": section if section else None,
            }
            if self._ffmpeg_path:
                options["ffmpeg_location"] = str(self._ffmpeg_path.parent)
                LOGGER.info("Using ffmpeg from: %s", self._ffmpeg_path)
            if cookiefile is not None:
                options["cookiefile"] = str(cookiefile)
            with YoutubeDL(options) as ydl:
                downloaded_info = ydl.extract_info(video.webpage_url, download=True)
                if not isinstance(downloaded_info, dict):
                    raise RuntimeError(f"Unable to download audio for {video.webpage_url}")
                
                prepared = Path(ydl.prepare_filename(downloaded_info))
                if prepared.exists():
                    return prepared

        candidates = sorted(
            path
            for path in temp_dir.iterdir()
            if path.is_file() and path.name != "youtube-cookies.txt"
        )
        if not candidates:
            raise RuntimeError(f"No audio file was downloaded for {video.video_id}")
        return candidates[0]

    def _truncate_audio(self, audio_path: Path, max_seconds: int) -> Path:
        """Trim audio to the first max_seconds using ffmpeg. Returns path to trimmed file."""
        # Force .mp3 extension for the trimmed file as we re-encode to libmp3lame
        trimmed_path = audio_path.with_name(audio_path.stem + "_trimmed.mp3")
        
        orig_size = audio_path.stat().st_size
        LOGGER.info("Truncating %s (%d bytes) to %ds...", audio_path.name, orig_size, max_seconds)
        LOGGER.debug("Current PATH: %s", os.environ.get("PATH"))
        
        try:
            # We use "ffmpeg" directly because _discover_ffmpeg has already injected the path into os.environ["PATH"]
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-t", str(max_seconds),
                    "-i", str(audio_path),
                    "-c:a", "libmp3lame", "-b:a", "48k",
                    str(trimmed_path)
                ],
                check=True,
                capture_output=True,
                text=True
            )
            LOGGER.debug("ffmpeg output: %s", result.stderr)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            stderr = getattr(exc, "stderr", "")
            if isinstance(exc, FileNotFoundError):
                raise RuntimeError(
                    f"Automatic cutting failed: ffmpeg not found in PATH. Path was: {os.environ.get('PATH')}"
                ) from exc
            raise RuntimeError(f"ffmpeg truncation failed: {exc}. Stderr: {stderr}") from exc
            
        new_size = trimmed_path.stat().st_size
        LOGGER.info("Truncation complete: %s (%d bytes)", trimmed_path.name, new_size)
        
        audio_path.unlink(missing_ok=True)
        return trimmed_path

    def _select_stt_slot(self) -> tuple[str, int]:
        """Pick the (model_id, key_index) slot with the most remaining request capacity.

        Checks both Whisper models across all API keys so both quotas are utilized.
        """
        now = time.monotonic()
        best_slot: tuple[str, int] | None = None
        best_remaining = -1
        for model_id in (PRIMARY_STT_MODEL_ID, FALLBACK_STT_MODEL_ID):
            for ki in range(len(self.groq_api_keys)):
                state = self._stt_rate_states.get((model_id, ki), {})
                reset_at = state.get("reset_at", 0.0)
                remaining = state.get("remaining_requests", 999)
                if reset_at > now:
                    # Within the rate-limit window: usable only if requests remain.
                    effective = remaining
                else:
                    # Window expired — treat as full capacity.
                    effective = 999
                if effective > best_remaining:
                    best_slot = (model_id, ki)
                    best_remaining = effective
        return best_slot or (PRIMARY_STT_MODEL_ID, 0)

    def _record_stt_response(self, model_id: str, key_index: int, response: Any) -> None:
        """Parse Groq rate-limit headers and update per-slot state."""
        now = time.monotonic()
        state = self._stt_rate_states.setdefault((model_id, key_index), {})
        headers = {k.lower(): v for k, v in response.headers.items()}
        try:
            remaining = int(headers.get("x-ratelimit-remaining-requests", ""))
            state["remaining_requests"] = remaining
        except (ValueError, TypeError):
            pass
        # x-ratelimit-reset-requests is formatted like "1s" or "500ms"
        reset_raw = headers.get("x-ratelimit-reset-requests", "")
        if reset_raw:
            try:
                if reset_raw.endswith("ms"):
                    state["reset_at"] = now + float(reset_raw[:-2]) / 1000
                elif reset_raw.endswith("s"):
                    state["reset_at"] = now + float(reset_raw[:-1])
            except (ValueError, TypeError):
                pass
        if response.status_code == 429:
            state["remaining_requests"] = 0
            retry_after = headers.get("retry-after", "")
            try:
                state["reset_at"] = now + float(retry_after)
            except (ValueError, TypeError):
                state["reset_at"] = now + 60.0

    def _transcribe_audio_with_fallback(self, audio_path: Path) -> tuple[TranscriptPayload | None, list[str]]:
        errors: list[str] = []
        num_slots = len(self.groq_api_keys) * 2  # both models × all keys
        tried_slots: set[tuple[str, int]] = set()

        for _ in range(num_slots):
            slot = self._select_stt_slot()
            if slot in tried_slots:
                break
            tried_slots.add(slot)
            model_id, key_index = slot
            try:
                payload = self._request_groq_transcription(audio_path, model_id=model_id, key_index=key_index)
            except Exception as exc:  # pragma: no cover - provider variability
                errors.append(f"{model_id}[key={key_index}]: {exc}")
                # Mark slot as exhausted so _select_stt_slot picks another.
                self._stt_rate_states.setdefault((model_id, key_index), {})["remaining_requests"] = 0
                continue

            transcript = _build_transcript_from_stt_payload(payload, model_id=model_id)
            if transcript is not None:
                return transcript, errors
            errors.append(f"{model_id}[key={key_index}]: returned unusable transcription output.")
        return None, errors

    def _request_groq_transcription(self, audio_path: Path, *, model_id: str, key_index: int = 0) -> dict[str, Any]:
        last_error: Exception | None = None
        current_ki = key_index

        for attempt_index in range(MAX_RETRIES):
            api_key = self.groq_api_keys[current_ki] if self.groq_api_keys else ""
            try:
                with audio_path.open("rb") as audio_file:
                    response = self.transcription_transport(
                        GROQ_AUDIO_TRANSCRIPTIONS_URL,
                        headers={"Authorization": f"Bearer {api_key}"},
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

            self._record_stt_response(model_id, current_ki, response)

            if response.status_code in TRANSIENT_STATUS_CODES:
                if response.status_code == 429 and len(self.groq_api_keys) > 1:
                    current_ki = (current_ki + 1) % len(self.groq_api_keys)
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

    def _fetch_feed_entries(self, channel: ChannelTarget) -> list[FlatPlaylistEntry]:
        response = self.http_get(
            YOUTUBE_FEED_URL_TEMPLATE.format(channel_id=channel.channel_id),
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return _parse_youtube_feed(
            response.text,
            channel=channel,
            recent_limit=self.recent_limit,
        )

    def _extract_info(self, url: str, *, cookiefile: Path | None = None) -> dict[str, Any]:
        options = {
            "quiet": True,
            "skip_download": True,
            "noplaylist": True,
            "ignoreerrors": False,
            "user_agent": DEFAULT_USER_AGENT,
            "http_headers": DEFAULT_HTTP_HEADERS,
        }
        if cookiefile is not None:
            options["cookiefile"] = str(cookiefile)
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        if not isinstance(info, dict):
            raise RuntimeError(f"Unable to extract metadata for {url}")
        return info

    def _temporary_cookie_file(self, temp_dir: Path):
        """Creates a Netscape-formatted cookie file with robust repair logic."""
        if not self.yt_cookies:
            return nullcontext(None)
        
        # Robust repair for Netscape format
        raw = self.yt_cookies.strip().strip("'\"")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        
        # Ensure header is present
        header = "# Netscape HTTP Cookie File"
        if not lines or header not in lines[0]:
            lines.insert(0, header)
            lines.insert(1, "# This file was automatically repaired by youtube-intake")
        
        # Write with standard Unix line endings
        cookie_path = temp_dir / "youtube-cookies.txt"
        cookie_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return nullcontext(cookie_path)


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


def _parse_youtube_feed(
    payload: str,
    *,
    channel: ChannelTarget,
    recent_limit: int,
) -> list[FlatPlaylistEntry]:
    root = ET.fromstring(payload)
    entries: list[FlatPlaylistEntry] = []
    for position, entry in enumerate(root.findall("atom:entry", ATOM_NAMESPACE), start=1):
        video_id = _xml_text(entry.find("yt:videoId", ATOM_NAMESPACE))
        if not video_id:
            continue
        title = _xml_text(entry.find("atom:title", ATOM_NAMESPACE))
        published_at = _xml_text(entry.find("atom:published", ATOM_NAMESPACE))
        description = _xml_text(entry.find("media:group/media:description", ATOM_NAMESPACE))
        thumbnail_url = None
        thumbnail = entry.find("media:group/media:thumbnail", ATOM_NAMESPACE)
        if thumbnail is not None:
            thumbnail_url = str(thumbnail.attrib.get("url") or "").strip() or None
        entries.append(
            FlatPlaylistEntry(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                title=title,
                source_tab="rss",
                position=position,
                channel_id=channel.channel_id,
                channel_name=_xml_text(entry.find("atom:author/atom:name", ATOM_NAMESPACE)) or channel.slug,
                published_at=published_at,
                published_timestamp=_parse_feed_published_timestamp(published_at),
                description=description,
                thumbnail_url=thumbnail_url,
            )
        )
        if len(entries) >= recent_limit:
            break
    return entries


def _xml_text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    value = element.text.strip()
    return value or None


def _parse_feed_published_timestamp(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


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


def _select_audio_format_id(
    formats: Iterable[Any], 
    *, 
    max_bytes: int,
    max_duration: int | None = None
) -> str | None:
    """Select the best audio format ID that fits within max_bytes.
    
    If max_duration is provided, estimates file size specifically for that duration.
    """
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

    within_limit = [
        item for item in audio_formats 
        if (_audio_format_size(item, max_duration) or 0) > 0 
        and (_audio_format_size(item, max_duration) or 0) <= max_bytes
    ]
    if within_limit:
        best = max(
            within_limit,
            key=lambda item: (
                _coerce_float(item.get("abr")) or _coerce_float(item.get("tbr")) or 0.0,
                -_audio_ext_priority(item),
                -float(_audio_format_size(item, max_duration) or max_bytes),
            ),
        )
        return str(best.get("format_id") or "").strip() or None

    known_size = [item for item in audio_formats if _audio_format_size(item, max_duration) is not None]
    if known_size:
        smallest = min(
            known_size,
            key=lambda item: (
                _audio_format_size(item, max_duration) or float("inf"),
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


def _audio_format_size(item: dict[str, Any], for_duration: int | None = None) -> int | None:
    """Estimated or actual file size in bytes."""
    if for_duration:
        # abr (average bitrate) is in kbps from yt-dlp.
        abr = _coerce_float(item.get("abr")) or _coerce_float(item.get("tbr"))
        if abr:
            # (abr * 1000 / 8) bytes/sec * duration. Added 5% overhead buffer.
            return int((abr * 1000 / 8) * for_duration * 1.05)
            
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
