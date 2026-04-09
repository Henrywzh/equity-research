from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from .config import get_data_dir
from .profiles import AnalysisProfile, get_profile
from .runtime_env import load_local_config, read_env
from .storage import load_json_document, write_json_document


GROQ_API_KEY_ENV = "GROQ_API_KEY"
PRIMARY_ANALYSIS_MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_ANALYSIS_MODEL_ID = "llama-3.3-70b-versatile"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = 3
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
TRANSIENT_EXCEPTION_TYPES = (
    requests.Timeout,
    requests.ConnectionError,
)
DEFAULT_BACKOFF_SECONDS = (2.0, 5.0, 10.0)


@dataclass(frozen=True)
class ModelLimits:
    rpm: int
    tpm: int
    safe_input_tokens: int
    reserve_output_tokens: int
    chunk_input_tokens: int


@dataclass
class CallResult:
    payload: dict[str, Any]
    model_id: str
    attempts: int
    estimated_input_tokens: int


@dataclass
class HeaderBudget:
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_requests_at: float | None = None
    reset_tokens_at: float | None = None


class RetryExhaustedError(RuntimeError):
    pass


class InputTooLargeForModel(RuntimeError):
    pass


MODEL_LIMITS: dict[str, ModelLimits] = {
    PRIMARY_ANALYSIS_MODEL_ID: ModelLimits(
        rpm=30,
        tpm=30_000,
        safe_input_tokens=22_000,
        reserve_output_tokens=1_200,
        chunk_input_tokens=19_000,
    ),
    FALLBACK_ANALYSIS_MODEL_ID: ModelLimits(
        rpm=30,
        tpm=12_000,
        safe_input_tokens=8_000,
        reserve_output_tokens=1_200,
        chunk_input_tokens=6_200,
    ),
}


class SlidingWindowRateLimiter:
    def __init__(
        self,
        *,
        model_limits: dict[str, ModelLimits],
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.model_limits = model_limits
        self.time_fn = time_fn or time.monotonic
        self.sleep_fn = sleep_fn or time.sleep
        self.event_sink = event_sink
        self.request_windows: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self.header_budgets: dict[str, HeaderBudget] = {}

    def wait_for_capacity(self, model_id: str, estimated_total_tokens: int) -> None:
        while True:
            wait_seconds, reasons = self._project_wait(model_id, estimated_total_tokens)
            if wait_seconds <= 0:
                return
            self._emit(
                {
                    "event_type": "proactive_wait",
                    "model_id": model_id,
                    "wait_seconds": round(wait_seconds, 3),
                    "reasons": reasons,
                }
            )
            self.sleep_fn(wait_seconds)

    def register_request(self, model_id: str, estimated_total_tokens: int) -> None:
        self._purge_expired(model_id)
        self.request_windows[model_id].append((self.time_fn(), estimated_total_tokens))

    def update_from_headers(self, model_id: str, headers: dict[str, Any]) -> None:
        budget = self.header_budgets.setdefault(model_id, HeaderBudget())
        now = self.time_fn()

        budget.remaining_requests = _parse_int_header(headers.get("x-ratelimit-remaining-requests"))
        budget.remaining_tokens = _parse_int_header(headers.get("x-ratelimit-remaining-tokens"))
        request_reset = _parse_duration_header(headers.get("x-ratelimit-reset-requests"))
        token_reset = _parse_duration_header(headers.get("x-ratelimit-reset-tokens"))
        budget.reset_requests_at = now + request_reset if request_reset is not None else budget.reset_requests_at
        budget.reset_tokens_at = now + token_reset if token_reset is not None else budget.reset_tokens_at

    def _project_wait(self, model_id: str, estimated_total_tokens: int) -> tuple[float, list[str]]:
        self._purge_expired(model_id)
        now = self.time_fn()
        limits = self.model_limits[model_id]
        entries = self.request_windows[model_id]
        waits: list[tuple[float, str]] = []

        if len(entries) >= limits.rpm and entries:
            waits.append((max(entries[0][0] + 60.0 - now, 0.0), "local_rpm"))

        current_tokens = sum(tokens for _, tokens in entries)
        if current_tokens + estimated_total_tokens > limits.tpm and entries:
            token_wait = _seconds_until_tokens_fit(entries, limits.tpm, estimated_total_tokens, now)
            waits.append((token_wait, "local_tpm"))

        header_budget = self.header_budgets.get(model_id)
        if header_budget:
            if (
                header_budget.remaining_requests is not None
                and header_budget.remaining_requests <= 0
                and header_budget.reset_requests_at is not None
            ):
                waits.append((max(header_budget.reset_requests_at - now, 0.0), "header_requests"))
            if (
                header_budget.remaining_tokens is not None
                and header_budget.remaining_tokens < estimated_total_tokens
                and header_budget.reset_tokens_at is not None
            ):
                waits.append((max(header_budget.reset_tokens_at - now, 0.0), "header_tokens"))

        if not waits:
            return 0.0, []
        wait_seconds = max(wait for wait, _ in waits)
        reasons = [reason for _, reason in waits if wait_seconds > 0]
        return wait_seconds, reasons

    def _purge_expired(self, model_id: str) -> None:
        now = self.time_fn()
        entries = self.request_windows[model_id]
        while entries and now - entries[0][0] >= 60.0:
            entries.popleft()

        header_budget = self.header_budgets.get(model_id)
        if header_budget is None:
            return
        if header_budget.reset_requests_at is not None and now >= header_budget.reset_requests_at:
            header_budget.remaining_requests = None
            header_budget.reset_requests_at = None
        if header_budget.reset_tokens_at is not None and now >= header_budget.reset_tokens_at:
            header_budget.remaining_tokens = None
            header_budget.reset_tokens_at = None

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_sink:
            self.event_sink(event)


class GroqAnalystClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        primary_model: str = PRIMARY_ANALYSIS_MODEL_ID,
        fallback_model: str = FALLBACK_ANALYSIS_MODEL_ID,
        request_timeout: int = 180,
        sleep_fn: Callable[[float], None] | None = None,
        time_fn: Callable[[], float] | None = None,
        jitter_fn: Callable[[], float] | None = None,
        transport: Callable[..., requests.Response] | None = None,
    ) -> None:
        load_local_config()
        self.api_key = (api_key or read_env(GROQ_API_KEY_ENV)).strip()
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.active_model = primary_model
        self.request_timeout = request_timeout
        self.sleep_fn = sleep_fn or time.sleep
        self.time_fn = time_fn or time.monotonic
        self.jitter_fn = jitter_fn or (lambda: random.uniform(0.0, 0.75))
        self.transport = transport or requests.post
        self.analysis_models_used: list[str] = []
        self.fallback_activated = False
        self.rate_limit_events: list[dict[str, Any]] = []
        self.run_notes: list[str] = []
        self.rate_limiter = SlidingWindowRateLimiter(
            model_limits=MODEL_LIMITS,
            time_fn=self.time_fn,
            sleep_fn=self.sleep_fn,
            event_sink=self._record_rate_limit_event,
        )

    @property
    def model_id(self) -> str:
        return self.active_model

    def _get_video_language_instruction(self, archive: dict[str, Any]) -> str:
        """Resolve the content language from the archive and return an LLM instruction."""
        lang_code = archive.get("transcript_language")
        if not lang_code:
            return (
                "Generate the summary and all descriptive fields in the same language "
                "as the video title and description. Keep JSON keys in English."
            )
        _LANG_NAMES: dict[str, str] = {
            "en": "English",
            "zh": "Chinese",
            "zh-hans": "Simplified Chinese",
            "zh-hant": "Traditional Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
        }
        name = _LANG_NAMES.get(lang_code.lower(), lang_code)
        return (
            f"IMPORTANT: Generate all text/descriptive JSON values "
            f"(executive_summary, signals, findings, etc.) in {name}, "
            f"matching the content language. Keep JSON keys in English."
        )

    def analyze_video(self, archive: dict[str, Any]) -> dict[str, Any]:
        source_basis = _resolve_source_basis(archive)
        profile = get_profile((archive.get("channel") or {}).get("profile"))
        if source_basis == "metadata_only":
            raw_result, call_meta = self._analyze_video_single_pass(archive, source_basis=source_basis, profile=profile)
            normalized = _normalize_video_analysis(raw_result, archive, profile=profile)
            normalized.update(
                {
                    "analysis_model": call_meta.model_id,
                    "analysis_attempts": call_meta.attempts,
                    "analysis_mode": "single_pass",
                }
            )
            return normalized

        force_chunked = False
        total_attempts = 0
        while True:
            current_limits = MODEL_LIMITS[self.active_model]
            transcript_segments = list(archive.get("transcript_segments") or [])
            estimated_input_tokens = _estimate_video_input_tokens(archive)
            should_chunk = force_chunked or estimated_input_tokens > current_limits.safe_input_tokens
            try:
                if should_chunk:
                    raw_result, chunk_attempts, final_meta = self._analyze_video_chunked(
                        archive,
                        source_basis=source_basis,
                        transcript_segments=transcript_segments,
                        current_limits=current_limits,
                        profile=profile,
                    )
                    total_attempts += chunk_attempts
                    normalized = _normalize_video_analysis(raw_result, archive, profile=profile)
                    normalized.update(
                        {
                            "analysis_model": final_meta.model_id,
                            "analysis_attempts": total_attempts,
                            "analysis_mode": "chunked",
                        }
                    )
                    return normalized

                raw_result, call_meta = self._analyze_video_single_pass(archive, source_basis=source_basis, profile=profile)
                total_attempts += call_meta.attempts
                normalized = _normalize_video_analysis(raw_result, archive, profile=profile)
                normalized.update(
                    {
                        "analysis_model": call_meta.model_id,
                        "analysis_attempts": total_attempts,
                        "analysis_mode": "single_pass",
                    }
                )
                return normalized
            except InputTooLargeForModel:
                force_chunked = True

    def synthesize_run(
        self,
        *,
        run_result: dict[str, Any],
        video_analyses: list[dict[str, Any]],
    ) -> dict[str, Any]:
        grouped_channels = defaultdict(list)
        for analysis in video_analyses:
            grouped_channels[analysis["channel_slug"]].append(
                {
                    "title": analysis["title"],
                    "executive_summary": analysis["executive_summary"],
                    "profile": analysis.get("profile", "macroeconomics"),
                    "tickers_mentioned": analysis.get("tickers_mentioned", []),
                    "topic_tags": analysis["topic_tags"],
                    "confidence": analysis["confidence"],
                }
            )

        user_payload = {
            "task": "run_synthesis",
            "run_started_at": run_result.get("run_started_at"),
            "channel_errors": list(run_result.get("errors") or []),
            "videos": [
                {
                    "channel_slug": item["channel_slug"],
                    "channel_name": item["channel_name"],
                    "title": item["title"],
                    "source_kind": item["source_kind"],
                    "profile": item.get("profile", "macroeconomics"),
                    "executive_summary": item["executive_summary"],
                    "tickers_mentioned": item.get("tickers_mentioned", []),
                    "profile_data": item.get("profile_data", {}),
                    "topic_tags": item["topic_tags"],
                    "confidence": item["confidence"],
                    "analysis_model": item.get("analysis_model"),
                }
                for item in video_analyses
            ],
            "output_schema": {
                "overall_day_summary": "string",
                "channel_summaries": [
                    {"channel_slug": "string", "summary": "string", "top_topics": ["string"]}
                ],
                "cross_video_themes": ["string"],
                "agreements": ["string"],
                "disagreements": ["string"],
                "top_claims_worth_watching": ["string"],
                "crowded_trades": ["string"],
                "contrarian_flags": ["string"],
            },
        }

        # Language logic for synthesis:
        # If all videos in this run share the same non-English transcript language, 
        # instruct the synthesizer to output in that language as well.
        _SYNTH_LANG_NAMES: dict[str, str] = {
            "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
            "de": "German", "fr": "French", "es": "Spanish",
            "it": "Italian", "pt": "Portuguese", "ru": "Russian",
        }
        video_languages = {
            (a.get("transcript_language") or "en").lower().split("-")[0]
            for a in video_analyses
        }
        if len(video_languages) == 1 and "en" not in video_languages:
            target_lang = list(video_languages)[0]
            lang_name = _SYNTH_LANG_NAMES.get(target_lang, target_lang)
            language_instruction = f"All content in this run is in {lang_name}. Generate the report and summaries in {lang_name}."
        else:
            language_instruction = "Generate the synthesis in English."

        call_result = self._chat_json(
            system_prompt=(
                "You are a finance research synthesizer. Combine the per-video analyses into a single run-level "
                "summary. Return JSON only. Highlight what matters most for a market watcher. "
                "Identify crowded trades (tickers or themes mentioned by 2+ sources in the same direction). "
                "Flag contrarian views that disagree with the majority consensus. "
                + language_instruction
            ),
            user_payload=user_payload,
        )
        normalized = _normalize_run_synthesis(
            call_result.payload,
            video_analyses,
            dict(grouped_channels),
            [],
        )
        normalized["summary_analysis_model"] = call_result.model_id
        normalized["summary_analysis_attempts"] = call_result.attempts
        return normalized

    def _analyze_video_single_pass(
        self,
        archive: dict[str, Any],
        *,
        source_basis: str,
        profile: AnalysisProfile,
    ) -> tuple[dict[str, Any], CallResult]:
        transcript_available = source_basis == "transcript"
        transcript_payload = {
            "segments": archive.get("transcript_segments") or [],
            "text": archive.get("transcript_text"),
        }
        user_payload = {
            "task": "video_analysis",
            "requirements": {
                "focus": list(profile.build_full_schema().keys()),
                "language_policy": self._get_video_language_instruction(archive),
                "timestamp_policy": (
                    "Only include key_timestamps when transcript segments are present. "
                    "Use direct evidence from transcript cues."
                ),
                "metadata_only_policy": (
                    "If there is no transcript, analyze title and description only, lower confidence, "
                    "and return an empty key_timestamps list."
                ),
            },
            "video": {
                "channel": archive.get("channel"),
                "video": archive.get("video"),
                "published_at": archive.get("published_at"),
                "description": archive.get("description"),
                "source_kind": archive.get("source_kind"),
                "transcript_status": archive.get("transcript_status"),
                "analysis_input_basis": source_basis,
                "transcript": transcript_payload if transcript_available else None,
            },
            "output_schema": profile.build_full_schema(),
        }
        call_result = self._chat_json(
            system_prompt=profile.system_prompt,
            user_payload=user_payload,
        )
        return call_result.payload, call_result

    def _analyze_video_chunked(
        self,
        archive: dict[str, Any],
        *,
        source_basis: str,
        transcript_segments: list[dict[str, Any]],
        current_limits: ModelLimits,
        profile: AnalysisProfile,
    ) -> tuple[dict[str, Any], int, CallResult]:
        chunk_token_limit = current_limits.chunk_input_tokens
        chunk_schema = dict(profile.build_full_schema())
        chunk_schema.pop("confidence", None)
        while True:
            chunks = _chunk_transcript_segments(
                transcript_segments,
                chunk_token_limit=chunk_token_limit,
            )
            if len(chunks) <= 1:
                if chunk_token_limit <= 400:
                    raise InputTooLargeForModel("Chunking could not reduce the transcript below the active model budget.")
                chunk_token_limit = max(chunk_token_limit // 2, 400)
                continue

            chunk_outputs: list[dict[str, Any]] = []
            total_attempts = 0
            try:
                lang_instruction = self._get_video_language_instruction(archive)
                for index, chunk in enumerate(chunks, start=1):
                    call_result = self._chat_json(
                        system_prompt=f"{profile.chunk_system_prompt} {lang_instruction}",
                        user_payload={
                            "task": "video_chunk_analysis",
                            "chunk_index": index,
                            "chunk_count": len(chunks),
                            "video": {
                                "channel": archive.get("channel"),
                                "video": archive.get("video"),
                                "published_at": archive.get("published_at"),
                                "description": archive.get("description"),
                                "source_kind": archive.get("source_kind"),
                                "transcript_status": archive.get("transcript_status"),
                                "analysis_input_basis": source_basis,
                            },
                            "chunk": {"segments": chunk},
                            "requirements": {"language_policy": lang_instruction},
                            "output_schema": chunk_schema,
                        },
                    )
                    total_attempts += call_result.attempts
                    chunk_outputs.append(call_result.payload)

                final_result = self._chat_json(
                    system_prompt=f"{profile.consolidation_system_prompt} {lang_instruction}",
                    user_payload={
                        "task": "video_chunk_consolidation",
                        "video": {
                            "channel": archive.get("channel"),
                            "video": archive.get("video"),
                            "published_at": archive.get("published_at"),
                            "description": archive.get("description"),
                            "source_kind": archive.get("source_kind"),
                            "transcript_status": archive.get("transcript_status"),
                            "analysis_input_basis": source_basis,
                        },
                        "chunk_analyses": chunk_outputs,
                        "requirements": {"language_policy": lang_instruction},
                        "output_schema": profile.build_full_schema(),
                    },
                )
                total_attempts += final_result.attempts
                return final_result.payload, total_attempts, final_result
            except InputTooLargeForModel:
                if chunk_token_limit <= 400:
                    raise
                chunk_token_limit = max(chunk_token_limit // 2, 400)

    def _chat_json(self, *, system_prompt: str, user_payload: dict[str, Any]) -> CallResult:
        if not self.api_key:
            raise RuntimeError(f"{GROQ_API_KEY_ENV} is required for youtube-intake analysis.")

        user_content = json.dumps(user_payload, ensure_ascii=False)
        estimated_input_tokens = _estimate_text_tokens(system_prompt) + _estimate_text_tokens(user_content)

        for model_id in self._candidate_models():
            limits = MODEL_LIMITS[model_id]
            if estimated_input_tokens > limits.safe_input_tokens:
                if model_id == self.primary_model and self.active_model == self.primary_model and self.fallback_model:
                    self._activate_fallback(
                        reason=(
                            f"Primary model input was too large for fallback-safe single pass; "
                            f"rerouting remaining work to {self.fallback_model}."
                        )
                    )
                    raise InputTooLargeForModel(f"Input too large for {model_id}; reroute through chunking.")
                raise InputTooLargeForModel(f"Estimated input {estimated_input_tokens} exceeds safe budget for {model_id}.")

            estimated_total_tokens = estimated_input_tokens + limits.reserve_output_tokens
            if estimated_total_tokens > limits.tpm:
                if model_id == self.primary_model and self.active_model == self.primary_model and self.fallback_model:
                    self._activate_fallback(
                        reason=f"Primary request budget exceeded; switching remaining work to {self.fallback_model}."
                    )
                    raise InputTooLargeForModel(f"Estimated request tokens exceed per-minute budget for {model_id}.")
                raise InputTooLargeForModel(f"Estimated request tokens exceed per-minute budget for {model_id}.")

            try:
                payload, attempts = self._send_request_with_retries(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    estimated_input_tokens=estimated_input_tokens,
                    estimated_total_tokens=estimated_total_tokens,
                    reserve_output_tokens=limits.reserve_output_tokens,
                )
                self._mark_model_used(model_id)
                return CallResult(
                    payload=payload,
                    model_id=model_id,
                    attempts=attempts,
                    estimated_input_tokens=estimated_input_tokens,
                )
            except RetryExhaustedError as exc:
                if model_id == self.primary_model and self.active_model == self.primary_model and self.fallback_model:
                    self._activate_fallback(
                        reason=(
                            f"Primary model {self.primary_model} exhausted retries; "
                            f"switching remaining work to {self.fallback_model}."
                        )
                    )
                    continue
                raise exc

        raise RetryExhaustedError("No analysis model succeeded for this request.")

    def _candidate_models(self) -> list[str]:
        if self.active_model == self.primary_model and self.fallback_model:
            return [self.primary_model, self.fallback_model]
        return [self.active_model]

    def _send_request_with_retries(
        self,
        *,
        model_id: str,
        system_prompt: str,
        user_content: str,
        estimated_input_tokens: int,
        estimated_total_tokens: int,
        reserve_output_tokens: int,
    ) -> tuple[dict[str, Any], int]:
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            self.rate_limiter.wait_for_capacity(model_id, estimated_total_tokens)
            self.rate_limiter.register_request(model_id, estimated_total_tokens)
            try:
                response = self.transport(
                    GROQ_CHAT_COMPLETIONS_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "temperature": 0.2,
                        "max_completion_tokens": reserve_output_tokens,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                    },
                    timeout=self.request_timeout,
                )
            except TRANSIENT_EXCEPTION_TYPES as exc:
                last_error = exc
                if attempt >= MAX_RETRIES:
                    raise RetryExhaustedError(f"{model_id} transient connection failure after {attempt} attempts: {exc}") from exc
                wait_seconds = DEFAULT_BACKOFF_SECONDS[min(attempt - 1, len(DEFAULT_BACKOFF_SECONDS) - 1)] + self.jitter_fn()
                self._record_rate_limit_event(
                    {
                        "event_type": "transient_exception",
                        "model_id": model_id,
                        "attempt": attempt,
                        "wait_seconds": round(wait_seconds, 3),
                        "detail": str(exc),
                    }
                )
                self.sleep_fn(wait_seconds)
                continue

            self.rate_limiter.update_from_headers(model_id, dict(response.headers))
            if response.status_code in TRANSIENT_STATUS_CODES:
                last_error = RetryExhaustedError(f"{model_id} returned HTTP {response.status_code}")
                wait_seconds = _compute_retry_wait_seconds(response, attempt, jitter_fn=self.jitter_fn)
                self._record_rate_limit_event(
                    {
                        "event_type": "http_retry",
                        "model_id": model_id,
                        "status_code": response.status_code,
                        "attempt": attempt,
                        "wait_seconds": round(wait_seconds, 3),
                    }
                )
                if attempt >= MAX_RETRIES:
                    raise RetryExhaustedError(
                        f"{model_id} returned HTTP {response.status_code} after {attempt} attempts."
                    )
                self.sleep_fn(wait_seconds)
                continue

            response.raise_for_status()
            payload = response.json()
            content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = _extract_json_object(content)
            if not isinstance(parsed, dict):
                raise RuntimeError("Groq response was not a JSON object.")
            return parsed, attempt

        raise RetryExhaustedError(str(last_error) if last_error else "Groq request failed.")

    def _mark_model_used(self, model_id: str) -> None:
        if model_id not in self.analysis_models_used:
            self.analysis_models_used.append(model_id)

    def _activate_fallback(self, *, reason: str) -> None:
        if not self.fallback_model or self.active_model == self.fallback_model:
            return
        self.active_model = self.fallback_model
        self.fallback_activated = True
        note = reason.strip()
        if note and note not in self.run_notes:
            self.run_notes.append(note)

    def _record_rate_limit_event(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", _utc_now())
        self.rate_limit_events.append(payload)


def analyze_run(
    *,
    result_path: str | Path,
    analysis_result_path: str | Path,
    data_dir: str | Path | None = None,
    client: GroqAnalystClient | None = None,
) -> dict[str, Any]:
    resolved_result_path = Path(result_path).expanduser().resolve()
    resolved_analysis_result_path = Path(analysis_result_path).expanduser().resolve()
    run_result = load_json_document(resolved_result_path)
    resolved_data_dir = get_data_dir(data_dir)
    client = client or GroqAnalystClient()

    analysis_started_at = run_result.get("analysis_started_at") or _utc_now()
    run_key = _make_run_key(str(run_result.get("run_started_at") or analysis_started_at))
    analysis_root = resolved_data_dir / "analysis" / run_key
    videos_dir = analysis_root / "videos"
    summary_path = analysis_root / "run-summary.json"

    result: dict[str, Any] = {
        "status": "success",
        "run_started_at": run_result.get("run_started_at"),
        "analysis_started_at": analysis_started_at,
        "analysis_completed_at": None,
        "analysis_model": getattr(client, "model_id", PRIMARY_ANALYSIS_MODEL_ID),
        "analysis_models_used": [],
        "fallback_activated": False,
        "rate_limit_events": [],
        "analysis_artifact_dir": str(analysis_root),
        "summary_analysis_path": str(summary_path),
        "summary_analysis_model": None,
        "summary_analysis_attempts": 0,
        "videos": [],
        "channels": {},
        "run_summary": {},
        "errors": [],
    }

    new_items = list(run_result.get("new_items") or [])
    if not new_items:
        result["status"] = "noop"
        result["analysis_completed_at"] = _utc_now()
        result["run_summary"] = {
            "overall_day_summary": "No newly archived videos were available for analysis.",
            "cross_video_themes": [],
            "agreements": [],
            "disagreements": [],
            "top_claims_worth_watching": [],
            "run_notes": [],
        }
        result["analysis_models_used"] = list(getattr(client, "analysis_models_used", []))
        result["fallback_activated"] = bool(getattr(client, "fallback_activated", False))
        result["rate_limit_events"] = list(getattr(client, "rate_limit_events", []))
        write_json_document(resolved_analysis_result_path, result)
        _update_run_result_with_analysis_paths(
            resolved_result_path,
            run_result,
            analysis_result_path=resolved_analysis_result_path,
            analysis_artifact_dir=analysis_root,
        )
        return result

    for item in new_items:
        archive_path = item.get("archive_path")
        if not archive_path:
            result["errors"].append(f"Missing archive path for {item.get('video_id') or 'unknown video'}.")
            result["status"] = "partial_success"
            continue
        try:
            archive = load_json_document(archive_path)
            maybe_analysis = client.analyze_video(archive)
            video_analysis = (
                maybe_analysis
                if "channel_slug" in maybe_analysis and "analysis_model" in maybe_analysis
                else _normalize_video_analysis(maybe_analysis, archive)
            )
            video_analysis.update(
                {
                    "archive_path": str(Path(archive_path).expanduser().resolve()),
                    "analysis_path": str(
                        write_json_document(
                            videos_dir / f"{video_analysis['channel_slug']}--{video_analysis['video_id']}.json",
                            video_analysis,
                        )
                    ),
                }
            )
            result["videos"].append(video_analysis)
        except Exception as exc:  # pragma: no cover - API/provider variability
            result["errors"].append(f"{item.get('channel_slug') or 'unknown'}:{item.get('video_id') or 'unknown'}: {exc}")
            result["status"] = "partial_success"

    if result["videos"]:
        try:
            run_summary = client.synthesize_run(run_result=run_result, video_analyses=result["videos"])
            result["channels"] = run_summary["channels"]
            result["run_summary"] = run_summary["run_summary"]
            result["summary_analysis_model"] = run_summary.get("summary_analysis_model")
            result["summary_analysis_attempts"] = run_summary.get("summary_analysis_attempts", 0)
            write_json_document(summary_path, run_summary)
        except Exception as exc:  # pragma: no cover - API/provider variability
            result["errors"].append(f"run-summary: {exc}")
            result["status"] = "partial_success"
            result["run_summary"] = {
                "overall_day_summary": "Analysis completed for some videos, but the run-level synthesis failed.",
                "cross_video_themes": [],
                "agreements": [],
                "disagreements": [],
                "top_claims_worth_watching": [],
                "run_notes": [],
            }
    else:
        result["run_summary"] = {
            "overall_day_summary": "Analysis could not be generated for the current run.",
            "cross_video_themes": [],
            "agreements": [],
            "disagreements": [],
            "top_claims_worth_watching": [],
            "run_notes": [],
        }

    run_notes = list(result["run_summary"].get("run_notes") or [])
    run_notes.extend(str(item) for item in getattr(client, "run_notes", []))
    for video in result["videos"]:
        for note in video.get("analysis_notes") or []:
            run_notes.append(f"{video.get('channel_slug')}/{video.get('video_id')}: {note}")
    if result["errors"]:
        run_notes.extend(result["errors"])
    result["run_summary"]["run_notes"] = _dedupe_preserve_order(run_notes)

    result["analysis_model"] = getattr(client, "model_id", result["analysis_model"])
    result["analysis_models_used"] = list(getattr(client, "analysis_models_used", []))
    result["fallback_activated"] = bool(getattr(client, "fallback_activated", False))
    result["rate_limit_events"] = list(getattr(client, "rate_limit_events", []))
    result["analysis_completed_at"] = _utc_now()
    write_json_document(resolved_analysis_result_path, result)
    _update_run_result_with_analysis_paths(
        resolved_result_path,
        run_result,
        analysis_result_path=resolved_analysis_result_path,
        analysis_artifact_dir=analysis_root,
    )
    return result


def _update_run_result_with_analysis_paths(
    result_path: Path,
    run_result: dict[str, Any],
    *,
    analysis_result_path: Path,
    analysis_artifact_dir: Path,
) -> None:
    updated = dict(run_result)
    updated["analysis_result_path"] = str(analysis_result_path)
    updated["analysis_artifact_dir"] = str(analysis_artifact_dir)
    write_json_document(result_path, updated)


def _normalize_video_analysis(raw: dict[str, Any], archive: dict[str, Any], *, profile: AnalysisProfile | None = None) -> dict[str, Any]:
    channel = archive.get("channel") or {}
    video = archive.get("video") or {}
    source_basis = _resolve_source_basis(archive)
    confidence = _coerce_confidence(raw.get("confidence"), fallback=0.78 if source_basis == "transcript" else 0.42)
    if source_basis == "metadata_only":
        confidence = min(confidence, 0.55)
    raw_key_timestamps = raw.get("key_timestamps")
    key_timestamps = _normalize_key_timestamps(raw_key_timestamps, archive=archive)
    analysis_notes: list[str] = []
    transcript_note = _extract_transcript_fallback_note(archive, source_basis=source_basis)
    if transcript_note:
        analysis_notes.append(transcript_note)
    if source_basis == "transcript" and isinstance(raw_key_timestamps, list) and raw_key_timestamps and not key_timestamps:
        analysis_notes.append(
            "Dropped model-provided key timestamps because they could not be validated against the transcript timeline."
        )

    resolved_profile = profile or get_profile(channel.get("profile"))
    profile_data: dict[str, Any] = {}
    for field_name in resolved_profile.extra_fields:
        value = raw.get(field_name)
        if isinstance(value, list):
            profile_data[field_name] = _normalize_string_list(value, limit=6)
        elif isinstance(value, str):
            profile_data[field_name] = _normalize_text(value, fallback="")
        elif value is not None:
            profile_data[field_name] = value

    return {
        "video_id": str(video.get("video_id") or ""),
        "channel_slug": str(channel.get("slug") or ""),
        "channel_name": channel.get("channel_name") or channel.get("slug"),
        "title": video.get("title") or "Untitled",
        "webpage_url": video.get("webpage_url"),
        "published_at": archive.get("published_at"),
        "source_kind": archive.get("source_kind"),
        "transcript_status": archive.get("transcript_status"),
        "transcript_language": archive.get("transcript_language"),
        "source_basis": source_basis,
        "profile": resolved_profile.name,
        "synthesis_section": resolved_profile.synthesis_section,
        "executive_summary": _normalize_text(
            raw.get("executive_summary") or raw.get("summary"),
            fallback=(archive.get("description") or "No summary generated."),
        ),
        "tickers_mentioned": _normalize_string_list(raw.get("tickers_mentioned"), limit=10),
        "profile_data": profile_data,
        "key_timestamps": key_timestamps,
        "topic_tags": _normalize_topic_tags(raw.get("topic_tags")),
        "confidence": confidence,
        "analysis_notes": analysis_notes,
    }


def _normalize_run_synthesis(
    raw: dict[str, Any],
    video_analyses: list[dict[str, Any]],
    grouped_channels: dict[str, list[dict[str, Any]]],
    run_errors: list[str],
) -> dict[str, Any]:
    channel_name_by_slug = {
        analysis["channel_slug"]: analysis["channel_name"]
        for analysis in video_analyses
        if analysis.get("channel_slug")
    }
    channel_summaries = raw.get("channel_summaries") or raw.get("channels") or []
    channel_summary_map: dict[str, dict[str, Any]] = {}
    if isinstance(channel_summaries, dict):
        iterable = channel_summaries.items()
    else:
        iterable = [
            (item.get("channel_slug"), item)
            for item in channel_summaries
            if isinstance(item, dict) and item.get("channel_slug")
        ]

    for slug, item in iterable:
        if not slug:
            continue
        channel_summary_map[str(slug)] = {
            "channel_name": channel_name_by_slug.get(str(slug), str(slug)),
            "video_count": len(grouped_channels.get(str(slug), [])),
            "summary": _normalize_text(item.get("summary"), fallback=""),
            "top_topics": _normalize_string_list(item.get("top_topics"), limit=4),
        }

    for slug, analyses in grouped_channels.items():
        channel_summary_map.setdefault(
            slug,
            {
                "channel_name": channel_name_by_slug.get(slug, slug),
                "video_count": len(analyses),
                "summary": analyses[0].get("executive_summary", "") if analyses else "",
                "top_topics": _derive_top_topics_from_analyses(analyses),
            },
        )

    return {
        "channels": channel_summary_map,
        "run_summary": {
            "overall_day_summary": _normalize_text(
                raw.get("overall_day_summary") or raw.get("executive_summary"),
                fallback="No run summary generated.",
            ),
            "cross_video_themes": _normalize_string_list(raw.get("cross_video_themes"), limit=5),
            "agreements": _normalize_string_list(raw.get("agreements"), limit=4),
            "disagreements": _normalize_string_list(raw.get("disagreements"), limit=4),
            "top_claims_worth_watching": _normalize_string_list(raw.get("top_claims_worth_watching"), limit=5),
            "crowded_trades": _normalize_string_list(raw.get("crowded_trades"), limit=5),
            "contrarian_flags": _normalize_string_list(raw.get("contrarian_flags"), limit=4),
            "run_notes": list(run_errors),
        },
    }


def _normalize_key_timestamps(value: Any, *, archive: dict[str, Any]) -> list[dict[str, str]]:
    source_basis = _resolve_source_basis(archive)
    if source_basis != "transcript":
        return []
    if not isinstance(value, list):
        return []

    video = archive.get("video") or {}
    duration_seconds = _coerce_duration_seconds(video.get("duration_seconds"))
    transcript_segments = list(archive.get("transcript_segments") or [])
    normalized: list[dict[str, str]] = []
    seen_timestamps: set[int] = set()
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        resolved_seconds = _resolve_key_timestamp_seconds(
            item,
            duration_seconds=duration_seconds,
            transcript_segments=transcript_segments,
        )
        if resolved_seconds is None or resolved_seconds in seen_timestamps:
            continue
        seen_timestamps.add(resolved_seconds)
        label = _normalize_text(item.get("label"), fallback="")
        snippet = _normalize_text(item.get("snippet"), fallback="")
        why = _normalize_text(item.get("why_it_matters"), fallback="")
        normalized.append(
            {
                "timestamp": _format_timestamp_seconds(resolved_seconds),
                "label": label,
                "snippet": snippet,
                "why_it_matters": why,
            }
        )
    return normalized


def _normalize_topic_tags(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value[:6]:
        if isinstance(item, dict):
            tag = _normalize_text(item.get("tag"), fallback="")
            score = _coerce_score(item.get("score"))
        else:
            tag = _normalize_text(item, fallback="")
            score = 50
        if not tag:
            continue
        normalized.append({"tag": tag, "score": score})
    return normalized


def _extract_transcript_fallback_note(archive: dict[str, Any], *, source_basis: str) -> str | None:
    transcript_source = str(archive.get("transcript_source") or "").strip()
    transcript_error = _normalize_text(archive.get("error"), fallback="")
    if transcript_source.startswith("groq_whisper_"):
        model_label = transcript_source.removeprefix("groq_").replace("_", "-")
        return f"Transcript was generated via Groq STT fallback ({model_label}) because YouTube captions were unavailable."
    if source_basis != "metadata_only" or not transcript_error:
        return None

    for part in (piece.strip() for piece in transcript_error.split(";")):
        lowered = part.lower()
        if lowered.startswith("groq stt skipped due to duration limit"):
            return part
        if lowered.startswith("groq stt skipped because"):
            return part
        if lowered.startswith("whisper-large-v3-turbo:") or lowered.startswith("whisper-large-v3:"):
            return "Groq STT fallback failed, so this video was analyzed from metadata only."
    return None


def _derive_top_topics_from_analyses(analyses: list[dict[str, Any]]) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for analysis in analyses:
        for item in analysis.get("topic_tags") or []:
            tag = _normalize_text(item.get("tag"), fallback="")
            if tag:
                ranked.append((_coerce_score(item.get("score")), tag))
    ranked.sort(reverse=True)
    seen: set[str] = set()
    topics: list[str] = []
    for _score, tag in ranked:
        if tag in seen:
            continue
        seen.add(tag)
        topics.append(tag)
        if len(topics) >= 4:
            break
    return topics


def _normalize_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value[:limit]:
        text = _normalize_text(item, fallback="")
        if text:
            items.append(text)
    return items


def _normalize_text(value: Any, *, fallback: str) -> str:
    if value is None:
        return fallback
    text = " ".join(str(value).split())
    return text if text else fallback


def _normalize_timestamp_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        total_seconds = max(int(float(value)), 0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    text = _normalize_text(value, fallback="")
    if not text:
        return ""
    if text.count(":") == 1:
        return f"00:{text}"
    return text


def _resolve_key_timestamp_seconds(
    item: dict[str, Any],
    *,
    duration_seconds: int | None,
    transcript_segments: list[dict[str, Any]],
) -> int | None:
    start_seconds = _coerce_start_seconds(item.get("start_seconds"))
    if _is_valid_timestamp_seconds(start_seconds, duration_seconds):
        return start_seconds

    timestamp_seconds = _parse_model_timestamp_seconds(item.get("timestamp") or item.get("time"))
    if _is_valid_timestamp_seconds(timestamp_seconds, duration_seconds):
        return timestamp_seconds

    snippet = _normalize_text(item.get("snippet"), fallback="")
    matched_segment_seconds = _match_snippet_to_segment_seconds(snippet, transcript_segments, duration_seconds=duration_seconds)
    if _is_valid_timestamp_seconds(matched_segment_seconds, duration_seconds):
        return matched_segment_seconds
    return None


def _parse_model_timestamp_seconds(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _coerce_start_seconds(value)

    text = _normalize_text(value, fallback="")
    if not text:
        return None
    parts = text.split(":")
    if len(parts) == 2:
        minutes = _safe_int(parts[0])
        seconds = _safe_int(parts[1])
        if minutes is None or seconds is None or seconds >= 60:
            return None
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours = _safe_int(parts[0])
        minutes = _safe_int(parts[1])
        seconds = _safe_int(parts[2])
        if hours is None or minutes is None or seconds is None or minutes >= 60 or seconds >= 60:
            return None
        return hours * 3600 + minutes * 60 + seconds
    return None


def _coerce_start_seconds(value: Any) -> int | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return int(round(parsed))


def _coerce_duration_seconds(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _is_valid_timestamp_seconds(value: int | None, duration_seconds: int | None) -> bool:
    if value is None or value < 0:
        return False
    if duration_seconds is None:
        return True
    return value <= duration_seconds


def _format_timestamp_seconds(value: int) -> str:
    total_seconds = max(int(value), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _match_snippet_to_segment_seconds(
    snippet: str,
    transcript_segments: list[dict[str, Any]],
    *,
    duration_seconds: int | None,
) -> int | None:
    if not snippet or not transcript_segments:
        return None

    snippet_text = _normalize_match_text(snippet)
    if not snippet_text:
        return None
    snippet_tokens = _match_tokens(snippet_text)
    if not snippet_tokens:
        return None

    best_score = 0.0
    best_seconds: int | None = None
    for segment in transcript_segments:
        segment_text = _normalize_match_text(segment.get("text") or "")
        if not segment_text:
            continue
        score = _segment_match_score(snippet_text, snippet_tokens, segment_text)
        if score <= best_score:
            continue
        candidate_seconds = _coerce_start_seconds(segment.get("start_seconds"))
        if not _is_valid_timestamp_seconds(candidate_seconds, duration_seconds):
            continue
        best_score = score
        best_seconds = candidate_seconds

    if best_score < 0.45:
        return None
    return best_seconds


def _segment_match_score(snippet_text: str, snippet_tokens: set[str], segment_text: str) -> float:
    if snippet_text in segment_text or segment_text in snippet_text:
        return 1.0
    segment_tokens = _match_tokens(segment_text)
    if not segment_tokens:
        return 0.0
    overlap = len(snippet_tokens & segment_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(snippet_tokens))


def _normalize_match_text(value: str) -> str:
    import re

    return re.sub(r"\s+", " ", str(value).strip().lower())


def _match_tokens(value: str) -> set[str]:
    import re

    return {token for token in re.findall(r"\w+", value) if len(token) >= 2}


def _safe_int(value: str) -> int | None:
    if not value.isdigit():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _resolve_source_basis(archive: dict[str, Any]) -> str:
    explicit = archive.get("analysis_input_basis")
    if explicit in {"transcript", "metadata_only"}:
        return explicit
    if archive.get("transcript_segments") or archive.get("transcript_text"):
        return "transcript"
    return "metadata_only"


def _coerce_confidence(value: Any, *, fallback: float) -> float:
    try:
        parsed = float(value)
        return max(0.0, min(parsed, 1.0))
    except (TypeError, ValueError):
        return fallback


def _coerce_score(value: Any) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return 50
    return max(0, min(parsed, 100))


def _extract_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 1)[1]
        cleaned = cleaned.replace("json", "", 1).strip()
        if "```" in cleaned:
            cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _estimate_text_tokens(value: str | None) -> int:
    if not value:
        return 0
    return max(1, math.ceil(len(value.encode("utf-8")) / 4))


def _estimate_video_input_tokens(archive: dict[str, Any]) -> int:
    payload = {
        "channel": archive.get("channel"),
        "video": archive.get("video"),
        "published_at": archive.get("published_at"),
        "description": archive.get("description"),
        "source_kind": archive.get("source_kind"),
        "transcript_status": archive.get("transcript_status"),
        "analysis_input_basis": _resolve_source_basis(archive),
        "transcript_segments": archive.get("transcript_segments") or [],
        "transcript_text": archive.get("transcript_text"),
    }
    return _estimate_text_tokens(json.dumps(payload, ensure_ascii=False))


def _chunk_transcript_segments(
    transcript_segments: list[dict[str, Any]],
    *,
    chunk_token_limit: int,
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []
    current_tokens = 0
    for segment in transcript_segments:
        segment_tokens = _estimate_text_tokens(segment.get("text") or "")
        if current_chunk and current_tokens + segment_tokens > chunk_token_limit:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(segment)
        current_tokens += segment_tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _join_chunk_text(chunk: list[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("text") or "").strip() for item in chunk if str(item.get("text") or "").strip())


def _seconds_until_tokens_fit(
    entries: deque[tuple[float, int]],
    token_budget: int,
    estimated_total_tokens: int,
    now: float,
) -> float:
    running_total = sum(tokens for _, tokens in entries)
    if running_total + estimated_total_tokens <= token_budget:
        return 0.0

    for timestamp, tokens in entries:
        running_total -= tokens
        if running_total + estimated_total_tokens <= token_budget:
            return max(timestamp + 60.0 - now, 0.0)
    return max(entries[0][0] + 60.0 - now, 0.0)


def _parse_int_header(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_duration_header(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("s") and text[:-1].replace(".", "", 1).isdigit():
        try:
            return float(text[:-1])
        except ValueError:
            return None
    total_seconds = 0.0
    remainder = text
    if "m" in remainder:
        minutes_text, remainder = remainder.split("m", 1)
        try:
            total_seconds += float(minutes_text) * 60.0
        except ValueError:
            return None
    remainder = remainder.strip()
    if remainder.endswith("s"):
        try:
            total_seconds += float(remainder[:-1])
        except ValueError:
            return None
        return total_seconds
    return None


def _compute_retry_wait_seconds(
    response: requests.Response,
    attempt: int,
    *,
    jitter_fn: Callable[[], float],
) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass

    token_reset = _parse_duration_header(response.headers.get("x-ratelimit-reset-tokens"))
    request_reset = _parse_duration_header(response.headers.get("x-ratelimit-reset-requests"))
    if response.status_code == 429:
        candidates = [value for value in (token_reset, request_reset) if value is not None]
        if candidates:
            return max(candidates)

    return DEFAULT_BACKOFF_SECONDS[min(attempt - 1, len(DEFAULT_BACKOFF_SECONDS) - 1)] + jitter_fn()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _make_run_key(value: str) -> str:
    return (
        value.replace(":", "-")
        .replace("+", "plus")
        .replace("/", "-")
        .replace(" ", "_")
    )
