# YouTube Intake

`youtube-intake` is the repository's YouTube source-collection layer for finance and market research workflows.

It monitors a configured set of channels, archives newly published videos and completed livestream replays, and stores repo-backed state so GitHub Actions can resume safely between runs.

## Commands

From inside `youtube-intake/`:

```bash
PYTHONPATH=src python -m youtube_intake smoke
PYTHONPATH=src python -m youtube_intake run
PYTHONPATH=src python -m youtube_intake analyze --result-path run-result.json --analysis-result-path analysis-result.json
PYTHONPATH=src python -m youtube_intake notify --result-path analysis-result.json
PYTHONPATH=src python -m youtube_intake test-email
```

## Data Layout

- `config/channels.json`: watched channel list
- `state/channels.json`: per-channel checkpoint manifest
- `data/youtube/<channel-slug>/videos/<video-id>.json`: archived source artifacts
- `data/analysis/<run-timestamp>/videos/<channel-slug>--<video-id>.json`: per-video analysis artifacts
- `data/analysis/<run-timestamp>/run-summary.json`: run-level and channel-level synthesis

## Analysis + Gmail Setup

The workflow now runs in this order:

1. `run`: archive newly detected videos and transcript cues
2. `analyze`: send the current run's new archives to Groq using `meta-llama/llama-4-scout-17b-16e-instruct`
3. `notify`: send one analyst-style Gmail digest for the analyzed items

When YouTube-native transcripts and caption tracks are both unavailable, `run` now tries a Groq speech-to-text fallback before giving up:

- primary STT model: `whisper-large-v3-turbo`
- fallback STT model: `whisper-large-v3`
- duration guardrail: only videos up to 60 minutes are transcribed this way
- successful STT output is normalized into the same `transcript_segments` schema used by YouTube captions
- temporary audio files are deleted immediately after the transcription call finishes

The analyst step is sequential and now includes:

- proactive RPM / TPM pacing per model
- retry and backoff for transient Groq failures
- automatic fallback from `meta-llama/llama-4-scout-17b-16e-instruct` to `llama-3.3-70b-versatile`
- chunked transcript analysis for large videos

### Groq

To enable analysis in GitHub Actions, create this repository secret:

- `GROQ_API_KEY`

For local runs, add this to an untracked `.config` file:

```bash
GROQ_API_KEY=your_groq_api_key
```

`GROQ_API_KEY` is now used by both:

- `run`, for speech-to-text fallback on no-transcript videos
- `analyze`, for Groq LLM summarization

### Gmail

To enable Gmail delivery in GitHub Actions, create these repository secrets:

- `YOUTUBE_INTAKE_GMAIL_SENDER`
- `YOUTUBE_INTAKE_GMAIL_APP_PASSWORD`
- `YOUTUBE_INTAKE_GMAIL_RECIPIENT`

Use a Gmail App Password, not your normal password:

1. Turn on 2-Step Verification for the sender Gmail account.
2. Go to Google Account `Security`.
3. Create an `App Password` for Mail.
4. Store the 16-character password in `YOUTUBE_INTAKE_GMAIL_APP_PASSWORD`.

The workflow sends one compact analyst email per run only when at least one new video was archived and analyzed.

For local runs, you can also use an untracked `.config` file at either:

- repo root: `/Users/henrywzh/Desktop/Quant/equity-research/.config`
- project root: `/Users/henrywzh/Desktop/Quant/equity-research/youtube-intake/.config`

Supported local keys:

```bash
GROQ_API_KEY=your_groq_api_key
YOUTUBE_INTAKE_GMAIL_SENDER=yourname@gmail.com
YOUTUBE_INTAKE_GMAIL_APP_PASSWORD=your_app_password
YOUTUBE_INTAKE_GMAIL_RECIPIENT=yourname@gmail.com
```

For convenience, local `.config` loading also accepts the legacy generic names:

```bash
GMAIL_SENDER=yourname@gmail.com
GMAIL_APP_PASSWORD=your_app_password
GMAIL_RECIPIENT=yourname@gmail.com
```
