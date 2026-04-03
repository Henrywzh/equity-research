# YouTube Intake

`youtube-intake` is the repository's YouTube source-collection layer for finance and market research workflows.

It monitors a configured set of channels, archives newly published videos and completed livestream replays, and stores repo-backed state so GitHub Actions can resume safely between runs.

## Commands

From inside `youtube-intake/`:

```bash
PYTHONPATH=src python -m youtube_intake smoke
PYTHONPATH=src python -m youtube_intake run
PYTHONPATH=src python -m youtube_intake test-email
```

## Data Layout

- `config/channels.json`: watched channel list
- `state/channels.json`: per-channel checkpoint manifest
- `data/youtube/<channel-slug>/videos/<video-id>.json`: archived source artifacts

## Gmail Delivery Setup

To enable Gmail delivery in GitHub Actions, create these repository secrets:

- `YOUTUBE_INTAKE_GMAIL_SENDER`
- `YOUTUBE_INTAKE_GMAIL_APP_PASSWORD`
- `YOUTUBE_INTAKE_GMAIL_RECIPIENT`

Use a Gmail App Password, not your normal password:

1. Turn on 2-Step Verification for the sender Gmail account.
2. Go to Google Account `Security`.
3. Create an `App Password` for Mail.
4. Store the 16-character password in `YOUTUBE_INTAKE_GMAIL_APP_PASSWORD`.

The workflow sends one compact Gmail summary per run only when new videos were archived.
