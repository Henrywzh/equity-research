# YouTube Intake

`youtube-intake` is the repository's YouTube source-collection layer for finance and market research workflows.

It monitors a configured set of channels, archives newly published videos and completed livestream replays, and stores repo-backed state so GitHub Actions can resume safely between runs.

## Commands

From inside `youtube-intake/`:

```bash
PYTHONPATH=src python -m youtube_intake smoke
PYTHONPATH=src python -m youtube_intake run
```

## Data Layout

- `config/channels.json`: watched channel list
- `state/channels.json`: per-channel checkpoint manifest
- `data/youtube/<channel-slug>/videos/<video-id>.json`: archived source artifacts
