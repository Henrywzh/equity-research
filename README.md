# Equity Research Workspace

This repository is a working space for daily finance summaries, equity research, macro research, and related data collection or monitoring tools.

The current projects focus on:
- collecting structured news inputs for daily market monitoring
- experimenting with research-oriented automation and scraping workflows
- monitoring event-driven signals that may matter for markets or geopolitics

## Projects

### `daily-macro/`

A reusable news scraping and storage project for daily research workflows.

Current focus:
- scrape HKEJ `instantnews`
- capture featured stories and the `最新` section
- normalize articles into SQLite
- store compact parsed JSON article backups
- support scheduled GitHub Actions runs

See [daily-macro/README.md](/Users/henrywzh/Desktop/Quant/equity-research/daily-macro/README.md) for commands and data layout.

### `marine-traffic-monitor/`

A project-specific maritime OSINT monitor for the Strait of Hormuz. It combines browser scraping, computer vision, rule-based alerting, and email delivery to track potentially meaningful vessel activity.

See [marine-traffic-monitor/PROJECT_README.md](/Users/henrywzh/Desktop/Quant/equity-research/marine-traffic-monitor/PROJECT_README.md) for the full project guide.

### `youtube-intake/`

A scheduled YouTube intake, transcript archiving, Groq-based video analysis, and Gmail delivery project for finance and market research workflows. It monitors configured channels, archives new videos and livestream replays into repo-backed JSON artifacts, produces per-video and run-level analyst summaries, and can send one compact Gmail digest when new content appears.

See [youtube-intake/README.md](/Users/henrywzh/Desktop/Quant/equity-research/youtube-intake/README.md) for commands, local config setup, and GitHub Actions email delivery details.

### Shared or supporting files

- `models.py`: model registry / shared model-selection utilities used by some experiments
- `not_in_use/`: archived or inactive scripts kept for reference rather than active workflows

## Repository Structure

```text
equity-research/
├── daily-macro/              # Daily news scraping and structured storage
├── marine-traffic-monitor/   # Maritime monitoring and alerting workflow
├── youtube-intake/           # YouTube channel intake and Gmail delivery
├── not_in_use/               # Archived or inactive experiments
├── models.py                 # Shared model-selection utilities
└── README.md                 # Repo-level overview
```

## Getting Started

Start from the subproject you want to work on:

- For daily news scraping and structured storage, go to [daily-macro/README.md](/Users/henrywzh/Desktop/Quant/equity-research/daily-macro/README.md)
- For the maritime monitor, go to [marine-traffic-monitor/PROJECT_README.md](/Users/henrywzh/Desktop/Quant/equity-research/marine-traffic-monitor/PROJECT_README.md)
- For scheduled YouTube channel intake and Gmail summaries, go to [youtube-intake/README.md](/Users/henrywzh/Desktop/Quant/equity-research/youtube-intake/README.md)
- For scheduled YouTube channel intake, Groq analysis, and Gmail delivery, go to [youtube-intake/README.md](/Users/henrywzh/Desktop/Quant/equity-research/youtube-intake/README.md)

This root README is intentionally broad. Project-specific setup, commands, and operational details live inside each subfolder’s own documentation.
