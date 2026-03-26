# Quantitative Equity Research — OSINT & Intelligence Tools

A collection of automated intelligence tools built for quantitative and geopolitical market research. The flagship project is a real-time Maritime OSINT monitor for the Strait of Hormuz.

---

## Projects

### `marine-traffic-monitor/` — Strait of Hormuz Maritime Monitor

An automated pipeline that continuously monitors vessel activity in the Strait of Hormuz and delivers structured intelligence alerts to your inbox.

**How it works:**
1. **Scrapes** live vessel-tracking maps every 15 minutes via Playwright
2. **Detects** ships inside the transit corridor using OpenCV computer vision
3. **Analyses** anomalies using two independent LLMs (vision + text) that produce structured evidence — not verdicts
4. **Decides** alert levels via a deterministic policy engine (zero LLM involvement in the final decision)
5. **Delivers** rich HTML email alerts with evidence breakdowns, hypothesis confidence scores, and geo-political news context

> **Design principle:** LLMs produce observations and hypotheses. A pure-Python policy engine makes all escalation decisions. A hallucinating model cannot trigger an alert alone.

---

## Repository Structure

```
equity-research/
├── models.py                        # LLM model registry (Groq, Anthropic, OpenRouter)
│
└── marine-traffic-monitor/
    ├── run.py                       # Entry point — start the monitor here
    ├── marine_traffic_monitor.py    # Playwright scraper + OpenCV CV detection
    ├── analyst.py                   # Evidence collection, consensus orchestration
    ├── policy_engine.py             # Deterministic 7-rule alert decision engine
    ├── state_manager.py             # Persistent 4-state machine (NORMAL → SURGE → ...)
    ├── news_fetcher.py              # Multi-source news: Google RSS + GDELT DOC 2.0
    ├── notifier.py                  # HTML alert emails + daily digest (Gmail SMTP)
    │
    ├── data/
    │   └── hormuz_traffic_log.csv   # Per-cycle detection log
    ├── state/
    │   ├── current_state.json       # Live state machine state
    │   ├── last_alert.json          # Alert cooldown timestamp
    │   └── rate_counters.json       # Groq daily API quota tracker
    └── logs/
        ├── monitor.log              # Full stdout/stderr log
        └── analyst_audit.jsonl      # Complete evidence + policy trace (replayable)
```

---

## Quickstart

### 1. Clone and set up the environment

```bash
git clone https://github.com/Henrywzh/equity-research.git
cd equity-research
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install playwright opencv-python groq anthropic openai feedparser python-dotenv requests
playwright install chromium
```

### 2. Configure API keys

Create a `.config` file in the repo root (this file is gitignored — never committed):

```bash
# .config
GROQ_API_KEY=your_groq_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key       # optional
OPENROUTER_API_KEY=your_openrouter_api_key     # optional

GMAIL_SENDER=yourname@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx         # Gmail App Password (not your login password)
GMAIL_RECIPIENT=yourname@gmail.com
```

> **Gmail App Password setup** (2 min): Google Account → Security → 2-Step Verification (must be ON) → App Passwords → create one named `marine-monitor`. Use the 16-character password above.

### 3. Run the monitor

```bash
cd marine-traffic-monitor

# Default: check every 15 min, call LLMs when ≥1 ship detected and count changed
python run.py

# Aggressive: 5-min interval, LLMs fire every single cycle, no email cooldown
python run.py --interval 300 --llm-threshold 0 --delta-threshold 0 --alert-cooldown 0

# Conservative: 30-min interval, 30-min cooldown between alerts
python run.py --interval 1800 --alert-cooldown 30
```

Press `Ctrl-C` to stop. All output is mirrored to `logs/monitor.log`.

---

## Testing Individual Components

Run these from inside `marine-traffic-monitor/`:

```bash
# Test the policy engine decision rules (no API calls, instant)
python policy_engine.py

# Test the news fetcher — shows which provider (Google / GDELT) is live
python news_fetcher.py

# Send a test alert email to verify HTML rendering and Gmail setup
python notifier.py --level ESCALATED    # amber — clear anomaly
python notifier.py --level DISPUTED     # purple — models disagree
python notifier.py --level REVIEW       # teal — low confidence, human required
python notifier.py --level CRITICAL     # red — maximum alert

# Run a single LLM analyst (uses latest screenshot)
python analyst.py --model llama_4_scout --count 2

# Run the full consensus pipeline (both models + policy engine + conditional email)
python analyst.py --model consensus --count 4
```

---

## Alert Levels

| Level | Colour | Email? | Human Review? | Meaning |
|---|---|---|---|---|
| `NORMAL` | — | No | No | Baseline; nothing unusual |
| `WATCH` | — | No | No | Noteworthy but inconclusive |
| `ESCALATED` ⚡ | Amber | **Yes** | No | Clear anomaly; both models agree |
| `CRITICAL` 🚨 | Red | **Yes** | No | Both models at maximum confidence |
| `DISPUTED` ❓ | Purple | **Yes** | **Yes** | Models disagree by ≥2 urgency levels |
| `REVIEW` 🔬 | Teal | **Yes** | **Yes** | Abstained or low confidence signal |

---

## Inspecting the Audit Log

Every LLM consensus cycle is fully logged for replay and debugging:

```bash
# Pretty-print the most recent audit entry
tail -1 marine-traffic-monitor/logs/analyst_audit.jsonl | python -m json.tool

# Check current state machine state
cat marine-traffic-monitor/state/current_state.json

# Check daily Groq API quota usage
cat marine-traffic-monitor/state/rate_counters.json
```

---

## Models Used

| Key | Model | Provider | Vision | Role |
|---|---|---|---|---|
| `llama_4_scout` | Llama 4 Scout 17B | Groq (free) | ✅ | Visual analyst — reads the screenshot |
| `llama_3_3_70b` | Llama 3.3 70B | Groq (free) | ❌ | Context analyst — reads CSV + news |
| `anthropic` | Claude 3.5 Sonnet | Anthropic | ✅ | Optional visual analyst |
| `openrouter_gpt4o` | GPT-4o | OpenRouter | ✅ | Optional visual analyst |

Default consensus pair: `llama_4_scout` (Model A, vision) + `llama_3_3_70b` (Model B, text).

Groq is free and sufficient for continuous monitoring. Anthropic/OpenRouter keys are optional.

---

## Detailed Documentation

Full technical documentation including the evidence schema, policy engine rule table, state machine transitions, email structure, and design decisions is in:

```
marine-traffic-monitor/PROJECT_README.md
```

---

## Notes

- `screenshots/` is gitignored — images are large and regenerated each run
- `.config` is gitignored — never commit API keys
- The `state/`, `data/`, and `logs/` directories are auto-created on first run — no manual setup needed
- Inactive or experimental scripts are in `not_in_use/`
