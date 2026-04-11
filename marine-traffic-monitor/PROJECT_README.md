# Strait of Hormuz — Maritime OSINT Monitor
### Automated Vessel Intelligence with Evidence-Based LLM Analysis

---

## Quickstart

### 1. Set up the environment

From the repository root:

```bash
python -m venv venv
source venv/bin/activate
pip install playwright opencv-python groq openai feedparser python-dotenv requests
playwright install chromium
```

### 2. Configure API keys

Create a `.config` file in the repository root:

```bash
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

GMAIL_SENDER=yourname@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password
GMAIL_RECIPIENT=yourname@gmail.com
```

### 3. Run the monitor

```bash
cd marine-traffic-monitor
python run.py
```

Common examples:

```bash
python run.py --interval 300 --llm-threshold 0 --delta-threshold 0 --alert-cooldown 0
python run.py --interval 1800 --alert-cooldown 30
```

### 4. Test individual components

Run these from inside `marine-traffic-monitor/`:

```bash
python policy_engine.py
python news_fetcher.py
python notifier.py --level ESCALATED
python analyst.py --model consensus --count 4
```

---

## 1. Project Overview

This system continuously monitors the **Strait of Hormuz** — one of the world's most geopolitically sensitive maritime chokepoints — for anomalous vessel activity. Every few minutes it:

1. Scrapes a live vessel-tracking map via headless browser
2. Runs computer vision to count ships inside a defined transit corridor
3. Gates expensive LLM calls behind two simultaneous conditions
4. Dispatches two independent AI analysts that each produce **structured evidence** (not verdicts)
5. Feeds that evidence to a **deterministic policy engine** which makes all escalation decisions
6. Sends rich HTML alert emails and a daily digest to a configured Gmail address

**Core architectural principle:** LLMs are unreliable decision-makers under sparse, ambiguous data. This system removes LLMs from the decision path entirely. They produce observations, hypotheses, and confidence scores. A pure-Python policy engine combines those with CV features and applies a fixed rule hierarchy to determine whether an alert should fire.

This separation means a hallucinating LLM cannot trigger a CRITICAL alert on its own — it needs corroborating evidence from the second model AND the policy engine's confidence threshold to be met.

---

## 2. System Architecture

```
┌─ DATA LAYER ────────────────────────────────────────────────┐
│  Playwright (non-headless, Cloudflare bypass)               │
│    → Screenshot of MarineTraffic live map                   │
│    → OpenCV: colorfulness detection + polygon zone count    │
│    → hormuz_traffic_log.csv  (every cycle, always)          │
└─────────────────────────────────────────────────────────────┘
                  ↓  ships_in_zone ≥ T  AND  |Δcount| ≥ ΔT
                  ↓  (both conditions must be true)
┌─ EVIDENCE LAYER (LLMs — produce observations, not verdicts) ┐
│                                                             │
│  ┌─ Model A: Llama 4 Scout 17B (Vision, Groq) ──────────┐  │
│  │  Role: visual_analyst                                 │  │
│  │  Input: screenshot + CSV history + news headlines     │  │
│  │  Output: direct_observations, hypotheses, risk_signals│  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─ Model B: Llama 3.3 70B (Text, Groq) ────────────────┐  │
│  │  Role: context_analyst                                │  │
│  │  Input: CSV history + news headlines (no image)       │  │
│  │  Output: historical_context, news_context, hypotheses │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  News context: 4-attempt cascade                           │
│    1. Google News RSS when:1d                              │
│    2. Google RSS retry (CDN cold-miss)                     │
│    3. GDELT DOC 2.0 API  ← independent academic provider  │
│    4. Google RSS when:2d (broadened fallback)              │
└─────────────────────────────────────────────────────────────┘
                  ↓  evidence_a + evidence_b + cv_features
┌─ DECISION LAYER (deterministic — zero LLM calls) ───────────┐
│  policy_engine.evaluate()                                   │
│    Applies 7-rule priority hierarchy                        │
│    → alert_level: NORMAL | WATCH | ESCALATED |             │
│                   DISPUTED | REVIEW | CRITICAL             │
│    → email_decision: bool                                   │
│    → human_review_needed: bool                              │
│    → proposed_state_transition                              │
└─────────────────────────────────────────────────────────────┘
                  ↓
┌─ ACTION LAYER ──────────────────────────────────────────────┐
│  state_manager.apply_transition()   (validated transitions) │
│  notifier.send_alert()              (HTML email, cooldown)  │
│  notifier.send_digest()             (daily summary, 10 UTC) │
│  analyst_audit.jsonl                (full replay log)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Walkthrough

### Step 1 — Live Map Scraping
Playwright opens a real browser window (`headless=False`, `slow_mo=50ms`) to bypass Cloudflare's bot detection on MarineTraffic. The viewport is held open for 8 seconds to allow map tiles and AIS vessel icons to fully render before a screenshot is taken.

### Step 2 — Computer Vision Detection
OpenCV analyses the screenshot using **colorfulness detection**:
- Vessel icons are identified by their hue/saturation properties distinct from the map background
- A pre-defined polygon defines the Hormuz transit corridor (the "zone")
- `cv2.pointPolygonTest()` determines which detected icons fall inside vs. outside the zone
- Result: `ships_in_zone` (int) and a processed screenshot with bounding boxes overlaid

### Step 3 — CSV Logging
Every cycle writes one row to `hormuz_traffic_log.csv` regardless of ship count:
```
timestamp, ships_in_zone, ships_outside, status_note
```
This creates an uninterrupted history used by both LLM analysts and the policy engine.

### Step 4 — LLM Gate (two conditions, both required)
```
CALL LLMs only if:
  ships_in_zone  >=  llm_threshold    (default: 1)
  AND
  |current_count - previous_count|  >=  delta_threshold  (default: 1)
```
Both conditions must be true simultaneously. This prevents sustained readings from re-triggering on unchanged counts, avoids wasting quota during quiet periods, and keeps per-cycle costs near zero during a genuine blockade (count stays elevated → no repeated LLM calls).

### Step 5 — Evidence Collection (two independent analysts)
Each model receives the same shared context (current macro-state, news headlines, last 12 CSV rows) but via different system prompts tailored to their role. They return a structured 11-field evidence dict — not a threat level or verdict. Neither model is told to make the escalation decision.

### Step 6 — Policy Engine Decision
`policy_engine.evaluate()` receives CV features + both evidence dicts and applies a strict 7-rule priority hierarchy (see Section 4d). The output is a `PolicyDecision` dict: `alert_level`, `email_decision`, `human_review_needed`, `proposed_transition`, `reasoning` (list of fired rules), and `avg_confidence`.

### Step 7 — State Transition + Dispatch
- `state_manager.apply_transition()` validates the suggested transition against the allowed graph
- `send_alert()` fires if `email_decision=True` (with cooldown enforcement)
- The full cycle — CV features, both evidence dicts, policy decision, reasoning chain — is appended to `analyst_audit.jsonl` for replay and debugging

---

## 4. LLM Integration

### 4a. Why the Architecture Changed

**The problem with the original approach:** LLMs were asked to output `"threat_level": "CRITICAL"` and `"proposed_state_transition": "SURGE -> BLOCKADE_ACTIVE"`. Under sparse data (count=1, limited CSV history), models hallucinated high-confidence verdicts. A single vessel icon would trigger CRITICAL alerts with 0.90 confidence. There was no circuit breaker.

**The solution:** Strict role separation.

LLMs are asked:
> *"What did you observe? What are the competing hypotheses? How confident are you? Should you abstain?"*

A separate deterministic Python module (`policy_engine.py`) is asked:
> *"Given these two evidence reports and these CV features, what is the alert level?"*

Hallucination can still occur in the evidence layer, but it cannot alone trigger an alert. The policy engine requires corroboration from both models at sufficient confidence, or it routes to `REVIEW` for human inspection.

---

### 4b. Two Model Roles

| Llama 4 Scout 17B | `llama_4_scout` | `visual_analyst` | Screenshot + CSV + news | ✅ Yes |
| Llama 3.3 70B Versatile | `llama_3_3_70b` | `context_analyst` | CSV + news only | ❌ No |
| GPT-4o (OpenRouter) | `openrouter_gpt4o` | `visual_analyst` | Screenshot + CSV + news | ✅ Yes |

The visual analyst sees the screenshot and is responsible for `direct_observations` — bounding box positions, icon shapes, zone geometry. The context analyst has no image access and focuses on CSV trends and news correlation.

Both models are explicitly told in their system prompts:
> *"You do NOT make final escalation decisions. A deterministic policy engine will combine your evidence with a second analyst's findings to determine the final alert level."*

---

### 4c. Evidence Schema (11 fields)

Each model returns **only** this JSON structure. No `threat_level`, no `is_true_positive`, no state transition proposals:

| Field | Type | Description |
|---|---|---|
| `model_role` | string | `"visual_analyst"` or `"context_analyst"` |
| `direct_observations` | list[str] | What was directly observed in the image. Context analyst always sets this to `["No image available..."]` |
| `historical_context` | list[str] | Key observations from the CSV: trend direction, spike/dip patterns, rate of change |
| `news_context` | list[str] | How injected news headlines relate to or corroborate the observed pattern |
| `hypotheses` | list[{statement, confidence}] | Competing explanations. At least 2 required. Each has a 0.0–1.0 confidence score |
| `risk_signals` | list[str] | Specific indicators of elevated risk — conservative, no narrative speculation |
| `uncertainties` | list[str] | Factors limiting confidence (image quality, limited history, no AIS, etc.) |
| `recommended_state` | string | `NORMAL` \| `WATCH` \| `ESCALATED` \| `CRITICAL` — a suggestion only, not a verdict |
| `recommended_action` | string | `monitor_only` \| `watch_closely` \| `escalate` \| `critical_alert` |
| `abstain` | bool | `true` if the model genuinely cannot determine what is happening — preferred over a hallucinated confident answer |
| `abstain_reason` | string\|null | Explanation if `abstain=true`, otherwise null |
| `overall_confidence` | float | 0.0–1.0. Reflects genuine uncertainty. A single bounding box should rarely exceed 0.65 |

Schema validated by `_validate_evidence()` in `analyst.py`. Malformed JSON or invalid fields trigger a self-correction retry. After two failures the model's evidence is replaced with a fallback `abstain=True` dict — the pipeline never blocks.

---

### 4d. Policy Engine — 7-Rule Hierarchy (`policy_engine.py`)

The policy engine maps each model's `recommended_state` to an urgency integer:

| State | Urgency |
|---|---|
| NORMAL | 0 |
| WATCH | 1 |
| ESCALATED | 2 |
| CRITICAL | 3 |

Rules are applied in strict priority order — the first matching rule fires:

| Priority | Condition | Alert Level | Email? | Human Review? |
|---|---|---|---|---|
| **P1** | Either model `abstain=True` | `REVIEW` | Yes | **Yes** |
| **P2** | Both urgency = 0 (NORMAL) | `NORMAL` | No | No |
| **P3** | Both urgency = 1 (WATCH), avg_conf ≥ 0.40 | `WATCH` | No | No |
| **P3b** | Both WATCH, avg_conf < 0.40 | `NORMAL` | No | No |
| **P4** | Both urgency ≥ 2 (ESCALATED+), avg_conf ≥ 0.55 | `ESCALATED` / `CRITICAL` | Yes | No |
| **P4b** | Both ESCALATED+, avg_conf < 0.55 | `REVIEW` | Yes | **Yes** |
| **P5** | Urgency gap ≥ 2 (e.g. CRITICAL vs NORMAL) | `DISPUTED` | Yes | **Yes** |
| **P6** | Gap = 1, avg_conf ≥ 0.50 | Higher urgency level | If ≥ ESCALATED | No |
| **P7** | Gap = 1, avg_conf < 0.50 | `REVIEW` | Yes | **Yes** |

Confidence thresholds: `CONF_WATCH_MIN=0.40`, `CONF_ESCALATE_MIN=0.55`, `CONF_MINOR_DISAGREE=0.50`

The policy engine also suggests state machine transitions (e.g. ESCALATED + current=NORMAL → `"NORMAL -> SURGE"`), but these are validated separately by `state_manager.apply_transition()` before being written.

---

### 4e. Alert Levels

| Level | Emoji | Colour | Email? | Human Review? | Meaning |
|---|---|---|---|---|---|
| `NORMAL` | — | — | No | No | Baseline; nothing unusual |
| `WATCH` | — | — | No | No | Noteworthy but inconclusive — logged only |
| `ESCALATED` | ⚡ | Amber `#d97706` | **Yes** | No | Clear anomaly; both models agree at sufficient confidence |
| `CRITICAL` | 🚨 | Red `#dc2626` | **Yes** | No | Both models at maximum urgency and confidence |
| `DISPUTED` | ❓ | Purple `#7c3aed` | **Yes** | **Yes** | Models disagree by ≥ 2 urgency levels — human must reconcile |
| `REVIEW` | 🔬 | Teal `#0891b2` | **Yes** | **Yes** | Abstain, low confidence, or conflicting minor signals |

---

### 4f. News Context — Multi-Source Cascade

Both models receive up to 5 recent Hormuz/Iran maritime headlines injected into their system prompt under `[RECENT GEO-POLITICAL NEWS]`. They are instructed to populate `news_context` with their interpretation of how headlines relate to the observed pattern.

**4-attempt fetch cascade** (stops at first non-empty result):

| Attempt | Source | Query | Trigger |
|---|---|---|---|
| 1 | Google News RSS | `when:1d` | Always |
| 2 | Google News RSS | `when:1d` (retry) | After 3s sleep — CDN cold-miss |
| 3 | **GDELT DOC 2.0** | `timespan=24h` | Only if Google returns empty |
| 4 | Google News RSS | `when:2d` | Last resort — broadened query |

**GDELT** (gdeltproject.org) is a free, no-API-key academic geopolitical intelligence system indexing thousands of global sources in real time. It serves as an independent fallback so a Google RSS cache miss never silently removes geopolitical context from the LLM analysis.

Each headline in the output includes a source label: `— Google News` or `— GDELT / cnbc.com`.

---

## 5. Safeguarding & Reliability

### 5.1 JSON Schema Validation + Self-Correction Loop
Every LLM response is parsed as JSON and validated against the 11-field evidence schema. Parsing failure or invalid field values trigger an automatic correction prompt sent to the same model with the specific error reason. This retry recovers ~90% of malformed outputs. After two failures, the model's evidence is replaced with a fallback `abstain=True` dict.

### 5.2 Abstain Logic — Hallucination Circuit Breaker
Models are explicitly instructed: *"Abstaining is ALWAYS preferred over hallucinating a confident answer."* If either model sets `abstain=True`, **P1 fires immediately** — alert level becomes `REVIEW` and a human-required email is sent, regardless of what the other model says. A confident model cannot override an abstaining model.

### 5.3 Rate Limit Guard (Groq)
Daily call counts are persisted in `rate_counters.json` and incremented on every Groq API call. At **80% of daily RPD quota**: warning logged to terminal. At **90%**: model auto-downgraded to `llama_3_1_8b` (14,400 RPD) for the remainder of the day. Counter resets at midnight UTC.

### 5.4 Alert Cooldown
`last_alert.json` stores the UTC timestamp of the last sent alert. If the time since last alert is less than `alert_cooldown` minutes (default 5), the email is suppressed and logged. Prevents inbox flooding during sustained high-ship-count periods.

### 5.5 State Transition Validation
`state_manager.apply_transition()` rejects any transition that is not in the allowed graph, is a no-op (current → current), is malformed, or references an unknown state. Only the 5 defined transitions are permitted — no skip-steps.

### 5.6 Two-Provider News Resilience
Google News RSS uses a server-side CDN with a separate TTL from its search index. Between rebuild waves, the feed returns 0 results even when matching articles exist. GDELT DOC 2.0 serves as a fully independent fallback, preventing silent context loss from the LLM analysis.

### 5.7 Image Retention Policy
Screenshots with 0 ships detected inside the zone are deleted after logging to prevent disk accumulation. Screenshots with 1+ ships are retained for LLM analysis and audit. Raw (un-annotated) screenshots are retained alongside CV-processed versions when ships are detected.

### 5.8 Full Audit Trail (Replayable)
Every consensus cycle appends one JSONL line to `analyst_audit.jsonl` containing the complete `evidence_a`, `evidence_b`, and `policy_decision` dicts. Any past decision can be fully reconstructed: what each model observed, what hypotheses were formed with what confidence, and which policy rules fired.

---

## 6. State Machine

The system maintains a persistent macro-state in `current_state.json` injected into every LLM system prompt as background context.

| State | Meaning | Typical trigger |
|---|---|---|
| `NORMAL` | Baseline; few or no ships expected | Default; post-recovery |
| `SURGE` | Unusual uptick; elevated monitoring | Policy ESCALATED from NORMAL |
| `BLOCKADE_ACTIVE` | Confirmed sustained breach; max alert | Policy ESCALATED from SURGE |
| `RECOVERY` | Count declining after active phase | Policy NORMAL from BLOCKADE_ACTIVE |

**Allowed transitions (strictly enforced — no skip-steps):**
```
NORMAL ──────────────→ SURGE
SURGE ───────────────→ BLOCKADE_ACTIVE
SURGE ───────────────→ NORMAL           (false alarm de-escalation)
BLOCKADE_ACTIVE ─────→ RECOVERY
RECOVERY ────────────→ NORMAL
```

Transitions are *suggested* by the policy engine and *validated* by `state_manager.apply_transition()`. A transition that skips a step (e.g. NORMAL → BLOCKADE_ACTIVE) is silently rejected.

---

## 7. Email Notification System

### 7a. Alert Emails
Sent when `policy_engine.email_decision=True` (levels: ESCALATED, CRITICAL, DISPUTED, REVIEW).

**HTML email structure:**
1. **Threat level banner** — full-width colour bar with level name + emoji
2. **Key stats pills** — ship count, Δcount, macro state, avg confidence, human review flag
3. **Verification Note** — model A/B confidence summary and abstain status
4. **Analyst Briefing** — combined visual + context summary with top hypotheses
5. **Model A Evidence** *(collapsible)* — direct observations, risk signals, uncertainties, hypothesis confidence bars (green ≥60%, amber ≥40%, grey <40%)
6. **Model B Evidence** *(collapsible)* — same structure from context analyst
7. **Policy Engine Reasoning** — alert level badge, avg confidence pills, ordered list of rules that fired
8. **Geo-Political Context** — live news headlines with source labels (Google News / GDELT)
9. **Traffic Table** — last 12 CSV rows as HTML table
10. **Raw JSON** — full evidence + policy decision for analysts who need it
11. **Attachments** — CV-annotated screenshot + raw screenshot

Plain-text fallback included for email clients without HTML support (`multipart/alternative`).

### 7b. Daily Digest
Sent once per day at `digest_hour` UTC (default 10:00). LLM-free — reads directly from `hormuz_traffic_log.csv`. Summarises total cycles, peak ship count, state transitions in the past 24h, and a traffic table.

---

## 8. File Reference

| File | Role |
|---|---|
| `run.py` | CLI entry point; argparse; Tee logger (stdout + log file); SIGTERM handler; main monitor loop |
| `marine_traffic_monitor.py` | Playwright scrape; OpenCV CV detection; CSV write; image retention policy |
| `analyst.py` | Evidence collection orchestration; `run_consensus_check()`; `_build_final_briefing()`; audit writer |
| **`policy_engine.py`** | **Deterministic 7-rule decision engine. Zero LLM calls. Single source of truth for all alert decisions** |
| `news_fetcher.py` | 4-attempt news cascade: Google News RSS (×3) + GDELT DOC 2.0 (no API key) |
| `state_manager.py` | Persistent 4-state machine; transition validation; state description for LLM prompts |
| `notifier.py` | Rich HTML alert emails + daily digest; MIME multipart/alternative + mixed; cooldown |
| `models.py` | Model registry: provider, model_id, vision support, daily RPD limits |
| `hormuz_traffic_log.csv` | Per-cycle detection log (written every cycle without exception) |
| `analyst_audit.jsonl` | Full evidence + policy trace per LLM consensus cycle (JSONL append-only) |
| `current_state.json` | Persisted state machine state + last transition timestamp |
| `last_alert.json` | Alert cooldown: UTC timestamp of last sent alert email |
| `rate_counters.json` | Groq daily API call counter (resets midnight UTC) |
| `logs/monitor.log` | Combined stdout + stderr log file via Tee |

---

## 9. CLI Reference

```bash
# Standard run — 15 min interval, default thresholds
python run.py

# Aggressive testing — 5 min interval, LLMs fire every cycle, no email cooldown
python run.py --interval 300 --llm-threshold 0 --delta-threshold 0 --alert-cooldown 0

# Conservative production — 30 min interval, 30 min alert cooldown
python run.py --interval 1800 --alert-cooldown 30

# Daily digest at 08:00 UTC instead of 10:00
python run.py --digest-hour 8

# ── Component tests (no monitor loop) ─────────────────────────────

# Test policy engine standalone — 4 scenarios covering all rule branches
python policy_engine.py

# Test news fetcher — shows which provider (Google / GDELT) is responding
python news_fetcher.py

# Send a dummy ESCALATED alert email (tests full HTML rendering)
python notifier.py --level ESCALATED

# Send a dummy DISPUTED email (purple, human review required)
python notifier.py --level DISPUTED

# Send a dummy REVIEW email (teal, abstain scenario)
python notifier.py --level REVIEW

# Test single analyst model (uses latest screenshot)
python analyst.py --model llama_4_scout --count 2

# Test full consensus pipeline (both models + policy engine + conditional email)
python analyst.py --model consensus --count 4

# ── Inspection ─────────────────────────────────────────────────────

# Inspect latest audit log entry (full evidence + policy trace)
tail -1 analyst_audit.jsonl | python -m json.tool

# Check current state machine state
cat current_state.json

# Check Groq daily call counter
cat rate_counters.json
```

---

## 10. Essay: Describe a Specific Instance Where You Encountered a Limitation of AI/ML Systems — and What You Did About It

*This section answers the Dymon Asia application question: "Describe a specific instance where you encountered a limitation of AI/ML systems. How did you address it, and what did you learn from that experience? (300+ words)"*

---

Building this Strait of Hormuz monitor exposed three compounding limitations of large language models and led to a significant architectural redesign that I believe generalises well beyond this project.

**The problem: LLMs issuing confident verdicts on ambiguous evidence**

The first version of the system asked each model to output a direct `threat_level` (`NONE`, `LOW`, `ELEVATED`, `CRITICAL`) and a `proposed_state_transition` (`NORMAL -> SURGE`). Two models would vote, agree or produce a DISPUTED outcome, and the consensus would determine whether an alert fired.

In testing, the flaw was immediate. With `ships_in_zone = 1` and only three rows of CSV history, Llama 4 Scout would output `"threat_level": "CRITICAL"` with `"overall_confidence": 0.90`. It wasn't lying — one ship in a blockade zone during heightened Iran tensions *could* be critical. But from an analyst's perspective, that is a single count reading with no baseline, no corroborating visual evidence, and a CV system that occasionally misidentifies map artefacts as vessels. The model had no mechanism to express *"I don't have enough data to know."*

**Fix 1: Output brittleness — self-correction loop**

Before addressing confidence calibration, I had to solve a mechanical problem: the models frequently returned malformed JSON — trailing commas, markdown fences, text outside the JSON block. I implemented a correction retry loop that sends the malformed output back to the same model with the specific parse error and asks it to fix only the JSON. This recovered approximately 90% of malformed responses without an additional full inference cost. It also forced me to define a strict schema up front — which turned out to be foundational for the larger fix.

**Fix 2: Architectural redesign — evidence-based pipeline with deterministic policy**

The real fix required a conceptual shift. I stopped asking the LLMs *"what is the threat level?"* and started asking: *"what did you observe, and what are the competing hypotheses?"*

Each model now returns an 11-field evidence schema: `direct_observations` (bounding box positions, icon shapes), `hypotheses` (competing explanations each with a 0.0–1.0 confidence score), `risk_signals`, `uncertainties`, and critically — an `abstain` boolean. The models are explicitly told: *"Abstaining is always preferred over hallucinating a confident answer."*

A new module, `policy_engine.py`, receives both evidence dicts plus CV features (ship count, delta from last reading) and applies a deterministic 7-rule hierarchy to make the final escalation decision. The rules incorporate confidence thresholds: both models must recommend ESCALATED with average confidence ≥ 0.55 before an alert fires (P4). If either model abstains, the system immediately routes to REVIEW requiring human inspection (P1), regardless of what the other model says. If models disagree by two or more urgency levels — one says CRITICAL, the other says NORMAL — the outcome is DISPUTED with mandatory human review (P5), not an average.

This means a single hallucinating model cannot trigger an alert. It needs the second model to corroborate, and the policy engine's confidence threshold to be cleared. The circuit breaker is architectural, not prompt-based.

**Fix 3: External data quality — multi-source news cascade**

A subtler issue emerged with news context injection. The system feeds both models recent Strait of Hormuz headlines to help them interpret count spikes in geopolitical context. But Google News RSS uses a server-side CDN with an inconsistent refresh cycle. Between cache rebuilds, the feed returns zero articles — so both models would receive "No recent news" even when highly relevant events (an Iran tanker strike, NATO Hormuz discussions) were being actively covered globally. The models would then assign low news context significance, systematically underweighting the most important geopolitical signal available.

The fix was to add GDELT DOC 2.0 — a free, no-API-key academic geopolitical intelligence monitor indexing thousands of global sources — as a secondary fallback provider. The news fetcher now tries Google RSS twice, then GDELT, then a broadened Google query. This 4-attempt cascade across two independent providers ensures a CDN cold-miss at Google never silently strips geo-political context from the LLM analysis. Each headline also carries a source label (`— Google News` or `— GDELT / reuters.com`) so analysts reading the audit log can see exactly what information the models had access to.

**What I learned**

The deepest lesson is that LLMs are evidence synthesisers, not decision engines. They excel at converting unstructured signals — screenshots, news, CSV patterns — into structured observations with competing hypotheses. But calibrated, auditable decisions under uncertainty require a deterministic rule layer that can express confidence thresholds, disagreement states, and explicit abstention. None of these can be reliably self-enforced by an LLM through prompt engineering alone.

The second lesson is about accountability. Every LLM evidence dict, policy decision, and reasoning chain is written to an append-only audit log (`analyst_audit.jsonl`). Any past alert can be fully reconstructed: what the models observed, what hypotheses they formed, what confidence they had, which rules fired. This is the difference between a system an analyst can trust and one they can only hope is working correctly.

In equity research, the principle is identical. An AI that summarises earnings call transcripts, flags covenant breaches, or identifies unusual shipping patterns is extremely useful. An AI that autonomously triggers portfolio adjustments based on its own confidence assessment — without a deterministic rule layer, confidence thresholds, and a human review gate for contested signals — is not a tool. It is a liability. The analyst's value in an AI-augmented world shifts from data retrieval to calibrating the boundary between automated action and human judgement, and understanding exactly where and why that boundary is drawn.

---

*System last updated: March 2026 | Author: Henry Wu*
