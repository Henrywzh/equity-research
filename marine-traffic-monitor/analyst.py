"""
analyst.py — Evidence-Based Multi-Model Maritime OSINT Analyst
===============================================================
Architecture: LLMs produce structured evidence; policy_engine.py makes decisions.

LLMs are NOT asked for final verdicts. They are asked to produce:
  - direct_observations  (what was seen)
  - hypotheses           (competing explanations with confidence)
  - risk_signals         (specific indicators)
  - uncertainties        (limiting factors)
  - recommended_state    (SUGGESTION ONLY)
  - overall_confidence   (0.0–1.0)
  - abstain              (true = "I cannot determine this")

The deterministic policy_engine.evaluate() combines both models' evidence
with CV features to make the final alert_level, email, and state decisions.

Key exports:
  run_analyst(image_path, reported_count, csv_path, model_key="llama_4_scout") -> dict
  run_consensus_check(image_path, reported_count, csv_path,
                      model_a="llama_4_scout", model_b="llama_3_3_70b") -> dict

Available model_key values:
  Groq (free):    "llama_4_scout" (vision), "llama_3_3_70b", "llama_3_1_8b"
  Anthropic:      "anthropic"
  OpenRouter:     "openrouter_gpt4o", "openrouter_gemini"
"""

import os
import sys
import json
import base64
import csv
import time
import anthropic
from datetime import datetime
from dotenv import load_dotenv
from notifier       import send_alert
from news_fetcher   import get_latest_news
from state_manager  import get_current_state, apply_transition
import policy_engine

# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------
_config_path = os.path.join(os.path.dirname(__file__), "..", ".config")
load_dotenv(dotenv_path=_config_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import get_analyst_model

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
_HERE             = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_HERE, "logs"),  exist_ok=True)
os.makedirs(os.path.join(_HERE, "state"), exist_ok=True)
os.makedirs(os.path.join(_HERE, "data"),  exist_ok=True)

AUDIT_LOG_PATH    = os.path.join(_HERE, "logs",  "analyst_audit.jsonl")
RATE_COUNTER_PATH = os.path.join(_HERE, "state", "rate_counters.json")
CSV_PATH          = os.path.join(_HERE, "data",  "hormuz_traffic_log.csv")

# ------------------------------------------------------------------
# Evidence Schema
# ------------------------------------------------------------------
EVIDENCE_REQUIRED_KEYS = {
    "model_role", "direct_observations", "historical_context", "news_context",
    "hypotheses", "risk_signals", "uncertainties",
    "recommended_state", "recommended_action", "abstain", "overall_confidence",
}
VALID_RECOMMENDED_STATES  = {"NORMAL", "WATCH", "ESCALATED", "CRITICAL"}
VALID_RECOMMENDED_ACTIONS = {"monitor_only", "watch_closely", "escalate", "critical_alert"}

# ------------------------------------------------------------------
# System Prompts (evidence-based — LLMs produce observations, not verdicts)
# ------------------------------------------------------------------

SYSTEM_PROMPT_VISUAL_ANALYST = """# SYSTEM DIRECTIVE: MARITIME VISUAL EVIDENCE ANALYST
## 1. YOUR ROLE
You are a Visual Evidence Analyst for a Maritime OSINT system monitoring the Strait of Hormuz.

CRITICAL: You do NOT make final escalation decisions. A separate deterministic policy engine
will combine your evidence with a second analyst's findings to determine the final alert level.
Your job is to produce an honest, structured evidence report — not a verdict.

## 2. YOUR INPUTS
1. A screenshot of the MarineTraffic live map. The red polygon = choke-point zone.
   Green/red bounding boxes = what OpenCV detected inside and outside the zone.
2. Reported ship count from the CV system.
3. Last 12 intervals (3 hours) of traffic history.

## 3. YOUR OUTPUT FIELDS

**direct_observations**
List only what you directly observe in the image. Be specific about bounding box shapes,
icon types, positions relative to the zone polygon.
Example: "1 green bounding box visible inside red polygon zone", "triangular icon pointing NE"

**historical_context**
Key observations from the CSV: trend direction, spike/dip patterns, rate of change.

**news_context**
How (if at all) the injected news headlines relate to the observed count or pattern.

**hypotheses**
Competing explanations for the observation. List at least 2 (even if one is unlikely).
Each must include a confidence score from 0.0 to 1.0.
Be honest: if you're 50/50, say so.

**risk_signals**
Specific indicators that suggest elevated risk. Be conservative — only list genuine signals,
not narrative speculation.

**uncertainties**
Factors that limit your confidence. Be explicit and complete.

**recommended_state**
Your suggestion: NORMAL | WATCH | ESCALATED | CRITICAL
- NORMAL:    consistent with baseline, no unusual pattern
- WATCH:     something worth noting but inconclusive evidence
- ESCALATED: clear anomaly with supporting evidence
- CRITICAL:  strong multi-factor signal, high confidence

This is a RECOMMENDATION only. The policy engine makes the final call.

**recommended_action**
One of: monitor_only | watch_closely | escalate | critical_alert

**abstain**
Set to TRUE if you genuinely cannot determine what is happening.
Reasons to abstain: image is too blurry, contradictory evidence, count too ambiguous.
ABSTAINING IS ALWAYS PREFERRED OVER HALLUCINATING A CONFIDENT ANSWER.

**abstain_reason**
If abstain=true, explain why clearly. Otherwise null.

**overall_confidence**
Float 0.0–1.0. Reflect genuine uncertainty. Do NOT be overconfident on ambiguous signals.
A single bounding box at count=1 should rarely exceed 0.65.

## 4. STRICT OUTPUT FORMAT (JSON ONLY)
Return ONLY valid JSON. No markdown fences (no ```json). No text outside the JSON.
{
  "model_role": "visual_analyst",
  "direct_observations": ["string", ...],
  "historical_context": ["string", ...],
  "news_context": ["string", ...],
  "hypotheses": [{"statement": "string", "confidence": 0.0}, ...],
  "risk_signals": ["string", ...],
  "uncertainties": ["string", ...],
  "recommended_state": "NORMAL | WATCH | ESCALATED | CRITICAL",
  "recommended_action": "monitor_only | watch_closely | escalate | critical_alert",
  "abstain": false,
  "abstain_reason": null,
  "overall_confidence": 0.0
}

## 5. GUARDRAILS
- DO NOT output threat_level, is_true_positive, or proposed_state_transition — those are not your fields.
- DO NOT speculate beyond what the image and CSV support.
- Any output outside the JSON structure will cause a critical pipeline failure."""


SYSTEM_PROMPT_CONTEXT_ANALYST = """# SYSTEM DIRECTIVE: MARITIME CONTEXT & TREND ANALYST
## 1. YOUR ROLE
You are a Context and Trend Analyst for a Maritime OSINT system monitoring the Strait of Hormuz.
You do NOT have access to the screenshot. A visual analyst model handles image verification.

CRITICAL: You do NOT make final escalation decisions. A deterministic policy engine
will combine your analysis with the visual analyst's findings to determine the final alert level.

## 2. YOUR INPUTS
1. Reported ship count from the CV system.
2. Last 12 intervals (3 hours) of traffic history.
(No image available for this call.)

## 3. YOUR OUTPUT FIELDS

**direct_observations**
Always: ["No image available — visual QA delegated to visual_analyst model."]

**historical_context**
Observations from the CSV: trend direction, spike/dip patterns, rate of change,
rolling pattern, comparison to prior intervals, anything noteworthy in the sequence.

**news_context**
How (if at all) the injected news headlines relate to or corroborate the CSV pattern.
Specifically: does any headline explain or predict the observed count change?

**hypotheses**
Competing explanations for the CSV pattern. Include at least 2.
Each must have a confidence score 0.0–1.0.

**risk_signals**
Specific CSV-derived indicators of elevated risk (e.g. sustained increase, acceleration,
count never returning to zero, time-of-day anomaly).

**uncertainties**
Factors limiting confidence (e.g. limited history, all zeros with no baseline,
CSV shows counts not vessel identity or AIS status).

**recommended_state**
NORMAL | WATCH | ESCALATED | CRITICAL — based on CSV + news context only.

**recommended_action**
monitor_only | watch_closely | escalate | critical_alert

**abstain**
True if the CSV history is insufficient to form any view (e.g. fewer than 3 data points,
all values identical with no variation, clearly corrupted data).

**abstain_reason**
If abstain=true, explain why. Otherwise null.

**overall_confidence**
Float 0.0–1.0. CSV-only analysis has inherent limits — be honest about them.

## 4. STRICT OUTPUT FORMAT (JSON ONLY)
Return ONLY valid JSON. No markdown fences. No text outside the JSON.
{
  "model_role": "context_analyst",
  "direct_observations": ["No image available — visual QA delegated to visual_analyst model."],
  "historical_context": ["string", ...],
  "news_context": ["string", ...],
  "hypotheses": [{"statement": "string", "confidence": 0.0}, ...],
  "risk_signals": ["string", ...],
  "uncertainties": ["string", ...],
  "recommended_state": "NORMAL | WATCH | ESCALATED | CRITICAL",
  "recommended_action": "monitor_only | watch_closely | escalate | critical_alert",
  "abstain": false,
  "abstain_reason": null,
  "overall_confidence": 0.0
}

## 5. GUARDRAILS
- DO NOT output threat_level, is_true_positive, or proposed_state_transition.
- Base your analysis strictly on the CSV data and news provided.
- Any output outside the JSON structure will cause a critical pipeline failure."""


# ------------------------------------------------------------------
# Audit trail
# ------------------------------------------------------------------

def _write_audit(model_key: str, latency_s: float, correction_needed: bool,
                 result: dict, consensus_label: str = "", news: str = "",
                 evidence_a: dict = None, evidence_b: dict = None,
                 policy_decision: dict = None) -> None:
    """Append one JSON line to analyst_audit.jsonl."""
    entry = {
        "ts":                   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_key":            model_key,
        "latency_s":            round(latency_s, 2),
        "correction_needed":    correction_needed,
        # Final decision
        "alert_level":          result.get("alert_level"),
        "threat_level":         result.get("threat_level"),
        "consensus":            consensus_label or None,
        "macro_state":          result.get("current_macro_state"),
        "state_transition":     result.get("proposed_state_transition"),
        "applied_macro_state":  result.get("applied_macro_state"),
        "avg_confidence":       result.get("avg_confidence"),
        "human_review_needed":  result.get("human_review_needed"),
        "news_headlines":       news.strip() if news.strip() else None,
        # Full evidence + policy trace for replay / debugging
        "evidence_a":           evidence_a,
        "evidence_b":           evidence_b,
        "policy_decision":      policy_decision,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------
# Escalation banner
# ------------------------------------------------------------------

def _print_escalation_banner(decision: dict, model_a: str = "", model_b: str = "") -> None:
    """Print a visible terminal banner for high-urgency alerts."""
    level = decision.get("alert_level", "")
    if level not in {"ESCALATED", "DISPUTED", "REVIEW", "CRITICAL"}:
        return
    reasoning = " | ".join(decision.get("reasoning", []))[:80]
    label = f"Models: {model_a} vs {model_b}" if model_a and model_b else ""
    hr = decision.get("human_review_needed", False)
    print("\n" + "╔" + "═" * 52 + "╗")
    if level == "ESCALATED":
        print("║  ⚡ ESCALATED — POLICY ENGINE DECISION              ║")
    elif level == "CRITICAL":
        print("║  🚨 CRITICAL  — POLICY ENGINE DECISION              ║")
    elif level == "DISPUTED":
        print("║  ❓ DISPUTED  — MODELS DISAGREE — HUMAN REVIEW      ║")
    else:
        print("║  🔬 REVIEW    — LOW CONFIDENCE — HUMAN REVIEW       ║")
    if label:
        print(f"║  {label[:52]:<52}  ║")
    print(f"║  Reasoning: {reasoning[:40]:<40}  ║")
    print(f"║  Human review needed: {'YES' if hr else 'NO':<30}  ║")
    print("╚" + "═" * 52 + "╝\n")


# ------------------------------------------------------------------
# Dynamic system prompt augmentation
# ------------------------------------------------------------------

def _augment_system_prompt(base: str, current_state: str, news: str) -> str:
    """
    Append live geo-political context to any system prompt.
    Note: models are NOT asked to propose state transitions — that is the policy engine's job.
    """
    news_section = news.strip() if news.strip() else "No recent news available."
    return base + f"""

## [CURRENT MACRO STATE — CONTEXT ONLY]
The monitoring system is currently in state: **{current_state}**

State reference:
  NORMAL           → Baseline, few/no ships expected.
  SURGE            → Unusual uptick; elevated monitoring.
  BLOCKADE_ACTIVE  → Confirmed sustained blockade breach; maximum alert.
  RECOVERY         → Ship count declining after an active phase.

Use this as background context for your recommended_state and hypotheses.
Do NOT include state transition proposals — the policy engine handles transitions.

## [RECENT GEO-POLITICAL NEWS]
{news_section}
Reference these headlines in your news_context field where relevant.
"""


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def load_csv_history(csv_path: str, n_rows: int = 12) -> str:
    """Read the last n_rows from the CSV and return as a plain-text table string."""
    abs_path = os.path.join(_HERE, csv_path) if not os.path.isabs(csv_path) else csv_path
    rows, header = [], None
    try:
        with open(abs_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        return "No CSV history file found."
    recent = rows[-n_rows:] if len(rows) >= n_rows else rows
    if not recent:
        return "No historical data recorded yet."
    lines = [",".join(header)] if header else []
    lines += [",".join(r) for r in recent]
    return "\n".join(lines)


def _get_last_csv_count(csv_path: str) -> int:
    """Read the last recorded ship count from the CSV (for delta calculation in policy engine)."""
    abs_path = os.path.join(_HERE, csv_path) if not os.path.isabs(csv_path) else csv_path
    try:
        with open(abs_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)          # skip header
            rows = list(reader)
        if rows:
            return int(rows[-1][1])
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return 0


def _validate_evidence(d: dict) -> tuple[bool, str]:
    """Check all required keys exist and have correct types for the evidence schema."""
    missing = EVIDENCE_REQUIRED_KEYS - d.keys()
    if missing:
        return False, f"Missing keys: {missing}"
    if d.get("recommended_state") not in VALID_RECOMMENDED_STATES:
        return False, f"'recommended_state' must be one of {VALID_RECOMMENDED_STATES}, got '{d.get('recommended_state')}'"
    if d.get("recommended_action") not in VALID_RECOMMENDED_ACTIONS:
        return False, f"'recommended_action' must be one of {VALID_RECOMMENDED_ACTIONS}, got '{d.get('recommended_action')}'"
    if not isinstance(d.get("abstain"), bool):
        return False, f"'abstain' must be a boolean, got {type(d.get('abstain')).__name__}"
    conf = d.get("overall_confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        return False, f"'overall_confidence' must be a float 0.0–1.0, got '{conf}'"
    hyps = d.get("hypotheses", [])
    if not isinstance(hyps, list):
        return False, "'hypotheses' must be a list"
    for h in hyps:
        if not isinstance(h, dict) or "statement" not in h or "confidence" not in h:
            return False, "Each hypothesis must have 'statement' and 'confidence' keys"
    return True, ""


def _parse_json_response(raw: str) -> dict | None:
    """Parse raw LLM text as JSON; strips markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------
# Rate limit guard (Groq)
# ------------------------------------------------------------------

def _check_and_increment_rate(model_key: str, cfg: dict) -> str:
    """
    Track daily Groq call count. If ≥ 90% of RPD limit, auto-downgrade
    to llama_3_1_8b (14,400 RPD). Returns the effective model_key to use.
    """
    if cfg.get("provider") != "groq":
        return model_key

    today    = datetime.utcnow().strftime("%Y-%m-%d")
    counters = {}
    try:
        with open(RATE_COUNTER_PATH) as f:
            counters = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if counters.get("date") != today:
        counters = {"date": today, "groq_calls": 0}

    groq_calls = counters.get("groq_calls", 0)
    rpd        = cfg.get("rpd") or 1000

    if groq_calls >= int(rpd * 0.9) and model_key != "llama_3_1_8b":
        print(f"[RATE GUARD] Groq quota at {groq_calls}/{rpd} — auto-downgrading to llama_3_1_8b.")
        model_key = "llama_3_1_8b"
    elif groq_calls >= int(rpd * 0.8):
        print(f"[RATE GUARD] ⚠️  Groq quota at {groq_calls}/{rpd} (80% used). Approaching daily limit.")

    counters["groq_calls"] = groq_calls + 1
    with open(RATE_COUNTER_PATH, "w") as f:
        json.dump(counters, f)

    return model_key


# ------------------------------------------------------------------
# Provider-specific API call helpers
# ------------------------------------------------------------------

def _build_groq_or_openrouter_messages(image_data: str | None, user_text: str,
                                        has_vision: bool,
                                        current_state: str = "NORMAL",
                                        news: str = "") -> list:
    """Build the messages list for Groq / OpenRouter (OpenAI-compatible format)."""
    base          = SYSTEM_PROMPT_VISUAL_ANALYST if has_vision else SYSTEM_PROMPT_CONTEXT_ANALYST
    system_prompt = _augment_system_prompt(base, current_state, news)
    user_content: list = []

    if has_vision and image_data:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data}"}
        })

    user_content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content}
    ]


def _run_groq(model_id: str, image_data: str | None, user_text: str,
              has_vision: bool, current_state: str = "NORMAL", news: str = "") -> str:
    """Call a Groq model."""
    from groq import Groq
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("GROK_API_KEY is not set in .config")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=_build_groq_or_openrouter_messages(
            image_data, user_text, has_vision, current_state, news),
        max_tokens=1024,
        temperature=0.1
    )
    return response.choices[0].message.content


def _run_openrouter(model_id: str, image_data: str | None, user_text: str,
                    has_vision: bool, current_state: str = "NORMAL", news: str = "") -> str:
    """Call a model via OpenRouter."""
    from openai import OpenAI
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .config")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model=model_id,
        messages=_build_groq_or_openrouter_messages(
            image_data, user_text, has_vision, current_state, news),
        max_tokens=1024,
        temperature=0.1
    )
    return response.choices[0].message.content


def _run_anthropic(model_id: str, image_data: str, user_text: str,
                   current_state: str = "NORMAL", news: str = "") -> str:
    """Call Anthropic Claude with vision + adaptive thinking."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-anthropic-api-key-here":
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured.\n"
            "  1. Get your key at https://console.anthropic.com\n"
            "  2. Open .config and replace 'your-anthropic-api-key-here' with your key."
        )
    system = _augment_system_prompt(SYSTEM_PROMPT_VISUAL_ANALYST, current_state, news)
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model=model_id,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
                {"type": "text", "text": user_text}
            ]
        }]
    ) as stream:
        final_message = stream.get_final_message()
    raw = next((b.text for b in final_message.content if b.type == "text"), None)
    if raw is None:
        raise RuntimeError("Anthropic returned no text block in the response.")
    return raw


def _retry_with_correction(cfg: dict, image_data: str | None, user_text: str,
                            malformed_response: str, error_reason: str,
                            current_state: str = "NORMAL", news: str = "") -> str:
    """Re-call the same provider+model with the malformed output + fix instruction."""
    correction_text = (
        f"{user_text}\n\n---\n"
        f"CORRECTION REQUIRED: Your previous response was invalid. Reason: {error_reason}\n"
        f"Previous output:\n{malformed_response}\n\n"
        f"Return ONLY the corrected raw JSON with no markdown, no explanation."
    )
    provider   = cfg["provider"]
    model_id   = cfg["model_id"]
    has_vision = cfg["supports_vision"]

    if provider == "groq":
        return _run_groq(model_id, image_data, correction_text, has_vision, current_state, news)
    elif provider == "openrouter":
        return _run_openrouter(model_id, image_data, correction_text, has_vision, current_state, news)
    else:
        return _run_anthropic(model_id, image_data, correction_text, current_state, news)


# ------------------------------------------------------------------
# Core: single-model evidence collector (private)
# ------------------------------------------------------------------

def _run_evidence_analyst(image_path: str, reported_count: int, csv_path: str,
                           model_key: str = "llama_4_scout",
                           current_state: str = "NORMAL", news: str = "") -> dict:
    """
    Call one LLM model and return a structured evidence dict.
    Returns {parse_error: ...} on total failure.
    """
    cfg        = get_analyst_model(model_key)
    has_vision = cfg.get("supports_vision", False)

    # Rate limit guard (Groq only)
    effective_key = _check_and_increment_rate(model_key, cfg)
    if effective_key != model_key:
        cfg        = get_analyst_model(effective_key)
        has_vision = cfg.get("supports_vision", False)

    # Load image (only if vision model)
    image_data: str | None = None
    if has_vision:
        abs_image = os.path.join(_HERE, image_path) if not os.path.isabs(image_path) else image_path
        if not os.path.exists(abs_image):
            print(f"[ANALYST ERROR] Image not found: {abs_image}")
            return {"parse_error": f"Image not found: {abs_image}"}
        with open(abs_image, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    csv_history = load_csv_history(csv_path)
    user_text = (
        f"Reported Count: {reported_count} ship(s) detected inside the transit corridor.\n\n"
        f"CSV History (last 12 intervals — 3 hours of data):\n{csv_history}"
    )

    role_label = "Visual Analyst" if has_vision else "Context Analyst"
    print(f"[ANALYST] Querying {cfg['name']} as {role_label} (vision={has_vision}, state={current_state})...")
    t0                = time.time()
    correction_needed = False

    try:
        provider = cfg["provider"]
        if provider == "groq":
            raw_json = _run_groq(cfg["model_id"], image_data, user_text,
                                 has_vision, current_state, news)
        elif provider == "anthropic":
            raw_json = _run_anthropic(cfg["model_id"], image_data, user_text,
                                      current_state, news)
        elif provider == "openrouter":
            raw_json = _run_openrouter(cfg["model_id"], image_data, user_text,
                                       has_vision, current_state, news)
        else:
            print(f"[ANALYST ERROR] Unknown provider '{provider}'.")
            return {"parse_error": f"Unknown provider: {provider}"}
    except RuntimeError as e:
        print(f"[ANALYST ERROR] {e}")
        return {"parse_error": str(e)}
    except Exception as e:
        print(f"[ANALYST ERROR] API call failed: {e}")
        return {"parse_error": str(e)}

    # Parse JSON
    evidence = _parse_json_response(raw_json)
    if evidence is None:
        correction_needed = True
        print("[ANALYST] JSON parse failed — retrying with correction prompt...")
        try:
            raw_json = _retry_with_correction(cfg, image_data, user_text, raw_json,
                                              "Invalid JSON syntax", current_state, news)
            evidence = _parse_json_response(raw_json)
        except Exception as e:
            return {"parse_error": f"JSON decode failed after retry: {e}"}
        if evidence is None:
            return {"parse_error": "JSON decode failed after retry"}

    # Validate evidence schema
    is_valid, error_reason = _validate_evidence(evidence)
    if not is_valid:
        correction_needed = True
        print(f"[ANALYST] Schema invalid ({error_reason}) — retrying with correction prompt...")
        try:
            raw_json = _retry_with_correction(cfg, image_data, user_text, raw_json,
                                              error_reason, current_state, news)
            evidence = _parse_json_response(raw_json)
        except Exception as e:
            return {"parse_error": f"Schema invalid after retry: {error_reason}"}
        if evidence is None:
            return {"parse_error": f"Schema invalid after retry: {error_reason}"}
        is_valid, error_reason = _validate_evidence(evidence)
        if not is_valid:
            return {"parse_error": f"Schema still invalid after retry: {error_reason}"}

    latency = round(time.time() - t0, 2)
    rec_state = evidence.get("recommended_state", "NORMAL")
    conf      = evidence.get("overall_confidence", 0.0)
    abstain   = evidence.get("abstain", False)
    print(f"[ANALYST] ✓ Evidence collected — recommended={rec_state} "
          f"conf={conf:.2f} abstain={abstain} latency={latency}s correction={correction_needed}")

    return evidence


# ------------------------------------------------------------------
# Final briefing builder (evidence + policy → notifier-compatible dict)
# ------------------------------------------------------------------

def _build_final_briefing(decision: dict, evidence_a: dict, evidence_b: dict,
                           reported_count: int, model_a: str, model_b: str,
                           current_state: str) -> dict:
    """
    Construct the final briefing dict from policy decision + evidence.
    Includes backward-compatible fields (threat_level, is_true_positive)
    so notifier.py works without changes.
    """
    alert_level = decision.get("alert_level", "REVIEW")

    # Map policy alert_level → threat_level for notifier backward compat
    threat_map = {
        "NORMAL":    "NONE",
        "WATCH":     "LOW",
        "ESCALATED": "ELEVATED",
        "CRITICAL":  "CRITICAL",
        "DISPUTED":  "DISPUTED",
        "REVIEW":    "REVIEW",
    }

    # Summarise observations and risk signals for analyst_briefing
    obs_a    = evidence_a.get("direct_observations", [])
    ctx_b    = evidence_b.get("historical_context", [])
    risk_a   = evidence_a.get("risk_signals", [])
    risk_b   = evidence_b.get("risk_signals", [])
    news_a   = evidence_a.get("news_context", [])

    visual_summary  = "; ".join(obs_a[:3] + risk_a[:2]) or "No visual observations."
    context_summary = "; ".join(ctx_b[:3] + risk_b[:2]) or "No context analysis."
    news_summary    = "; ".join(news_a[:2]) if news_a else "No relevant news context."

    # Hypothesis summary
    def _fmt_hyps(hyps: list, name: str) -> str:
        if not hyps:
            return f"[{name}]: No hypotheses provided."
        parts = []
        for h in hyps[:3]:
            conf = h.get("confidence", "?")
            stmt = h.get("statement", "?")
            parts.append(f"{stmt} (conf={conf:.2f})" if isinstance(conf, (int, float)) else stmt)
        return f"[{name}]: " + " | ".join(parts)

    hyps_a = evidence_a.get("hypotheses", [])
    hyps_b = evidence_b.get("hypotheses", [])

    analyst_briefing = (
        f"Visual Analysis: {visual_summary}\n"
        f"Trend Context: {context_summary}\n"
        f"News Context: {news_summary}\n"
        f"Hypotheses — {_fmt_hyps(hyps_a, model_a)} || {_fmt_hyps(hyps_b, model_b)}"
    )

    verification_note = (
        f"[{model_a}] conf={evidence_a.get('overall_confidence', 0):.2f} "
        f"abstain={evidence_a.get('abstain', False)} | "
        f"[{model_b}] conf={evidence_b.get('overall_confidence', 0):.2f} "
        f"abstain={evidence_b.get('abstain', False)} | "
        f"avg_conf={decision.get('avg_confidence', 0):.2f}"
    )

    return {
        # Backward-compatible fields for notifier.py
        "threat_level":         threat_map.get(alert_level, "DISPUTED"),
        "event_classification": f"{alert_level} — Policy Engine Decision",
        "is_true_positive":     True  if alert_level in ("ESCALATED", "CRITICAL") else
                                None  if alert_level in ("DISPUTED", "REVIEW") else False,
        "consensus":            alert_level not in ("DISPUTED", "REVIEW"),
        "consensus_note":       " | ".join(decision.get("reasoning", [])),
        "analyst_briefing":     analyst_briefing,
        "verification_note":    verification_note,
        # New fields
        "alert_level":                  alert_level,
        "policy_reasoning":             decision.get("reasoning", []),
        "avg_confidence":               decision.get("avg_confidence", 0.0),
        "current_macro_state":          decision.get("applied_state", current_state),
        "proposed_state_transition":    decision.get("proposed_transition", "NONE"),
        "applied_macro_state":          decision.get("applied_state", current_state),
        "human_review_needed":          decision.get("human_review_needed", False),
        "evidence_a":                   evidence_a,
        "evidence_b":                   evidence_b,
    }


# ------------------------------------------------------------------
# Public API: single model (returns evidence dict)
# ------------------------------------------------------------------

def run_analyst(image_path: str, reported_count: int, csv_path: str,
                model_key: str = "llama_4_scout",
                current_state: str = "NORMAL", news: str = "") -> dict:
    """
    Public wrapper: run a single evidence analyst and return the evidence dict.
    Useful for CLI testing and single-model analysis.
    """
    if not current_state:
        current_state = get_current_state()
    if not news:
        news = get_latest_news()
    return _run_evidence_analyst(image_path, reported_count, csv_path,
                                  model_key=model_key, current_state=current_state, news=news)


# ------------------------------------------------------------------
# Public API: full consensus + policy cycle
# ------------------------------------------------------------------

def run_consensus_check(image_path: str, reported_count: int, csv_path: str,
                        model_a: str = "llama_4_scout",
                        model_b: str = "llama_3_3_70b",
                        raw_image_path: str = "",
                        alert_cooldown: int = 5) -> dict:
    """
    Run two evidence analysts, pass their output to the policy engine,
    apply any validated state transition, then send alert if warranted.

    Returns the final briefing dict (policy decision + evidence summary).
    """
    csv_snapshot = load_csv_history(csv_path, n_rows=12)
    image_paths  = [p for p in [raw_image_path, image_path] if p]

    # Fetch shared context once
    current_state = get_current_state()
    news          = get_latest_news()
    news_lines    = [l for l in news.splitlines() if l.strip()]
    print(f"[CONSENSUS] Current macro state : {current_state}")
    print(f"[CONSENSUS] News headlines fetched: {len(news_lines)}")
    if news_lines:
        print("[CONSENSUS] Recent geo-political headlines:")
        for line in news_lines:
            print(f"            {line}")
    else:
        print("[CONSENSUS] No Hormuz/Iran maritime news in the last 24h.")

    # Collect evidence from both models
    print(f"[CONSENSUS] Collecting evidence — Model A: {model_a}...")
    evidence_a = _run_evidence_analyst(image_path, reported_count, csv_path,
                                       model_key=model_a, current_state=current_state, news=news)

    print(f"[CONSENSUS] Collecting evidence — Model B: {model_b}...")
    evidence_b = _run_evidence_analyst(image_path, reported_count, csv_path,
                                       model_key=model_b, current_state=current_state, news=news)

    # Handle model failures gracefully — create fallback evidence with abstain=True
    a_failed = "parse_error" in evidence_a
    b_failed = "parse_error" in evidence_b

    if a_failed and b_failed:
        print("[CONSENSUS] Both models failed — returning error.")
        return {"alert_level": "REVIEW", "threat_level": "REVIEW",
                "error": "Both models failed to produce evidence."}

    _FALLBACK_EVIDENCE = lambda name, err: {
        "model_role": name, "direct_observations": [], "historical_context": [],
        "news_context": [], "hypotheses": [], "risk_signals": [], "uncertainties": [],
        "recommended_state": "NORMAL", "recommended_action": "monitor_only",
        "abstain": True, "abstain_reason": f"Model failed: {err}",
        "overall_confidence": 0.0,
    }
    if a_failed:
        evidence_a = _FALLBACK_EVIDENCE("visual_analyst", evidence_a.get("parse_error"))
    if b_failed:
        evidence_b = _FALLBACK_EVIDENCE("context_analyst", evidence_b.get("parse_error"))

    # Build CV features for policy engine (delta from last CSV row)
    last_count  = _get_last_csv_count(csv_path)
    delta       = abs(reported_count - last_count)
    cv_features = {
        "ships_in_zone": reported_count,
        "delta":         delta,
        "last_count":    last_count,
        "timestamp":     datetime.utcnow().isoformat(),
    }

    # Policy engine — deterministic decision
    print(f"[POLICY] Evaluating: ships={reported_count}, delta={delta}, state={current_state}")
    decision = policy_engine.evaluate(cv_features, evidence_a, evidence_b, current_state)
    print(f"[POLICY] Alert level  : {decision['alert_level']}")
    print(f"[POLICY] Email        : {decision['email_decision']}")
    print(f"[POLICY] Avg conf     : {decision['avg_confidence']:.2f}")
    print(f"[POLICY] Transition   : {decision['proposed_transition']}")
    print(f"[POLICY] Reasoning    : {' | '.join(decision['reasoning'])}")

    # Apply state transition (validated by state_manager)
    proposed = decision.get("proposed_transition", "NONE")
    if proposed != "NONE":
        changed, new_state = apply_transition(proposed)
        decision["applied_state"] = new_state
        if changed:
            print(f"[POLICY] ✓ State transition applied: {proposed} → now {new_state}")
        else:
            print(f"[POLICY] Transition '{proposed}' rejected by state_manager.")

    # Build final briefing
    final_briefing = _build_final_briefing(
        decision, evidence_a, evidence_b, reported_count, model_a, model_b, current_state
    )

    # Audit log
    _write_audit(
        f"{model_a}+{model_b}", 0.0, False,
        final_briefing,
        consensus_label=decision["alert_level"],
        news=news,
        evidence_a=evidence_a,
        evidence_b=evidence_b,
        policy_decision=decision,
    )

    # Escalation banner
    _print_escalation_banner(decision, model_a, model_b)

    # Send alert if policy engine says so
    if decision["email_decision"]:
        send_alert(
            final_briefing,
            model_a=model_a, model_b=model_b,
            reported_count=reported_count,
            image_paths=image_paths,
            csv_snapshot=csv_snapshot,
            alert_cooldown=alert_cooldown,
            news=news,
            policy_decision=decision,
        )
    else:
        print(f"[POLICY] No email — alert_level={decision['alert_level']} does not warrant notification.")

    return final_briefing


# ------------------------------------------------------------------
# Standalone CLI entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Maritime OSINT Analyst — standalone test")
    parser.add_argument(
        "--model", default="llama_4_scout",
        help="Model key or 'consensus'. Options: llama_4_scout, llama_3_3_70b, "
             "llama_3_1_8b, anthropic, openrouter_gpt4o, openrouter_gemini, consensus"
    )
    parser.add_argument("--model-a", default="llama_4_scout",
                        help="Model A for consensus (default: llama_4_scout)")
    parser.add_argument("--model-b", default="llama_3_3_70b",
                        help="Model B for consensus (default: llama_3_3_70b)")
    parser.add_argument("--count", type=int, default=2,
                        help="Reported ship count for the test (default: 2)")
    args = parser.parse_args()

    import glob as _glob
    screenshots_dir = os.path.join(_HERE, "screenshots")
    candidates = sorted(
        _glob.glob(os.path.join(screenshots_dir, "**", "detected_ships_*.png"), recursive=True)
    ) if os.path.isdir(screenshots_dir) else []

    if not candidates:
        print("[ANALYST] No detected_ships screenshots found. Run marine_traffic_monitor.py first.")
        sys.exit(1)

    latest = candidates[-1]
    print(f"[ANALYST] Using screenshot: {latest}\n")

    if args.model == "consensus":
        result = run_consensus_check(
            latest, args.count, CSV_PATH,
            model_a=args.model_a, model_b=args.model_b
        )
    else:
        result = run_analyst(latest, args.count, CSV_PATH,
                             model_key=args.model)

    if result:
        print("\n[ANALYST OUTPUT]")
        print(json.dumps(result, indent=2, default=str))
