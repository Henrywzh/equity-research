"""
policy_engine.py — Deterministic Policy Engine
================================================
The single source of truth for all escalation decisions.

This module takes structured evidence from two LLM analysts and CV features,
then applies a deterministic rule set to decide:
  - alert_level     : NORMAL | WATCH | ESCALATED | DISPUTED | REVIEW
  - email_decision  : bool
  - human_review    : bool
  - proposed_transition : state machine suggestion (validated separately)

IMPORTANT: This module contains NO LLM calls. All logic is pure Python.
The LLMs produce evidence; this engine produces decisions.

Urgency map:
  NORMAL    = 0  (no action)
  WATCH     = 1  (log only, no email)
  ESCALATED = 2  (alert email)
  CRITICAL  = 3  (alert email, human review)

Policy rules (applied in strict priority order):
  P1  Either model abstains → REVIEW
  P2  Both agree NORMAL     → NORMAL
  P3  Both agree WATCH      → WATCH  (if avg_conf ≥ 0.4, else NORMAL)
  P4  Both agree ESCALATED+ → ESCALATED (if avg_conf ≥ 0.55, else REVIEW)
  P5  Gap ≥ 2 urgency levels → DISPUTED
  P6  Gap == 1, avg_conf ≥ 0.5 → take higher
  P7  Gap == 1, avg_conf < 0.5  → REVIEW

Standalone test:
  python policy_engine.py
"""

from datetime import datetime, timezone

# ------------------------------------------------------------------
# Urgency mapping
# ------------------------------------------------------------------
URGENCY: dict[str, int] = {
    "NORMAL":    0,
    "WATCH":     1,
    "ESCALATED": 2,
    "CRITICAL":  3,
}

# Minimum confidence thresholds
CONF_WATCH_MIN      = 0.40   # below this, WATCH → NORMAL
CONF_ESCALATE_MIN   = 0.55   # below this, ESCALATED agreement → REVIEW
CONF_MINOR_DISAGREE = 0.50   # below this, minor disagreement → REVIEW


# ------------------------------------------------------------------
# State transition suggestion (pure logic, no file I/O)
# ------------------------------------------------------------------

def _suggest_transition(alert_level: str, current_state: str) -> str:
    """
    Suggest a state machine transition based on the policy decision.
    The suggestion is still validated and written by state_manager.apply_transition().
    """
    if alert_level in ("ESCALATED", "CRITICAL"):
        if current_state == "NORMAL":
            return "NORMAL -> SURGE"
        elif current_state == "SURGE":
            return "SURGE -> BLOCKADE_ACTIVE"
        # BLOCKADE_ACTIVE stays, RECOVERY re-escalates to SURGE via NORMAL first
        elif current_state == "RECOVERY":
            return "RECOVERY -> NORMAL"   # policy engine de-escalates first
    elif alert_level == "WATCH":
        if current_state == "NORMAL":
            return "NORMAL -> SURGE"      # tentative upward move
        # If already elevated, stay (policy engine doesn't de-escalate on WATCH alone)
    elif alert_level in ("NORMAL", "REVIEW"):
        if current_state == "SURGE":
            return "SURGE -> NORMAL"
        elif current_state == "RECOVERY":
            return "RECOVERY -> NORMAL"
        elif current_state == "BLOCKADE_ACTIVE":
            return "BLOCKADE_ACTIVE -> RECOVERY"
    return "NONE"


# ------------------------------------------------------------------
# Internal decision builder
# ------------------------------------------------------------------

def _decision(alert_level: str, current_state: str, avg_conf: float,
              email: bool, human_review: bool,
              reasoning: list[str], ships: int) -> dict:
    """Assemble the final PolicyDecision dict."""
    return {
        "alert_level":          alert_level,
        "proposed_transition":  _suggest_transition(alert_level, current_state),
        "email_decision":       email,
        "human_review_needed":  human_review,
        "reasoning":            reasoning,
        "avg_confidence":       round(avg_conf, 3),
        "ships_in_zone":        ships,
        "applied_state":        current_state,   # updated by caller after transition
        "evaluated_at":         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def evaluate(
    cv_features:    dict,   # {ships_in_zone, delta, last_count, timestamp}
    evidence_a:     dict,   # from visual_analyst model
    evidence_b:     dict,   # from context_analyst model
    current_state:  str,    # from state_manager.get_current_state()
) -> dict:
    """
    Apply the deterministic policy rule set and return a PolicyDecision dict.

    Args:
        cv_features:   CV-derived features for this cycle.
        evidence_a:    Evidence dict from Model A (visual_analyst).
        evidence_b:    Evidence dict from Model B (context_analyst).
        current_state: Current macro state from state_manager.

    Returns:
        PolicyDecision dict with keys:
          alert_level, proposed_transition, email_decision,
          human_review_needed, reasoning, avg_confidence,
          ships_in_zone, applied_state, evaluated_at
    """
    # Extract model recommendations & confidence
    a_state   = evidence_a.get("recommended_state", "NORMAL")
    b_state   = evidence_b.get("recommended_state", "NORMAL")
    a_conf    = float(evidence_a.get("overall_confidence", 0.5))
    b_conf    = float(evidence_b.get("overall_confidence", 0.5))
    a_abstain = bool(evidence_a.get("abstain", False))
    b_abstain = bool(evidence_b.get("abstain", False))

    # Guard against unknown states
    if a_state not in URGENCY:
        a_state = "NORMAL"
    if b_state not in URGENCY:
        b_state = "NORMAL"

    ships       = int(cv_features.get("ships_in_zone", 0))
    delta       = int(cv_features.get("delta", 0))
    avg_conf    = (a_conf + b_conf) / 2
    a_urgency   = URGENCY[a_state]
    b_urgency   = URGENCY[b_state]
    urgency_gap = abs(a_urgency - b_urgency)
    reasoning: list[str] = []

    # ── P1: Abstain override ──────────────────────────────────────
    if a_abstain or b_abstain:
        who = ("model_a" if a_abstain else "") + ("+" if a_abstain and b_abstain else "") + ("model_b" if b_abstain else "")
        reasoning.append(f"p1_abstain: {who} abstained")
        reasoning.append(f"p1_reason_a: {evidence_a.get('abstain_reason') or 'n/a'}")
        reasoning.append(f"p1_reason_b: {evidence_b.get('abstain_reason') or 'n/a'}")
        return _decision("REVIEW", current_state, avg_conf, True, True, reasoning, ships)

    # ── P2: Both agree NORMAL ─────────────────────────────────────
    if a_urgency == 0 and b_urgency == 0:
        reasoning.append("p2_both_normal")
        return _decision("NORMAL", current_state, avg_conf, False, False, reasoning, ships)

    # ── P3: Both agree WATCH ──────────────────────────────────────
    if a_urgency == 1 and b_urgency == 1:
        if avg_conf >= CONF_WATCH_MIN:
            reasoning.append(f"p3_both_watch_conf={avg_conf:.2f}")
            return _decision("WATCH", current_state, avg_conf, False, False, reasoning, ships)
        else:
            reasoning.append(f"p3_both_watch_but_low_conf={avg_conf:.2f}<{CONF_WATCH_MIN} → NORMAL")
            return _decision("NORMAL", current_state, avg_conf, False, False, reasoning, ships)

    # ── P4: Both agree ESCALATED or CRITICAL ─────────────────────
    if a_urgency >= 2 and b_urgency >= 2:
        if avg_conf >= CONF_ESCALATE_MIN:
            # Use the higher of the two as the final label
            final = "CRITICAL" if (a_urgency == 3 and b_urgency == 3) else "ESCALATED"
            reasoning.append(f"p4_both_escalated final={final} conf={avg_conf:.2f}")
            return _decision(final, current_state, avg_conf, True, False, reasoning, ships)
        else:
            reasoning.append(f"p4_both_escalated_but_low_conf={avg_conf:.2f}<{CONF_ESCALATE_MIN} → REVIEW")
            return _decision("REVIEW", current_state, avg_conf, True, True, reasoning, ships)

    # ── P5: Major disagreement (gap ≥ 2) ─────────────────────────
    if urgency_gap >= 2:
        reasoning.append(
            f"p5_major_disagreement: model_a={a_state}(u={a_urgency}) "
            f"vs model_b={b_state}(u={b_urgency}) gap={urgency_gap}"
        )
        return _decision("DISPUTED", current_state, avg_conf, True, True, reasoning, ships)

    # ── P6/P7: Minor disagreement (gap == 1) ─────────────────────
    higher_urgency = max(a_urgency, b_urgency)
    higher_state   = a_state if a_urgency >= b_urgency else b_state
    reasoning.append(
        f"p6_minor_disagreement: model_a={a_state} vs model_b={b_state} → higher={higher_state}"
    )

    if avg_conf >= CONF_MINOR_DISAGREE:
        email_needed = higher_urgency >= 2
        reasoning.append(f"p6_conf_ok={avg_conf:.2f} email={email_needed}")
        # Return the higher-urgency state
        final = higher_state if higher_urgency >= 1 else "WATCH"
        return _decision(final, current_state, avg_conf, email_needed, False, reasoning, ships)
    else:
        reasoning.append(f"p7_low_conf={avg_conf:.2f}<{CONF_MINOR_DISAGREE} → REVIEW")
        return _decision("REVIEW", current_state, avg_conf, True, True, reasoning, ships)


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  Policy Engine — Standalone Test")
    print("=" * 60)

    cv = {"ships_in_zone": 3, "delta": 2, "last_count": 1, "timestamp": "2026-03-17T10:00:00Z"}

    # Test 1: Both escalated, high confidence → ESCALATED
    ea = {"recommended_state": "ESCALATED", "overall_confidence": 0.72,
          "abstain": False, "abstain_reason": None,
          "hypotheses": [{"statement": "Convoy transit", "confidence": 0.72}]}
    eb = {"recommended_state": "ESCALATED", "overall_confidence": 0.65,
          "abstain": False, "abstain_reason": None,
          "hypotheses": [{"statement": "Unusual clustering", "confidence": 0.65}]}
    d = evaluate(cv, ea, eb, "NORMAL")
    print(f"\nTest 1 (both ESCALATED, high conf): alert={d['alert_level']} email={d['email_decision']}")
    print(f"  Transition: {d['proposed_transition']} | Reasoning: {d['reasoning']}")

    # Test 2: Major disagreement → DISPUTED
    ea2 = {"recommended_state": "CRITICAL", "overall_confidence": 0.80,
           "abstain": False, "abstain_reason": None, "hypotheses": []}
    eb2 = {"recommended_state": "NORMAL",   "overall_confidence": 0.70,
           "abstain": False, "abstain_reason": None, "hypotheses": []}
    d2 = evaluate(cv, ea2, eb2, "SURGE")
    print(f"\nTest 2 (CRITICAL vs NORMAL):         alert={d2['alert_level']} email={d2['email_decision']}")
    print(f"  Reasoning: {d2['reasoning']}")

    # Test 3: Abstain from model A → REVIEW
    ea3 = {"recommended_state": "WATCH", "overall_confidence": 0.30,
           "abstain": True, "abstain_reason": "Image too blurry to classify",
           "hypotheses": []}
    eb3 = {"recommended_state": "WATCH", "overall_confidence": 0.60,
           "abstain": False, "abstain_reason": None, "hypotheses": []}
    d3 = evaluate(cv, ea3, eb3, "NORMAL")
    print(f"\nTest 3 (model A abstains):           alert={d3['alert_level']} human_review={d3['human_review_needed']}")

    # Test 4: Both NORMAL → NORMAL, no email
    ea4 = {"recommended_state": "NORMAL", "overall_confidence": 0.85,
           "abstain": False, "abstain_reason": None, "hypotheses": []}
    eb4 = {"recommended_state": "NORMAL", "overall_confidence": 0.80,
           "abstain": False, "abstain_reason": None, "hypotheses": []}
    d4 = evaluate({"ships_in_zone": 0, "delta": 0, "last_count": 0}, ea4, eb4, "NORMAL")
    print(f"\nTest 4 (both NORMAL):                alert={d4['alert_level']} email={d4['email_decision']}")

    print("\n" + "=" * 60)
    checks = [
        d["alert_level"]  == "ESCALATED",
        d2["alert_level"] == "DISPUTED",
        d3["alert_level"] == "REVIEW",
        d4["alert_level"] == "NORMAL",
    ]
    print("✓  All 4 tests passed." if all(checks) else f"⚠  {checks.count(False)} test(s) failed.")
    print("=" * 60)
