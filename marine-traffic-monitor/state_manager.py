"""
state_manager.py — Temporal macro-state tracker for the Hormuz choke point.
=============================================================================
Replaces per-cycle "snapshot" reasoning with a persistent, validated state
machine so the LLM accumulates situational awareness across many intervals.

States
------
  NORMAL           Baseline. Few or no ships expected.
  SURGE            Unusual uptick detected; elevated monitoring.
  BLOCKADE_ACTIVE  Confirmed sustained blockade breach; maximum alert.
  RECOVERY         Ship count declining after an active state.

Allowed transitions (enforced):
  NORMAL          → SURGE
  SURGE           → BLOCKADE_ACTIVE | NORMAL
  BLOCKADE_ACTIVE → RECOVERY
  RECOVERY        → NORMAL

Storage: current_state.json (same directory as this file).

Usage:
  from state_manager import get_current_state, apply_transition

  state = get_current_state()                          # → "NORMAL"
  changed, new = apply_transition("NORMAL -> SURGE")   # → (True, "SURGE")
  changed, new = apply_transition("NONE")              # → (False, "SURGE")

Standalone test:
  python state_manager.py
"""

import json
import os
from datetime import datetime, timezone

_HERE      = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_HERE, "state"), exist_ok=True)
STATE_PATH = os.path.join(_HERE, "state", "current_state.json")

VALID_STATES  = {"NORMAL", "SURGE", "BLOCKADE_ACTIVE", "RECOVERY"}
DEFAULT_STATE = "NORMAL"

# Destination states reachable from each source state
VALID_TRANSITIONS: dict[str, set] = {
    "NORMAL":          {"SURGE"},                       # can only escalate
    "SURGE":           {"BLOCKADE_ACTIVE", "NORMAL"},   # confirm or de-escalate
    "BLOCKADE_ACTIVE": {"RECOVERY"},                    # must go through recovery
    "RECOVERY":        {"NORMAL"},                      # full de-escalation
}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_current_state() -> str:
    """
    Read the current macro state from disk.
    Returns DEFAULT_STATE ('NORMAL') on missing file, corrupt JSON, or unknown value.
    Never raises.
    """
    try:
        with open(STATE_PATH) as f:
            data = json.load(f)
        state = data.get("state", DEFAULT_STATE)
        return state if state in VALID_STATES else DEFAULT_STATE
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_STATE


def apply_transition(proposed_transition: str) -> tuple[bool, str]:
    """
    Parse and apply a state transition proposed by the LLM.

    Accepted input formats:
      "NONE"                   → no-op
      "NORMAL -> SURGE"        → validated & applied if rules allow
      "normal -> surge"        → case-insensitive

    Rejection rules (logged, never raise):
      - Blank / "NONE"         → no-op
      - Malformed string       → ignored
      - Unknown state names    → ignored
      - from_state ≠ current   → stale, ignored (race-condition safety)
      - to_state not in allowed → invalid, ignored

    Returns:
      (changed: bool, new_state: str)
        changed   = True if the state was actually written to disk
        new_state = the effective current state after this call
    """
    current = get_current_state()

    if not proposed_transition or proposed_transition.strip().upper() == "NONE":
        return False, current

    parts = [p.strip().upper() for p in proposed_transition.split("->")]
    if len(parts) != 2:
        print(f"[STATE] Malformed transition string: '{proposed_transition}' — ignored.")
        return False, current

    from_state, to_state = parts

    if from_state not in VALID_STATES or to_state not in VALID_STATES:
        print(f"[STATE] Unknown state(s) in '{proposed_transition}' — ignored.")
        return False, current

    if from_state != current:
        print(f"[STATE] Stale transition '{proposed_transition}' "
              f"(current is '{current}') — ignored.")
        return False, current

    allowed = VALID_TRANSITIONS.get(current, set())
    if to_state not in allowed:
        print(f"[STATE] Invalid transition {current} → {to_state} "
              f"(allowed from {current}: {allowed}) — ignored.")
        return False, current

    _write_state(to_state)
    print(f"[STATE] ✓ Transition applied: {current} → {to_state}")
    return True, to_state


def get_state_description() -> str:
    """Return a human-readable description of the current state for prompt injection."""
    state = get_current_state()
    desc = {
        "NORMAL":          "Baseline. Few or no ships expected. No active threat.",
        "SURGE":           "Unusual uptick detected. Elevated monitoring in effect.",
        "BLOCKADE_ACTIVE": "Confirmed sustained blockade breach. Maximum alert.",
        "RECOVERY":        "Ship count declining after active blockade phase.",
    }
    return f"{state} — {desc.get(state, '')}"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _write_state(state: str) -> None:
    """Persist state to disk with a UTC timestamp."""
    with open(STATE_PATH, "w") as f:
        json.dump({
            "state":      state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    def _check(label: str, expected_changed: bool, expected_state: str,
               changed: bool, new_state: str) -> bool:
        ok = (changed == expected_changed and new_state == expected_state)
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {label}: changed={changed}, state={new_state}"
              + ("" if ok else f"  ← EXPECTED changed={expected_changed}, state={expected_state}"))
        return ok

    print("=" * 60)
    print("  state_manager.py — Transition Logic Test")
    print("=" * 60)

    # Reset to clean slate
    _write_state("NORMAL")
    print(f"\nInitial state: {get_current_state()}\n")

    results = []
    # Valid escalation path
    results.append(_check("NORMAL → SURGE (valid)",          True,  "SURGE",           *apply_transition("NORMAL -> SURGE")))
    results.append(_check("SURGE → BLOCKADE_ACTIVE (valid)", True,  "BLOCKADE_ACTIVE", *apply_transition("SURGE -> BLOCKADE_ACTIVE")))
    results.append(_check("BLOCKADE_ACTIVE → RECOVERY",      True,  "RECOVERY",        *apply_transition("BLOCKADE_ACTIVE -> RECOVERY")))
    results.append(_check("RECOVERY → NORMAL",               True,  "NORMAL",          *apply_transition("RECOVERY -> NORMAL")))

    # Invalid / edge cases
    results.append(_check("'NONE' → no-op",                  False, "NORMAL",          *apply_transition("NONE")))
    results.append(_check("Stale transition (SURGE→X from NORMAL)", False, "NORMAL",  *apply_transition("SURGE -> BLOCKADE_ACTIVE")))
    results.append(_check("Skip step (NORMAL → BLOCKADE_ACTIVE)", False, "NORMAL",    *apply_transition("NORMAL -> BLOCKADE_ACTIVE")))
    results.append(_check("Case-insensitive 'normal -> surge'",     True,  "SURGE",   *apply_transition("normal -> surge")))

    # Reset
    _write_state("NORMAL")

    print(f"\nFinal state reset to: {get_current_state()}")
    passed = sum(results)
    total  = len(results)
    print(f"\n{'=' * 60}")
    print(f"  Tests: {passed}/{total} passed {'✓' if passed == total else '✗'}")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)
