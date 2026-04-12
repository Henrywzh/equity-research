import csv
import glob
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(
    os.environ.get("EQUITY_RESEARCH_ROOT", Path(__file__).resolve().parents[1])
).resolve()
HUB_DATA_DIR = ROOT / "hub" / "data"
SIGNALS_OUTPUT_PATH = HUB_DATA_DIR / "signals.json"
HORMUZ_OUTPUT_PATH = HUB_DATA_DIR / "hormuz.json"

MARINE_ROOT = ROOT / "marine-traffic-monitor"
TRAFFIC_LOG_PATH = MARINE_ROOT / "data" / "hormuz_traffic_log.csv"
AUDIT_LOG_PATH = MARINE_ROOT / "logs" / "analyst_audit.jsonl"
CURRENT_STATE_PATH = MARINE_ROOT / "state" / "current_state.json"
LAST_ALERT_PATH = MARINE_ROOT / "state" / "last_alert.json"
RATE_COUNTERS_PATH = MARINE_ROOT / "state" / "rate_counters.json"
SCREENSHOTS_DIR = MARINE_ROOT / "screenshots"


def _safe_load_json(path: Path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    except Exception as exc:
        print(f"JSON parse error for {path}: {exc}")
    return default


def _safe_mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def _parse_hormuz_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            continue
    return None


def _read_traffic_rows() -> list[dict[str, object]]:
    if not TRAFFIC_LOG_PATH.exists():
        return []

    rows: list[dict[str, object]] = []
    try:
        with TRAFFIC_LOG_PATH.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                detected_ships = int(raw.get("Detected_Ships") or 0)
                timestamp_raw = str(raw.get("Timestamp") or "")
                timestamp_dt = _parse_hormuz_timestamp(timestamp_raw)
                rows.append(
                    {
                        "timestamp": timestamp_raw,
                        "timestamp_iso": timestamp_dt.isoformat() if timestamp_dt else None,
                        "date": timestamp_raw.split("_")[0] if "_" in timestamp_raw else timestamp_raw[:10],
                        "detected_ships": detected_ships,
                        "status_note": str(raw.get("Status_Note") or "No status note"),
                    }
                )
    except Exception as exc:
        print(f"Maritime parse error: {exc}")
    return rows


def _load_recent_alerts(limit: int = 8) -> list[dict[str, object]]:
    if not AUDIT_LOG_PATH.exists():
        return []

    try:
        lines = [line for line in AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        print(f"Audit parse error: {exc}")
        return []

    alerts: list[dict[str, object]] = []
    for raw in reversed(lines[-limit:]):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        policy_decision = payload.get("policy_decision") or {}
        alerts.append(
            {
                "timestamp": payload.get("ts"),
                "alert_level": payload.get("alert_level") or "UNKNOWN",
                "threat_level": payload.get("threat_level") or "UNKNOWN",
                "consensus": payload.get("consensus") or "UNKNOWN",
                "macro_state": payload.get("macro_state") or "UNKNOWN",
                "state_transition": payload.get("state_transition") or "NONE",
                "applied_state": payload.get("applied_macro_state") or payload.get("macro_state") or "UNKNOWN",
                "avg_confidence": float(payload.get("avg_confidence") or 0),
                "human_review_needed": bool(payload.get("human_review_needed")),
                "ships_in_zone": int(policy_decision.get("ships_in_zone") or 0),
                "summary": "; ".join(policy_decision.get("reasoning") or []) or payload.get("news_headlines") or "No supporting context recorded.",
                "news_headlines": payload.get("news_headlines") or "",
            }
        )
    return alerts


def _load_screenshots(limit: int = 8) -> list[dict[str, str | None]]:
    if not SCREENSHOTS_DIR.exists():
        return []

    screenshots = sorted(
        (path for path in SCREENSHOTS_DIR.glob("*") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    items: list[dict[str, str | None]] = []
    for path in screenshots[:limit]:
        stem = path.stem
        screenshot_type = "screenshot"
        timestamp = stem
        label = path.name
        if stem.startswith("detected_ships_"):
            screenshot_type = "detected"
            timestamp = stem.removeprefix("detected_ships_")
            label = "Detected ships overlay"
        elif stem.startswith("raw_map_"):
            screenshot_type = "raw"
            timestamp = stem.removeprefix("raw_map_")
            label = "Raw map capture"

        timestamp_dt = _parse_hormuz_timestamp(timestamp)
        items.append(
            {
                "name": path.name,
                "label": label,
                "type": screenshot_type,
                "timestamp": timestamp,
                "timestamp_iso": timestamp_dt.isoformat() if timestamp_dt else None,
                "relative_path": f"../marine-traffic-monitor/screenshots/{path.name}",
            }
        )
    return items


def build_hormuz_payload() -> dict[str, object]:
    traffic_rows = _read_traffic_rows()
    latest_row = traffic_rows[-1] if traffic_rows else {}

    current_state = _safe_load_json(CURRENT_STATE_PATH, {})
    last_alert = _safe_load_json(LAST_ALERT_PATH, {})
    rate_counters = _safe_load_json(RATE_COUNTERS_PATH, {})
    recent_alerts = _load_recent_alerts()
    screenshots = _load_screenshots()

    daily_buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in traffic_rows:
        daily_buckets[str(row.get("date") or "unknown")].append(row)

    daily_history: list[dict[str, object]] = []
    for day in sorted(daily_buckets):
        rows = daily_buckets[day]
        counts = [int(row.get("detected_ships") or 0) for row in rows]
        statuses = [str(row.get("status_note") or "") for row in rows]
        daily_history.append(
            {
                "date": day,
                "run_count": len(rows),
                "max_ships": max(counts) if counts else 0,
                "avg_ships": round(sum(counts) / len(counts), 2) if counts else 0.0,
                "latest_ships": counts[-1] if counts else 0,
                "warning_count": sum(1 for status in statuses if "warning" in status.lower()),
                "clear_zone_count": sum(1 for status in statuses if "clear zone" in status.lower()),
            }
        )

    latest_alert = recent_alerts[0] if recent_alerts else {}
    freshness = {
        "traffic_log": {"path": str(TRAFFIC_LOG_PATH.relative_to(ROOT)), "modified_at": _safe_mtime_iso(TRAFFIC_LOG_PATH)},
        "audit_log": {"path": str(AUDIT_LOG_PATH.relative_to(ROOT)), "modified_at": _safe_mtime_iso(AUDIT_LOG_PATH)},
        "current_state": {"path": str(CURRENT_STATE_PATH.relative_to(ROOT)), "modified_at": _safe_mtime_iso(CURRENT_STATE_PATH)},
        "last_alert": {"path": str(LAST_ALERT_PATH.relative_to(ROOT)), "modified_at": _safe_mtime_iso(LAST_ALERT_PATH)},
        "rate_counters": {"path": str(RATE_COUNTERS_PATH.relative_to(ROOT)), "modified_at": _safe_mtime_iso(RATE_COUNTERS_PATH)},
        "screenshots_dir": {"path": str(SCREENSHOTS_DIR.relative_to(ROOT)), "modified_at": _safe_mtime_iso(SCREENSHOTS_DIR) if SCREENSHOTS_DIR.exists() else None},
    }

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "current": {
            "latest_ship_count": int(latest_row.get("detected_ships") or 0),
            "latest_status_note": latest_row.get("status_note") or "No Recent Data",
            "latest_timestamp": latest_row.get("timestamp"),
            "latest_timestamp_iso": latest_row.get("timestamp_iso"),
            "macro_state": current_state.get("state") or "UNKNOWN",
            "macro_state_updated_at": current_state.get("updated_at"),
            "last_alert_timestamp": last_alert.get("ts"),
            "rate_counter_date": rate_counters.get("date"),
            "groq_calls_today": int(rate_counters.get("groq_calls") or 0),
            "latest_alert_level": latest_alert.get("alert_level") or "NONE",
            "latest_alert_confidence": latest_alert.get("avg_confidence") or 0,
        },
        "recent_cycles": traffic_rows[-16:],
        "daily_history": daily_history,
        "recent_alerts": recent_alerts,
        "screenshots": screenshots,
        "freshness": freshness,
    }


def get_latest_maritime() -> dict[str, object]:
    hormuz = build_hormuz_payload()
    current = hormuz["current"]
    assert isinstance(current, dict)
    return {
        "count": current.get("latest_ship_count", 0),
        "status": current.get("latest_status_note", "No Recent Data"),
        "updated": current.get("latest_timestamp", "N/A"),
        "updated_iso": current.get("latest_timestamp_iso"),
    }


def get_latest_market():
    data = {"sections": [], "updated": "N/A", "top_mover_val": 0, "top_mover_ticker": "N/A"}
    try:
        summary_glob = str(ROOT / "daily-market" / "data" / "summaries" / "*" / "*.json")
        files = glob.glob(summary_glob)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            with open(files[0], "r", encoding="utf-8") as handle:
                raw_data = json.load(handle)
                data["sections"] = raw_data.get("sections", [])
                data["updated"] = raw_data.get("date", "N/A")

                max_change = 0
                max_ticker = "N/A"
                for section in data["sections"]:
                    for row in section.get("rows", []):
                        val = abs(float(row.get("pct_change", 0)))
                        if val > max_change:
                            max_change = val
                            max_ticker = row.get("ticker", "N/A")
                            data["top_mover_val"] = row.get("pct_change", 0)
                            data["top_mover_ticker"] = max_ticker
    except Exception as exc:
        print(f"Market parse error: {exc}")
    return data


def get_latest_macro():
    data = {"alerts": [], "sentiment": "Pending LLM", "summary": [], "updated": "N/A"}
    try:
        analysis_glob = str(ROOT / "daily-macro" / "data" / "analyses" / "*" / "*.json")
        files = glob.glob(analysis_glob)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            with open(files[0], "r", encoding="utf-8") as handle:
                raw_data = json.load(handle)
                data["summary"] = raw_data.get("executive_summary", [])
                data["updated"] = raw_data.get("report_date", "N/A")

                alerts = []
                for category in raw_data.get("categories", []):
                    developments = category.get("key_developments", [])
                    if developments:
                        alerts.extend(developments)
                data["alerts"] = alerts
                data["sentiment"] = "Pending LLM"
    except Exception as exc:
        print(f"Macro parse error: {exc}")
    return data


def get_latest_youtube():
    data = {"signals": [], "total_analyzed": 0, "updated": "N/A"}
    try:
        run_glob = str(ROOT / "youtube-intake" / "data" / "analysis" / "*")
        runs = [Path(path) for path in glob.glob(run_glob) if os.path.isdir(path)]
        if runs:
            runs.sort(key=os.path.getmtime, reverse=True)
            latest_run = runs[0]
            summary_path = latest_run / "run-summary.json"
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as handle:
                    raw_data = json.load(handle)
                    run_info = raw_data.get("run_summary", {})

                    signals = []
                    for claim in run_info.get("top_claims_worth_watching", []):
                        signals.append({"title": claim, "channel_name": "Multi-Source Alpha"})
                    data["signals"] = signals

                    total = 0
                    for channel in raw_data.get("channels", {}).values():
                        total += channel.get("video_count", 0)
                    data["total_analyzed"] = total
                    data["updated"] = latest_run.name
    except Exception as exc:
        print(f"YouTube parse error: {exc}")
    return data


def bake_cake():
    print("Aggregation started...")
    hormuz_data = build_hormuz_payload()
    hub_data = {
        "maritime": get_latest_maritime(),
        "market": get_latest_market(),
        "macro": get_latest_macro(),
        "youtube": get_latest_youtube(),
        "last_baked": datetime.now(tz=UTC).isoformat(),
    }

    HUB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with SIGNALS_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(hub_data, handle, indent=2)
    with HORMUZ_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(hormuz_data, handle, indent=2)

    print(f"Aggregated signal cake baked at {SIGNALS_OUTPUT_PATH}")
    print(f"Hormuz dashboard payload baked at {HORMUZ_OUTPUT_PATH}")


if __name__ == "__main__":
    bake_cake()
