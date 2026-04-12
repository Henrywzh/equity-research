import csv
import glob
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

# Add internal package paths for direct imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "daily-macro" / "src"))

ROOT = Path(
    os.environ.get("EQUITY_RESEARCH_ROOT", Path(__file__).resolve().parents[1])
).resolve()
HUB_DATA_DIR = ROOT / "hub" / "data"
SIGNALS_OUTPUT_PATH = HUB_DATA_DIR / "signals.json"
HORMUZ_OUTPUT_PATH = HUB_DATA_DIR / "hormuz.json"
POLYMARKET_OUTPUT_PATH = HUB_DATA_DIR / "polymarket.json"

MARINE_ROOT = ROOT / "marine-traffic-monitor"
TRAFFIC_LOG_PATH = MARINE_ROOT / "data" / "hormuz_traffic_log.csv"
AUDIT_LOG_PATH = MARINE_ROOT / "logs" / "analyst_audit.jsonl"
CURRENT_STATE_PATH = MARINE_ROOT / "state" / "current_state.json"
LAST_ALERT_PATH = MARINE_ROOT / "state" / "last_alert.json"
RATE_COUNTERS_PATH = MARINE_ROOT / "state" / "rate_counters.json"
SCREENSHOTS_DIR = MARINE_ROOT / "screenshots"
HUB_SCREENSHOTS_DIR = ROOT / "hub" / "screenshots"
POLYMARKET_RUNS_DIR = ROOT / "daily-market" / "data" / "polymarket_runs"


def _safe_load_json(path: Path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    except Exception as exc:
        print(f"JSON parse error for {path}: {exc}")
    return default


def _safe_mtime_iso(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _parse_hormuz_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _read_traffic_rows() -> List[Dict[str, object]]:
    if not TRAFFIC_LOG_PATH.exists():
        return []

    rows: List[Dict[str, object]] = []
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


def _load_recent_alerts(limit: int = 8) -> List[Dict[str, object]]:
    if not AUDIT_LOG_PATH.exists():
        return []

    try:
        lines = [line for line in AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        print(f"Audit parse error: {exc}")
        return []

    alerts: List[Dict[str, object]] = []
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


def _load_screenshots(limit: int = 8) -> List[Dict[str, Union[str, None]]]:
    if not SCREENSHOTS_DIR.exists():
        return []

    screenshots = sorted(
        (path for path in SCREENSHOTS_DIR.glob("*") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    items: List[Dict[str, Union[str, None]]] = []
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
                "relative_path": f"screenshots/{path.name}",
            }
        )
    return items


def build_hormuz_payload() -> Dict[str, object]:
    traffic_rows = _read_traffic_rows()
    latest_row = traffic_rows[-1] if traffic_rows else {}

    current_state = _safe_load_json(CURRENT_STATE_PATH, {})
    last_alert = _safe_load_json(LAST_ALERT_PATH, {})
    rate_counters = _safe_load_json(RATE_COUNTERS_PATH, {})
    recent_alerts = _load_recent_alerts()
    screenshots = _load_screenshots()

    daily_buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in traffic_rows:
        daily_buckets[str(row.get("date") or "unknown")].append(row)

    daily_history: List[Dict[str, object]] = []
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
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
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


def get_latest_maritime() -> Dict[str, object]:
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


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _load_polymarket_runs() -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    # Use rglob to find all run_*.json files in date-partitioned folders
    run_paths = sorted(POLYMARKET_RUNS_DIR.rglob("*.json"), key=os.path.getmtime, reverse=True)
    for path in run_paths:
        payload = _safe_load_json(path, None)
        if not isinstance(payload, dict):
            continue
        run = payload.get("run") or {}
        if not isinstance(run, dict):
            continue
        started_at = _parse_iso(run.get("started_at"))
        if started_at is None:
            continue
        payload["_started_at"] = started_at
        try:
            payload["_path"] = str(path.relative_to(ROOT))
        except ValueError:
            payload["_path"] = str(path)
        runs.append(payload)
    runs.sort(key=lambda item: item["_started_at"], reverse=True)
    return runs


def _find_prior_probability(
    history: List[tuple[datetime, float]],
    current_time: datetime,
    minimum_age: timedelta,
) -> Optional[float]:
    for ts, value in reversed(history):
        if current_time - ts >= minimum_age:
            return value
    return None


def _prepare_polymarket_rows(
    latest_payload: Dict[str, object],
    prior_runs: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    snapshots = latest_payload.get("snapshots") or []
    markets = latest_payload.get("markets") or []
    if not isinstance(snapshots, list):
        return []
    market_meta: Dict[str, Dict[str, object]] = {}
    if isinstance(markets, list):
        for market in markets:
            if not isinstance(market, dict):
                continue
            slug = str(market.get("market_slug") or "")
            if slug:
                market_meta[slug] = market

    history_by_market: Dict[str, List[tuple[datetime, float]]] = defaultdict(list)
    for payload in prior_runs:
        for snap in payload.get("snapshots") or []:
            if not isinstance(snap, dict):
                continue
            slug = str(snap.get("market_slug") or "")
            ts = _parse_iso(snap.get("fetched_at"))
            prob = snap.get("implied_probability")
            if not slug or ts is None or prob is None:
                continue
            try:
                history_by_market[slug].append((ts, float(prob)))
            except (TypeError, ValueError):
                continue

    for values in history_by_market.values():
        values.sort(key=lambda item: item[0])

    rows: List[Dict[str, object]] = []
    for snap in snapshots:
        if not isinstance(snap, dict):
            continue
        slug = str(snap.get("market_slug") or "")
        fetched_at = _parse_iso(snap.get("fetched_at"))
        if not slug or fetched_at is None:
            continue

        prob_raw = snap.get("implied_probability")
        try:
            probability = float(prob_raw) if prob_raw is not None else None
        except (TypeError, ValueError):
            probability = None

        delta_1d = None
        delta_7d = None
        pulse_history = []
        if probability is not None:
            history = history_by_market.get(slug, [])
            prior_1d = _find_prior_probability(history, fetched_at, timedelta(days=1))
            prior_7d = _find_prior_probability(history, fetched_at, timedelta(days=7))
            delta_1d = probability - prior_1d if prior_1d is not None else None
            delta_7d = probability - prior_7d if prior_7d is not None else None

            # Extract full time-series for "Pulse" markets (QQQ/BTC daily)
            if snap.get("group_key") in ("qqq_daily", "btc_daily"):
                cutoff = fetched_at - timedelta(hours=48)
                pulse_history = [
                    {"t": ts.isoformat(), "v": round(v * 100, 1)}
                    for ts, v in history
                    if ts >= cutoff
                ]
                pulse_history.append({"t": fetched_at.isoformat(), "v": round(probability * 100, 1)})

        rows.append(
            {
                "market_id": snap.get("market_id"),
                "market_slug": slug,
                "group_key": snap.get("group_key"),
                "asset": snap.get("asset"),
                "horizon": snap.get("horizon"),
                "question": snap.get("question") or market_meta.get(slug, {}).get("question"),
                "probability": probability,
                "probability_pct": round(probability * 100, 1) if probability is not None else None,
                "best_bid": snap.get("best_bid"),
                "best_ask": snap.get("best_ask"),
                "spread": snap.get("spread"),
                "last_trade_price": snap.get("last_trade_price"),
                "liquidity": snap.get("liquidity"),
                "volume": snap.get("volume"),
                "volume_24h": snap.get("volume_24h"),
                "expiry": snap.get("expiry_timestamp"),
                "market_status": snap.get("market_status"),
                "source_url": snap.get("source_url") or market_meta.get(slug, {}).get("source_url"),
                "delta_1d": round(delta_1d, 4) if delta_1d is not None else None,
                "delta_1d_pct": round(delta_1d * 100, 1) if delta_1d is not None else None,
                "delta_7d": round(delta_7d, 4) if delta_7d is not None else None,
                "delta_7d_pct": round(delta_7d * 100, 1) if delta_7d is not None else None,
                "history": pulse_history,
            }
        )
    return rows


def _group_polymarket_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    labels = {
        "fed_rates": "Fed rates",
        "qqq_daily": "QQQ daily",
        "btc_daily": "BTC daily",
        "spx_thresholds": "SPX thresholds",
        "btc_thresholds": "BTC thresholds",
        "gold_thresholds": "Gold thresholds",
        "oil_thresholds": "Oil thresholds",
    }
    ordered_keys = [
        "fed_rates",
        "qqq_daily",
        "btc_daily",
        "spx_thresholds",
        "btc_thresholds",
        "gold_thresholds",
        "oil_thresholds",
    ]
    grouped: List[Dict[str, object]] = []
    for key in ordered_keys:
        group_rows = [row for row in rows if row.get("group_key") == key]
        if group_rows:
            grouped.append({"group_key": key, "label": labels.get(key, key), "rows": group_rows})
    return grouped


def get_latest_polymarket() -> tuple[Dict[str, object], Dict[str, object]]:
    compact: Dict[str, object] = {
        "status": "missing",
        "updated": "N/A",
        "freshness_minutes": None,
        "error_count": 0,
        "errors": [],
        "fed_rates": [],
        "qqq_daily": None,
        "btc_daily": None,
        "largest_movers": [],
    }
    detailed: Dict[str, object] = {
        "status": "missing",
        "updated": "N/A",
        "freshness_minutes": None,
        "error_count": 0,
        "errors": [],
        "groups": [],
        "source_run_path": None,
    }

    try:
        runs = _load_polymarket_runs()
        if not runs:
            return compact, detailed

        latest = runs[0]
        run = latest.get("run") or {}
        errors = latest.get("errors") or []
        started_at = _parse_iso(run.get("started_at"))
        freshness_minutes = (
            round((datetime.now(tz=timezone.utc) - started_at).total_seconds() / 60)
            if started_at is not None
            else None
        )
        rows = _prepare_polymarket_rows(latest, runs[1:])

        compact.update(
            {
                "status": run.get("status") or "missing",
                "updated": run.get("started_at") or "N/A",
                "freshness_minutes": freshness_minutes,
                "error_count": len(errors),
                "errors": list(errors)[:5],
                "fed_rates": sorted(
                    [row for row in rows if row.get("group_key") == "fed_rates"],
                    key=lambda row: row.get("probability") or -1,
                    reverse=True,
                )[:4],
                "qqq_daily": next((row for row in rows if row.get("group_key") == "qqq_daily"), None),
                "btc_daily": next((row for row in rows if row.get("group_key") == "btc_daily"), None),
            }
        )

        movers = [row for row in rows if row.get("delta_1d_pct") is not None]
        movers.sort(key=lambda row: abs(row.get("delta_1d_pct") or 0), reverse=True)
        compact["largest_movers"] = movers[:5]

        detailed.update(
            {
                "status": run.get("status") or "missing",
                "updated": run.get("started_at") or "N/A",
                "freshness_minutes": freshness_minutes,
                "error_count": len(errors),
                "errors": list(errors),
                "groups": _group_polymarket_rows(rows),
                "source_run_path": latest.get("_path"),
            }
        )
    except Exception as exc:
        print(f"Polymarket parse error: {exc}")

    return compact, detailed


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


def get_latest_fred() -> Dict[str, object]:
    data: Dict[str, object] = {"items": [], "updated": "N/A", "status": "missing"}
    try:
        from daily_macro.release_calendar import build_release_digest

        # Get high-impact releases for the next 7 days
        digest = build_release_digest(start_date=datetime.now(timezone.utc).date(), days_ahead=7)
        data["items"] = [
            item for item in digest.get("items") or [] if item.get("impact") == "high"
        ][:5]
        data["updated"] = digest.get("generated_at", "N/A")
        data["status"] = digest.get("fetch_status", "success")
    except Exception as exc:
        print(f"FRED parse error: {exc}")
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
    polymarket_compact, polymarket_detail = get_latest_polymarket()
    hub_data = {
        "maritime": get_latest_maritime(),
        "market": get_latest_market(),
        "polymarket": polymarket_compact,
        "macro": get_latest_macro(),
        "fred": get_latest_fred(),
        "youtube": get_latest_youtube(),
        "last_baked": datetime.now(tz=timezone.utc).isoformat(),
    }

    HUB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    HUB_SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy latest screenshots to hub/screenshots/ for deployment
    for shot in hormuz_data.get("screenshots", []):
        src = SCREENSHOTS_DIR / shot["name"]
        dst = HUB_SCREENSHOTS_DIR / shot["name"]
        if src.exists():
            shutil.copy2(src, dst)

    with SIGNALS_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(hub_data, handle, indent=2)
    with HORMUZ_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(hormuz_data, handle, indent=2)
    with POLYMARKET_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(polymarket_detail, handle, indent=2)

    print(f"Aggregated signal cake baked at {SIGNALS_OUTPUT_PATH}")
    print(f"Hormuz dashboard payload baked at {HORMUZ_OUTPUT_PATH}")
    print(f"Polymarket dashboard payload baked at {POLYMARKET_OUTPUT_PATH}")


if __name__ == "__main__":
    bake_cake()
