import os
import json
import csv
import glob
from pathlib import Path
from datetime import datetime

ROOT = Path(
    os.environ.get("EQUITY_RESEARCH_ROOT", Path(__file__).resolve().parents[1])
).resolve()
OUTPUT_PATH = ROOT / "hub" / "data" / "signals.json"

def get_latest_maritime():
    log_path = ROOT / "marine-traffic-monitor" / "data" / "hormuz_traffic_log.csv"
    data = {"count": "0", "status": "No Recent Data", "updated": "N/A"}
    try:
        if log_path.exists():
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    latest = rows[-1]
                    data = {
                        "count": latest.get("Detected_Ships", "0"),
                        "status": latest.get("Status_Note", "Normal Activity"),
                        "updated": latest.get("Timestamp", "N/A")
                    }
    except Exception as e:
        print(f"Maritime parse error: {e}")
    return data

def get_latest_market():
    data = {"sections": [], "updated": "N/A", "top_mover_val": 0, "top_mover_ticker": "N/A"}
    try:
        summary_glob = str(ROOT / "daily-market" / "data" / "summaries" / "*" / "*.json")
        files = glob.glob(summary_glob)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            with open(files[0], 'r') as f:
                raw_data = json.load(f)
                data["sections"] = raw_data.get("sections", [])
                data["updated"] = raw_data.get("date", "N/A")
                
                # Still find the top mover for the KPI card
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
    except Exception as e:
        print(f"Market parse error: {e}")
    return data

def get_latest_macro():
    data = {"alerts": [], "sentiment": "Pending LLM", "summary": [], "updated": "N/A"}
    try:
        analysis_glob = str(ROOT / "daily-macro" / "data" / "analyses" / "*" / "*.json")
        files = glob.glob(analysis_glob)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            with open(files[0], 'r') as f:
                raw_data = json.load(f)
                data["summary"] = raw_data.get("executive_summary", [])
                data["updated"] = raw_data.get("report_date", "N/A")
                
                # Extract alerts from key developments in categories
                alerts = []
                for cat in raw_data.get("categories", []):
                    developments = cat.get("key_developments", [])
                    if developments:
                        alerts.extend(developments)
                data["alerts"] = alerts # Frontend will slice for 'top 5'
                
                # Sentiment is now 'Pending' until LLM scoring is integrated
                data["sentiment"] = "Pending LLM"
    except Exception as e:
        print(f"Macro parse error: {e}")
    return data

def get_latest_youtube():
    data = {"signals": [], "total_analyzed": 0, "updated": "N/A"}
    try:
        run_glob = str(ROOT / "youtube-intake" / "data" / "analysis" / "*")
        runs = [Path(p) for p in glob.glob(run_glob) if os.path.isdir(p)]
        if runs:
            runs.sort(key=os.path.getmtime, reverse=True)
            latest_run = runs[0]
            summary_path = latest_run / "run-summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    raw_data = json.load(f)
                    run_info = raw_data.get("run_summary", {})
                    
                    # Map top claims to signals
                    claims = run_info.get("top_claims_worth_watching", [])
                    signals = []
                    for claim in claims:
                        signals.append({
                            "title": claim,
                            "channel_name": "Multi-Source Alpha"
                        })
                    data["signals"] = signals
                    
                    # Count total videos analyzed
                    total = 0
                    for ch in raw_data.get("channels", {}).values():
                        total += ch.get("video_count", 0)
                    data["total_analyzed"] = total
                    data["updated"] = latest_run.name
    except Exception as e:
        print(f"YouTube parse error: {e}")
    return data

def bake_cake():
    print("Aggregation started...")
    hub_data = {
        "maritime": get_latest_maritime(),
        "market": get_latest_market(),
        "macro": get_latest_macro(),
        "youtube": get_latest_youtube(),
        "last_baked": datetime.now().isoformat()
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open('w') as f:
        json.dump(hub_data, f, indent=2)
    print(f"Aggregated signal cake baked at {OUTPUT_PATH}")

if __name__ == "__main__":
    bake_cake()
