import os
import json
import csv
import glob
from pathlib import Path
from datetime import datetime

ROOT = Path("/Users/henrywzh/Desktop/Quant/equity-research")
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
    data = {"movers": [], "snapshots": [], "updated": "N/A"}
    try:
        summary_glob = str(ROOT / "daily-market" / "data" / "summaries" / "*" / "*.json")
        files = glob.glob(summary_glob)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            with open(files[0], 'r') as f:
                raw_data = json.load(f)
                
                # Extract all rows from all sections
                all_rows = []
                indices = []
                for section in raw_data.get("sections", []):
                    asset_class = section.get("asset_class", "unknown")
                    for row in section.get("rows", []):
                        row["asset_class"] = asset_class
                        all_rows.append(row)
                        if asset_class == "index":
                            indices.append(row)

                # Top Movers: Sort by absolute percentage change
                sorted_movers = sorted(all_rows, key=lambda x: abs(float(x.get("pct_change", 0))), reverse=True)
                data["movers"] = sorted_movers[:5]
                
                # Snapshot: Use Indices
                data["snapshots"] = indices[:5]
                data["updated"] = raw_data.get("date", "N/A")
    except Exception as e:
        print(f"Market parse error: {e}")
    return data

def get_latest_macro():
    data = {"alerts": [], "sentiment": "Neutral", "summary": [], "updated": "N/A"}
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
                data["alerts"] = alerts[:8] # Top 8 developments as alerts

                # Sentiment is now 'Pending' until LLM scoring is integrated to avoid brittle regex matching
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
                    data["updated"] = latest_run.name # Timestamp in folder name
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
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(hub_data, f, indent=2)
    print(f"Aggregated signal cake baked at {OUTPUT_PATH}")

if __name__ == "__main__":
    bake_cake()
