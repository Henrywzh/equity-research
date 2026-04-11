import os
import json
import csv
import glob
from datetime import datetime
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
HUB_DATA = ROOT / "hub" / "data" / "signals.json"

def get_latest_maritime():
    csv_file = ROOT / "marine-traffic-monitor" / "data" / "hormuz_traffic_log.csv"
    if not csv_file.exists():
        return {"count": 0, "status": "Unknown", "updated": "N/A"}
    try:
        with open(csv_file, 'r') as f:
            lines = list(csv.reader(f))
            if len(lines) < 2:
                return {"count": 0, "status": "Empty Log", "updated": "N/A"}
            last_row = lines[-1]
            return {
                "count": last_row[1],
                "status": last_row[2],
                "updated": last_row[0]
            }
    except Exception as e:
        return {"error": str(e)}

def get_latest_market():
    # Looks for highest date folder, then latest json
    summary_glob = str(ROOT / "daily-market" / "data" / "summaries" / "*" / "*.json")
    files = glob.glob(summary_glob)
    if not files:
        return {"movers": [], "updated": "N/A"}
    
    files.sort(reverse=True) # Sort lexicographically (date folders work well here)
    latest_file = Path(files[0])
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
            # Assuming standard summary format with 'top_movers'
            return {
                "movers": data.get("top_movers", [])[:5],
                "snapshot": data.get("snapshots", [])[:5],
                "updated": latest_file.parent.name
            }
    except Exception as e:
        return {"error": str(e)}

def get_latest_macro():
    analysis_glob = str(ROOT / "daily-macro" / "data" / "analyses" / "*" / "hkej-news-analysis.json")
    files = glob.glob(analysis_glob)
    if not files:
        return {"alerts": [], "sentiment": "Neutral", "updated": "N/A"}
    
    files.sort(reverse=True)
    latest_file = Path(files[0])
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
            return {
                "alerts": data.get("top_alerts", [])[:5],
                "sentiment": data.get("overall_sentiment", "Neutral"),
                "summary": data.get("executive_summary", [])[:3],
                "updated": latest_file.parent.name
            }
    except Exception as e:
        return {"error": str(e)}

def get_latest_youtube():
    run_glob = str(ROOT / "youtube-intake" / "data" / "analysis" / "*" / "run-summary.json")
    files = glob.glob(run_glob)
    if not files:
        return {"signals": [], "updated": "N/A"}
    
    files.sort(reverse=True)
    latest_file = Path(files[0])
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
            # Get high attention videos
            videos = data.get("videos", [])
            signals = [v for v in videos if v.get("attention_tier") == "high"]
            return {
                "signals": signals[:5],
                "total_analyzed": data.get("totals", {}).get("analyzed", 0),
                "updated": latest_file.parent.name
            }
    except Exception as e:
        return {"error": str(e)}

def main():
    print("Aggregation started...")
    payload = {
        "maritime": get_latest_maritime(),
        "market": get_latest_market(),
        "macro": get_latest_macro(),
        "youtube": get_latest_youtube(),
        "last_baked": datetime.now().isoformat()
    }
    
    os.makedirs(HUB_DATA.parent, exist_ok=True)
    with open(HUB_DATA, 'w') as f:
        json.dump(payload, f, indent=2)
    
    print(f"Aggregated signal cake baked at {HUB_DATA}")

if __name__ == "__main__":
    main()
