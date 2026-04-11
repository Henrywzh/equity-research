import os
import asyncio
import json
import csv
from datetime import datetime
from ais_client import get_ais_snapshot
from analyst import run_consensus_check, load_csv_history
from notifier import send_digest, send_alert

# --- Configuration ---
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = os.path.join(_HERE, "data", "hormuz_traffic_log.csv")
LLM_THRESHOLD = 1
DELTA_THRESHOLD = 1

def _last_csv_count(csv_filename):
    try:
        with open(csv_filename, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) <= 1:
            return 0
        return int(rows[-1][1]) 
    except (FileNotFoundError, IndexError, ValueError):
        return 0

async def main():
    print(f"--- Marine Monitor API Runner (GHA Mode) ---")
    api_key = os.getenv("AIS_STREAM_API_KEY")
    if not api_key:
        print("Error: AIS_STREAM_API_KEY not found.")
        return

    # 1. Capture AIS Snapshot
    vessels = await get_ais_snapshot(api_key, duration=60)
    ships_in_zone = len(vessels)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 2. Determine Status
    status = "Active Blockade / Clear Zone"
    if ships_in_zone > 5:
        status = f"BREACH DETECTED: {ships_in_zone} ships transiting choke point!"
    elif ships_in_zone > 0:
        status = f"WARNING: {ships_in_zone} vessel(s) in choke point."

    # 3. Update CSV
    last_count = _last_csv_count(CSV_FILENAME)
    os.makedirs(os.path.dirname(CSV_FILENAME), exist_ok=True)
    file_exists = os.path.exists(CSV_FILENAME)
    
    with open(CSV_FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Detected_Ships", "Status_Note"])
        writer.writerow([timestamp_str, ships_in_zone, status])
    
    print(f"[{timestamp_str}] AIS Result: {ships_in_zone} ships. Status: {status}")

    # 4. LLM Analysis
    delta = abs(ships_in_zone - last_count)
    should_call_llm = (ships_in_zone >= LLM_THRESHOLD) and (delta >= DELTA_THRESHOLD)
    
    if should_call_llm:
        print(f"[ANALYST] Triggering context-only analysis...")
        # We pass image_path="" to trigger the image-less mode we just added to analyst.py
        briefing = run_consensus_check(
            image_path="", 
            reported_count=ships_in_zone, 
            csv_path=CSV_FILENAME,
            model_a="llama_4_scout", # This will abstain from vision but still provide meta
            model_b="llama_3_3_70b"   # This is a context analyst anyway
        )
        
        # Format the manifest for the briefing
        manifest_text = "\n".join([f"- {v['name']} ({v['type']}) | SPD: {v['speed']}kts" for v in vessels[:10]])
        if len(vessels) > 10:
            manifest_text += f"\n- ... and {len(vessels)-10} more."
            
        if briefing:
            # Inject manifest into the briefing for the notification
            briefing['analyst_briefing'] += f"\n\n[AIS MANIFEST]\n{manifest_text}"
            print(f"[ANALYST BRIEFING] Generated.")
    else:
        print(f"[ANALYST] Skipped (Count: {ships_in_zone}, Delta: {delta})")

    print("--- Run Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
