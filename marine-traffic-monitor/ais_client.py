import asyncio
import json
import os
import time
import websockets
import ssl
import certifi
from datetime import datetime

# --- Configuration ---
# Strait of Hormuz Chokepoint Bounding Box
# Format: [[MinLat, MinLon], [MaxLat, MaxLon]]
HORMUZ_BBOX = [[26.0, 56.0], [27.0, 57.0]]

async def get_ais_snapshot(api_key, duration=60):
    """
    Connects to AISStream, subscribes to the Hormuz bounding box, 
    and collects vessel data for 'duration' seconds.
    Returns a list of unique vessel dictionaries.
    """
    url = "wss://stream.aisstream.io/v0/stream"
    vessels = {} # MMSI -> Vessel Info

    # Create SSL context (common fix for macOS local cert issues)
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    print(f"[AIS] Connecting to AISStream (Snapshot duration: {duration}s)...")
    
    try:
        async with websockets.connect(url, ssl=ssl_context) as websocket:
            subscribe_msg = {
                "APIKey": api_key,
                "BoundingBoxes": [HORMUZ_BBOX],
                "FilterMessageTypes": ["PositionReport", "ShipStaticData"]
            }
            
            await websocket.send(json.dumps(subscribe_msg))
            
            start_time = time.time()
            while time.time() - start_time < duration:
                try:
                    # Set a timeout for recv so we don't hang if no data flows
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    msg = json.loads(raw_msg)
                    
                    mmsi = msg.get("MetaData", {}).get("MMSI")
                    if not mmsi:
                        continue
                        
                    # Initialize vessel record if new
                    if mmsi not in vessels:
                        vessels[mmsi] = {
                            "mmsi": mmsi,
                            "name": msg.get("MetaData", {}).get("ShipName", "Unknown").strip(),
                            "lat": None,
                            "lon": None,
                            "speed": 0.0,
                            "course": 0.0,
                            "type": "Unknown",
                            "last_seen": datetime.utcnow().isoformat()
                        }
                    
                    # Update dynamic info from PositionReport
                    if msg.get("MessageType") == "PositionReport":
                        pos = msg.get("Message", {}).get("PositionReport", {})
                        vessels[mmsi]["lat"] = pos.get("Latitude")
                        vessels[mmsi]["lon"] = pos.get("Longitude")
                        vessels[mmsi]["speed"] = pos.get("Sog", 0.0)
                        vessels[mmsi]["course"] = pos.get("Cog", 0.0)
                        
                    # Update static info from ShipStaticData (if available)
                    elif msg.get("MessageType") == "ShipStaticData":
                        static = msg.get("Message", {}).get("ShipStaticData", {})
                        vessels[mmsi]["type"] = static.get("ShipType", "Unknown")
                        if vessels[mmsi]["name"] == "Unknown":
                            vessels[mmsi]["name"] = static.get("Name", "Unknown").strip()

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"[AIS] Error receiving message: {e}")
                    break
                    
    except Exception as e:
        print(f"[AIS] Connection failed: {e}")
        return []

    # Filter out entries that never got a position during the snapshot
    active_vessels = [v for v in vessels.values() if v["lat"] is not None]
    print(f"[AIS] Snapshot complete. Found {len(active_vessels)} active vessels in zone.")
    return active_vessels

if __name__ == "__main__":
    # Local test entry point
    KEY = os.getenv("AIS_STREAM_API_KEY")
    if not KEY:
        print("Error: AIS_STREAM_API_KEY not found in environment.")
    else:
        results = asyncio.run(get_ais_snapshot(KEY, duration=10))
        print(json.dumps(results, indent=2))
