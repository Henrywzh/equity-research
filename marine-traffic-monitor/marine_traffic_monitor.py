import time
import cv2
import csv
import json
import os
import numpy as np
from datetime import datetime
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from analyst import run_consensus_check, load_csv_history
from notifier import send_digest

# --- Configuration ---
_HERE = os.path.dirname(os.path.abspath(__file__))

URL = "https://www.marinetraffic.com/en/ais/home/centerx:56.3/centery:26.4/zoom:10"
CHECK_INTERVAL = 900  # Check every 15 minutes (900 seconds)
NORMAL_TRAFFIC_THRESHOLD = 10  # Baseline for a "surge" in current conditions

os.makedirs(os.path.join(_HERE, "data"), exist_ok=True)
CSV_FILENAME = os.path.join(_HERE, "data", "hormuz_traffic_log.csv")

# 1. Initialize the CSV file with headers if it doesn't exist yet
if not os.path.exists(CSV_FILENAME):
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Detected_Ships", "Status_Note"])


# Module-level state: track which date the digest was last sent (resets on restart)
_last_digest_date: str = ""


def _should_send_digest(digest_hour: int) -> bool:
    """Return True once per day when the current UTC hour has reached digest_hour."""
    global _last_digest_date
    now   = datetime.utcnow()
    today = now.strftime("%Y-%m-%d")
    if now.hour >= digest_hour and _last_digest_date != today:
        _last_digest_date = today
        return True
    return False


def _last_csv_count(csv_filename: str) -> int:
    """Return the Detected_Ships value from the most recent CSV row, or 0 if not found."""
    try:
        with open(csv_filename, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) <= 1:   # empty or header-only
            return 0
        return int(rows[-1][1])   # column index 1 = Detected_Ships
    except (FileNotFoundError, IndexError, ValueError):
        return 0


def _cleanup_old_screenshots(csv_filename: str) -> None:
    """
    Disk hygiene: delete screenshots older than 24h where ships_in_zone == 0.

    Retention rule:
      KEEP  — any image whose timestamp appears in a CSV row with Detected_Ships > 0
      DELETE — raw + debug images from zero-ship intervals after 24 hours

    Scans all *.png files recursively under screenshots/ to handle the
    YYYY/MM/DD subfolder hierarchy.
    """
    import glob as _glob
    from datetime import timedelta

    # Build set of timestamps with ships > 0 from CSV
    keep_ts: set[str] = set()
    try:
        with open(csv_filename, newline="") as f:
            rows = list(csv.reader(f))
        for row in rows[1:]:   # skip header
            if len(row) >= 2:
                try:
                    if int(row[1]) > 0:
                        keep_ts.add(row[0])   # e.g. "2026-03-15_18-17-38"
                except ValueError:
                    pass
    except FileNotFoundError:
        return

    cutoff      = datetime.now() - timedelta(hours=24)
    screen_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
    deleted     = 0

    for filepath in _glob.glob(os.path.join(screen_root, "**", "*.png"), recursive=True):
        fname = os.path.basename(filepath)
        matched = False
        for prefix in ("raw_map_", "detected_ships_"):
            if fname.startswith(prefix):
                ts_str = fname[len(prefix):-4]   # strip prefix + ".png"
                try:
                    file_dt = datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S")
                except ValueError:
                    break
                if file_dt < cutoff and ts_str not in keep_ts:
                    try:
                        os.remove(filepath)
                        deleted += 1
                    except OSError as e:
                        print(f"[CLEANUP] Could not delete {filepath}: {e}")
                matched = True
                break

    if deleted:
        print(f"[CLEANUP] Deleted {deleted} old zero-ship screenshot(s).")
    else:
        print("[CLEANUP] No stale zero-ship screenshots to remove.")


def monitor_and_log(llm_threshold=1, delta_threshold=1,
                    alert_cooldown=5, digest_hour=10):
    # Generate a clean timestamp for file names
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    time_display = now.strftime("%H:%M:%S")

    # Setup folder and file paths — hierarchical daily subfolders
    date_folder    = now.strftime("%Y/%m/%d")
    screen_dir     = os.path.join(_HERE, "screenshots", date_folder)
    os.makedirs(screen_dir, exist_ok=True)
    raw_img_path   = os.path.join(screen_dir, f"raw_map_{timestamp_str}.png")
    debug_img_path = os.path.join(screen_dir, f"detected_ships_{timestamp_str}.png")

    # Read previous count BEFORE this cycle's screenshot (for delta comparison later)
    last_count = _last_csv_count(CSV_FILENAME)

    print(f"[{time_display}] Fetching live map data...")

    # 2. Take the Original Screenshot (UPDATED WITH STEALTH)
    try:
        with sync_playwright() as p:
            # headless=False + stealth bypasses Cloudflare fingerprinting.
            # RESIDENTIAL_PROXY routes browser traffic through a residential IP
            # so datacenter runners (GitHub Actions) aren't blocked by Cloudflare.
            _proxy_url = os.getenv("RESIDENTIAL_PROXY", "").strip()
            _proxy_cfg = {"server": _proxy_url} if _proxy_url else None
            browser = p.chromium.launch(headless=False, slow_mo=50,
                                        proxy=_proxy_cfg)

            # Added realistic browser context/headers
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080}
            )

            page = context.new_page()

            # Apply the stealth patch to the page
            stealth = Stealth()
            stealth.apply_stealth_sync(page)

            page.goto(URL)
            time.sleep(7)  # Increased from 10 to 15 to give Cloudflare time to clear
            page.screenshot(path=raw_img_path)
            browser.close()
    except Exception as e:
        print(f"Error fetching map: {e}")
        return

    # 3. Process the image with OpenCV
    img = cv2.imread(raw_img_path)
    if img is None:
        print("Error: Could not read the screenshot.")
        return

    # A. Crop out the UI and Cookie Banner
    height, width, _ = img.shape
    roi = img[80:height - 250, 80:width - 80]
    output_img = roi.copy()

    # --- NEW: Adjusted Angled "Choke Point" Polygon ---
    # Shifted to avoid Hengam Island and extended 10% to the right.
    ZONE_POLY = np.array([
        [620, 280],  # Top-Left (Moved down and right to clear Hengam Island)
        [1150, 120],  # Top-Right (Extended ~100px right)
        [1250, 480],  # Bottom-Right (Extended ~100px right)
        [720, 580]  # Bottom-Left (Shifted right to keep the angle)
    ], np.int32)

    # Draw the slanted polygon on the image
    cv2.polylines(output_img, [ZONE_POLY], isClosed=True, color=(0, 0, 255), thickness=3)
    cv2.putText(output_img, "TRANSIT CORRIDOR", (620, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # B. Calculate "Colorfulness"
    b, g, r = cv2.split(roi)
    max_c = np.maximum(np.maximum(b, g), r).astype(np.int16)
    min_c = np.minimum(np.minimum(b, g), r).astype(np.int16)
    color_diff = max_c - min_c

    _, binary_mask = cv2.threshold(color_diff.astype(np.uint8), 25, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ships_in_zone = 0
    ships_outside = 0

    # 4. Filter contours and check location
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 500:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + (w // 2)
            center_y = y + (h // 2)

            # Check if the center of the ship is INSIDE our Polygon
            # returns +1 if inside, 0 if on the line, -1 if outside
            is_inside = cv2.pointPolygonTest(ZONE_POLY, (center_x, center_y), False)

            if is_inside >= 0:
                ships_in_zone += 1
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for inside
            else:
                ships_outside += 1
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for outside

    # Save the processed image
    cv2.imwrite(debug_img_path, output_img)

    # 5. Determine Geopolitical Status based on the ZONE
    status = "Active Blockade / Clear Zone"
    if ships_in_zone > 5:
        status = f"BREACH DETECTED: {ships_in_zone} ships transiting choke point!"
    elif ships_in_zone > 0:
        status = f"WARNING: {ships_in_zone} dark/escorted vessel(s) in choke point."

    # 6. Save the data to the CSV file (FIXED ERROR HERE)
    with open(CSV_FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Replaced the broken 'ship_count' with 'ships_in_zone'
        writer.writerow([timestamp_str, ships_in_zone, status])

    print(f"[{time_display}] Result: {ships_in_zone} in zone, {ships_outside} outside. Status: {status}")
    print(f"Saved Processed: {debug_img_path}\n")

    # 7a. Daily digest — fires once per day at digest_hour UTC, no LLM needed
    if _should_send_digest(digest_hour):
        snap = load_csv_history(CSV_FILENAME, n_rows=96)  # ~24h at 15-min intervals
        send_digest(csv_snapshot=snap)
        _cleanup_old_screenshots(CSV_FILENAME)   # purge stale zero-ship images once/day

    # 7b. LLM Analyst QA pass — smart gating: absolute minimum + delta change
    delta = abs(ships_in_zone - last_count)
    should_call_llm = (ships_in_zone >= llm_threshold) and (delta >= delta_threshold)

    if should_call_llm:
        briefing = run_consensus_check(
            debug_img_path, ships_in_zone, CSV_FILENAME,
            raw_image_path=raw_img_path,
            alert_cooldown=alert_cooldown,
        )
        if briefing:
            print(f"[ANALYST BRIEFING]\n{json.dumps(briefing, indent=2)}\n")
    else:
        if ships_in_zone < llm_threshold:
            reason = f"ship count {ships_in_zone} < threshold {llm_threshold}"
        else:
            reason = f"no change (delta={delta}, threshold={delta_threshold})"
        print(f"[ANALYST] Skipped — {reason}\n")


# --- Main Loop ---
print("--- Strait of Hormuz Data Logger Initialized ---")
print(f"Logging data to {CSV_FILENAME} every {CHECK_INTERVAL // 60} minutes.")


def run_monitor(check_interval=CHECK_INTERVAL, iteration=1,
                llm_threshold=1, delta_threshold=1,
                alert_cooldown=5, digest_hour=10):
    if iteration is None:
        while True:
            monitor_and_log(llm_threshold=llm_threshold,
                            delta_threshold=delta_threshold,
                            alert_cooldown=alert_cooldown,
                            digest_hour=digest_hour)
            time.sleep(check_interval)

    for i in range(iteration):
        monitor_and_log(llm_threshold=llm_threshold,
                        delta_threshold=delta_threshold,
                        alert_cooldown=alert_cooldown,
                        digest_hour=digest_hour)
        if iteration > 1:
            time.sleep(check_interval)

    print('---- Done ----')


if __name__ == '__main__':
    run_monitor(iteration=None)