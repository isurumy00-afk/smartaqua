#!/usr/bin/env python3
"""Dataset Image Collector for YOLO Retraining.

Captures 100 images from USB Camera 0 to build a training dataset for custom YOLO fish detection.
Supports both manual capture (SPACE) and automatic burst mode ('a').
"""

import os
import time
from datetime import datetime
from pathlib import Path
import cv2

# Configuration
CAMERA_INDEX = 0
TARGET_IMAGE_COUNT = 100
AUTO_CAPTURE_INTERVAL = 5.0  # seconds between auto snapshots (1 photo / 5 sec)

# Output Directory
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "images"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print(" Smart Aquarium — USB Camera Dataset Collector (100 Images)")
    print("=" * 60)
    print(f" Output Directory : {DATASET_DIR}")
    print(f" Target Count     : {TARGET_IMAGE_COUNT} images")
    print(" Controls:")
    print("   [SPACE] : Capture single image")
    print("   [A]     : Toggle Auto-Capture mode (every 5.0 sec)")
    print("   [Q/ESC] : Quit")
    print("=" * 60)

    # Open USB Camera Index 0
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap or not cap.isOpened():
        if os.name == "nt":
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap or not cap.isOpened():
        print(f"[ERROR] Could not open USB Camera at index {CAMERA_INDEX}.")
        print("Please verify your USB webcam is plugged in and accessible.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Count existing images in output folder to resume smoothly
    existing = list(DATASET_DIR.glob("img_*.jpg"))
    saved_count = len(existing)
    print(f"[INFO] Found {saved_count} existing images in dataset directory.")

    auto_mode = False
    last_auto_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Failed to read frame from USB camera. Retrying...")
            time.sleep(0.1)
            continue

        now = time.time()
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Auto-capture logic
        if auto_mode and (now - last_auto_time >= AUTO_CAPTURE_INTERVAL):
            saved_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"img_{saved_count:03d}_{timestamp}.jpg"
            filepath = DATASET_DIR / filename
            cv2.imwrite(str(filepath), frame)
            print(f" [{saved_count}/{TARGET_IMAGE_COUNT}] Saved (Auto): {filepath.name}")
            last_auto_time = now

        # Top Overlay Banner
        cv2.rectangle(vis, (0, 0), (w, 50), (30, 30, 30), -1)
        progress_text = f"Collected: {saved_count} / {TARGET_IMAGE_COUNT} images"
        progress_color = (0, 255, 0) if saved_count >= TARGET_IMAGE_COUNT else (0, 255, 255)
        cv2.putText(vis, progress_text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, progress_color, 2)

        # Mode Indicator
        mode_text = "AUTO CAPTURE: ON" if auto_mode else "MANUAL MODE"
        mode_color = (0, 255, 0) if auto_mode else (200, 200, 200)
        cv2.putText(vis, mode_text, (w - 220, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)

        # Bottom Instructions
        cv2.rectangle(vis, (0, h - 35), (w, h), (20, 20, 20), -1)
        instructions = "SPACE: Snap | A: Toggle Auto | Q: Quit"
        cv2.putText(vis, instructions, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("USB Camera - Dataset Collector", vis)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            print("[INFO] Collection stopped by user.")
            break
        elif key == ord(' '):  # SPACEBAR: Capture single image
            saved_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"img_{saved_count:03d}_{timestamp}.jpg"
            filepath = DATASET_DIR / filename
            cv2.imwrite(str(filepath), frame)
            print(f" [{saved_count}/{TARGET_IMAGE_COUNT}] Saved (Manual): {filepath.name}")

            # Brief visual green flash on screen
            flash = vis.copy()
            cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), 10)
            cv2.imshow("USB Camera - Dataset Collector", flash)
            cv2.waitKey(50)

        elif key == ord('a') or key == ord('A'):  # Toggle Auto-Capture
            auto_mode = not auto_mode
            status = "ENABLED (1 photo / 5 sec)" if auto_mode else "DISABLED"
            print(f"[MODE] Auto-Capture {status}")
            last_auto_time = time.time()

        if saved_count >= TARGET_IMAGE_COUNT:
            print("=" * 60)
            print(f" SUCCESS: Target reached! {saved_count} images collected in {DATASET_DIR}")
            print("=" * 60)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
