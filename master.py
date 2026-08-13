"""Single Main Orchestrator for the Smart Aquarium Monitoring System on Raspberry Pi 4B.

Executes all monitoring, ML inference, disease detection, hunger feeding, water quality prediction,
and cloud synchronization sequentially in a simple, clean, linear loop.
Prints clear real-time progress to the terminal and saves all state to JSON.
"""

import math
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import cv2
import numpy as np
from config import (
    DATA_DIR,
    TOP_REGION_PERCENT,
    BOTTOM_REGION_PERCENT,
)
from health.watchdog import Watchdog
from storage.json_store import load_json, save_json
from utils.logger import get_logger

# Module imports
from vision.side_camera import SideCamera
from vision.top_camera import TopCamera
from vision.fish_tracker import FishTracker
from vision.fish_behavior import BehaviorAnalyzer
from vision.disease_detector import DiseaseDetector
from vision.hunger_detector import detect as detect_hunger
from ml.water_quality_predictor import WaterQualityPredictor
from ml.stress_classifier import (
    classify as classify_stress,
    classify_stress as classify_fish_stress,
    classify_tank_stress,
)
from ml.shap_explainer import explain as explain_shap
from ml.disease_fusion import fuse as fuse_disease
from nlp.symptom_input import process as process_symptoms
from firebase.fetch_user_symptoms import fetch_user_symptom
from feeding.servo import FeederServo

from firebase.upload_sensor_data import upload_latest as upload_sensor
from firebase.upload_behavior import upload_latest as upload_behavior
from firebase.upload_water_quality import upload_latest as upload_wq
from firebase.upload_disease import upload_latest as upload_disease


LOG = get_logger(__name__)
WATCHDOG = Watchdog()

# Hardware & Model Component Instances
SIDE_CAMERA = SideCamera()
TOP_CAMERA = TopCamera()
FISH_TRACKER = FishTracker()
BEHAVIOR_ANALYZER = BehaviorAnalyzer()
DISEASE_DETECTOR = DiseaseDetector()
WQ_PREDICTOR = WaterQualityPredictor()
FEEDER_SERVO = FeederServo()


def print_banner():
    """Print clean startup header."""
    print("=" * 60)
    print(" Smart Aquarium Monitoring System — Master CLI")
    print("=" * 60)


FISH_STATES: Dict[int, Dict[str, Any]] = {}

def make_fish_state():
    """Create tracking state dictionary for a single fish."""
    return {
        "last": None, "cross": 0, "top": 0.0,
        "bottom": 0.0, "freeze": 0.0, "tracked": 0.0,
        "last_region": "middle", "current_bottom": 0.0,
        "longest_bottom": 0.0, "bottom_entries": 0,
        "surface_visits": 0, "last_top_visit_time": None,
        "top_visit_intervals": [], "immobility_events": 0,
        "current_immobile_seconds": 0.0
    }

def _draw_analysis_overlay(frame, tracks, stress_data, remaining_secs, behavior, current_fps=30.0, dt=0.033):
    """Draw bounding boxes, region lines, 3-line fish tags, countdown timer, FPS, and tank stress on frame."""
    vis = frame.copy()
    fh, fw = vis.shape[:2]

    top_line = int(fh * TOP_REGION_PERCENT)
    bottom_line = int(fh * (1.0 - BOTTOM_REGION_PERCENT))

    # ── Region boundary lines ──
    cv2.line(vis, (0, top_line), (fw, top_line), (255, 0, 0), 2)
    cv2.putText(vis, "TOP FEEDING REGION", (10, top_line - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

    cv2.line(vis, (0, bottom_line), (fw, bottom_line), (0, 0, 255), 2)
    cv2.putText(vis, "BOTTOM DWELLING REGION", (10, bottom_line + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    fish_scores = []

    # ── Draw per-fish bounding boxes, multi-line status tags, & individual stress ──
    for fish in tracks:
        bbox = fish.get("bbox")
        tid = fish.get("fish_id", 1)

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            s = FISH_STATES.setdefault(tid, make_fish_state())
            s["tracked"] += dt

            # Compute spatial freezing immobility (centroid displacement < 5px)
            if s["last"] is not None:
                disp = math.dist(s["last"], (cx, cy))
                if disp < 5.0:
                    s["freeze"] += dt
                    s["current_immobile_seconds"] += dt
                    if s["current_immobile_seconds"] >= 2.0 and (s["current_immobile_seconds"] - dt) < 2.0:
                        s["immobility_events"] += 1
                else:
                    s["current_immobile_seconds"] = 0.0

            # Region assignment & entries
            prev_region = s["last_region"]
            if cy < top_line:
                region = "top"
                s["top"] += dt
                s["current_bottom"] = 0.0
                if prev_region != "top":
                    s["surface_visits"] += 1
                    if s["last_top_visit_time"] is not None:
                        interval = s["tracked"] - s["last_top_visit_time"]
                        s["top_visit_intervals"].append(interval)
                    s["last_top_visit_time"] = s["tracked"]
            elif cy > bottom_line:
                region = "bottom"
                s["bottom"] += dt
                s["current_bottom"] += dt
                s["longest_bottom"] = max(s["longest_bottom"], s["current_bottom"])
                if prev_region != "bottom":
                    s["bottom_entries"] += 1
            else:
                region = "middle"
                s["current_bottom"] = 0.0

            if region != prev_region and region != "middle" and prev_region != "middle":
                s["cross"] += 1

            s["last_region"] = region
            s["last"] = (cx, cy)

            avg_top_interval = (
                float(np.mean(s["top_visit_intervals"]))
                if s["top_visit_intervals"]
                else 0.0
            )

            # Classify fish stress using all 9 measurable features from table
            score, label, color, reason = classify_fish_stress(
                s["top"], s["bottom"], s["freeze"], s["longest_bottom"],
                s["bottom_entries"], s["surface_visits"], avg_top_interval,
                s["immobility_events"], s["tracked"], current_region=region
            )
            fish_scores.append(score)

            # 1. Draw colored bounding box matching stress state (Green / Yellow / Red)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 2. Draw 3 lines of tags above bounding box
            for text, y in ((f"ID {tid}", y1 - 45),
                            (f"{label} ({score:.2f})", y1 - 25),
                            (reason, y1 - 5)):
                cv2.putText(vis, text, (int(x1), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.50, color, 2)

    # ── Whole Tank Stress Status ──
    tank_score, tank_label, tank_color = classify_tank_stress(fish_scores)

    # Top-left Tank Stress Header matching user layout
    cv2.putText(vis, f"Tank Stress: {tank_label} ({tank_score:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, tank_color, 2)

    # Top-right Countdown Timer & FPS readout
    mins = int(remaining_secs // 60)
    secs = int(remaining_secs % 60)
    timer_text = f"Time Left: {mins:02d}:{secs:02d}"
    cv2.putText(vis, timer_text, (fw - 250, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(vis, f"FPS: {current_fps:.1f}", (fw - 250, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    return vis


def stage_sensors_and_stress():
    """Stage 1: Read sensors, then run a 3-minute visual stress observation with live video at 30 FPS."""
    print("\n[1/5] Reading Sensors & Starting 3-Minute Visual Stress Analysis...")

    # ── 1. Read Sensors (once at the start) ──
    try:
        from sensors.arduino_reader import read as read_arduino
        arduino_readings = read_arduino()
    except Exception as exc:
        LOG.warning("Arduino serial read failed: %s", exc)
        ts = datetime.now(timezone.utc).isoformat()
        arduino_readings = {
            "temperature": {"value": None, "unit": "C", "timestamp": ts, "error": str(exc)},
            "ph":          {"value": None, "unit": "pH", "timestamp": ts, "error": str(exc)},
            "turbidity":   {"value": None, "unit": "NTU", "timestamp": ts, "error": str(exc)},
        }

    try:
        from sensors.ionconcentration_reader import read as read_ionconc
        ionconc_reading = read_ionconc()
    except Exception as exc:
        LOG.warning("Ion concentration read failed: %s", exc)
        ts = datetime.now(timezone.utc).isoformat()
        ionconc_reading = {"value": None, "unit": "us/cm", "timestamp": ts, "error": str(exc)}

    sensor_readings = {
        "temperature":      arduino_readings["temperature"],
        "ph":               arduino_readings["ph"],
        "turbidity":        arduino_readings["turbidity"],
        "ionconcentration": ionconc_reading,
    }
    save_json(DATA_DIR / "latest_sensor.json", sensor_readings)

    temp_val = sensor_readings['temperature'].get('value', 'N/A')
    ph_val = sensor_readings['ph'].get('value', 'N/A')
    turb_val = sensor_readings['turbidity'].get('value', 'N/A')
    ion_val = sensor_readings['ionconcentration'].get('value', 'N/A')
    print(f"  |-- Temp: {temp_val} C | pH: {ph_val} | Turbidity: {turb_val} NTU | Ion: {ion_val} uS/cm")

    # ── 2. Three-minute continuous visual stress observation at 30 FPS ──
    OBSERVATION_DURATION = 180   # 3 minutes
    STRESS_UPDATE_INTERVAL = 5   # Refresh stress HUD every 5 seconds
    TARGET_FPS = 30.0
    TARGET_FRAME_TIME = 1.0 / TARGET_FPS  # 33.3 ms per frame
    WINDOW_NAME = "AquaMonitor - Step 1: Live Stress Analysis (3 min @ 30 FPS)"

    start_time = time.time()
    last_stress_update = 0.0
    running_stress = {"tank_stress_score": 0.0, "tank_stress_level": "Healthy"}
    behavior_metrics = {"fish_count": 0}
    frame_count = 0
    current_fps = 30.0
    last_frame_time = time.time()
    actual_dt = 0.033
    tracks = []

    print(f"  |-- Starting 3-minute visual stress observation at ~30 FPS (press 'q' to skip)...")

    try:
        while True:
            frame_start = time.time()
            elapsed = frame_start - start_time
            remaining = max(0.0, OBSERVATION_DURATION - elapsed)

            if elapsed >= OBSERVATION_DURATION:
                break

            # Capture frame
            frame = SIDE_CAMERA.read()
            if frame is None:
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "CAMERA FEED UNAVAILABLE", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            frame_h = frame.shape[0]
            frame_count += 1

            # Run fish detection & tracking
            tracks = FISH_TRACKER.track(frame)

            # Accumulate behavioral metrics across frames
            behavior_metrics = BEHAVIOR_ANALYZER.analyze(tracks, frame_height=frame_h)

            # Periodically recalculate stress for the live HUD
            if elapsed - last_stress_update >= STRESS_UPDATE_INTERVAL:
                running_stress = classify_stress(behavior_metrics, sensor_readings)
                last_stress_update = elapsed

            # Draw annotated frame and display
            vis_frame = _draw_analysis_overlay(
                frame, tracks, running_stress, remaining, behavior_metrics, current_fps, dt=max(0.01, actual_dt)
            )
            cv2.imshow(WINDOW_NAME, vis_frame)

            # Dynamic 30 FPS loop timing control
            proc_duration = time.time() - frame_start
            wait_ms = max(1, int((TARGET_FRAME_TIME - proc_duration) * 1000))

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                print("  |-- Observation terminated early by user.")
                break

            # Compute smoothed moving average FPS
            actual_dt = time.time() - last_frame_time
            last_frame_time = time.time()
            if actual_dt > 0:
                current_fps = 0.9 * current_fps + 0.1 * (1.0 / actual_dt)

    except Exception as exc:
        LOG.warning("Visual observation loop error (continuing with accumulated data): %s", exc)
    finally:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass

    # ── 3. Final stress classification on fully-accumulated behavior data ──
    stress_results = classify_stress(behavior_metrics, sensor_readings)
    save_json(DATA_DIR / "latest_behavior.json", behavior_metrics)
    save_json(DATA_DIR / "latest_stress.json", stress_results)

    observation_secs = round(time.time() - start_time, 1)
    print(f"  |-- Observation complete: {observation_secs}s, {frame_count} frames analyzed")
    print(f"  |-- Tracked Fish: {behavior_metrics.get('fish_count', 0)}")
    print(f"  +-- Tank Stress Score: {stress_results.get('tank_stress_score', 0)} ({stress_results.get('tank_stress_level', 'Healthy')})")


def stage_disease():
    """Stage 2: Visual CV disease detection & NLP symptom fusion."""
    print("\n[2/5] Disease Detection & Symptom Fusion...")
    frame = SIDE_CAMERA.read()
    tracks = FISH_TRACKER.track(frame)
    detection = DISEASE_DETECTOR.detect(frame, tracks=tracks)

    user_symptom_data = fetch_user_symptom()
    symptom_text = user_symptom_data.get("text", "")
    nlp_results = process_symptoms(symptom_text)

    fused_result = fuse_disease(detection, nlp_results)
    save_json(DATA_DIR / "latest_disease.json", fused_result)

    print(f"  |-- Detected Disease: {fused_result.get('disease_class', 'Healthy')}")
    print(f"  +-- Diagnosis Confidence: {round(fused_result.get('confidence', 1.0) * 100, 1)}%")


def stage_hunger_and_feeding():
    """Stage 3: Top view hunger detection & automatic feeding."""
    print("\n[3/5] Top Camera Hunger Detection & Servo Control...")
    frame = TOP_CAMERA.read()
    hunger_result = detect_hunger(frame)
    save_json(DATA_DIR / "latest_hunger.json", hunger_result)

    hungry_count = hunger_result.get("hungry_count", 0)
    feed_result = FEEDER_SERVO.dispense(hungry_count)
    save_json(DATA_DIR / "latest_feed.json", feed_result)

    print(f"  |-- Hungry Fish Count: {hungry_count} ({hunger_result.get('hunger_level', 'Normal')})")
    print(f"  +-- Dispensed Portion: {feed_result.get('dispensed', False)}")


def stage_water_quality_and_shap():
    """Stage 4: ML Water Quality prediction & SHAP XAI explanation."""
    print("\n[4/5] Water Quality Prediction & SHAP Explanation...")
    sensor_data = load_json(DATA_DIR / "latest_sensor.json", default={})
    prediction = WQ_PREDICTOR.predict(sensor_data)
    save_json(DATA_DIR / "latest_water_quality.json", prediction)

    shap_result = explain_shap(prediction)
    save_json(DATA_DIR / "latest_shap.json", shap_result)

    print(f"  |-- Water Quality: {prediction.get('water_quality', 'Good')}")
    print(f"  +-- Est. Hours until Water Change: {prediction.get('estimated_hours_until_water_change', 'N/A')}")


def stage_firebase_sync():
    """Stage 5: Upload all JSON states to Firebase Realtime Database."""
    print("\n[5/5] Uploading State to Firebase Cloud...")
    sensor_data = load_json(DATA_DIR / "latest_sensor.json", default={})
    behavior_data = load_json(DATA_DIR / "latest_behavior.json", default={})
    stress_data = load_json(DATA_DIR / "latest_stress.json", default={})
    disease_data = load_json(DATA_DIR / "latest_disease.json", default={})
    wq_data = load_json(DATA_DIR / "latest_water_quality.json", default={})
    shap_data = load_json(DATA_DIR / "latest_shap.json", default={})

    if sensor_data:
        upload_sensor(sensor_data)
    if behavior_data or stress_data:
        upload_behavior({"behavior": behavior_data, "stress": stress_data})
    if disease_data:
        upload_disease(disease_data)
    if wq_data or shap_data:
        upload_wq({"water_quality": wq_data, "shap": shap_data})

    print("  +-- Firebase synchronization completed.")


def run_stage(stage_name: str, func):
    """Execute a single pipeline stage inside Watchdog monitoring."""
    save_json(DATA_DIR / "latest_pipeline_stage.json", {"active_stage": stage_name, "timestamp": datetime.now(timezone.utc).isoformat()})
    with WATCHDOG.monitor(stage_name):
        func()


def main():
    """Single main loop calling all python scripts one by one."""
    print_banner()
    LOG.info("Starting Master Sequential Pipeline...")

    stages = [
        ("sensors_and_stress", stage_sensors_and_stress),
        ("disease_detection", stage_disease),
        ("hunger_and_feeding", stage_hunger_and_feeding),
        ("water_quality_shap", stage_water_quality_and_shap),
        ("firebase_sync", stage_firebase_sync),
    ]

    cycle_count = 1
    while True:
        print(f"\n==================== Cycle #{cycle_count} ({datetime.now().strftime('%H:%M:%S')}) ====================")
        for stage_name, stage_func in stages:
            run_stage(stage_name, stage_func)
            time.sleep(0.5)
        
        WATCHDOG.check_health()
        cycle_count += 1
        time.sleep(1.0)


if __name__ == "__main__":
    main()
