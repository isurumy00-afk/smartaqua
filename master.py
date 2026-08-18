"""Single Main Orchestrator for the Smart Aquarium Monitoring System on Raspberry Pi 4B.

Executes all monitoring, ML inference, disease detection, hunger feeding, water quality prediction,
and cloud synchronization sequentially in a simple, clean, linear loop.
Prints clear real-time progress to the terminal and saves all state to JSON.
"""

import os
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
    SEND_STRESS_ROI_TO_DISEASE,
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
                    s["current_immobile_seconds"] += dt
                    # Only count as freeze time after 10s of continuous immobility
                    if s["current_immobile_seconds"] >= 10.0:
                        s["freeze"] += dt
                        if (s["current_immobile_seconds"] - dt) < 10.0:
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

            # 2. Draw 2 lines of tags above bounding box
            for text, y in ((f"{label} ({score:.2f})", y1 - 20),
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


def _draw_hunger_overlay(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    current_count: int,
    avg_count: float,
    presence_ratio: float,
    remaining: float,
    fps: float,
) -> np.ndarray:
    """Draw Top Camera feeding HUD with detected bounding boxes, stats, countdown timer, and FPS."""
    vis = frame.copy()
    fh, fw = vis.shape[:2]

    # Draw detected fish bounding boxes in cyan/yellow
    for det in detections:
        bbox = det.get("bbox")
        conf = det.get("confidence", 0.0)
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            tag = f"Hungry Fish {int(conf * 100)}%"
            cv2.putText(vis, tag, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # Top HUD Banner with semi-transparent background
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 65), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    rem_m = int(remaining) // 60
    rem_s = int(remaining) % 60
    timer_str = f"{rem_m:02d}:{rem_s:02d}"

    # Line 1: Title & Timer & FPS
    cv2.putText(vis, "Step 3: Top Camera Hunger Monitoring (3 min)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Remaining: {timer_str} | {fps:.1f} FPS", (fw - 230, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # Line 2: Real-Time Stats
    stats_str = f"Instant: {current_count} fish | 3-Min Avg: {avg_count:.2f} fish | Surface Attendance: {presence_ratio * 100:.1f}%"
    status_color = (0, 255, 0) if presence_ratio >= 0.30 and avg_count >= 0.5 else (200, 200, 200)
    cv2.putText(vis, stats_str, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

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

    # Ensure DISPLAY default for Linux desktop popups
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["DISPLAY"] = ":0"

    can_display = os.environ.get("HEADLESS") != "1"
    if can_display:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 800, 500)
            try:
                cv2.startWindowThread()
            except Exception:
                pass
        except Exception as exc:
            LOG.warning("Could not initialize desktop GUI popup window (%s): %s", WINDOW_NAME, exc)
            can_display = False

    start_time = time.time()
    last_stress_update = 0.0
    running_stress = {"tank_stress_score": 0.0, "tank_stress_level": "Healthy"}
    behavior_metrics = {"fish_count": 0}
    frame_count = 0
    current_fps = 30.0
    last_frame_time = time.time()
    actual_dt = 0.667  # ~1.5 FPS on Pi 4B; self-corrects from frame 2 onward
    tracks = []

    print(f"  |-- Starting 3-minute visual stress observation at ~30 FPS (press 'q' in popup to skip)...")

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

            # Accumulate behavioral metrics across frames (dt = real elapsed time per frame)
            behavior_metrics = BEHAVIOR_ANALYZER.analyze(tracks, frame_height=frame_h, dt=max(0.01, actual_dt))

            # Periodically recalculate stress for the live HUD
            if elapsed - last_stress_update >= STRESS_UPDATE_INTERVAL:
                running_stress = classify_stress(behavior_metrics, sensor_readings)
                last_stress_update = elapsed

            # Dynamic 30 FPS loop timing control
            proc_duration = time.time() - frame_start
            wait_ms = max(1, int((TARGET_FRAME_TIME - proc_duration) * 1000))

            # 1. Always generate annotated inference preview frame
            vis_frame = _draw_analysis_overlay(
                frame, tracks, running_stress, remaining, behavior_metrics, current_fps, dt=max(0.01, actual_dt)
            )

            # 2. Persist latest frame for Web UI Live Model Preview
            try:
                cv2.imwrite(str(DATA_DIR / "latest_stress_frame.jpg"), vis_frame)
            except Exception:
                pass

            # 3. Display in desktop GUI window if graphical environment is available
            if can_display:
                try:
                    cv2.imshow(WINDOW_NAME, vis_frame)
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == ord('q'):
                        print("  |-- Observation terminated early by user.")
                        break
                except Exception:
                    can_display = False
                    time.sleep(wait_ms / 1000.0)
            else:
                time.sleep(wait_ms / 1000.0)

            # Compute smoothed moving average FPS
            actual_dt = time.time() - last_frame_time
            last_frame_time = time.time()
            if actual_dt > 0:
                current_fps = 0.9 * current_fps + 0.1 * (1.0 / actual_dt)

    except Exception as exc:
        LOG.warning("Visual observation loop error (continuing with accumulated data): %s", exc)
    finally:
        if can_display:
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

    # Dynamic check in case configuration was updated at runtime
    from config import SEND_STRESS_ROI_TO_DISEASE
    if SEND_STRESS_ROI_TO_DISEASE:
        tracks = FISH_TRACKER.track(frame)
        print(f"  |-- Visual Stress ROI forwarding: ENABLED ({len(tracks)} fish tracks forwarded as ROI crops)")
        detection = DISEASE_DETECTOR.detect(frame, tracks=tracks, use_roi=True)
    else:
        print("  |-- Visual Stress ROI forwarding: DISABLED (performing whole-frame disease diagnosis)")
        detection = DISEASE_DETECTOR.detect(frame, tracks=None, use_roi=False)

    user_symptom_data = fetch_user_symptoms() if "fetch_user_symptoms" in globals() else fetch_user_symptom()
    symptom_text = user_symptom_data.get("text", "")
    nlp_results = process_symptoms(symptom_text)

    fused_result = fuse_disease(detection, nlp_results)
    save_json(DATA_DIR / "latest_disease.json", fused_result)

    print(f"  |-- Detected Disease: {fused_result.get('disease_class', 'Healthy')}")
    print(f"  +-- Diagnosis Confidence: {round(fused_result.get('confidence', 1.0) * 100, 1)}%")


def stage_hunger_and_feeding():
    """Stage 3: Top view 3-minute continuous hunger monitoring, temporal averaging & automatic feeding."""
    # ── 1. Check Post-Dispense Cooldown Setting ──
    in_cooldown, remaining_secs = FEEDER_SERVO.is_in_cooldown()
    if in_cooldown:
        rem_mins = round(remaining_secs / 60.0, 1)
        print(f"\n[3/5] Top Camera Hunger Observation (Post-Feed Cooldown: {rem_mins} min remaining)...")
        print(f"  |-- Post-feeding cooldown active ({rem_mins}m / {int(remaining_secs)}s remaining).")
        print(f"  +-- Skipping feeding activity check until cooldown expires ({getattr(FEEDER_SERVO.config, 'post_feed_cooldown_minutes', 30)} min total).")
        hunger_summary = {
            "hungry_count": 0,
            "average_count": 0.0,
            "presence_ratio": 0.0,
            "is_truly_hungry": False,
            "hunger_level": "Cooldown",
            "confidence": 0.0,
            "observation_duration_seconds": 0.0,
            "frames_analyzed": 0,
            "cooldown_active": True,
            "cooldown_remaining_seconds": int(remaining_secs),
            "cooldown_total_minutes": getattr(FEEDER_SERVO.config, "post_feed_cooldown_minutes", 30),
            "source": "post_feed_cooldown_bypass",
        }
        save_json(DATA_DIR / "latest_hunger.json", hunger_summary)
        return

    print("\n[3/5] Top Camera Hunger Observation (3 min continuous monitoring)...")

    HUNGER_OBSERVATION_DURATION = 180  # 3 minutes
    TARGET_FPS = 30.0
    TARGET_FRAME_TIME = 1.0 / TARGET_FPS
    WINDOW_NAME = "AquaMonitor - Step 3: Top Camera Hunger Monitoring (3 min @ 30 FPS)"

    # Ensure DISPLAY default for Linux desktop popups
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["DISPLAY"] = ":0"

    can_display = os.environ.get("HEADLESS") != "1"
    if can_display:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 800, 500)
            try:
                cv2.startWindowThread()
            except Exception:
                pass
        except Exception as exc:
            LOG.warning("Could not initialize desktop GUI popup window (%s): %s", WINDOW_NAME, exc)
            can_display = False

    start_time = time.time()
    sample_counts: List[int] = []
    sample_confidences: List[float] = []
    latest_detections: List[Dict[str, Any]] = []
    frame_count = 0
    current_fps = 30.0
    last_frame_time = time.time()
    actual_dt = 0.667

    print("  |-- Monitoring Top Camera for 3 minutes to evaluate sustained hunger (press 'q' in popup to skip)...")

    try:
        while True:
            frame_start = time.time()
            elapsed = frame_start - start_time
            remaining = max(0.0, HUNGER_OBSERVATION_DURATION - elapsed)

            if elapsed >= HUNGER_OBSERVATION_DURATION:
                break

            frame = TOP_CAMERA.read()
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "TOP CAMERA FEED UNAVAILABLE", (130, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            frame_count += 1

            # Run Top Camera YOLOv8 hunger detection
            latest_hunger_result = detect_hunger(frame)
            current_count = latest_hunger_result.get("hungry_count", 0)
            conf = latest_hunger_result.get("confidence", 0.0)
            latest_detections = latest_hunger_result.get("detections", [])

            sample_counts.append(current_count)
            if conf > 0:
                sample_confidences.append(conf)

            # Compute running stats
            running_avg = float(np.mean(sample_counts)) if sample_counts else 0.0
            presence_frames = sum(1 for c in sample_counts if c > 0)
            presence_ratio = (presence_frames / len(sample_counts)) if sample_counts else 0.0

            # Dynamic 30 FPS loop timing control
            proc_duration = time.time() - frame_start
            wait_ms = max(1, int((TARGET_FRAME_TIME - proc_duration) * 1000))

            # 1. Always generate annotated hunger inference preview frame
            vis_frame = _draw_hunger_overlay(
                frame,
                latest_detections,
                current_count,
                running_avg,
                presence_ratio,
                remaining,
                current_fps,
            )

            # 2. Persist latest frame for Web UI Live Model Preview
            try:
                cv2.imwrite(str(DATA_DIR / "latest_hunger_frame.jpg"), vis_frame)
            except Exception:
                pass

            # 3. Display in desktop GUI window if graphical environment is available
            if can_display:
                try:
                    cv2.imshow(WINDOW_NAME, vis_frame)
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == ord('q'):
                        print("  |-- Hunger observation finalized early by user.")
                        break
                except Exception:
                    can_display = False
                    time.sleep(wait_ms / 1000.0)
            else:
                time.sleep(wait_ms / 1000.0)

            actual_dt = time.time() - last_frame_time
            last_frame_time = time.time()
            if actual_dt > 0:
                current_fps = 0.9 * current_fps + 0.1 * (1.0 / actual_dt)

    except Exception as exc:
        LOG.warning("Top Camera hunger observation loop error: %s", exc)
    finally:
        if can_display:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass

    # ── 2. Calculate 3-minute Temporal Average & True Hunger Classification ──
    total_samples = len(sample_counts)
    if total_samples > 0:
        avg_hungry_count = float(np.mean(sample_counts))
        presence_frames = sum(1 for c in sample_counts if c > 0)
        presence_ratio = round(presence_frames / total_samples, 3)
        mean_confidence = round(float(np.mean(sample_confidences)), 3) if sample_confidences else 0.0
    else:
        avg_hungry_count = 0.0
        presence_ratio = 0.0
        mean_confidence = 0.0

    # Sustained surface attendance criteria:
    # Fish must be present at the top feeding zone for at least 30% of the 3-minute observation
    # with an average count of at least 0.5 fish to confirm genuine hunger.
    is_truly_hungry = (presence_ratio >= 0.30) and (avg_hungry_count >= 0.5)

    if is_truly_hungry:
        final_hungry_count = max(1, min(4, int(round(avg_hungry_count))))
        hunger_level = (
            "Low" if final_hungry_count == 1
            else ("Moderate" if final_hungry_count == 2
                  else "High")
        )
    else:
        final_hungry_count = 0
        hunger_level = "Normal"

    observation_secs = round(time.time() - start_time, 1)

    # Compile 3-minute temporal hunger report
    hunger_summary = {
        "hungry_count": final_hungry_count,
        "average_count": round(avg_hungry_count, 2),
        "presence_ratio": presence_ratio,
        "is_truly_hungry": is_truly_hungry,
        "hunger_level": hunger_level,
        "confidence": mean_confidence,
        "observation_duration_seconds": observation_secs,
        "frames_analyzed": total_samples,
        "source": "top_cam_yolov8_temporal_average",
    }
    save_json(DATA_DIR / "latest_hunger.json", hunger_summary)

    # ── 3. Feeder Servo Dispense ──
    feed_result = FEEDER_SERVO.dispense(final_hungry_count)
    save_json(DATA_DIR / "latest_feed.json", feed_result)

    print(f"  |-- Observation Complete: {observation_secs}s ({total_samples} frames sampled)")
    print(f"  |-- Avg Surface Fish: {avg_hungry_count:.2f} | Surface Attendance: {presence_ratio * 100:.1f}%")
    print(f"  |-- Hunger Status: {'CONFIRMED HUNGRY' if is_truly_hungry else 'NOT HUNGRY (Transient/Normal)'} (Count: {final_hungry_count}, Level: {hunger_level})")
    if feed_result.get("dispensed", False):
        cooldown_mins = getattr(FEEDER_SERVO.config, "post_feed_cooldown_minutes", 30)
        print(f"  +-- Dispensed Portion: True (Rounds: {feed_result.get('rounds', 0)}) -> Cooldown active: no feeding activity checks for {cooldown_mins} mins.")
    else:
        print(f"  +-- Dispensed Portion: False (Rounds: 0)")


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
