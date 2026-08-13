"""Single Main Orchestrator for the Smart Aquarium Monitoring System on Raspberry Pi 4B.

Executes all monitoring, ML inference, disease detection, hunger feeding, water quality prediction,
and cloud synchronization sequentially in a simple, clean, linear loop.
Prints clear real-time progress to the terminal and saves all state to JSON.
"""

import time
from datetime import datetime, timezone

from config import DATA_DIR
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
from ml.stress_classifier import classify as classify_stress
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


def stage_sensors_and_stress():
    """Stage 1: Read sensors (Arduino + Modbus), track fish & classify stress."""
    print("\n[1/5] Reading Sensors & Analyzing Fish Behavior/Stress...")
    
    # 1. Read Sensors
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

    # 2. Side Camera Capture & Tracking
    frame = SIDE_CAMERA.read()
    tracks = FISH_TRACKER.track(frame)

    # Render OpenCV window with live feed, region boundary lines, and bounding boxes
    if frame is not None:
        try:
            import cv2
            from config import TOP_REGION_PERCENT, BOTTOM_REGION_PERCENT
            vis_frame = frame.copy()
            fh, fw = vis_frame.shape[:2]

            top_line = int(fh * TOP_REGION_PERCENT)
            bottom_line = int(fh * (1.0 - BOTTOM_REGION_PERCENT))

            # Draw Top Feeding Region boundary line (Magenta)
            cv2.line(vis_frame, (0, top_line), (fw, top_line), (255, 0, 255), 2)
            cv2.putText(vis_frame, "TOP FEEDING REGION", (10, top_line - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

            # Draw Bottom Dwelling Region boundary line (Yellow)
            cv2.line(vis_frame, (0, bottom_line), (fw, bottom_line), (0, 255, 255), 2)
            cv2.putText(vis_frame, "BOTTOM DWELLING REGION", (10, bottom_line + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Draw Bounding Boxes & Tags
            for fish in tracks:
                bbox = fish.get("bbox")
                fid = fish.get("fish_id", "?")
                conf = fish.get("confidence", 0.0)
                speed = fish.get("speed", 0.0)
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    # Draw cyan bounding box
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    label = f"Fish #{fid} ({int(conf*100)}%) {speed}px/s"
                    cv2.putText(vis_frame, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow("AquaMonitor - Step 1: Live Fish Tracking", vis_frame)
            cv2.waitKey(1)
        except Exception as exc:
            LOG.debug("OpenCV GUI display skipped (headless environment): %s", exc)

    # 3. Behavior & Stress
    frame_h = frame.shape[0] if frame is not None else 480
    behavior_metrics = BEHAVIOR_ANALYZER.analyze(tracks, frame_height=frame_h)
    save_json(DATA_DIR / "latest_behavior.json", behavior_metrics)

    stress_results = classify_stress(behavior_metrics, sensor_readings)
    save_json(DATA_DIR / "latest_stress.json", stress_results)

    temp_val = sensor_readings['temperature'].get('value', 'N/A')
    ph_val = sensor_readings['ph'].get('value', 'N/A')
    turb_val = sensor_readings['turbidity'].get('value', 'N/A')
    ion_val = sensor_readings['ionconcentration'].get('value', 'N/A')

    print(f"  |-- Temp: {temp_val} C | pH: {ph_val} | Turbidity: {turb_val} NTU | Ion: {ion_val} uS/cm")
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
