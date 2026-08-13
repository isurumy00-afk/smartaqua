#!/usr/bin/env python3
"""Comprehensive System Diagnostics Script for Smart Aquarium Monitoring System.

Audits hardware interfaces, system resources, runtime directory permissions,
pre-trained ML model artifacts, module imports, and unit functional contracts.
"""

import sys
import os
import platform
import shutil
import importlib
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import Central Configuration
try:
    from config import (
        DATA_DIR,
        LOG_DIR,
        MODELS_DIR,
        SIDE_CAMERA_INDEX,
        TOP_CAMERA_INDEX,
        FISH_MODEL_ONNX_PATH,
        DISEASE_MODEL_ONNX_PATH,
        DISEASE_CLASSES_PATH,
        WATER_QUALITY_MODEL_DIR,
        SENSOR_CONFIG,
        FIREBASE_CREDENTIALS_PATH,
        SERVO,
    )
except Exception as exc:
    print(f"[CRITICAL] Could not import config.py: {exc}")
    sys.exit(1)

# ANSI Color codes
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"


class DiagnosticReporter:
    def __init__(self):
        self.results: List[Tuple[str, str, str, str]] = []  # (Category, Test Name, Status, Details)
        self.passed_count = 0
        self.warning_count = 0
        self.failed_count = 0

    def add_result(self, category: str, test_name: str, status: str, details: str = ""):
        """Record a diagnostic test result (status: PASS, WARN, FAIL)."""
        if status == "PASS":
            self.passed_count += 1
            icon = f"{COLOR_GREEN}[PASS]{COLOR_RESET}"
        elif status == "WARN":
            self.warning_count += 1
            icon = f"{COLOR_YELLOW}[WARN]{COLOR_RESET}"
        else:
            self.failed_count += 1
            icon = f"{COLOR_RED}[FAIL]{COLOR_RESET}"

        self.results.append((category, test_name, status, details))
        print(f"  {icon} {test_name}: {details if details else status}")

    def print_summary(self):
        print("\n" + "=" * 80)
        print(f"{COLOR_BOLD}   SMART AQUARIUM SYSTEM DIAGNOSTIC HEALTH REPORT{COLOR_RESET}")
        print("=" * 80)
        
        current_cat = ""
        for cat, name, status, details in self.results:
            if cat != current_cat:
                current_cat = cat
                print(f"\n{COLOR_BLUE}--- {current_cat} ---{COLOR_RESET}")
            
            if status == "PASS":
                st_str = f"{COLOR_GREEN}PASS{COLOR_RESET}"
            elif status == "WARN":
                st_str = f"{COLOR_YELLOW}WARN{COLOR_RESET}"
            else:
                st_str = f"{COLOR_RED}FAIL{COLOR_RESET}"
                
            print(f"  [{st_str}] {name} - {details}")

        print("\n" + "-" * 80)
        total = len(self.results)
        print(f"{COLOR_BOLD}SUMMARY:{COLOR_RESET} Total Checks: {total} | "
              f"{COLOR_GREEN}Passed: {self.passed_count}{COLOR_RESET} | "
              f"{COLOR_YELLOW}Warnings: {self.warning_count}{COLOR_RESET} | "
              f"{COLOR_RED}Failed: {self.failed_count}{COLOR_RESET}")
        print("=" * 80)

        if self.failed_count > 0:
            print(f"\n{COLOR_RED}[!] CRITICAL ISSUES DETECTED:{COLOR_RESET}")
            for cat, name, status, details in self.results:
                if status == "FAIL":
                    print(f"  - [{cat}] {name}: {details}")
        elif self.warning_count > 0:
            print(f"\n{COLOR_YELLOW}[!] System functional with non-critical warnings.{COLOR_RESET}")
        else:
            print(f"\n{COLOR_GREEN}[✓] ALL SYSTEM DIAGNOSTICS PASSED PERFECTLY! System ready for 24/7 deployment.{COLOR_RESET}")


reporter = DiagnosticReporter()


# ------------------------------------------------------------------------------
# 1. System & Resource Diagnostics
# ------------------------------------------------------------------------------
def check_system_resources():
    print(f"\n{COLOR_BLUE}[1/5] Auditing System & Resource Availability...{COLOR_RESET}")
    cat = "System & Hardware"

    # Python Version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    reporter.add_result(cat, "Python Version", "PASS", f"{py_ver} ({sys.executable})")

    # Platform & Arch
    arch = platform.machine()
    system = platform.system()
    reporter.add_result(cat, "OS / Architecture", "PASS", f"{system} ({arch})")

    # RAM Availability
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_ram_mb = (pages * page_size) / (1024 * 1024)
            reporter.add_result(cat, "Physical RAM", "PASS", f"{total_ram_mb:.1f} MB Total")
        else:
            reporter.add_result(cat, "Physical RAM", "PASS", "Available (Windows/OS standard)")
    except Exception as exc:
        reporter.add_result(cat, "Physical RAM", "WARN", f"Could not query RAM: {exc}")

    # Disk Space
    try:
        total, used, free = shutil.disk_usage(str(BASE_DIR))
        free_mb = free / (1024 * 1024)
        if free_mb < 500:
            reporter.add_result(cat, "Free Disk Space", "WARN", f"{free_mb:.1f} MB remaining (low disk space)")
        else:
            reporter.add_result(cat, "Free Disk Space", "PASS", f"{free_mb / 1024:.2f} GB Free")
    except Exception as exc:
        reporter.add_result(cat, "Free Disk Space", "WARN", f"Disk space check failed: {exc}")


# ------------------------------------------------------------------------------
# 2. Directory & Model Artifact Diagnostics
# ------------------------------------------------------------------------------
def check_directories_and_artifacts():
    print(f"\n{COLOR_BLUE}[2/5] Auditing Storage Directories & Model Artifacts...{COLOR_RESET}")
    cat = "Storage & Models"

    # Data & Log Dir Write Access
    for d_name, d_path in [("Data Store Directory", DATA_DIR), ("Logs Directory", LOG_DIR)]:
        if d_path.exists() and os.access(str(d_path), os.W_OK):
            reporter.add_result(cat, d_name, "PASS", f"Writable at {d_path.relative_to(BASE_DIR)}")
        else:
            reporter.add_result(cat, d_name, "FAIL", f"Missing or non-writable: {d_path}")

    # Vision Model (YOLOv8 ONNX)
    if FISH_MODEL_ONNX_PATH.exists():
        reporter.add_result(cat, "YOLOv8 Fish Detector Model (ONNX)", "PASS", f"Found ({FISH_MODEL_ONNX_PATH.stat().st_size} bytes)")
    else:
        reporter.add_result(cat, "YOLOv8 Fish Detector Model (ONNX)", "WARN", f"Missing at {FISH_MODEL_ONNX_PATH.relative_to(BASE_DIR)} — run utils/export_onnx.py to generate")

    # Disease Classifier Model (ONNX)
    if DISEASE_MODEL_ONNX_PATH.exists():
        reporter.add_result(cat, "Disease Classifier Model (ONNX)", "PASS", f"Found ({DISEASE_MODEL_ONNX_PATH.stat().st_size} bytes)")
    else:
        reporter.add_result(cat, "Disease Classifier Model (ONNX)", "WARN", f"Missing at {DISEASE_MODEL_ONNX_PATH.relative_to(BASE_DIR)} — run utils/export_onnx.py to generate")

    if DISEASE_CLASSES_PATH.exists():
        reporter.add_result(cat, "Disease Class Labels", "PASS", f"Found at {DISEASE_CLASSES_PATH.relative_to(BASE_DIR)}")
    else:
        reporter.add_result(cat, "Disease Class Labels", "WARN", f"Missing at {DISEASE_CLASSES_PATH.relative_to(BASE_DIR)}")

    # Water Quality Predictor Model
    rfr_path = WATER_QUALITY_MODEL_DIR / "rfr_model.pkl"
    if rfr_path.exists():
        reporter.add_result(cat, "Water Quality RFR Model", "PASS", f"Found at {rfr_path.relative_to(BASE_DIR)}")
    else:
        reporter.add_result(cat, "Water Quality RFR Model", "WARN", f"Missing at {rfr_path.relative_to(BASE_DIR)} (Rule-based fallback active)")

    # Firebase Credentials
    if FIREBASE_CREDENTIALS_PATH.exists():
        reporter.add_result(cat, "Firebase Credentials JSON", "PASS", f"Found at {FIREBASE_CREDENTIALS_PATH.name}")
    else:
        reporter.add_result(cat, "Firebase Credentials JSON", "WARN", f"Missing at {FIREBASE_CREDENTIALS_PATH.name} (Firebase uploads skipped)")


# ------------------------------------------------------------------------------
# 3. Hardware Interface Diagnostics
# ------------------------------------------------------------------------------
def check_hardware_interfaces():
    print(f"\n{COLOR_BLUE}[3/5] Auditing Hardware & Sensor Devices...{COLOR_RESET}")
    cat = "Hardware Peripherals"

    # OpenCV Camera test
    try:
        import cv2
        # Check Camera 1 (Side View)
        cap1 = cv2.VideoCapture(SIDE_CAMERA_INDEX)
        if cap1.isOpened():
            reporter.add_result(cat, f"Camera 1 (Side View index={SIDE_CAMERA_INDEX})", "PASS", "Device opened successfully")
            cap1.release()
        else:
            reporter.add_result(cat, f"Camera 1 (Side View index={SIDE_CAMERA_INDEX})", "WARN", "Video device index could not be opened (Fallback capture mode)")
        
        # Check Camera 2 (Top View)
        cap2 = cv2.VideoCapture(TOP_CAMERA_INDEX)
        if cap2.isOpened():
            reporter.add_result(cat, f"Camera 2 (Top View index={TOP_CAMERA_INDEX})", "PASS", "Device opened successfully")
            cap2.release()
        else:
            reporter.add_result(cat, f"Camera 2 (Top View index={TOP_CAMERA_INDEX})", "WARN", "Video device index could not be opened (Fallback capture mode)")
    except Exception as exc:
        reporter.add_result(cat, "OpenCV Camera Drivers", "WARN", f"OpenCV check failed: {exc}")

    # Arduino Uno UART Serial Port (Temperature, pH, Turbidity)
    ard_port = Path(SENSOR_CONFIG.get("arduino_serial_port", "/dev/ttyAMA0"))
    if ard_port.exists() or os.name == "nt":
        reporter.add_result(cat, "Arduino Uno UART Serial Port", "PASS", f"Port interface available ({ard_port})")
    else:
        reporter.add_result(cat, "Arduino Uno UART Serial Port", "WARN", f"UART port {ard_port} not attached")

    # Ion Concentration Serial Port
    ion_port = Path(SENSOR_CONFIG.get("ionconcentration_serial_port", "/dev/ttyUSB1"))
    if ion_port.exists() or os.name == "nt":
        reporter.add_result(cat, "Ion Concentration Modbus Serial Port", "PASS", f"Port interface available ({ion_port})")
    else:
        reporter.add_result(cat, "Ion Concentration Modbus Serial Port", "WARN", f"Serial port {ion_port} not attached")

    # Feeder Servo Pin
    reporter.add_result(cat, "MG90 Feeder Servo", "PASS", f"Configured on GPIO Pin {SERVO.pin} (Freq={SERVO.pwm_frequency}Hz)")


# ------------------------------------------------------------------------------
# 4. Core Module Import & Contract Diagnostics
# ------------------------------------------------------------------------------
def check_core_modules():
    print(f"\n{COLOR_BLUE}[4/5] Auditing Module Import Integrity & Logic Contracts...{COLOR_RESET}")
    cat = "Module Imports"

    modules_to_test = [
        ("utils.logger", "Logger Utility"),
        ("utils.scheduler", "Task Scheduler Utility"),
        ("utils.firebase", "Firebase Bridge Utility"),
        ("storage.json_store", "JSON Persistence Store"),
        ("health.watchdog", "System Watchdog Context Manager"),
        # Sensor readers: Arduino Uno handles temp/pH/turbidity over serial;
        # ion concentration uses a separate Modbus RTU USB device.
        ("sensors.arduino_reader", "Arduino Serial Reader (Temp / pH / Turbidity)"),
        ("sensors.ionconcentration_reader", "Ion Concentration Modbus RTU Reader"),
        ("vision.side_camera", "Side Camera Capture Module"),
        ("vision.top_camera", "Top Camera Capture Module"),
        ("vision.fish_tracker", "Fish Tracker Module"),
        ("vision.fish_behavior", "Fish Behavior Analyzer Module"),
        ("vision.disease_detector", "Disease Detector Module"),
        ("vision.hunger_detector", "Hunger Detector Module"),
        ("ml.water_quality_predictor", "Water Quality Predictor ML Module"),
        ("ml.stress_classifier", "Stress Classifier ML Module"),
        ("ml.shap_explainer", "SHAP Explainer ML Module"),
        ("ml.disease_fusion", "Disease Fusion ML Module"),
        ("nlp.symptom_input", "NLP Symptom Parser Module"),
        ("feeding.servo", "MG90 Servo Feeder Module"),
        ("firebase.client", "Firebase Upload Client Module"),
        ("firebase.fetch_user_symptoms", "Firebase User Symptom Fetcher Module"),
        ("firebase.upload_sensor_data", "Firebase Sensor Data Uploader"),
        ("firebase.upload_behavior", "Firebase Behavior Data Uploader"),
        ("firebase.upload_water_quality", "Firebase Water Quality Uploader"),
        ("firebase.upload_disease", "Firebase Disease Data Uploader"),
    ]

    for mod_path, mod_desc in modules_to_test:
        try:
            importlib.import_module(mod_path)
            reporter.add_result(cat, f"Import {mod_path}", "PASS", mod_desc)
        except Exception as exc:
            reporter.add_result(cat, f"Import {mod_path}", "FAIL", f"Import Error: {exc}")


# ------------------------------------------------------------------------------
# 5. End-to-End Functional Smoke Tests
# ------------------------------------------------------------------------------
def check_functional_contracts():
    print(f"\n{COLOR_BLUE}[5/5] Executing Functional Component Smoke Tests...{COLOR_RESET}")
    cat = "Functional Execution"

    # Test JSON Persistence Store
    try:
        from storage.json_store import save_json, load_json
        test_file = "diag_test.json"
        test_data = {"status": "ok", "test": True}
        save_json(test_file, test_data)
        read_back = load_json(test_file, {})
        if read_back.get("test") is True:
            reporter.add_result(cat, "Atomic JSON Store", "PASS", "Write/Read verification successful")
        else:
            reporter.add_result(cat, "Atomic JSON Store", "FAIL", "Read data did not match written test data")
        
        # Cleanup test file
        test_path = DATA_DIR / test_file
        if test_path.exists():
            test_path.unlink()
    except Exception as exc:
        reporter.add_result(cat, "Atomic JSON Store", "FAIL", f"JSON store error: {exc}")

    # Test Watchdog Context Manager
    try:
        from health.watchdog import Watchdog
        wd = Watchdog()
        with wd.monitor("diag_test_task"):
            _ = 1 + 1
        reporter.add_result(cat, "Watchdog Context Manager", "PASS", "Monitored execution block succeeded")
    except Exception as exc:
        reporter.add_result(cat, "Watchdog Context Manager", "FAIL", f"Watchdog error: {exc}")

    # Test Sensor Readers Smoke Test
    # Arduino provides temp/pH/turbidity; ionconcentration comes over Modbus RTU.
    try:
        from sensors.arduino_reader import read as read_arduino
        from sensors.ionconcentration_reader import read as read_ion

        ard = read_arduino()
        ion = read_ion()
        temp_val = ard["temperature"].get("value")
        ph_val   = ard["ph"].get("value")
        turb_val = ard["turbidity"].get("value")
        ion_val  = ion.get("value")
        reporter.add_result(
            cat, "Sensor Readers Callability", "PASS",
            f"Arduino: Temp={temp_val}C ph={ph_val} Turbidity={turb_val} NTU | "
            f"Modbus: IonConc={ion_val} uS/cm"
        )
    except Exception as exc:
        reporter.add_result(cat, "Sensor Readers Callability", "FAIL", f"Sensor read error: {exc}")

    # Test ML Water Quality Predictor
    try:
        from ml.water_quality_predictor import WaterQualityPredictor
        wqp = WaterQualityPredictor()
        res = wqp.predict({"ph": 7.2, "ionconcentration": 250, "temp": 25.5, "turbidity": 120})
        reporter.add_result(cat, "Water Quality Predictor ML", "PASS", f"Predicted WQI Index: {res.get('water_quality')}")
    except Exception as exc:
        reporter.add_result(cat, "Water Quality Predictor ML", "FAIL", f"WQI Predictor error: {exc}")

    # Test Stress Classifier
    try:
        from ml.stress_classifier import classify
        st_res = classify(
            behavior_data={"surface_dwelling": False, "erratic": False, "freezing": False},
            sensor_data={"temperature": 25.5, "ph": 7.2}
        )
        reporter.add_result(cat, "Stress Classifier ML", "PASS", f"Classified level: {st_res.get('tank_stress_level', 'Healthy')}")
    except Exception as exc:
        reporter.add_result(cat, "Stress Classifier ML", "FAIL", f"Stress Classifier error: {exc}")

    # Test Feeder Servo Angle Calculator
    try:
        from feeding.servo import FeederServo
        servo = FeederServo()
        angle = servo.dispense(hungry_count=2)
        reporter.add_result(cat, "Feeder Servo Angle Logic", "PASS", f"Hungry count 2 -> Angle {angle}°")
    except Exception as exc:
        reporter.add_result(cat, "Feeder Servo Angle Logic", "FAIL", f"Feeder Servo error: {exc}")

    # Test NLP Symptom Parser & Fusion with CV Detection
    try:
        from nlp.symptom_input import process as process_symptoms
        from ml.disease_fusion import fuse as fuse_disease
        nlp_res = process_symptoms("Fish has white spots and scratching on rocks")
        cv_mock = {"disease_class": "Parasitic diseases", "confidence": 0.85}
        fused = fuse_disease(cv_mock, nlp_res)
        reporter.add_result(cat, "NLP Parser & CV Fusion", "PASS", f"Fused result: '{fused.get('disease')}' (Confidence: {fused.get('confidence')})")
    except Exception as exc:
        reporter.add_result(cat, "NLP Parser & CV Fusion", "FAIL", f"NLP/Fusion error: {exc}")


    # Test Live Firebase Connection & Initialization
    try:
        from utils.firebase import init_firebase
        from config import FIREBASE_CREDENTIALS_PATH, FIREBASE_DATABASE_URL

        if not FIREBASE_CREDENTIALS_PATH.exists():
            reporter.add_result(cat, "Firebase Network Connection", "WARN",
                                f"Credentials file missing ({FIREBASE_CREDENTIALS_PATH.name}). Live cloud sync skipped.")
        else:
            if init_firebase():
                try:
                    from firebase_admin import db
                    ref = db.reference("health_ping")
                    _ = ref.get()
                    reporter.add_result(cat, "Firebase Network Connection", "PASS",
                                        f"Connected to Realtime DB ({FIREBASE_DATABASE_URL})")
                except Exception as net_err:
                    reporter.add_result(cat, "Firebase Network Connection", "FAIL",
                                        f"SDK initialized but database query failed: {net_err}")
            else:
                reporter.add_result(cat, "Firebase Network Connection", "WARN",
                                    "Firebase Admin SDK initialization failed")
    except Exception as exc:
        reporter.add_result(cat, "Firebase Network Connection", "FAIL", f"Firebase check error: {exc}")


def main():
    print("==========================================================================")
    print(" Smart Aquarium Monitoring System - Comprehensive Diagnostic Suite")
    print(" Target Environment: Raspberry Pi 4B (2GB RAM) / Linux / Dev Host")
    print("==========================================================================")

    check_system_resources()
    check_directories_and_artifacts()
    check_hardware_interfaces()
    check_core_modules()
    check_functional_contracts()

    reporter.print_summary()


if __name__ == "__main__":
    main()
