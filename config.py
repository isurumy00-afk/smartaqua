"""Central configuration for the Smart Aquarium Monitoring System on Raspberry Pi 4B.

Configuration values are dynamically loaded from data/config.json.
Settings can be queried or updated via the Web UI dashboard or REST API.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
CONFIG_JSON_PATH = DATA_DIR / "config.json"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Default configuration tree
DEFAULT_CONFIG: Dict[str, Any] = {
    "cameras": {
        "side_camera_index": 0,
        "top_camera_index": 1,
    },
    "sensor_config": {
        "arduino_serial_port": "/dev/ttyAMA0",
        "arduino_baudrate": 9600,
        "ionconcentration_serial_port": "/dev/ttyUSB1",
        "ionconcentration_baudrate": 9600,
        "ionconcentration_bytesize": 8,
        "ionconcentration_parity": "N",
        "ionconcentration_stopbits": 1,
        "ionconcentration_timeout": 1,
        "ionconcentration_device_id": 1,
        "ionconcentration_address": 20,
    },
    "task_intervals": {
        "side_stream": 0.2,
        "sensor": 1,
        "top_stream": 2,
        "manual_feed": 1,
        "disease": 5,
        "hunger": 30,
        "water_quality": 300,
        "watchdog": 10,
    },
    "vision": {
        "fish_confidence": 0.20,
        "max_tracked_fish": 4,
        "top_region_percent": 0.30,
        "bottom_region_percent": 0.25,
        "freeze_speed_threshold": 5.0,
        "abnormal_speed_threshold": 100.0,
        "abnormal_speed_duration": 3.0,
    },
    "servo": {
        "pin": 18,
        "minimum_angle": 0,
        "maximum_angle": 65,
        "feed_angles": [0, 20, 35, 50, 65],
        "max_daily_feedings": 10,
        "pwm_frequency": 50,
    },
    "dashboard": {
        "dashboard_host": "0.0.0.0",
        "dashboard_port": 5000,
        "dashboard_enabled": True,
    },
    "firebase": {
        "firebase_database_url": "https://smart-aquarium-default-rtdb.firebaseio.com",
    },
}


def _deep_merge(default: dict, override: dict) -> dict:
    """Recursively merge override dictionary into default dictionary."""
    result = dict(default)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json, creating it with defaults if missing."""
    if not CONFIG_JSON_PATH.exists():
        try:
            CONFIG_JSON_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        except Exception:
            pass
        return DEFAULT_CONFIG

    try:
        data = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        return _deep_merge(DEFAULT_CONFIG, data)
    except Exception:
        return DEFAULT_CONFIG


def save_config(new_config: Dict[str, Any]) -> bool:
    """Save updated configuration to config.json and reload in-memory exports."""
    try:
        merged = _deep_merge(get_all_config(), new_config)
        CONFIG_JSON_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        _apply_config_globals(merged)
        return True
    except Exception:
        return False


def get_all_config() -> Dict[str, Any]:
    """Return current structured configuration dictionary."""
    return load_config()


@dataclass
class ServoConfig:
    """Servo hardware settings for MG90 Micro Servo feeder."""
    pin: int = 18
    minimum_angle: int = 0
    maximum_angle: int = 65
    feed_angles: tuple = (0, 20, 35, 50, 65)
    max_daily_feedings: int = 10
    pwm_frequency: int = 50


# Static Model Paths (Fixed system structure)
FISH_MODEL_ONNX_PATH = MODELS_DIR / "vision" / "fish_detector_yolov8.onnx"
DISEASE_MODEL_ONNX_PATH = MODELS_DIR / "disease" / "fish_disease_model.onnx"
DISEASE_CLASSES_PATH = MODELS_DIR / "disease" / "class_names.json"
WATER_QUALITY_MODEL_DIR = MODELS_DIR / "water_quality"
FIREBASE_CREDENTIALS_PATH = BASE_DIR / "firebase_credentials.json"


# Global variables exported for modules
SIDE_CAMERA_INDEX: int = 0
TOP_CAMERA_INDEX: int = 1

FISH_CONFIDENCE: float = 0.20
MAX_TRACKED_FISH: int = 4
TOP_REGION_PERCENT: float = 0.30
BOTTOM_REGION_PERCENT: float = 0.25
FREEZE_SPEED_THRESHOLD: float = 5.0
ABNORMAL_SPEED_THRESHOLD: float = 100.0
ABNORMAL_SPEED_DURATION: float = 3.0

TASK_INTERVALS: Dict[str, int] = {}
SENSOR_CONFIG: Dict[str, Any] = {}

DASHBOARD_HOST: str = "0.0.0.0"
DASHBOARD_PORT: int = 5000
DASHBOARD_ENABLED: bool = True

FIREBASE_DATABASE_URL: str = ""

SERVO: ServoConfig = ServoConfig()


def _apply_config_globals(cfg: Dict[str, Any]) -> None:
    """Populate module-level globals from dictionary."""
    global SIDE_CAMERA_INDEX, TOP_CAMERA_INDEX
    global FISH_CONFIDENCE, MAX_TRACKED_FISH
    global TOP_REGION_PERCENT, BOTTOM_REGION_PERCENT
    global FREEZE_SPEED_THRESHOLD, ABNORMAL_SPEED_THRESHOLD, ABNORMAL_SPEED_DURATION
    global TASK_INTERVALS, SENSOR_CONFIG
    global DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_ENABLED
    global FIREBASE_DATABASE_URL, SERVO

    cams = cfg.get("cameras", {})
    SIDE_CAMERA_INDEX = int(cams.get("side_camera_index", 0))
    TOP_CAMERA_INDEX = int(cams.get("top_camera_index", 1))

    vis = cfg.get("vision", {})
    FISH_CONFIDENCE = float(vis.get("fish_confidence", 0.20))
    MAX_TRACKED_FISH = int(vis.get("max_tracked_fish", 4))
    TOP_REGION_PERCENT = float(vis.get("top_region_percent", 0.30))
    BOTTOM_REGION_PERCENT = float(vis.get("bottom_region_percent", 0.25))
    FREEZE_SPEED_THRESHOLD = float(vis.get("freeze_speed_threshold", 5.0))
    ABNORMAL_SPEED_THRESHOLD = float(vis.get("abnormal_speed_threshold", 100.0))
    ABNORMAL_SPEED_DURATION = float(vis.get("abnormal_speed_duration", 3.0))

    TASK_INTERVALS.clear()
    TASK_INTERVALS.update(cfg.get("task_intervals", {}))

    SENSOR_CONFIG.clear()
    SENSOR_CONFIG.update(cfg.get("sensor_config", {}))

    dash = cfg.get("dashboard", {})
    DASHBOARD_HOST = str(dash.get("dashboard_host", "0.0.0.0"))
    DASHBOARD_PORT = int(dash.get("dashboard_port", 5000))
    DASHBOARD_ENABLED = bool(dash.get("dashboard_enabled", True))

    fb = cfg.get("firebase", {})
    FIREBASE_DATABASE_URL = str(fb.get("firebase_database_url", ""))

    srv = cfg.get("servo", {})
    angles = srv.get("feed_angles", [0, 20, 35, 50, 65])
    if isinstance(angles, list):
        angles = tuple(angles)

    SERVO = ServoConfig(
        pin=int(srv.get("pin", 18)),
        minimum_angle=int(srv.get("minimum_angle", 0)),
        maximum_angle=int(srv.get("maximum_angle", 65)),
        feed_angles=angles,
        max_daily_feedings=int(srv.get("max_daily_feedings", 10)),
        pwm_frequency=int(srv.get("pwm_frequency", 50)),
    )


# Initial load on module import
_apply_config_globals(load_config())
