"""Arduino Uno Serial Reader for Smart Aquarium Monitoring System.

Reads the JSON telemetry payload transmitted by the Arduino Uno sensor node
over Hardware UART (Serial). The Arduino handles all physical sensor acquisition:
  - DS18B20 temperature via 1-Wire (Digital Pin 2)
  - PH-4502C pH via 10-sample trimmed-mean ADC (Analog Pin A0)
  - Turbidity sensor via 10-sample trimmed-mean ADC (Analog Pin A1)

Transmission detail — the Arduino builds the JSON across sequential Serial.print()
calls and terminates with Serial.println("}") which appends \r\n:

    Serial.print("{\"temp\":");      Serial.print(tempC, 2);
    Serial.print(",\"ph\":");        Serial.print(ph_act, 2);
    Serial.print(",\"turbidity\":"); Serial.print(turbidity_ntu, 1);
    Serial.println("}");             // <-- sends }\r\n

pyserial's readline() buffers all incoming bytes until it sees \n, so the
full JSON is always assembled into one complete line before it is returned.
"""

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict

from config import SENSOR_CONFIG
from utils.logger import get_logger

LOG = get_logger(__name__)

# Module-level cache with lock — safe for concurrent access from ThreadPoolExecutor
_lock = threading.Lock()
_cached_payload: Dict[str, Any] = {}
_last_read_time: float = 0.0
_CACHE_TTL = 1.5  # seconds


def _read_serial_once() -> Dict[str, Any]:
    """Open Serial port, read one complete JSON line from the Arduino Uno.

    reset_input_buffer() is called immediately after opening the port to
    discard any partial line that arrived before the port was opened — e.g.
    if the port opened mid-cycle while the Arduino was halfway through its
    Serial.print() sequence. After flushing, readline() waits cleanly for the
    next complete \r\n-terminated packet.

    Returns parsed dict on success, empty dict on any error.
    """
    port = SENSOR_CONFIG.get("arduino_serial_port", "/dev/ttyAMA0")
    baudrate = SENSOR_CONFIG.get("arduino_baudrate", 9600)
    try:
        import serial
        with serial.Serial(port, baudrate, timeout=2.0) as ser:
            ser.reset_input_buffer()          # Discard stale / partial line
            line = ser.readline()             # Block until \n — always a full packet
            line = line.decode("utf-8", errors="ignore").strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
            LOG.debug("Arduino: unexpected line format: %r", line)
    except Exception as exc:
        LOG.debug("Arduino serial read error on %s: %s", port, exc)
    return {}


def _get_payload() -> Dict[str, Any]:
    """Return cached payload if fresh, otherwise read a new line from serial."""
    global _cached_payload, _last_read_time
    now = time.time()
    with _lock:
        if _cached_payload and (now - _last_read_time) < _CACHE_TTL:
            return _cached_payload
        fresh = _read_serial_once()
        if fresh:
            fresh["timestamp"] = datetime.now(timezone.utc).isoformat()
            _cached_payload = fresh
            _last_read_time = now
        return _cached_payload


def read() -> Dict[str, Dict[str, Any]]:
    """Read all Arduino sensor values in one call.

    Returns a dict of three standardized readings (system telemetry format):
    {
        "temperature": {"value": float|None, "unit": "C",   "timestamp": str},
        "ph":          {"value": float|None, "unit": "pH",  "timestamp": str},
        "turbidity":   {"value": float|None, "unit": "NTU", "timestamp": str},
    }
    All three readings share a single serial read (cached for _CACHE_TTL seconds).
    """
    payload = _get_payload()
    timestamp = payload.get("timestamp", datetime.now(timezone.utc).isoformat())

    def _val(key: str, decimals: int):
        raw = payload.get(key)
        if raw is None:
            return None
        try:
            return round(float(raw), decimals)
        except (ValueError, TypeError):
            return None

    return {
        "temperature": {
            "value": _val("temp", 2),
            "unit": "C",
            "timestamp": timestamp,
            "source": "arduino_serial",
        },
        "ph": {
            "value": _val("ph", 2),
            "unit": "pH",
            "timestamp": timestamp,
            "source": "arduino_serial",
        },
        "turbidity": {
            "value": _val("turbidity", 1),
            "unit": "NTU",
            "timestamp": timestamp,
            "source": "arduino_serial",
        },
    }
