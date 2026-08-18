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

# Module-level cache and persistent serial connection with lock
_lock = threading.Lock()
_cached_payload: Dict[str, Any] = {}
_last_read_time: float = 0.0
_CACHE_TTL = 2.0  # seconds
_ser = None


def _get_candidate_ports(primary_port: str) -> list:
    """Return ordered list of candidate serial ports on Linux/Debian & Windows."""
    candidates = [primary_port]
    import os
    if os.name != "nt":
        # Arduino Uno frequently connects as /dev/ttyACM0 or /dev/ttyACM1 on Debian Linux
        alt_ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB1", "/dev/ttyUSB0", "/dev/ttyAMA0"]
        for p in alt_ports:
            if p not in candidates:
                candidates.append(p)
    return candidates


def _get_serial_connection(port: str, baudrate: int):
    """Obtain or reconnect to persistent Serial port with automatic fallback probing."""
    global _ser
    if _ser is not None:
        try:
            if _ser.is_open:
                return _ser
        except Exception:
            _ser = None

    import serial
    candidates = _get_candidate_ports(port)

    for p in candidates:
        import os
        from pathlib import Path
        # On Linux, only attempt ports that exist in /dev
        if os.name != "nt" and not Path(p).exists():
            continue

        try:
            ser_obj = serial.Serial(
                port=p,
                baudrate=baudrate,
                timeout=2.0,
                dsrdtr=False,
                rtscts=False,
            )
            ser_obj.reset_input_buffer()
            _ser = ser_obj
            LOG.info("Connected to Arduino Uno on serial port %s (%d baud)", p, baudrate)
            return _ser
        except Exception as exc:
            LOG.debug("Could not connect to Arduino Uno on port %s: %s", p, exc)

    _ser = None
    return None


def _read_serial_once() -> Dict[str, Any]:
    """Read one complete JSON telemetry line from the Arduino Uno via persistent serial."""
    global _ser
    port = SENSOR_CONFIG.get("arduino_serial_port", "/dev/ttyUSB1")
    baudrate = SENSOR_CONFIG.get("arduino_baudrate", 9600)
    try:
        ser = _get_serial_connection(port, baudrate)
        if ser is None or not ser.is_open:
            return {}

        line = ser.readline()
        line = line.decode("utf-8", errors="ignore").strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
        if line:
            LOG.debug("Arduino: unexpected line format: %r", line)
    except Exception as exc:
        LOG.debug("Arduino serial read error on %s: %s", port, exc)
        if _ser is not None:
            try:
                _ser.close()
            except Exception:
                pass
            _ser = None
    return {}


def close() -> None:
    """Close the persistent serial connection."""
    global _ser
    with _lock:
        if _ser is not None:
            try:
                _ser.close()
            except Exception:
                pass
            _ser = None


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
