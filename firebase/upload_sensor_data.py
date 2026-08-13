"""Upload Sensor Data module.

Reads latest_sensor.json and uploads payload to Firebase node 'sensors/latest'.
"""

from typing import Dict, Any
from firebase.client import upload


def upload_latest(payload: Dict[str, Any]) -> bool:
    """Upload sensor payload JSON to Firebase."""
    return upload("sensors/latest", payload)
