"""Upload Behavior Data module.

Reads latest_behavior.json / latest_stress.json and uploads payload to Firebase node 'behavior/latest'.
"""

from typing import Dict, Any
from firebase.client import upload


def upload_latest(payload: Dict[str, Any]) -> bool:
    """Upload behavioral payload JSON to Firebase."""
    return upload("behavior/latest", payload)
