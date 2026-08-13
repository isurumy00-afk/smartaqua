"""Upload Disease Data module.

Reads latest_disease.json and uploads payload to Firebase node 'disease/latest'.
"""

from typing import Dict, Any
from firebase.client import upload


def upload_latest(payload: Dict[str, Any]) -> bool:
    """Upload disease prediction payload JSON to Firebase."""
    return upload("disease/latest", payload)
