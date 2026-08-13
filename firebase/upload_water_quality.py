"""Upload Water Quality Data module.

Reads latest_water_quality.json / latest_shap.json and uploads payload to Firebase node 'water_quality/latest'.
"""

from typing import Dict, Any
from firebase.client import upload


def upload_latest(payload: Dict[str, Any]) -> bool:
    """Upload water quality payload JSON to Firebase."""
    return upload("water_quality/latest", payload)
