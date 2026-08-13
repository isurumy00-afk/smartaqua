"""Fetch User Symptom Input module from Firebase Realtime Database.

Retrieves symptom descriptions set by the Android application from Realtime Database key
(e.g., 'symptoms/user_input' or 'symptom_input').
Falls back to local json storage (storage/data/latest_symptom.json) if Firebase is offline.
"""

from typing import Dict, Any
from config import DATA_DIR
from firebase.client import download
from storage.json_store import load_json, save_json
from utils.logger import get_logger

LOG = get_logger(__name__)


def fetch_user_symptom() -> Dict[str, Any]:
    """Fetch symptom input text set by Android application from Firebase DB.

    Returns dictionary:
    {
        "text": "fish has red spots and gasping",
        "source": "firebase" / "local_file" / "default",
        "timestamp": str
    }
    """
    # 1. Attempt to download from Firebase Realtime Database
    firebase_paths = ["symptoms/user_input", "symptom_input", "user_symptoms"]
    for path in firebase_paths:
        data = download(path)
        if data is not None:
            text = ""
            if isinstance(data, str):
                text = data.strip()
            elif isinstance(data, dict):
                text = str(data.get("text") or data.get("symptoms") or data.get("description", "")).strip()

            if text:
                LOG.info("Retrieved user symptom input from Firebase path '%s': '%s'", path, text)
                payload = {
                    "text": text,
                    "source": "firebase",
                    "path": path,
                }
                save_json(DATA_DIR / "latest_symptom.json", payload)
                return payload

    # 2. Fallback to local JSON persistence
    local_data = load_json(DATA_DIR / "latest_symptom.json", default={})
    if local_data:
        text = str(local_data.get("text") or local_data.get("symptoms", "")).strip()
        if text:
            LOG.debug("Loaded user symptom input from local storage: '%s'", text)
            return {
                "text": text,
                "source": "local_file",
            }

    # 3. Default empty
    return {
        "text": "",
        "source": "default",
    }
