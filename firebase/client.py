"""Isolated Firebase Realtime Database Client module.

Network communications remain isolated from core business modules.
"""

from typing import Dict, Any
from utils.firebase import init_firebase
from utils.logger import get_logger

LOG = get_logger(__name__)


def upload(path: str, payload: Dict[str, Any]) -> bool:
    """Upload a payload dictionary to Firebase Realtime Database path."""
    if not payload:
        return False

    if not init_firebase():
        LOG.debug("Firebase upload skipped for '%s': Client unconfigured", path)
        return False

    try:
        from firebase_admin import db
        ref = db.reference(path)
        ref.set(payload)
        LOG.info("Uploaded data successfully to Firebase node '%s'", path)
        return True
    except Exception as exc:
        LOG.warning("Firebase upload to '%s' failed: %s", path, exc)
        return False


def download(path: str) -> Any:
    """Download data from a Firebase Realtime Database path."""
    if not init_firebase():
        LOG.debug("Firebase download skipped for '%s': Client unconfigured", path)
        return None

    try:
        from firebase_admin import db
        ref = db.reference(path)
        data = ref.get()
        LOG.info("Downloaded data successfully from Firebase node '%s'", path)
        return data
    except Exception as exc:
        LOG.warning("Firebase download from '%s' failed: %s", path, exc)
        return None

