"""Firebase Admin initialization helper for Realtime Database uploads."""

from config import FIREBASE_CREDENTIALS_PATH, FIREBASE_DATABASE_URL
from utils.logger import get_logger

LOG = get_logger(__name__)

_app_initialized = False


def init_firebase():
    """Safely initialize Firebase Admin SDK if credentials are valid."""
    global _app_initialized
    if _app_initialized:
        return True

    if not FIREBASE_CREDENTIALS_PATH.exists():
        LOG.debug("Firebase credentials not found at %s. Skipping network sync.", FIREBASE_CREDENTIALS_PATH)
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials

        cred = credentials.Certificate(str(FIREBASE_CREDENTIALS_PATH))
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        _app_initialized = True
        LOG.info("Firebase app initialized successfully.")
        return True
    except Exception as exc:
        LOG.warning("Failed to initialize Firebase: %s", exc)
        return False
