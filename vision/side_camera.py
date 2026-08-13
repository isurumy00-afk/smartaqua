"""Single-owner side-camera (Camera 1) stream module.

Camera 1 is used strictly for Side View analysis:
- Fish detection
- Fish tracking
- Fish ID
- Behaviour analysis
- Stress classification
- Disease detection

The camera device is opened once and shared across downstream side-view modules.
"""

import threading
import cv2
from typing import Optional
from config import SIDE_CAMERA_INDEX
from utils.logger import get_logger

LOG = get_logger(__name__)


class SideCamera:
    """Manages Camera 1 side-view video capture stream."""

    def __init__(self, index: int = SIDE_CAMERA_INDEX):
        self.index = index
        self.capture: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._failed = False

    def _open_unlocked(self) -> None:
        """Open camera capture device (caller must already hold self._lock)."""
        import os
        if os.environ.get("MOCK_CAMERAS") == "1" or self.index is None:
            self._failed = True
            return

        if self._failed:
            return
        if self.capture is None or not self.capture.isOpened():
            try:
                self.capture = cv2.VideoCapture(self.index)
                if not self.capture or not self.capture.isOpened():
                    LOG.warning("Side Camera (index %s) is unavailable", self.index)
                    self._failed = True
            except Exception as exc:
                LOG.warning("Failed to open Side Camera (index %s): %s", self.index, exc)
                self.capture = None
                self._failed = True

    def open(self) -> None:
        """Open camera capture device if not already open."""
        with self._lock:
            self._open_unlocked()

    def read(self):
        """Read a frame from Side Camera 1.
        
        Returns frame numpy array, or None if unavailable.
        """
        with self._lock:
            if self._failed:
                return None
            self._open_unlocked()
            if self.capture and self.capture.isOpened():
                ok, frame = self.capture.read()
                if ok:
                    self._latest_frame = frame
                    return frame
                else:
                    self._failed = True
            return None

    def close(self) -> None:
        """Release camera hardware resources."""
        with self._lock:
            if self.capture:
                self.capture.release()
                self.capture = None
                LOG.info("Side Camera closed.")
