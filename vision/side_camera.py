"""Single-owner side-camera (Camera 1) stream module.

Camera 1 is used strictly for Side View analysis:
- Fish detection
- Fish tracking
- Fish ID
- Behaviour analysis
- Stress classification
- Disease detection

The camera device is opened once and shared across downstream side-view modules.
Includes automatic synthetic aquarium video stream fallback when physical hardware camera is not available.
"""

import math
import os
import time
import threading
from typing import Optional
import cv2
import numpy as np
from config import SIDE_CAMERA_INDEX
from utils.logger import get_logger

LOG = get_logger(__name__)


class SideCamera:
    """Manages Camera 1 side-view video capture stream with synthetic fallback."""

    def __init__(self, index: int = SIDE_CAMERA_INDEX):
        self.index = index
        self.capture: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._failed_hw = False
        self._start_time = time.time()

    def _open_unlocked(self) -> None:
        """Open hardware camera capture device (caller must hold self._lock)."""
        if os.environ.get("MOCK_CAMERAS") == "1" or self.index is None:
            self._failed_hw = True
            return

        if self._failed_hw:
            return

        if self.capture is None or not self.capture.isOpened():
            try:
                # Attempt primary camera index
                self.capture = cv2.VideoCapture(self.index)
                if not self.capture or not self.capture.isOpened():
                    # Attempt Windows DirectShow fallback or index 0
                    if os.name == 'nt':
                        self.capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
                    if not self.capture or not self.capture.isOpened():
                        LOG.warning("Hardware Camera (index %s) is unavailable. Switching to live synthetic video feed.", self.index)
                        self._failed_hw = True
            except Exception as exc:
                LOG.warning("Failed to open Camera (index %s): %s", self.index, exc)
                self.capture = None
                self._failed_hw = True

    def open(self) -> None:
        """Open camera capture device if not already open."""
        with self._lock:
            self._open_unlocked()

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate synthetic aquarium video frame with animated swimming fish."""
        t = time.time() - self._start_time
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Aquatic blue gradient background (light cyan top to deep blue bottom)
        for y in range(h):
            ratio = y / float(h)
            b = int(120 * (1 - ratio) + 40 * ratio)
            g = int(80 * (1 - ratio) + 20 * ratio)
            r = int(20 * (1 - ratio) + 5 * ratio)
            frame[y, :] = (b, g, r)

        # Draw tank water line (top) and gravel line (bottom)
        cv2.line(frame, (0, 30), (w, 30), (255, 200, 100), 1)
        cv2.rectangle(frame, (0, h - 25), (w, h), (40, 60, 80), -1)
        cv2.putText(frame, "AQUARIUM LIVE STREAM [DEMO/SYNTHETIC FEED]", (15, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 255), 1)

        # Simulated fish 1 (Golden swimming fish)
        f1_x = int((w / 2) + math.sin(t * 0.8) * 200)
        f1_y = int((h / 2) + math.cos(t * 1.2) * 80)
        cv2.ellipse(frame, (f1_x, f1_y), (35, 18), int(math.sin(t * 0.8) * 15), 0, 360, (0, 165, 255), -1)
        cv2.circle(frame, (f1_x + 15, f1_y - 3), 3, (255, 255, 255), -1)
        cv2.circle(frame, (f1_x + 16, f1_y - 3), 1, (0, 0, 0), -1)

        # Simulated fish 2 (Blue active fish)
        f2_x = int((w / 2) + math.cos(t * 1.1) * 220)
        f2_y = int((h * 0.35) + math.sin(t * 1.5) * 50)
        cv2.ellipse(frame, (f2_x, f2_y), (28, 14), int(math.cos(t * 1.1) * 20), 0, 360, (255, 100, 0), -1)
        cv2.circle(frame, (f2_x + 10, f2_y - 2), 3, (255, 255, 255), -1)

        # Simulated fish 3 (Bottom dwelling fish)
        f3_x = int((w / 2) + math.sin(t * 0.4) * 160)
        f3_y = int((h * 0.82) + math.sin(t * 0.6) * 15)
        cv2.ellipse(frame, (f3_x, f3_y), (40, 16), 0, 0, 360, (50, 180, 50), -1)

        self._latest_frame = frame
        return frame

    def read(self) -> np.ndarray:
        """Read a frame from Side Camera 1.
        
        Returns BGR numpy array (hardware camera frame or synthetic feed).
        """
        with self._lock:
            if not self._failed_hw:
                self._open_unlocked()
                if self.capture and self.capture.isOpened():
                    ok, frame = self.capture.read()
                    if ok:
                        self._latest_frame = frame
                        return frame
                    else:
                        LOG.warning("Side Camera read frame failed. Falling back to synthetic feed.")
                        self._failed_hw = True

            return self._generate_synthetic_frame()

    def close(self) -> None:
        """Release camera hardware resources."""
        with self._lock:
            if self.capture:
                self.capture.release()
                self.capture = None
                LOG.info("Side Camera closed.")
