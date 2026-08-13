"""Hungry Fish Detection module for Top View (Camera 2) only.

Detects hungry fish hovering in the feeding zone of Camera 2.
Never used for side-view stress analysis.
"""

from typing import Dict, Any, List
import numpy as np
from utils.logger import get_logger

LOG = get_logger(__name__)


def detect(frame) -> Dict[str, Any]:
    """Detect hungry fish from Top Camera 2 frame.
    
    Returns:
    {
        "hungry_fish_ids": List[int],
        "hungry_count": int,
        "confidence": float
    }
    """
    if frame is None:
        return {"hungry_fish_ids": [], "hungry_count": 0, "confidence": 0.0}

    try:
        # Computer vision analysis of feeding zone on top view
        import cv2

        h, w = frame.shape[:2]
        # Define central top feeding area ROI (middle 50% width, top 60% height)
        roi = frame[0:int(h * 0.6), int(w * 0.25):int(w * 0.75)]
        
        # Motion / contour / surface presence detection in ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size corresponding to fish visible from top view
        fish_contours = [c for c in contours if 300 < cv2.contourArea(c) < 15000]
        hungry_count = min(len(fish_contours), 4)
        hungry_ids = [i + 1 for i in range(hungry_count)]
        confidence = round(float(np.clip(0.70 + (0.05 * hungry_count), 0.0, 0.95)), 2)

        return {
            "hungry_fish_ids": hungry_ids,
            "hungry_count": hungry_count,
            "confidence": confidence,
        }
    except Exception as exc:
        LOG.warning("Hunger detection evaluation failed: %s", exc)
        return {"hungry_fish_ids": [], "hungry_count": 0, "confidence": 0.0, "error": str(exc)}
