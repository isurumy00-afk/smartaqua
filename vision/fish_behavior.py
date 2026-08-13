"""Fish Behaviour Analysis module.

Processes tracking results from Camera 1 to calculate behavioral metrics:
- Bottom dwelling ratio & continuous bottom stay duration
- Surface dwelling ratio & surface visit frequency
- Freezing duration (speed < threshold)
- Erratic swimming events & region crossings
- Average speed & total distance travelled
- Shoaling score (spatial dispersion of fish group)
"""

import math
from typing import List, Dict, Any
import numpy as np
from config import (
    TOP_REGION_PERCENT,
    BOTTOM_REGION_PERCENT,
    FREEZE_SPEED_THRESHOLD,
    ABNORMAL_SPEED_THRESHOLD,
)


class BehaviorAnalyzer:
    """Computes spatial and kinematic behavioral metrics from fish tracks."""

    def __init__(self):
        self.history: Dict[int, Dict[str, Any]] = {}

    def _make_fish_state(self):
        return {
            "last_pos": None,
            "total_distance": 0.0,
            "crossings": 0,
            "top_seconds": 0.0,
            "bottom_seconds": 0.0,
            "freeze_seconds": 0.0,
            "tracked_seconds": 0.0,
            "last_region": "middle",
            "current_bottom_seconds": 0.0,
            "longest_bottom_seconds": 0.0,
            "surface_visits": 0,
            "high_speed_duration": 0.0,
        }

    def analyze(self, tracks: List[Dict[str, Any]], frame_height: int = 480, dt: float = 1.0) -> Dict[str, Any]:
        """Analyze frame tracking results and update continuous metrics.
        
        Returns aggregated tank and per-fish behavioral metrics.
        """
        top_line = int(frame_height * TOP_REGION_PERCENT)
        bottom_line = int(frame_height * (1.0 - BOTTOM_REGION_PERCENT))

        if not tracks:
            return {
                "fish_count": 0,
                "bottom_ratio": 0.0,
                "surface_ratio": 0.0,
                "average_speed": 0.0,
                "freeze_seconds": 0.0,
                "erratic_events": 0,
                "shoaling_score": 1.0,
                "continuous_bottom_duration": 0.0,
                "surface_visit_frequency": 0.0,
                "fish_details": [],
            }

        bottom_count = 0
        surface_count = 0
        speeds = []
        centers = []
        fish_details = []

        for track in tracks:
            tid = track["fish_id"]
            cx, cy = track["center"]
            speed = track.get("speed", 0.0)
            centers.append((cx, cy))
            speeds.append(speed)

            state = self.history.setdefault(tid, self._make_fish_state())
            state["tracked_seconds"] += dt
            state["total_distance"] += speed * dt

            if speed < FREEZE_SPEED_THRESHOLD:
                state["freeze_seconds"] += dt

            if speed >= ABNORMAL_SPEED_THRESHOLD:
                state["high_speed_duration"] += dt
            else:
                state["high_speed_duration"] = 0.0

            # Region assignment
            if cy < top_line:
                region = "top"
                surface_count += 1
                state["top_seconds"] += dt
                state["current_bottom_seconds"] = 0.0
            elif cy > bottom_line:
                region = "bottom"
                bottom_count += 1
                state["bottom_seconds"] += dt
                state["current_bottom_seconds"] += dt
                state["longest_bottom_seconds"] = max(
                    state["longest_bottom_seconds"], state["current_bottom_seconds"]
                )
            else:
                region = "middle"
                state["current_bottom_seconds"] = 0.0

            # Crossing & surface visit counts
            prev_region = state["last_region"]
            if region != prev_region and region != "middle" and prev_region != "middle":
                state["crossings"] += 1
            if prev_region != "top" and region == "top":
                state["surface_visits"] += 1

            state["last_region"] = region
            state["last_pos"] = (cx, cy)

            fish_details.append({
                "fish_id": tid,
                "bbox": track.get("bbox", []),
                "confidence": track.get("confidence", 0.0),
                "current_speed": speed,
                "region": region,
                "total_distance": round(state["total_distance"], 1),
                "tracked_seconds": round(state["tracked_seconds"], 1),
                "freeze_seconds": round(state["freeze_seconds"], 1),
                "longest_bottom_seconds": round(state["longest_bottom_seconds"], 1),
                "surface_visits": state["surface_visits"],
                "crossings": state["crossings"],
            })

        count = len(tracks)
        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        # Compute shoaling score (inverse normalized variance of fish positions)
        if count >= 2:
            pts = np.array(centers)
            centroid = np.mean(pts, axis=0)
            avg_dist_to_centroid = float(np.mean(np.linalg.norm(pts - centroid, axis=1)))
            # Higher score (closer to 1.0) means tighter group/shoal
            shoaling_score = round(float(np.clip(1.0 - (avg_dist_to_centroid / (frame_height * 1.5)), 0.0, 1.0)), 2)
        else:
            shoaling_score = 1.0

        avg_freeze = float(np.mean([f["freeze_seconds"] for f in fish_details])) if fish_details else 0.0
        avg_longest_bottom = float(np.max([f["longest_bottom_seconds"] for f in fish_details])) if fish_details else 0.0
        total_crossings = int(np.sum([f["crossings"] for f in fish_details])) if fish_details else 0

        return {
            "fish_count": count,
            "bottom_ratio": round(bottom_count / count, 2),
            "surface_ratio": round(surface_count / count, 2),
            "average_speed": round(avg_speed, 2),
            "freeze_seconds": round(avg_freeze, 1),
            "erratic_events": total_crossings,
            "shoaling_score": shoaling_score,
            "continuous_bottom_duration": round(avg_longest_bottom, 1),
            "surface_visit_frequency": round(float(np.mean([f["surface_visits"] for f in fish_details])) if fish_details else 0.0, 1),
            "fish_details": fish_details,
        }
