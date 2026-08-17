"""Fish Behaviour Analysis module.

Processes tracking results from Camera 1 to calculate behavioral metrics across 3 categories:
1. Bottom Dwelling & Stay (% time in bottom zone, longest bottom stay, number of bottom entries)
2. Top Feeding Area Activity (% time in top zone, top visits/min, time between top visits)
3. Freezing / Spatial Immobility (immobility duration, immobility events/min)
"""

import math
from typing import List, Dict, Any
import numpy as np
from config import (
    TOP_REGION_PERCENT,
    BOTTOM_REGION_PERCENT,
)


class BehaviorAnalyzer:
    """Computes spatial behavioral metrics from fish tracks."""

    def __init__(self):
        self.history: Dict[int, Dict[str, Any]] = {}

    def _make_fish_state(self):
        return {
            "last_pos": None,
            "crossings": 0,
            "top_seconds": 0.0,
            "bottom_seconds": 0.0,
            "freeze_seconds": 0.0,
            "tracked_seconds": 0.0,
            "last_region": "middle",
            "current_bottom_seconds": 0.0,
            "longest_bottom_seconds": 0.0,
            "bottom_entries": 0,
            "surface_visits": 0,
            "last_top_visit_time": None,
            "top_visit_intervals": [],
            "immobility_events": 0,
            "current_immobile_seconds": 0.0,
            "last_seen": 0.0,
        }

    def analyze(self, tracks: List[Dict[str, Any]], frame_height: int = 480, dt: float = 1.0) -> Dict[str, Any]:
        """Analyze frame tracking results and update continuous metrics with TTL eviction."""
        top_line = int(frame_height * TOP_REGION_PERCENT)
        bottom_line = int(frame_height * (1.0 - BOTTOM_REGION_PERCENT))

        import time
        now = time.time()

        # Prune stale fish metrics inactive for > 60 seconds (prevents 24/7 memory leak)
        stale_cutoff = now - 60.0
        self.history = {
            fid: s for fid, s in self.history.items()
            if s.get("last_seen", now) >= stale_cutoff
        }

        if not tracks:
            return {
                "fish_count": 0,
                "bottom_ratio": 0.0,
                "surface_ratio": 0.0,
                "freeze_seconds": 0.0,
                "erratic_events": 0,
                "shoaling_score": 1.0,
                "continuous_bottom_duration": 0.0,
                "surface_visit_frequency": 0.0,
                "fish_details": [],
            }

        bottom_count = 0
        surface_count = 0
        centers = []
        fish_details = []

        for track in tracks:
            tid = track["fish_id"]
            cx, cy = track["center"]
            centers.append((cx, cy))

            state = self.history.setdefault(tid, self._make_fish_state())
            state["tracked_seconds"] += dt
            state["last_seen"] = now

            # ── Freezing / Immobility Tracking ──
            if state["last_pos"] is not None:
                disp = math.dist(state["last_pos"], (cx, cy))
                if disp < 5.0:
                    state["current_immobile_seconds"] += dt
                    # Only count as freeze time after 5s of continuous immobility
                    if state["current_immobile_seconds"] >= 10.0:
                        state["freeze_seconds"] += dt
                        # Increment immobility episode once at the 10s crossing
                        if (state["current_immobile_seconds"] - dt) < 10.0:
                            state["immobility_events"] += 1
                else:
                    state["current_immobile_seconds"] = 0.0

            # ── Region Assignment & Entries ──
            prev_region = state["last_region"]
            if cy < top_line:
                region = "top"
                surface_count += 1
                state["top_seconds"] += dt
                state["current_bottom_seconds"] = 0.0

                if prev_region != "top":
                    state["surface_visits"] += 1
                    if state["last_top_visit_time"] is not None:
                        interval = state["tracked_seconds"] - state["last_top_visit_time"]
                        state["top_visit_intervals"].append(interval)
                    state["last_top_visit_time"] = state["tracked_seconds"]

            elif cy > bottom_line:
                region = "bottom"
                bottom_count += 1
                state["bottom_seconds"] += dt
                state["current_bottom_seconds"] += dt
                state["longest_bottom_seconds"] = max(
                    state["longest_bottom_seconds"], state["current_bottom_seconds"]
                )
                if prev_region != "bottom":
                    state["bottom_entries"] += 1
            else:
                region = "middle"
                state["current_bottom_seconds"] = 0.0

            if region != prev_region and region != "middle" and prev_region != "middle":
                state["crossings"] += 1

            state["last_region"] = region
            state["last_pos"] = (cx, cy)

            avg_top_interval = (
                float(np.mean(state["top_visit_intervals"]))
                if state["top_visit_intervals"]
                else 0.0
            )

            fish_details.append({
                "fish_id": tid,
                "bbox": track.get("bbox", []),
                "confidence": track.get("confidence", 0.0),
                "region": region,
                "tracked_seconds": round(state["tracked_seconds"], 1),
                "top_seconds": round(state["top_seconds"], 1),
                "bottom_seconds": round(state["bottom_seconds"], 1),
                "freeze_seconds": round(state["freeze_seconds"], 1),
                "longest_bottom_seconds": round(state["longest_bottom_seconds"], 1),
                "bottom_entries": state["bottom_entries"],
                "surface_visits": state["surface_visits"],
                "time_between_top_visits": round(avg_top_interval, 1),
                "immobility_events": state["immobility_events"],
                "crossings": state["crossings"],
            })

        count = len(tracks)

        # Compute shoaling score (inverse normalized variance of fish positions)
        if count >= 2:
            pts = np.array(centers)
            centroid = np.mean(pts, axis=0)
            avg_dist_to_centroid = float(np.mean(np.linalg.norm(pts - centroid, axis=1)))
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
            "freeze_seconds": round(avg_freeze, 1),
            "erratic_events": total_crossings,
            "shoaling_score": shoaling_score,
            "continuous_bottom_duration": round(avg_longest_bottom, 1),
            "surface_visit_frequency": round(float(np.mean([f["surface_visits"] for f in fish_details])) if fish_details else 0.0, 1),
            "fish_details": fish_details,
        }
