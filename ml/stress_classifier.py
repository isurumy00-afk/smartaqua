"""Stress Classifier module combining behavioral metrics and environmental sensors.

Calculates individual fish stress and overall tank stress using behavioral formulas:
- Bottom Dwelling (0.35)
- Erratic Swimming (0.20)
- Low Surface Activity (0.15)
- Prolonged Bottom Stay (0.18)
- Frequent Surfacing (0.12)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from utils.logger import get_logger

LOG = get_logger(__name__)


def classify_stress(
    top_time: float,
    bottom_time: float,
    crossings: int,
    longest_bottom: float,
    surface_visits: int,
    total_time: float,
    current_region: str = None,
) -> Tuple[float, str, Tuple[int, int, int], str]:
    """Calculate exact per-fish stress score, label, status color, and primary reason."""
    if total_time <= 0:
        return 0.0, "Healthy", (0, 255, 0), "Normal"

    bottom_ratio, top_ratio = bottom_time / total_time, top_time / total_time
    bottom_score = min(bottom_ratio / 0.70, 1.0)
    if current_region is not None and current_region != "bottom":
        bottom_score = 0.0

    components = {
        "Bottom Dwelling": 0.35 * bottom_score,
        "Erratic Swimming": 0.20 * min(crossings / 15.0, 1.0),
        "Low Surface Activity": 0.15 * (1 - min(top_ratio / 0.30, 1.0)),
        "Prolonged Bottom Stay": 0.18 * min(longest_bottom / 30.0, 1.0),
        "Frequent Surfacing": 0.12 * min(surface_visits / 20.0, 1.0),
    }

    score = max(0.0, min(sum(components.values()), 1.0))
    reason = max(components, key=components.get)

    if score < 0.30:
        return score, "Healthy", (0, 255, 0), "Normal"
    if score < 0.60:
        return score, "Mild Stress", (0, 255, 255), reason
    return score, "High Stress", (0, 0, 255), reason


def classify_tank_stress(scores: List[float]) -> Tuple[float, str, Tuple[int, int, int]]:
    """Return a live whole-tank score from the fish scores visible in this frame."""
    if not scores:
        return 0.0, "No fish detected", (180, 180, 180)
    score = float(np.mean(scores))
    if score < 0.30:
        return score, "Healthy", (0, 255, 0)
    if score < 0.60:
        return score, "Mild Stress", (0, 255, 255)
    return score, "High Stress", (0, 0, 255)


def classify(behavior_data: Dict[str, Any], sensor_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Wrapper function returning standard dictionary format for system JSON persistence."""
    fish_details = behavior_data.get("fish_details", [])
    per_fish_stress = []
    scores = []

    for fish in fish_details:
        fid = fish.get("fish_id", 0)
        top_t = fish.get("top_seconds", 0.0)
        bot_t = fish.get("bottom_seconds", 0.0)
        trk_t = fish.get("tracked_seconds", 1.0)
        cross = fish.get("crossings", 0)
        long_bot = fish.get("longest_bottom_seconds", 0.0)
        surf_vis = fish.get("surface_visits", 0)
        region = fish.get("region", "middle")

        score, label, color, reason = classify_stress(
            top_t, bot_t, cross,
            long_bot, surf_vis, trk_t,
            current_region=region
        )
        scores.append(score)
        per_fish_stress.append({
            "fish_id": fid,
            "stress_score": round(score, 3),
            "stress_level": label,
            "primary_reason": reason
        })

    tank_score, tank_level, _ = classify_tank_stress(scores)

    return {
        "tank_stress_score": round(tank_score, 3),
        "tank_stress_level": tank_level,
        "per_fish_stress": per_fish_stress
    }
