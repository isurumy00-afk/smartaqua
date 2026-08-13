"""Stress Classifier module combining behavioral metrics and environmental sensors.

Calculates individual fish stress and overall tank stress using exact multi-component formulas:
- Bottom Dwelling (0.22)
- Freezing (0.22)
- Abnormal Speed (0.16)
- Erratic Swimming (0.10)
- Low Surface Activity (0.10)
- Prolonged Bottom Stay (0.12)
- Frequent Surfacing (0.08)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from config import ABNORMAL_SPEED_DURATION
from utils.logger import get_logger

LOG = get_logger(__name__)


def classify_stress(
    top_time: float,
    bottom_time: float,
    freeze_time: float,
    mean_speed: float,
    crossings: int,
    longest_bottom: float,
    surface_visits: int,
    total_time: float,
    current_region: str = None,
    current_speed: float = None,
    high_speed_duration: float = 0.0,
) -> Tuple[float, str, Tuple[int, int, int], str]:
    """Calculate exact per-fish stress score, label, status color, and primary reason."""
    if total_time <= 0:
        return 0.0, "Healthy", (0, 255, 0), "Normal"

    bottom_ratio, top_ratio = bottom_time / total_time, top_time / total_time
    bottom_score = min(bottom_ratio / 0.70, 1.0)
    if current_region is not None and current_region != "bottom":
        bottom_score = 0.0

    speed_for_status = mean_speed if current_speed is None else current_speed
    speed_score = min(abs(speed_for_status - 40.0) / 40.0, 1.0)
    high_speed = high_speed_duration >= ABNORMAL_SPEED_DURATION

    components = {
        "Bottom Dwelling": 0.22 * bottom_score,
        "Freezing": 0.22 * min(freeze_time / 20.0, 1.0),
        "Abnormal Speed": 0.16 * speed_score,
        "Erratic Swimming": 0.10 * min(crossings / 15.0, 1.0),
        "Low Surface Activity": 0.10 * (1 - min(top_ratio / 0.30, 1.0)),
        "Prolonged Bottom Stay": 0.12 * min(longest_bottom / 30.0, 1.0),
        "Frequent Surfacing": 0.08 * min(surface_visits / 20.0, 1.0),
    }

    score = max(0.0, min(sum(components.values()), 1.0))
    reason = max(components, key=components.get)

    if high_speed:
        return max(score, 0.60), "High Stress", (0, 0, 255), "Abnormal Speed"
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
        frz_t = fish.get("freeze_seconds", 0.0)
        trk_t = fish.get("tracked_seconds", 1.0)
        spd = fish.get("current_speed", 0.0)
        cross = fish.get("crossings", 0)
        long_bot = fish.get("longest_bottom_seconds", 0.0)
        surf_vis = fish.get("surface_visits", 0)
        region = fish.get("region", "middle")
        high_speed_dur = fish.get("high_speed_duration", 0.0)
        dist = fish.get("total_distance", 0.0)
        mean_speed = dist / max(trk_t, 1e-6)

        score, label, color, reason = classify_stress(
            top_t, bot_t, frz_t, mean_speed, cross,
            long_bot, surf_vis, trk_t,
            current_region=region, current_speed=spd,
            high_speed_duration=high_speed_dur
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
