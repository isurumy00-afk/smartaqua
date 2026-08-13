"""Stress Classifier module.

Calculates individual fish stress level based on:
1. Bottom Dwelling & Stay (40%)
2. Top Feeding Area Activity (35%)
3. Freezing / Spatial Immobility (25%)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from utils.logger import get_logger

LOG = get_logger(__name__)


def classify_stress(
    top_time: float,
    bottom_time: float,
    freeze_time: float,
    longest_bottom: float,
    surface_visits: int,
    total_time: float,
    current_region: str = None,
) -> Tuple[float, str, Tuple[int, int, int], str]:
    """Determine individual fish stress level based on bottom dwelling, top feeding, and freezing immobility."""
    if total_time <= 0:
        return 0.0, "Healthy", (0, 255, 0), "Normal"

    bottom_ratio = bottom_time / total_time
    top_ratio = top_time / total_time

    # 1. Bottom Dwelling Score (Weight: 0.40)
    bottom_score = min(bottom_ratio / 0.60, 1.0)
    if current_region != "bottom":
        bottom_score *= 0.5
    bottom_stay_score = min(longest_bottom / 20.0, 1.0)
    bottom_component = 0.28 * bottom_score + 0.12 * bottom_stay_score

    # 2. Top Feeding Area Activity (Weight: 0.35) - Low surface activity indicates stress
    top_activity_score = min(top_ratio / 0.30, 1.0)
    visit_score = min(surface_visits / 10.0, 1.0)
    top_component = 0.35 * (1.0 - (0.7 * top_activity_score + 0.3 * visit_score))

    # 3. Freezing / Spatial Immobility (Weight: 0.25)
    freeze_score = min(freeze_time / 15.0, 1.0)
    freeze_component = 0.25 * freeze_score

    components = {
        "Bottom Dwelling": bottom_component,
        "Low Top Feeding Activity": top_component,
        "Freezing Motion": freeze_component,
    }

    score = max(0.0, min(bottom_component + top_component + freeze_component, 1.0))
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
        frz_t = fish.get("freeze_seconds", 0.0)
        trk_t = fish.get("tracked_seconds", 1.0)
        long_bot = fish.get("longest_bottom_seconds", 0.0)
        surf_vis = fish.get("surface_visits", 0)
        region = fish.get("region", "middle")

        score, label, color, reason = classify_stress(
            top_t, bot_t, frz_t, long_bot, surf_vis, trk_t,
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
