"""Stress Classifier module.

Calculates individual fish stress level based on 3 behavioral categories and 9 measurable features:
1. Bottom Dwelling & Stay (Weight: 35%)
   - % time in bottom zone
   - Longest bottom stay
   - Number of bottom entries

2. Top Feeding Area Activity (Weight: 25%)
   - % time in top zone
   - Top visits/min
   - Time between top visits

3. Freezing / Spatial Immobility (Weight: 40%)
   - Immobility duration
   - Immobility events/min
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
    bottom_entries: int,
    surface_visits: int,
    time_between_top_visits: float,
    immobility_events: int,
    total_time: float,
    current_region: str = None,
) -> Tuple[float, str, Tuple[int, int, int], str]:
    """Determine individual fish stress using all 9 behavioral features."""
    if total_time <= 0:
        return 0.0, "Healthy", (0, 255, 0), "Normal"

    minutes = max(total_time / 60.0, 0.05)
    bottom_ratio = bottom_time / total_time
    top_ratio = top_time / total_time

    # ── Category 1: Bottom Dwelling & Stay (Weight: 35%) ──
    f1_bottom_pct = min(bottom_ratio / 0.60, 1.0)
    if current_region != "bottom":
        f1_bottom_pct *= 0.5
    f2_longest_stay = min(longest_bottom / 20.0, 1.0)
    f3_bottom_entries = min((bottom_entries / minutes) / 6.0, 1.0)

    bottom_score = 0.45 * f1_bottom_pct + 0.35 * f2_longest_stay + 0.20 * f3_bottom_entries
    bottom_component = 0.35 * bottom_score

    # ── Category 2: Top Feeding Area Activity (Weight: 25%) ──
    f4_top_pct = min(top_ratio / 0.30, 1.0)
    f5_top_visits_rate = min((surface_visits / minutes) / 4.0, 1.0)
    f6_time_between_visits = min(time_between_top_visits / 45.0, 1.0)

    top_activity_score = 0.45 * f4_top_pct + 0.35 * f5_top_visits_rate + 0.20 * (1.0 - f6_time_between_visits)
    top_component = 0.25 * (1.0 - top_activity_score)

    # ── Category 3: Freezing / Spatial Immobility (Weight: 40%) ──
    f7_immobility_dur = min(freeze_time / 10.0, 1.0)
    f8_immobility_events_rate = min((immobility_events / minutes) / 3.0, 1.0)

    freeze_score = 0.60 * f7_immobility_dur + 0.40 * f8_immobility_events_rate
    freeze_component = 0.40 * freeze_score

    # ── Overall Stress Score & Primary Stress Reason ──
    components = {
        "Bottom Dwelling & Stay": bottom_component,
        "Low Top Feeding Activity": top_component,
        "Freezing / Spatial Immobility": freeze_component,
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
        bot_ent = fish.get("bottom_entries", 0)
        surf_vis = fish.get("surface_visits", 0)
        t_btw_vis = fish.get("time_between_top_visits", 0.0)
        immob_evts = fish.get("immobility_events", 0)
        region = fish.get("region", "middle")

        score, label, color, reason = classify_stress(
            top_t, bot_t, frz_t, long_bot, bot_ent,
            surf_vis, t_btw_vis, immob_evts, trk_t,
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
