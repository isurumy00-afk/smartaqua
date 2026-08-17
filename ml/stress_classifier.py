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

Sensor Fusion Layer (applied on top of visual score):
- Visual behavior stress:  60% weight
- pH sensor stress:        20% weight
- EC (ion conc.) stress:   20% weight

Healthy ranges used for sensor scoring:
- pH:  6.5 – 7.5  (saturates at ±1.5 pH units from boundary)
- EC:  100 – 500 μS/cm (saturates at 100 below / 300 above boundary)
"""

from typing import Dict, Any, List, Tuple, Optional
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


def score_ph(ph_value: float) -> float:
    """Convert a pH reading into a normalized stress score [0.0 – 1.0].

    Healthy range: 6.5 – 7.5 pH
    Score saturates at 1.0 when 1.5 pH units outside the healthy boundary.
    Returns 0.0 if ph_value is None or invalid.
    """
    if ph_value is None:
        return 0.0
    try:
        ph = float(ph_value)
    except (TypeError, ValueError):
        return 0.0

    if 6.5 <= ph <= 7.5:
        return 0.0
    deviation = (6.5 - ph) if ph < 6.5 else (ph - 7.5)
    return float(min(deviation / 1.5, 1.0))


def score_ec(ec_value: float) -> float:
    """Convert an EC / ion-concentration reading into a normalized stress score [0.0 – 1.0].

    Healthy range: 100 – 500 μS/cm
    Below 100: saturates at 1.0 when EC reaches 0.
    Above 500: saturates at 1.0 when EC reaches 800 (300 μS/cm above upper boundary).
    Returns 0.0 if ec_value is None or invalid.
    """
    if ec_value is None:
        return 0.0
    try:
        ec = float(ec_value)
    except (TypeError, ValueError):
        return 0.0

    if 100.0 <= ec <= 500.0:
        return 0.0
    if ec < 100.0:
        return float(min((100.0 - ec) / 100.0, 1.0))
    return float(min((ec - 500.0) / 300.0, 1.0))


def _level_label(score: float) -> str:
    """Map a stress score to a human-readable label."""
    if score < 0.30:
        return "Healthy"
    if score < 0.60:
        return "Mild Stress"
    return "High Stress"


def _extract_sensor_val(raw: Any) -> Optional[float]:
    """Safely extract numerical sensor reading from dictionary telemetry or scalar value."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        val = raw.get("value")
        return float(val) if val is not None else None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def classify(behavior_data: Dict[str, Any], sensor_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Wrapper function returning standard dictionary format for system JSON persistence.

    Produces two stress results:
    - Visual-only:  pure behavior-based tank stress (existing pipeline)
    - Fused:        visual (60%) + pH sensor (20%) + EC sensor (20%)

    If sensor_data is None or a sensor value is missing, that component
    contributes 0 to the fused score (graceful degradation).
    """
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

    # ── Visual-only tank score ──
    tank_score, tank_level, _ = classify_tank_stress(scores)

    # ── Sensor stress components ──
    sd = sensor_data or {}
    ph_val = _extract_sensor_val(sd.get("ph"))
    ec_val = _extract_sensor_val(sd.get("ionconcentration"))

    ph_score = score_ph(ph_val)
    ec_score = score_ec(ec_val)

    sensors_available = (ph_val is not None) or (ec_val is not None)

    # ── Sensor fusion (60% visual + 20% pH + 20% EC) ──
    fused_score = round(
        max(0.0, min(0.60 * tank_score + 0.20 * ph_score + 0.20 * ec_score, 1.0)),
        3
    )
    fused_level = _level_label(fused_score)

    # Determine primary reason for fused result
    fused_components = {
        "Behavior": 0.60 * tank_score,
        "pH Out of Range": 0.20 * ph_score,
        "EC Out of Range": 0.20 * ec_score,
    }
    fused_reason = max(fused_components, key=fused_components.get)

    LOG.debug(
        "Stress — visual=%.3f  pH=%.3f (raw=%s)  EC=%.3f (raw=%s)  fused=%.3f",
        tank_score, ph_score, ph_val, ec_score, ec_val, fused_score,
    )

    return {
        # ── Visual-only result (unchanged pipeline) ──
        "tank_stress_score": round(tank_score, 3),
        "tank_stress_level": tank_level,
        "per_fish_stress": per_fish_stress,

        # ── Sensor stress components ──
        "ph_stress_score": round(ph_score, 3),
        "ec_stress_score": round(ec_score, 3),
        "sensors_used": sensors_available,

        # ── Fused result ──
        "fused_stress_score": fused_score,
        "fused_stress_level": fused_level,
        "fused_primary_reason": fused_reason,
    }
