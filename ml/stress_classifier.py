"""Stress Classifier module combining behavioral metrics and sensor readings.

Predicts individual fish stress and overall tank stress:
- Stress levels: Healthy (<0.30), Mild (0.30-0.50), Moderate (0.50-0.70), High (0.70-0.85), Critical (>0.85).
"""

from typing import Dict, Any, List
import numpy as np
from utils.logger import get_logger

LOG = get_logger(__name__)


def classify(behavior_data: Dict[str, Any], sensor_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Calculate fish and tank stress scores based on behavioral metrics and environmental sensors.
    
    Returns:
    {
        "tank_stress_score": float,
        "tank_stress_level": str,
        "primary_stressor": str,
        "per_fish_stress": List[dict]
    }
    """
    sensor_data = sensor_data or {}
    temperature = sensor_data.get("temperature")
    if isinstance(temperature, dict):
        temperature = temperature.get("value")

    ionconcentration = sensor_data.get("ionconcentration")
    if isinstance(ionconcentration, dict):
        ionconcentration = ionconcentration.get("value")

    fish_details = behavior_data.get("fish_details", [])
    per_fish_stress: List[Dict[str, Any]] = []
    scores = []

    for fish in fish_details:
        fid = fish.get("fish_id", 0)
        speed = fish.get("current_speed", 0.0)
        freeze_sec = fish.get("freeze_seconds", 0.0)
        longest_bottom = fish.get("longest_bottom_seconds", 0.0)
        crossings = fish.get("crossings", 0)
        region = fish.get("region", "middle")

        # Multi-factor stress formula weighted by physiological stress indicators
        bottom_score = 0.25 * min(longest_bottom / 30.0, 1.0) if region == "bottom" else 0.0
        freeze_score = 0.25 * min(freeze_sec / 20.0, 1.0)
        speed_score = 0.20 * min(abs(speed - 40.0) / 40.0, 1.0)
        erratic_score = 0.15 * min(crossings / 15.0, 1.0)

        temp_score = 0.0
        if temperature is not None:
            # Ideal aquarium temperature range: 24 - 28 C
            temp_deviation = abs(temperature - 26.0)
            temp_score = 0.15 * min(temp_deviation / 8.0, 1.0)

        total_score = float(np.clip(bottom_score + freeze_score + speed_score + erratic_score + temp_score, 0.0, 1.0))
        scores.append(total_score)

        level = _score_to_level(total_score)
        primary_reason = _identify_primary_stressor(bottom_score, freeze_score, speed_score, erratic_score, temp_score)

        per_fish_stress.append({
            "fish_id": fid,
            "stress_score": round(total_score, 3),
            "stress_level": level,
            "primary_reason": primary_reason
        })

    if scores:
        tank_score = float(np.mean(scores))
    else:
        tank_score = 0.0

    tank_level = _score_to_level(tank_score)

    return {
        "tank_stress_score": round(tank_score, 3),
        "tank_stress_level": tank_level,
        "per_fish_stress": per_fish_stress
    }


def _score_to_level(score: float) -> str:
    if score < 0.30:
        return "Healthy"
    elif score < 0.50:
        return "Mild"
    elif score < 0.70:
        return "Moderate"
    elif score < 0.85:
        return "High"
    else:
        return "Critical"


def _identify_primary_stressor(bottom: float, freeze: float, speed: float, erratic: float, temp: float) -> str:
    factors = {
        "Prolonged Bottom Dwelling": bottom,
        "Freezing Motion": freeze,
        "Abnormal Swimming Speed": speed,
        "Erratic Swimming Pattern": erratic,
        "Temperature Deviation": temp
    }
    return max(factors, key=factors.get)
