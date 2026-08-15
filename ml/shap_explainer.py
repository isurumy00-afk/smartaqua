"""Explainable AI (SHAP) module.

Generates feature importance, contribution percentages, and human-readable natural-language
explanations for water quality predictions and system telemetry.
"""

from typing import Dict, Any
import numpy as np
from utils.logger import get_logger

LOG = get_logger(__name__)

# Baseline standard values for ideal aquarium water
REFERENCE_VALUES = {
    "PH": 7.25,
    "IONCONCENTRATION": 345.0,
    "TEMP": 26.5,
    "TURBIDITY": 820.0,
}


def explain(water_quality_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate feature contributions and natural-language XAI explanations.
    
    Accepts result dictionary from WaterQualityPredictor (or raw sensor data).
    
    Returns:
    {
        "feature_importance": {"TURBIDITY": 41.0, "PH": 24.0, ...},
        "contribution_percentages": {...},
        "natural_language_explanation": str,
        "primary_factor": str,
        "primary_contribution_percentage": float,
        "shap_backend": str
    }
    """
    inputs = water_quality_data.get("inputs_used", {})
    if not inputs:
        inputs = REFERENCE_VALUES

    shap_values = water_quality_data.get("shap_values")
    
    # If exact SHAP values are present from KernelExplainer, use them
    if isinstance(shap_values, dict) and shap_values:
        total_abs = sum(abs(v) for v in shap_values.values()) or 1e-6
        contributions = {
            k: round(float((abs(v) / total_abs) * 100.0), 1)
            for k, v in shap_values.items()
        }

        sorted_factors = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
        primary_factor, primary_pct = sorted_factors[0]

        explanation_parts = []
        for factor, pct in sorted_factors:
            val = inputs.get(factor, REFERENCE_VALUES.get(factor, 0.0))
            ref = REFERENCE_VALUES.get(factor, 0.0)
            qualifier = "High" if val > ref else ("Low" if val < ref else "Normal")
            shap_val = shap_values.get(factor, 0.0)
            direction = "BAD" if shap_val > 0 else ("GOOD" if shap_val < 0 else "neutral")
            explanation_parts.append(f"{qualifier} {factor.lower()} contributed {pct}% toward {direction} water quality.")

        nl_explanation = " ".join(explanation_parts)
        backend = "KernelExplainer (SHAP)"
    else:
        # Fallback heuristic calculation
        deviations = {}
        for feature, ref in REFERENCE_VALUES.items():
            val = float(inputs.get(feature, ref))
            dev = abs(val - ref) / ref if ref != 0 else abs(val)
            deviations[feature] = dev

        total_dev = sum(deviations.values()) or 1e-6
        contributions = {
            feature: round(float((dev / total_dev) * 100.0), 1)
            for feature, dev in deviations.items()
        }

        sorted_factors = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
        primary_factor, primary_pct = sorted_factors[0]

        explanation_parts = []
        for factor, pct in sorted_factors:
            val = inputs.get(factor, REFERENCE_VALUES.get(factor))
            ref = REFERENCE_VALUES.get(factor, 0.0)
            qualifier = "High" if val > ref else ("Low" if val < ref else "Normal")
            explanation_parts.append(f"{qualifier} {factor.lower()} contributed {pct}%.")

        nl_explanation = " ".join(explanation_parts)
        backend = "Heuristic SHAP approximator"

    return {
        "feature_importance": contributions,
        "contribution_percentages": contributions,
        "natural_language_explanation": nl_explanation,
        "primary_factor": primary_factor,
        "primary_contribution_percentage": primary_pct,
        "xai_effect": water_quality_data.get("xai_effect", "Calculated"),
        "shap_backend": backend,
    }
