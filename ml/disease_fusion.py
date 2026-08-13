"""Disease Fusion module combining visual disease detection and NLP symptom probabilities.

Fuses ONNX Runtime visual disease classification with NLP symptom text probabilities to produce a unified
risk assessment dynamically aligned with models/disease/class_names.json.
"""

from typing import Dict, Any
from utils.logger import get_logger

LOG = get_logger(__name__)


def fuse(vision_data: Dict[str, Any] = None, nlp_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Fuse vision-based disease classification with NLP symptom processing results.
    
    Returns:
    {
        "disease": str,
        "disease_probability": float,
        "confidence": float,
        "reason": str,
        "breakdown": {
            "yolo_class": str,
            "yolo_confidence": float,
            "symptom_confidence": float
        }
    }
    """
    vision_data = vision_data or {}
    nlp_data = nlp_data or {}

    v_class = vision_data.get("disease_class", "Healthy Fish")
    v_conf = float(vision_data.get("confidence", 0.0))

    symptom_probs = nlp_data.get("probabilities", {})

    # Accumulate evidence scores across official class names
    fused_scores = {}
    display_names = {}

    # 1. Visual evidence (weight 0.50)
    if v_class and "healthy" not in v_class.lower():
        key = v_class.lower()
        fused_scores[key] = fused_scores.get(key, 0.0) + (0.50 * v_conf)
        display_names[key] = v_class

    # 2. NLP Symptom evidence (weight 0.50)
    for disease_cls, prob in symptom_probs.items():
        if "healthy" in disease_cls.lower():
            continue
        key = disease_cls.lower()
        fused_scores[key] = fused_scores.get(key, 0.0) + (0.50 * float(prob))
        display_names[key] = disease_cls

    if not fused_scores:
        return {
            "disease": "Healthy Fish",
            "disease_probability": 0.0,
            "confidence": 1.0,
            "reason": "No abnormal visual symptoms or reported text symptoms detected.",
            "breakdown": {"yolo_class": v_class, "yolo_confidence": v_conf, "symptom_confidence": 0.0},
        }

    top_key = max(fused_scores, key=fused_scores.get)
    top_disease_name = display_names.get(top_key, top_key.title())
    combined_score = round(float(fused_scores[top_key]), 3)

    nlp_score = 0.0
    for cls_name, prob in symptom_probs.items():
        if cls_name.lower() == top_key:
            nlp_score = float(prob)
            break

    reason = (
        f"Fused evidence for '{top_disease_name}': "
        f"Visual Model ({v_class} {int(v_conf * 100)}%), "
        f"NLP Symptoms ({int(nlp_score * 100)}%)"
    )

    return {
        "disease": top_disease_name,
        "disease_probability": combined_score,
        "confidence": combined_score,
        "reason": reason,
        "breakdown": {
            "yolo_class": v_class,
            "yolo_confidence": round(v_conf, 2),
            "symptom_confidence": round(nlp_score, 2),
        },
    }
