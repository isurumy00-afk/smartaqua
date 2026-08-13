"""NLP Symptom Processing module using target classes from models/disease/class_names.json.

Parses natural language text symptom descriptions (e.g. 'red spots', 'swollen gills', 'white tail')
and maps them directly to official disease classes in models/disease/class_names.json.
Optimized for Raspberry Pi 4B execution.
"""

import json
from typing import Dict, Any, List
from config import DISEASE_CLASSES_PATH
from utils.logger import get_logger

LOG = get_logger(__name__)

# Core symptom mappings mapped directly to classes in class_names.json
SYMPTOM_KNOWLEDGE_BASE = {
    "Bacterial Red disease": [
        "red spot", "red spots", "red patch", "ulcer", "bleeding", "red sores", "skin reddening", "red streak", "redness"
    ],
    "Bacterial diseases - Aeromoniasis": [
        "aeromonas", "pop eye", "popeye", "swollen belly", "dropsy", "exophthalmia", "hemorrhage", "fin erosion", "bloated"
    ],
    "Bacterial gill disease": [
        "gill", "gills", "swollen gills", "pale gills", "flared operculum", "gasping", "rapid breathing", "heavy respiration", "mucus on gills"
    ],
    "Fungal diseases Saprolegniasis": [
        "fungus", "fungal", "cotton", "cotton-like", "white tufts", "fuzzy patches", "saprolegnia", "mold", "fuzzy"
    ],
    "Parasitic diseases": [
        "ich", "white spot", "white spots", "flashing", "scratching", "rubbing", "velvet", "gold dust", "clamped fins", "parasite", "dots"
    ],
    "Viral diseases White tail disease": [
        "white tail", "tail whitening", "opaque tail", "muscle opacity", "viral", "tail rot", "white tail disease"
    ],
    "Healthy Fish": [
        "active", "normal", "healthy", "good appetite", "clear eyes", "vibrant", "smooth fins"
    ],
}


def load_disease_classes() -> List[str]:
    """Dynamically load disease class names from models/disease/class_names.json."""
    if DISEASE_CLASSES_PATH.exists():
        try:
            classes = json.loads(DISEASE_CLASSES_PATH.read_text(encoding="utf-8"))
            return classes
        except Exception as exc:
            LOG.warning("Failed to load class_names.json: %s", exc)
    return list(SYMPTOM_KNOWLEDGE_BASE.keys())


def _try_ml_model_prediction(text: str, official_classes: List[str]) -> Dict[str, float]:
    """Attempt ML model inference using models/NLP/fish_disease_nlp_model.pkl if available."""
    ml_probs = {cls: 0.0 for cls in official_classes}
    try:
        from pathlib import Path
        import joblib
        from config import MODELS_DIR

        model_path = MODELS_DIR / "NLP" / "fish_disease_nlp_model.pkl"
        if not model_path.exists():
            return ml_probs

        bundle = joblib.load(str(model_path))
        model = bundle.get("model")
        disease_labels = bundle.get("disease_labels", {})

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            classes_in_model = model.classes_
            for idx, prob in zip(classes_in_model, probs):
                disease_name = disease_labels.get(int(idx))
                if disease_name in ml_probs:
                    ml_probs[disease_name] = round(float(prob), 4)
    except Exception as exc:
        LOG.debug("ML model prediction skipped: %s", exc)
    return ml_probs


def process(text: str) -> Dict[str, Any]:
    """Process symptom text description and calculate disease probabilities matching class_names.json.
    
    Returns:
    {
        "input_text": str,
        "probabilities": {"Parasitic diseases": 0.85, "Bacterial Red disease": 0.0, ...},
        "detected_symptoms": list,
        "top_disease": str
    }
    """
    if not text:
        return {"input_text": "", "probabilities": {}, "detected_symptoms": [], "top_disease": "Healthy Fish"}

    lowered = text.lower().strip()
    official_classes = load_disease_classes()
    probabilities = {cls: 0.0 for cls in official_classes}
    detected_symptoms = []

    # 1. Rule-based / Knowledge-base Keyword Extraction
    for disease_cls in official_classes:
        keywords = SYMPTOM_KNOWLEDGE_BASE.get(disease_cls, [])
        if not keywords:
            keywords = [w.lower() for w in disease_cls.split() if len(w) > 3]

        matched = [kw for kw in keywords if kw in lowered]

        if matched:
            detected_symptoms.extend(matched)
            score = round(min(0.60 + (0.20 * len(matched)), 0.95), 2)
            probabilities[disease_cls] = score

    # Token fallback if no explicit symptoms matched
    total_matched = sum(probabilities.values())
    if total_matched == 0:
        for disease_cls in official_classes:
            tokens = [t.lower() for t in disease_cls.split()]
            matched_tokens = [t for t in tokens if t in lowered]
            if matched_tokens:
                probabilities[disease_cls] = 0.50
                detected_symptoms.extend(matched_tokens)

    # 2. Try ML Model Prediction & Blend Results
    ml_probs = _try_ml_model_prediction(text, official_classes)
    if any(p > 0 for p in ml_probs.values()):
        # Blended score: 60% ML Model + 40% Keyword Matcher (or pure ML if keywords empty)
        for cls_name in official_classes:
            kw_p = probabilities[cls_name]
            ml_p = ml_probs[cls_name]
            if kw_p > 0:
                blended = round(0.60 * ml_p + 0.40 * kw_p, 3)
            else:
                blended = round(ml_p, 3)
            probabilities[cls_name] = blended

    # Determine top predicted disease
    if max(probabilities.values(), default=0.0) > 0:
        top_disease = max(probabilities, key=probabilities.get)
    else:
        top_disease = "Healthy Fish"
        probabilities["Healthy Fish"] = 1.0

    return {
        "input_text": text,
        "probabilities": probabilities,
        "detected_symptoms": list(set(detected_symptoms)),
        "top_disease": top_disease,
        "official_classes": official_classes,
    }

