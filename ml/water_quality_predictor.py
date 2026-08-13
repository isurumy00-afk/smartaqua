"""Water Quality Predictor module using trained machine learning models.

Predicts:
- Water Quality Index (Good, Fair, Poor, Critical)
- Bad Water Probability
- Prediction Confidence
- Estimated hours until water change required
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import joblib
from config import WATER_QUALITY_MODEL_DIR
from utils.logger import get_logger

LOG = get_logger(__name__)

FEATURES = ["PH", "IONCONCENTRATION", "TEMP", "TURBIDITY"]


class WaterQualityPredictor:
    """Predicts water quality health from sensor reading inputs."""

    def __init__(self, model_dir: Path = WATER_QUALITY_MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.metadata = None

    def _load_artifacts(self) -> bool:
        """Load scaler and pre-trained Random Forest or regression model."""
        if self.model is not None:
            return True

        if not self.model_dir.exists():
            LOG.warning("Water quality model directory missing: %s", self.model_dir)
            return False

        try:
            metadata_path = self.model_dir / "model_metadata.json"
            if metadata_path.exists():
                self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            # Try loading Random Forest / best model pkl
            for model_filename in ("rfr_model.pkl", "best_water_quality_model.pkl", "lr_model.pkl"):
                path = self.model_dir / model_filename
                if path.exists():
                    self.model = joblib.load(path)
                    LOG.info("Loaded water quality model: %s", model_filename)
                    break

            return self.model is not None
        except Exception as exc:
            LOG.error("Error loading water quality ML artifacts: %s", exc)
            return False

    def predict(self, sensor_readings: Dict[str, Any]) -> Dict[str, Any]:
        """Perform water quality prediction based on sensor data.
        
        Expected input readings keys: ph, ionconcentration, temp/temperature, turbidity.
        """
        # Extract features with sensible defaults
        ph = self._extract_value(sensor_readings, ["ph"], default=7.2)
        ionconcentration = self._extract_value(sensor_readings, ["ionconcentration", "ammonia"], default=250.0)
        temp = self._extract_value(sensor_readings, ["temperature", "temp"], default=25.5)
        turbidity = self._extract_value(sensor_readings, ["turbidity"], default=200.0)

        import pandas as pd
        # Model artifact trained expecting columns ["PH", "AMMONIA", "TEMP", "TURBIDITY"]
        df_input = pd.DataFrame([[ph, ionconcentration, temp, turbidity]], columns=["PH", "AMMONIA", "TEMP", "TURBIDITY"])

        if self._load_artifacts() and self.scaler is not None and self.model is not None:
            try:
                scaled_vector = self.scaler.transform(df_input)
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(scaled_vector)[0]
                    bad_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                else:
                    pred = self.model.predict(scaled_vector)[0]
                    bad_prob = float(np.clip(pred, 0.0, 1.0))
            except Exception as exc:
                LOG.error("Model prediction execution error: %s", exc)
                bad_prob = self._rule_based_fallback(ph, ionconcentration, temp, turbidity)
        else:
            bad_prob = self._rule_based_fallback(ph, ionconcentration, temp, turbidity)

        good_prob = round(1.0 - bad_prob, 4)
        bad_prob = round(bad_prob, 4)
        confidence = round(float(max(good_prob, bad_prob)), 4)

        if bad_prob < 0.25:
            label = "Good"
        elif bad_prob < 0.50:
            label = "Fair"
        elif bad_prob < 0.75:
            label = "Poor"
        else:
            label = "Critical"

        estimated_hours = round(max(0.0, 72.0 * (1.0 - bad_prob)), 1)

        return {
            "water_quality": label,
            "bad_probability": bad_prob,
            "good_probability": good_prob,
            "confidence": confidence,
            "estimated_hours_until_water_change": estimated_hours,
            "inputs_used": {
                "PH": ph,
                "IONCONCENTRATION": ionconcentration,
                "TEMP": temp,
                "TURBIDITY": turbidity,
            },
        }

    def _rule_based_fallback(self, ph: float, ionconcentration: float, temp: float, turbidity: float) -> float:
        """Heuristic calculation if ML model artifacts are absent."""
        score = 0.0
        score += 0.3 * (abs(ph - 7.0) / 2.0)
        score += 0.4 * min(abs(ionconcentration - 250.0) / 1000.0, 1.0)
        score += 0.15 * (abs(temp - 25.0) / 10.0)
        score += 0.15 * min(turbidity / 1000.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))

    def _extract_value(self, data: Dict[str, Any], keys: list, default: float) -> float:
        for k in keys:
            if k in data:
                val = data[k]
                if isinstance(val, dict):
                    val = val.get("value")
                if val is None or isinstance(val, dict):
                    continue
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return default
