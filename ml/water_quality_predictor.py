"""Water Quality Predictor and SHAP Explainable AI Module.

Predicts water quality using pre-trained ML models (LSTM, Random Forest Regressor, Logistic Regression)
with features: ["PH", "IONCONCENTRATION", "TEMP", "TURBIDITY"].

Supports:
- Ion concentration from RS485 Modbus RTU reader
- pH, Temperature, and Turbidity from Arduino Uno serial reader
- Explainable AI (SHAP) feature contribution analysis and visualizations
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Union, Optional

import numpy as np
import pandas as pd
import joblib

from config import WATER_QUALITY_MODEL_DIR
from utils.logger import get_logger

LOG = get_logger(__name__)

warnings.filterwarnings("ignore")

FEATURES = ["PH", "IONCONCENTRATION", "TEMP", "TURBIDITY"]
DEFAULT_THRESHOLD = 0.5


def safe_filename(text: str) -> str:
    """Makes filenames safe for chart exports."""
    text = str(text).lower().replace(" ", "_").replace("-", "_")
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
    return "".join(ch for ch in text if ch in allowed)


def _clean_filename(path_or_str: Union[str, Path]) -> str:
    """Normalizes Windows backslashes and POSIX forward slashes to extract bare filename."""
    return str(path_or_str).replace("\\", "/").rstrip("/").split("/")[-1]


def normalize_shap_values(shap_values_raw: Any) -> np.ndarray:
    """Normalizes SHAP output into shape: (rows, features)."""
    if isinstance(shap_values_raw, list):
        shap_values_raw = shap_values_raw[0]

    shap_values_array = np.array(shap_values_raw)

    if shap_values_array.ndim == 3:
        if shap_values_array.shape[2] == 1:
            shap_values_array = shap_values_array[:, :, 0]
        elif shap_values_array.shape[2] == 2:
            shap_values_array = shap_values_array[:, :, 1]

    return shap_values_array


class WaterQualityPredictor:
    """Water Quality ML Predictor & SHAP Explainer."""

    def __init__(self, model_dir: Union[str, Path] = WATER_QUALITY_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.output_dir = self.model_dir / "xai_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = None
        self.shap_background = None
        self.metadata = {}
        self.model_runtime_type: str = "unknown"
        self.model_path: Optional[Path] = None
        self.threshold: float = DEFAULT_THRESHOLD
        self._tree_explainer = None

        self._load_artifacts()

    def _get_candidate_model_paths(self) -> list:
        """Returns ordered list of candidate model paths to attempt loading."""
        candidates = []

        # 1. Preferred model from metadata if available
        meta_best = self.metadata.get("best_model_path")
        if meta_best:
            filename = _clean_filename(meta_best)
            meta_path = self.model_dir / filename
            if meta_path.exists() and meta_path not in candidates:
                candidates.append(meta_path)

        # 2. Standard model artifacts in priority order (Keras LSTM -> RFR -> LR)
        default_names = [
            "best_water_quality_model.keras",
            "lstm_model.keras",
            "rfr_model.pkl",
            "best_water_quality_model.pkl",
            "lr_model.pkl",
        ]
        for name in default_names:
            p = self.model_dir / name
            if p.exists() and p not in candidates:
                candidates.append(p)

        return candidates

    def _load_single_model(self, model_path: Path) -> bool:
        """Attempts to load a single model file (.keras or .pkl)."""
        model_path_str = str(model_path)
        if model_path_str.endswith(".keras"):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path_str)
                self.model_runtime_type = "keras"
                self.model_path = model_path
                return True
            except Exception as exc:
                LOG.debug("Could not load Keras model from %s: %s", model_path.name, exc)
                return False
        else:
            try:
                # Provide module alias for legacy gradient boosting pickles referencing _loss
                try:
                    import sklearn._loss
                    import sys
                    sys.modules.setdefault("_loss", sklearn._loss)
                except Exception:
                    pass

                loaded = joblib.load(model_path_str)
                self.model = loaded
                self.model_path = model_path
                if hasattr(self.model, "predict_proba"):
                    self.model_runtime_type = "sklearn_classifier"
                else:
                    self.model_runtime_type = "sklearn_regressor"
                return True
            except Exception as exc:
                LOG.debug("Could not load scikit-learn model from %s: %s", model_path.name, exc)
                return False

    def _load_artifacts(self) -> bool:
        """Load scaler, model, metadata, and SHAP background data."""
        if self.model is not None and self.scaler is not None:
            return True

        if not self.model_dir.exists():
            LOG.warning("Water quality model directory missing: %s", self.model_dir)
            return False

        try:
            metadata_path = self.model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                try:
                    self.scaler = joblib.load(scaler_path)
                except Exception as se:
                    LOG.warning("Failed to load scaler from %s: %s", scaler_path, se)
            else:
                LOG.warning("Scaler not found at %s", scaler_path)

            shap_bg_path = self.model_dir / "shap_background.pkl"
            if shap_bg_path.exists():
                try:
                    self.shap_background = joblib.load(shap_bg_path)
                except Exception as be:
                    LOG.warning("Failed to load SHAP background from %s: %s", shap_bg_path, be)
            else:
                LOG.warning("SHAP background not found at %s", shap_bg_path)

            # Attempt loading models in priority order
            candidates = self._get_candidate_model_paths()
            model_loaded = False
            for candidate in candidates:
                if self._load_single_model(candidate):
                    model_loaded = True
                    break

            if not model_loaded:
                LOG.error("Failed to load any valid water quality model artifact from %s", self.model_dir)
                return False

            best_model_name = self.metadata.get("best_model")
            models_trained = self.metadata.get("models_trained", {})
            if best_model_name in models_trained:
                self.threshold = float(models_trained[best_model_name].get("threshold", DEFAULT_THRESHOLD))
            else:
                self.threshold = DEFAULT_THRESHOLD

            LOG.info(
                "Loaded Water Quality Model (%s, runtime=%s, threshold=%.2f)",
                self.model_path.name if self.model_path else "unknown",
                self.model_runtime_type,
                self.threshold
            )
            return True
        except Exception as exc:
            LOG.error("Error loading water quality model artifacts: %s", exc)
            return False

    def prepare_raw_dataframe(self, input_data: Union[pd.DataFrame, Dict[str, Any], list]) -> pd.DataFrame:
        """Converts SHAP/model input into a clean DataFrame with correct feature order."""
        if isinstance(input_data, pd.DataFrame):
            raw_df = input_data.copy()
        elif isinstance(input_data, dict):
            raw_df = pd.DataFrame([input_data])
        else:
            raw_df = pd.DataFrame(input_data, columns=FEATURES)

        for col in FEATURES:
            if col not in raw_df.columns:
                raise ValueError(f"Missing input column: {col}")

        raw_df = raw_df[FEATURES].copy()

        for col in FEATURES:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

        if raw_df.isnull().any().any():
            raise ValueError("Input contains invalid numeric values.")

        return raw_df

    def predict_bad_probability(self, input_data: Union[pd.DataFrame, Dict[str, Any], list]) -> np.ndarray:
        """Returns BAD water probability array for Keras, sklearn classifier, or sklearn regressor."""
        if self.model is None or self.scaler is None:
            if not self._load_artifacts():
                raise RuntimeError("Water Quality Predictor model or scaler not loaded.")

        raw_df = self.prepare_raw_dataframe(input_data)
        X_scaled = self.scaler.transform(raw_df[FEATURES])

        # Keras / LSTM model
        if self.model_runtime_type == "keras":
            X_lstm = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            bad_prob = self.model.predict(X_lstm, verbose=0).ravel()
            return np.clip(bad_prob, 0.0, 1.0)

        # Sklearn classifier (Logistic Regression, etc.)
        if self.model_runtime_type == "sklearn_classifier":
            probabilities = self.model.predict_proba(X_scaled)
            classes = list(self.model.classes_)
            bad_class_index = classes.index(1) if 1 in classes else 1
            return probabilities[:, bad_class_index]

        # Sklearn regressor (Random Forest Regressor, etc.)
        raw_pred = self.model.predict(X_scaled)
        return np.clip(raw_pred, 0.0, 1.0)

    def extract_feature_values(self, sensor_readings: Dict[str, Any]) -> Dict[str, float]:
        """Extracts standard feature values from sensor reading payload.
        
        Reads:
        - PH: from Arduino (key 'ph' / 'PH')
        - IONCONCENTRATION: from Modbus RTU (key 'ionconcentration' / 'IONCONCENTRATION')
        - TEMP: from Arduino (key 'temperature' / 'temp' / 'TEMP')
        - TURBIDITY: from Arduino (key 'turbidity' / 'TURBIDITY')
        """
        ph = self._extract_value(sensor_readings, ["ph", "PH"], default=7.25)
        ionconc = self._extract_value(sensor_readings, ["ionconcentration", "IONCONCENTRATION"], default=345.0)
        temp = self._extract_value(sensor_readings, ["temperature", "temp", "TEMP"], default=26.5)
        turbidity = self._extract_value(sensor_readings, ["turbidity", "TURBIDITY"], default=820.0)

        return {
            "PH": ph,
            "IONCONCENTRATION": ionconc,
            "TEMP": temp,
            "TURBIDITY": turbidity,
        }

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

    def predict(
        self,
        sensor_readings: Dict[str, Any],
        run_shap: bool = True,
        save_xai: bool = False
    ) -> Dict[str, Any]:
        """Perform water quality prediction & SHAP explanation on sensor data."""
        features_dict = self.extract_feature_values(sensor_readings)
        df_input = pd.DataFrame([features_dict])[FEATURES]

        try:
            bad_prob_arr = self.predict_bad_probability(df_input)
            bad_prob = float(bad_prob_arr[0])
        except Exception as exc:
            LOG.error("Model prediction failed: %s. Using rule-based fallback.", exc)
            bad_prob = self._rule_based_fallback(features_dict)

        good_prob = float(np.clip(1.0 - bad_prob, 0.0, 1.0))
        confidence = float(max(good_prob, bad_prob))
        predicted_label = 1 if bad_prob >= self.threshold else 0
        water_quality_str = "BAD WATER QUALITY" if predicted_label == 1 else "GOOD WATER QUALITY"

        # Human-friendly level label
        if bad_prob < 0.25:
            quality_level = "Good"
        elif bad_prob < 0.50:
            quality_level = "Fair"
        elif bad_prob < 0.75:
            quality_level = "Poor"
        else:
            quality_level = "Critical"

        estimated_hours = round(max(0.0, 72.0 * (1.0 - bad_prob)), 1)

        result: Dict[str, Any] = {
            "water_quality": quality_level,
            "water_quality_status": water_quality_str,
            "predicted_label": predicted_label,
            "bad_probability": round(bad_prob, 4),
            "good_probability": round(good_prob, 4),
            "confidence": round(confidence, 4),
            "threshold": self.threshold,
            "estimated_hours_until_water_change": estimated_hours,
            "inputs_used": features_dict,
        }

        # Run SHAP Explanation if requested
        if run_shap:
            shap_info = self.explain_shap(df_input, save_xai=save_xai)
            result.update(shap_info)

        return result

    def explain_shap(
        self,
        samples_df: pd.DataFrame,
        sample_names: Optional[list] = None,
        save_xai: bool = False
    ) -> Dict[str, Any]:
        """Calculates SHAP explanations for input samples using TreeExplainer, KernelExplainer, or Heuristic fallback."""
        try:
            import shap
            shap_lib_available = True
        except ImportError:
            shap_lib_available = False

        if not shap_lib_available or self.shap_background is None or self.shap_background.empty:
            from ml.shap_explainer import explain as explain_heuristic
            sample_dict = samples_df[FEATURES].iloc[0].to_dict() if len(samples_df) > 0 else {}
            return explain_heuristic({"inputs_used": sample_dict})

        try:
            background = self.shap_background[FEATURES].copy()
            for col in FEATURES:
                background[col] = pd.to_numeric(background[col], errors="coerce")
            background = background.dropna()

            # For Tree models (Random Forest), TreeExplainer is orders of magnitude faster
            if self.model_runtime_type == "sklearn_regressor" and hasattr(self.model, "estimators_"):
                if self._tree_explainer is None:
                    self._tree_explainer = shap.TreeExplainer(self.model)
                X_scaled = self.scaler.transform(samples_df[FEATURES]) if self.scaler is not None else samples_df[FEATURES]
                shap_values_raw = self._tree_explainer.shap_values(X_scaled)
                shap_values = normalize_shap_values(shap_values_raw)
            else:
                # KernelExplainer for Keras / LogisticRegression
                bg_sample = background.head(30)
                explainer = shap.KernelExplainer(
                    self.predict_bad_probability,
                    bg_sample,
                    link="identity"
                )
                shap_values_raw = explainer.shap_values(samples_df[FEATURES], nsamples=50)
                shap_values = normalize_shap_values(shap_values_raw)

            # Process first sample for dict output
            first_shap = shap_values[0]
            top_index = int(np.argmax(np.abs(first_shap)))
            top_feature = FEATURES[top_index]
            top_value = float(first_shap[top_index])
            top_contrib_pct = float(abs(top_value) * 100)

            if top_value > 0:
                effect = "Increased BAD probability"
            elif top_value < 0:
                effect = "Reduced BAD probability"
            else:
                effect = "No effect"

            shap_per_feature = {
                feature: float(first_shap[idx]) for idx, feature in enumerate(FEATURES)
            }

            xai_summary = {
                "most_contributed_feature": top_feature,
                "top_shap_value": round(top_value, 6),
                "top_contribution_probability_points": round(top_contrib_pct, 3),
                "xai_effect": effect,
                "shap_values": shap_per_feature,
                "shap_status": "active",
            }

            if save_xai and len(samples_df) > 0:
                self._save_xai_outputs(samples_df, shap_values, sample_names)

            return xai_summary
        except Exception as exc:
            LOG.debug("SHAP computation failed: %s, falling back to heuristic", exc)
            from ml.shap_explainer import explain as explain_heuristic
            sample_dict = samples_df[FEATURES].iloc[0].to_dict() if len(samples_df) > 0 else {}
            return explain_heuristic({"inputs_used": sample_dict})

    def _save_xai_outputs(
        self,
        samples_df: pd.DataFrame,
        shap_values: np.ndarray,
        sample_names: Optional[list] = None
    ) -> None:
        """Saves XAI bar charts and CSV summary files to xai_outputs directory."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            names = sample_names or [f"Sample_{i+1}" for i in range(len(samples_df))]

            for i, name in enumerate(names):
                plt.figure(figsize=(8, 4))
                plt.bar(FEATURES, shap_values[i])
                plt.axhline(0, color="black", linewidth=0.8)
                plt.ylabel("SHAP value toward BAD probability")
                plt.title(f"{name} - SHAP Contributions")
                plt.tight_layout()

                chart_filename = safe_filename(name) + "_shap_bar.png"
                chart_path = self.output_dir / chart_filename
                plt.savefig(chart_path)
                plt.close()

            # Global importance
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            global_importance = pd.DataFrame({
                "feature": FEATURES,
                "mean_abs_shap": mean_abs_shap,
                "mean_abs_probability_points": mean_abs_shap * 100
            }).sort_values(by="mean_abs_shap", ascending=False)

            plt.figure(figsize=(8, 4))
            plt.bar(global_importance["feature"], global_importance["mean_abs_probability_points"])
            plt.ylabel("Mean absolute SHAP value")
            plt.title("Overall SHAP Importance")
            plt.tight_layout()
            plt.savefig(self.output_dir / "overall_shap_importance.png")
            plt.close()

            global_importance.to_csv(self.output_dir / "xai_global_importance.csv", index=False)
            LOG.info("Saved XAI charts and CSV outputs to %s", self.output_dir)
        except Exception as exc:
            LOG.warning("Failed to save XAI visualization outputs: %s", exc)

    def _rule_based_fallback(self, features: Dict[str, float]) -> float:
        """Fallback score calculator if ML model fails."""
        ph = features.get("PH", 7.25)
        ionconc = features.get("IONCONCENTRATION", 345.0)
        temp = features.get("TEMP", 26.5)
        turbidity = features.get("TURBIDITY", 820.0)

        score = 0.0
        score += 0.30 * (abs(ph - 7.25) / 2.0)
        score += 0.40 * min(max(0.0, ionconc - 350.0) / 200.0, 1.0)
        score += 0.15 * (abs(temp - 26.5) / 10.0)
        score += 0.15 * min(turbidity / 1000.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))


if __name__ == "__main__":
    print("Testing Water Quality Predictor & SHAP XAI Module...")

    predictor = WaterQualityPredictor()

    samples = pd.DataFrame([
        {
            "sample_name": "Sample 1 - Expected GOOD Water",
            "PH": 7.25,
            "IONCONCENTRATION": 345.0,
            "TEMP": 26.50,
            "TURBIDITY": 820
        },
        {
            "sample_name": "Sample 2 - Expected BAD Water",
            "PH": 5.40,
            "IONCONCENTRATION": 560.0,
            "TEMP": 34.80,
            "TURBIDITY": 280
        }
    ])

    print("\nRunning predictions...")
    bad_probs = predictor.predict_bad_probability(samples)
    
    for i, row in samples.iterrows():
        prob = bad_probs[i]
        label = "BAD WATER QUALITY" if prob >= predictor.threshold else "GOOD WATER QUALITY"
        print(f"\n{row['sample_name']}:")
        print(f"  PH: {row['PH']}, ION: {row['IONCONCENTRATION']}, TEMP: {row['TEMP']}, TURB: {row['TURBIDITY']}")
        print(f"  BAD Probability: {prob:.4f} -> {label}")

    print("\nRunning SHAP Explanation on 2 samples...")
    shap_info = predictor.explain_shap(
        samples[FEATURES],
        sample_names=samples["sample_name"].tolist(),
        save_xai=True
    )
    print("SHAP Summary Output:", json.dumps(shap_info, indent=2))
