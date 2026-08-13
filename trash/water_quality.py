import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError:
    raise ImportError(
        "SHAP is not installed. Install it using: pip install shap"
    )


MODEL_DIR = "models/water_quality"
OUTPUT_DIR = os.path.join(MODEL_DIR, "xai_outputs")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
SHAP_BACKGROUND_PATH = os.path.join(MODEL_DIR, "shap_background.pkl")

FEATURES = ["PH", "AMMONIA", "TEMP", "TURBIDITY"]

DEFAULT_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = {}

if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    print("\nLoaded model metadata")
    print("Best model:", metadata.get("best_model"))
    print("Best model type:", metadata.get("best_model_type"))
    print("Features:", metadata.get("features"))
else:
    print("\nWarning: model_metadata.json not found.")


def resolve_best_model_path():
    """
    Supports:
    - best_water_quality_model.pkl
    - best_water_quality_model.keras
    - metadata best_model_path
    """

    metadata_model_path = metadata.get("best_model_path")

    if metadata_model_path and os.path.exists(metadata_model_path):
        return metadata_model_path

    pkl_path = os.path.join(MODEL_DIR, "best_water_quality_model.pkl")
    keras_path = os.path.join(MODEL_DIR, "best_water_quality_model.keras")

    if os.path.exists(pkl_path):
        return pkl_path

    if os.path.exists(keras_path):
        return keras_path

    raise FileNotFoundError(
        "Could not find best model file. Expected one of:\n"
        f"{pkl_path}\n"
        f"{keras_path}\n"
        "or a valid best_model_path inside model_metadata.json"
    )


MODEL_PATH = resolve_best_model_path()

print("\nUsing model:")
print(MODEL_PATH)


if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

if not os.path.exists(SHAP_BACKGROUND_PATH):
    raise FileNotFoundError(f"SHAP background not found: {SHAP_BACKGROUND_PATH}")

scaler = joblib.load(SCALER_PATH)
shap_background = joblib.load(SHAP_BACKGROUND_PATH)

if MODEL_PATH.endswith(".keras"):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required to load LSTM .keras model.\n"
            "Install it using: pip install tensorflow"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    model_runtime_type = "keras"

else:
    model = joblib.load(MODEL_PATH)

    if hasattr(model, "predict_proba"):
        model_runtime_type = "sklearn_classifier"
    else:
        model_runtime_type = "sklearn_regressor"


print("\nDetected runtime model type:", model_runtime_type)


best_model_name = metadata.get("best_model")
models_trained = metadata.get("models_trained", {})

threshold = DEFAULT_THRESHOLD

if best_model_name in models_trained:
    threshold = models_trained[best_model_name].get("threshold", DEFAULT_THRESHOLD)

print("Classification threshold:", threshold)


samples = pd.DataFrame([
    {
        "sample_name": "Sample 1 - Expected GOOD Water",
        "PH": 7.25,
        "AMMONIA": 0.030,
        "TEMP": 26.50,
        "TURBIDITY": 820
    },
    {
        "sample_name": "Sample 2 - Expected BAD Water",
        "PH": 5.40,
        "AMMONIA": 1.250,
        "TEMP": 34.80,
        "TURBIDITY": 280
    }
])


missing_cols = []

for col in FEATURES:
    if col not in samples.columns:
        missing_cols.append(col)

if missing_cols:
    raise ValueError(f"Missing input columns: {missing_cols}")

samples_features = samples[FEATURES].copy()

for col in FEATURES:
    samples_features[col] = pd.to_numeric(samples_features[col], errors="coerce")

if samples_features.isnull().any().any():
    raise ValueError("Sample data contains invalid numeric values.")


def prepare_raw_dataframe(input_data):
    """
    Converts SHAP/model input into a clean DataFrame with correct feature order.
    """

    if isinstance(input_data, pd.DataFrame):
        raw_df = input_data.copy()
    else:
        raw_df = pd.DataFrame(input_data, columns=FEATURES)

    raw_df = raw_df[FEATURES].copy()

    for col in FEATURES:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    if raw_df.isnull().any().any():
        raise ValueError("Input contains invalid numeric values.")

    return raw_df


def predict_bad_probability(input_data):
    """
    Returns BAD water probability for:
    - sklearn classifier: uses predict_proba()
    - sklearn regressor/RFR: uses clipped regression output
    - keras/LSTM: uses sigmoid output
    """

    raw_df = prepare_raw_dataframe(input_data)
    X_scaled = scaler.transform(raw_df[FEATURES])

    # ------------------------------
    # LSTM / Keras model
    # ------------------------------
    if model_runtime_type == "keras":
        X_lstm = X_scaled.reshape(
            X_scaled.shape[0],
            X_scaled.shape[1],
            1
        )

        bad_prob = model.predict(X_lstm, verbose=0).ravel()
        bad_prob = np.clip(bad_prob, 0.0, 1.0)

        return bad_prob

    # ------------------------------
    # Sklearn classifier: LR, RFC, etc.
    # ------------------------------
    if model_runtime_type == "sklearn_classifier":
        probabilities = model.predict_proba(X_scaled)

        classes = list(model.classes_)

        if 1 in classes:
            bad_class_index = classes.index(1)
        else:
            bad_class_index = 1

        bad_prob = probabilities[:, bad_class_index]
        return bad_prob

    # ------------------------------
    # Sklearn regressor: RFR
    # ------------------------------
    raw_pred = model.predict(X_scaled)

    # Since label is 0/1, RFR output is treated like BAD probability.
    bad_prob = np.clip(raw_pred, 0.0, 1.0)

    return bad_prob


def normalize_shap_values(shap_values_raw):
    """
    Normalizes SHAP output into shape:
    rows x features
    """

    if isinstance(shap_values_raw, list):
        shap_values_raw = shap_values_raw[0]

    shap_values_array = np.array(shap_values_raw)

    if shap_values_array.ndim == 3:
        if shap_values_array.shape[2] == 1:
            shap_values_array = shap_values_array[:, :, 0]
        elif shap_values_array.shape[2] == 2:
            shap_values_array = shap_values_array[:, :, 1]

    return shap_values_array


def safe_filename(text):
    """
    Makes chart filenames safe.
    """

    text = str(text)
    text = text.lower()
    text = text.replace(" ", "_")
    text = text.replace("-", "_")

    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"

    return "".join(ch for ch in text if ch in allowed)


bad_probability = predict_bad_probability(samples_features)
good_probability = 1.0 - bad_probability

predictions = (bad_probability >= threshold).astype(int)

confidence = np.maximum(good_probability, bad_probability)


print("\nCalculating SHAP explanations...")
print("This may take a few seconds.")

background = shap_background[FEATURES].copy()

for col in FEATURES:
    background[col] = pd.to_numeric(background[col], errors="coerce")

background = background.dropna()

explainer = shap.KernelExplainer(
    predict_bad_probability,
    background,
    link="identity"
)

shap_values_raw = explainer.shap_values(
    samples_features,
    nsamples=100
)

shap_values = normalize_shap_values(shap_values_raw)

expected_bad_probability = explainer.expected_value

if isinstance(expected_bad_probability, list):
    expected_bad_probability = expected_bad_probability[0]

if isinstance(expected_bad_probability, np.ndarray):
    expected_bad_probability = float(expected_bad_probability.flatten()[0])
else:
    expected_bad_probability = float(expected_bad_probability)


results = samples.copy()

results["predicted_label"] = predictions

results["water_quality"] = results["predicted_label"].map({
    0: "GOOD WATER QUALITY",
    1: "BAD WATER QUALITY"
})

results["good_probability"] = good_probability
results["bad_probability"] = bad_probability
results["confidence"] = confidence

top_features = []
top_shap_values = []
top_effects = []
top_contribution_percent = []

for i in range(len(samples)):
    sample_shap_values = shap_values[i]

    top_index = int(np.argmax(np.abs(sample_shap_values)))
    top_feature = FEATURES[top_index]
    top_value = float(sample_shap_values[top_index])

    if top_value > 0:
        effect = "Increased BAD probability"
    elif top_value < 0:
        effect = "Reduced BAD probability"
    else:
        effect = "No effect"

    top_features.append(top_feature)
    top_shap_values.append(top_value)
    top_contribution_percent.append(abs(top_value) * 100)
    top_effects.append(effect)

results["most_contributed_feature"] = top_features
results["top_shap_value"] = top_shap_values
results["top_contribution_probability_points"] = top_contribution_percent
results["xai_effect"] = top_effects

for feature_index, feature in enumerate(FEATURES):
    results[f"SHAP_{feature}"] = shap_values[:, feature_index]


print("\nPrediction Results with SHAP XAI")
print("================================")

display_cols = [
    "sample_name",
    "PH",
    "AMMONIA",
    "TEMP",
    "TURBIDITY",
    "predicted_label",
    "water_quality",
    "good_probability",
    "bad_probability",
    "confidence",
    "most_contributed_feature",
    "top_shap_value",
    "top_contribution_probability_points",
    "xai_effect"
]

print(results[display_cols].to_string(index=False))



print("\nReadable XAI Explanation")
print("========================")

for i, row in results.iterrows():
    print(f"\n{row['sample_name']}")
    print("-" * 50)

    print("PH        :", row["PH"])
    print("AMMONIA   :", row["AMMONIA"], "mg/l")
    print("TEMP      :", row["TEMP"], "C")
    print("TURBIDITY :", row["TURBIDITY"])

    print("\nPrediction:", row["water_quality"])
    print("Label     :", int(row["predicted_label"]))
    print("Confidence:", round(row["confidence"] * 100, 2), "%")
    print("Good probability:", round(row["good_probability"] * 100, 2), "%")
    print("Bad probability :", round(row["bad_probability"] * 100, 2), "%")

    print("\nMost contributed feature:", row["most_contributed_feature"])
    print("SHAP value:", round(row["top_shap_value"], 6))
    print(
        "Contribution strength:",
        round(row["top_contribution_probability_points"], 3),
        "probability points"
    )
    print("Effect:", row["xai_effect"])

    print("\nAll SHAP feature contributions toward BAD class:")

    for feature in FEATURES:
        shap_value = row[f"SHAP_{feature}"]

        if shap_value > 0:
            direction = "toward BAD"
        elif shap_value < 0:
            direction = "toward GOOD"
        else:
            direction = "neutral"

        print(f"  {feature:<10}: {shap_value:+.6f}  {direction}")


mean_abs_shap = np.abs(shap_values).mean(axis=0)

global_importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": mean_abs_shap,
    "mean_abs_probability_points": mean_abs_shap * 100
}).sort_values(
    by="mean_abs_shap",
    ascending=False
)

print("\nOverall SHAP Importance for These 2 Samples")
print("===========================================")
print(global_importance.to_string(index=False))

overall_top_feature = global_importance.iloc[0]["feature"]

print("\nOverall most contributed feature:", overall_top_feature)


csv_path = os.path.join(OUTPUT_DIR, "xai_shap_results_2_samples.csv")
results.to_csv(csv_path, index=False)

importance_path = os.path.join(OUTPUT_DIR, "xai_global_importance_for_2_samples.csv")
global_importance.to_csv(importance_path, index=False)


for i in range(len(samples)):
    plt.figure(figsize=(8, 4))
    plt.bar(FEATURES, shap_values[i])
    plt.axhline(0)
    plt.ylabel("SHAP value toward BAD probability")
    plt.title(f"{results.loc[i, 'sample_name']} - SHAP Contributions")
    plt.tight_layout()

    filename = safe_filename(results.loc[i, "sample_name"])
    chart_path = os.path.join(OUTPUT_DIR, f"{filename}_shap_bar.png")

    plt.savefig(chart_path)
    plt.close()


plt.figure(figsize=(8, 4))
plt.bar(
    global_importance["feature"],
    global_importance["mean_abs_probability_points"]
)
plt.ylabel("Mean absolute SHAP value")
plt.title("Overall SHAP Importance for 2 Samples")
plt.tight_layout()

global_chart_path = os.path.join(
    OUTPUT_DIR,
    "overall_shap_importance_2_samples.png"
)

plt.savefig(global_chart_path)
plt.close()



print("\nSaved XAI files:")
print(csv_path)
print(importance_path)

for i in range(len(samples)):
    filename = safe_filename(results.loc[i, "sample_name"])
    print(os.path.join(OUTPUT_DIR, f"{filename}_shap_bar.png"))

print(global_chart_path)
