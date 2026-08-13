import os
import joblib

# ============================================================
# Fish Disease NLP Predictor
# Input : fish symptoms text
# Output: predicted disease + confidence
# ============================================================

MODEL_PATH = "fish_disease_nlp_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} not found. Run train_model.py first."
    )

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
disease_labels = bundle["disease_labels"]

def predict_fish_disease(symptom_text):
    if not symptom_text or not symptom_text.strip():
        return {
            "label": None,
            "disease": "Invalid input",
            "confidence": 0.0
        }

    symptom_text = symptom_text.strip()

    predicted_label = int(model.predict([symptom_text])[0])

    confidence = None

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba([symptom_text])[0]
        confidence = float(max(probabilities)) * 100

    disease_name = disease_labels[predicted_label]

    return {
        "label": predicted_label,
        "disease": disease_name,
        "confidence": round(confidence, 2) if confidence is not None else None
    }


# ------------------------------------------------------------
# Sample predictions
# ------------------------------------------------------------

sample_inputs = [
    "fish has red patches on body and bloody wounds near fins",
    "fish is gasping at surface and gills are pale with mucus",
    "white cotton like fungus is growing on fish body",
    "fish is active eating well and swimming normally",
    "fish has white spots and rubbing body against rocks",
    "tail is turning white and fish is swimming weakly",
    "fish has swollen belly red ulcers and popeye"
]

print("\nSample Predictions")
print("=" * 60)

for text in sample_inputs:
    result = predict_fish_disease(text)

    print("\nInput:", text)
    print("Predicted Label:", result["label"])
    print("Predicted Disease:", result["disease"])
    print("Confidence:", result["confidence"], "%")


# ------------------------------------------------------------
# Manual input mode
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("Manual Prediction Mode")
print("Type fish symptoms and press Enter.")
print("Type 'exit' to stop.")
print("=" * 60)

while True:
    user_input = input("\nEnter fish symptoms: ")

    if user_input.lower().strip() == "exit":
        print("Stopped.")
        break

    result = predict_fish_disease(user_input)

    print("\nPrediction Result")
    print("-----------------")
    print("Label:", result["label"])
    print("Disease:", result["disease"])
    print("Confidence:", result["confidence"], "%")