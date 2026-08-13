import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# Fish Disease NLP Model Trainer
# Input : fish_symptom_dataset.csv
# Output: fish_disease_nlp_model.pkl
# ============================================================

DATASET_PATH = "fish_symptom_dataset.csv"
MODEL_PATH = "fish_disease_nlp_model.pkl"

disease_labels = {
    0: "Bacterial Red disease",
    1: "Bacterial diseases - Aeromoniasis",
    2: "Bacterial gill disease",
    3: "Fungal diseases Saprolegniasis",
    4: "Healthy Fish",
    5: "Parasitic diseases",
    6: "Viral diseases White tail disease"
}

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"{DATASET_PATH} not found. Run generate_dataset.py first."
    )

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print()

required_cols = ["symptoms", "label"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

df = df.dropna(subset=["symptoms", "label"])

X = df["symptoms"].astype(str)
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),

        # Helps reduce memorizing rare synthetic phrases
        min_df=3,
        max_df=0.90,

        # Reduce model complexity
        max_features=2500,

        # More stable for noisy user text
        sublinear_tf=True
    )),

    ("classifier", LogisticRegression(
        max_iter=1500,
        class_weight="balanced",

        # Lower C = stronger regularization
        C=0.6,

        solver="liblinear"
    ))
])

print("Training model...")
model.fit(X_train, y_train)

print("Training completed")
print()

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("Training Accuracy:", round(train_acc * 100, 2), "%")
print("Testing Accuracy :", round(test_acc * 100, 2), "%")
print("Accuracy Gap     :", round((train_acc - test_acc) * 100, 2), "%")
print()

cv_scores = cross_val_score(model, X, y, cv=5)
print("5-Fold CV Scores:", [round(s * 100, 2) for s in cv_scores])
print("Mean CV Accuracy:", round(cv_scores.mean() * 100, 2), "%")
print()

print("Classification Report:")
print(classification_report(
    y_test,
    test_pred,
    target_names=[disease_labels[i] for i in sorted(disease_labels.keys())]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

model_bundle = {
    "model": model,
    "disease_labels": disease_labels
}

joblib.dump(model_bundle, MODEL_PATH)

print()
print("Model saved successfully!")
print("File:", MODEL_PATH)