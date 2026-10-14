"""Compare Layer 2 thresholds on CHASM validation data only.

Place in backend/scripts and run from backend:
    python scripts/check_layer2_validation_thresholds.py

This script makes no changes to the saved model, API, or locked test data.
"""

from pathlib import Path

import joblib
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from evaluate_chasm_test import THRESHOLD, load_data


BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATION_PATH = BASE_DIR / "data" / "processed" / "chasm_validation.jsonl"
VECTORIZER_PATH = BASE_DIR / "models" / "chasm_vectorizer.joblib"
CLASSIFIER_PATH = BASE_DIR / "models" / "chasm_classifier.joblib"


def main():
    """Score validation posts with the saved model at several cutoffs."""
    # These are the same saved vectorizer and classifier used by the evaluator.
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    texts, labels = load_data(VALIDATION_PATH)
    if len(texts) == 0 or set(labels.tolist()) != {0, 1}:
        raise ValueError("Validation data needs both label 0 and label 1.")

    # Column 1 means the score for label 1 (covert advertisement).
    probabilities = classifier.predict_proba(vectorizer.transform(texts))[:, 1]
    print("CHASM VALIDATION-ONLY THRESHOLD CHECK")
    print(f"Rows: {len(labels)} | label 0: {sum(labels == 0)} | label 1: {sum(labels == 1)}")
    print(f"Current evaluator threshold: {THRESHOLD:.2f}")
    print("threshold  precision  recall     F1     TN   FP   FN   TP")
    # Sweep 0.30, 0.35, ..., 0.70 without modifying any model settings.
    for step in range(30, 71, 5):
        threshold = step / 100
        # The confusion matrix orders rows as true labels and columns as predictions.
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        print(
            f"{threshold:>8.2f}  "
            f"{precision_score(labels, predictions, zero_division=0):>9.4f}  "
            f"{recall_score(labels, predictions, zero_division=0):>6.4f}  "
            f"{f1_score(labels, predictions, zero_division=0):>6.4f}  "
            f"{tn:>5} {fp:>4} {fn:>4} {tp:>4}"
        )
    print("Review the tradeoff before changing a threshold; the API may use a separate value.")


if __name__ == "__main__":
    main()
