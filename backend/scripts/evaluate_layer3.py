"""Evaluate the trained Layer 3 model used by the API."""

import json
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)

# Find the backend folder so this script can import the production detector.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Use the same model prediction function as main.py.
from scripts.layer3_detector import predict_layer3

# This file currently contains six examples, so treat results as a sanity check.
DATA_PATH = BASE_DIR / "data" / "layer3_eval.json"

with DATA_PATH.open("r", encoding="utf-8") as file:
    dataset = json.load(file)

if not dataset:
    raise ValueError(f"No evaluation examples found in {DATA_PATH}")

y_true = []
y_pred = []
scores = []
thresholds = []

print("=" * 60)
print("LAYER 3 TRAINED-MODEL EVALUATION")
print("=" * 60)

for index, example in enumerate(dataset, start=1):
    # Pass the same post ID and comment list shape used by the API.
    group = {
        "id": example.get("id", f"eval_{index:03d}"),
        "comments": example["comments"],
    }

    # This loads layer3_model.joblib and uses its saved threshold.
    result = predict_layer3(group)

    if not result.get("available"):
        raise RuntimeError(
            result.get("message", "The trained Layer 3 model is unavailable.")
        )

    true_label = int(example["label"])
    score = float(result["score"])
    predicted_label = int(result["label"])
    threshold = float(result["threshold"])

    y_true.append(true_label)
    y_pred.append(predicted_label)
    scores.append(score)
    thresholds.append(threshold)

    print(
        f'{group["id"]:16} '
        f"true={true_label} "
        f"predicted={predicted_label} "
        f"score={score:.4f} "
        f"threshold={threshold:.4f} "
        f'risk={result["risk"]}'
    )

if len(set(y_true)) < 2:
    raise ValueError("Evaluation data must contain both labels: 0 and 1.")

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Examples: {len(dataset)}")
print(f"Model threshold: {thresholds[0]:.4f}")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_true, scores):.4f}")
print(f"Brier:     {brier_score_loss(y_true, scores):.4f}")

print()
print("Confusion matrix (rows=true labels [0, 1], columns=predicted [0, 1]):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print()
print("Classification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "coordinated"],
        digits=4,
        zero_division=0,
    )
)

if len(dataset) < 30:
    print()
    print(
        "NOTE: This is a small sanity check, not a reliable estimate "
        "of real-world performance."
    )