import json
import os
import sys

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from comment_detector import analyze_comment_coordination


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "layer3_eval.json",
)

THRESHOLD = 0.25


with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)


y_true = []
y_pred = []

print("=" * 60)
print("LAYER 3 COMMENT COORDINATION EVALUATION")
print("=" * 60)

for example in dataset:

    result = analyze_comment_coordination(
        example["comments"]
    )

    score = result["coordination_score"]

    prediction = 1 if score >= THRESHOLD else 0

    y_true.append(example["label"])
    y_pred.append(prediction)

    print(
        f'{example["id"]:12} '
        f'label={example["label"]} '
        f'pred={prediction} '
        f'score={score:.4f} '
        f'risk={result["risk"]}'
    )


print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Examples: {len(dataset)}")
print(f"Threshold: {THRESHOLD}")

print(
    f"Accuracy: "
    f"{accuracy_score(y_true, y_pred):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
)

print(
    f"Recall: "
    f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
)

print(
    f"F1: "
    f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
)

print()
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))

print()
print("Classification report:")
print(
    classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0,
    )
)