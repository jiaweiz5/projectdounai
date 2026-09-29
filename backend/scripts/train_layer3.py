import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "layer3_features.json"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "layer3_model.joblib"


FEATURE_NAMES = [
    "comment_count",
    "unique_comment_ratio",
    "duplicate_ratio",
    "avg_comment_length",
    "min_comment_length",
    "max_comment_length",
    "short_comment_ratio",
    "emoji_like_ratio",
    "exclamation_ratio",
    "question_ratio",
]


def main():
    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    labels = [item["label"] for item in data]

    unique_labels = set(labels)

    if len(unique_labels) < 2:
        print("=" * 65)
        print("LAYER 3 TRAINING NOT READY")
        print("=" * 65)

        print(
            "Training requires both label 0 and label 1 examples."
        )

        print(f"Labels currently present: {unique_labels}")

        print(
            "\nWait until layer3_positive.json contains "
            "positive examples."
        )

        return

    X = []

    y = []

    for item in data:
        features = item["features"]

        row = [
            features[name]
            for name in FEATURE_NAMES
        ]

        X.append(row)
        y.append(item["label"])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print("=" * 65)
    print("LAYER 3 RESULTS")
    print("=" * 65)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    MODEL_DIR.mkdir(
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
        },
        MODEL_PATH,
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()