import json
from pathlib import Path
from collections import Counter

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


TEST_PATH = Path("data/processed/chasm_test.jsonl")

VECTORIZER_PATH = Path(
    "models/chasm_vectorizer.joblib"
)

CLASSIFIER_PATH = Path(
    "models/chasm_classifier.joblib"
)


# 0.5 is ok
THRESHOLD = 0.5


def load_data(path):
    texts = []
    labels = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            texts.append(row["text"])
            labels.append(row["label"])

    return texts, np.array(labels)


def main():
    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    texts, labels = load_data(
        TEST_PATH
    )

    X = vectorizer.transform(texts)

    probabilities = classifier.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    print("=" * 60)
    print("LOCKED CHASM TEST RESULTS")
    print("=" * 60)

    print("Examples:", len(texts))
    print("Label distribution:", Counter(labels))
    print("Threshold:", THRESHOLD)

    print(
        "Accuracy:",
        f"{accuracy_score(labels, predictions):.4f}",
    )

    print(
        "Precision:",
        f"{precision_score(labels, predictions, zero_division=0):.4f}",
    )

    print(
        "Recall:",
        f"{recall_score(labels, predictions, zero_division=0):.4f}",
    )

    print(
        "F1:",
        f"{f1_score(labels, predictions, zero_division=0):.4f}",
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            labels,
            predictions,
        )
    )

    print("\nClassification report:")

    print(
        classification_report(
            labels,
            predictions,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()