import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


DATA_PATH = Path("data/processed/chasm_validation.jsonl")
VECTORIZER_PATH = Path("models/chasm_vectorizer.joblib")
CLASSIFIER_PATH = Path("models/chasm_classifier.joblib")


def load_data(path):
    texts = []
    labels = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                texts.append(row["text"])
                labels.append(row["label"])

    return texts, np.array(labels)


def main():
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)

    texts, labels = load_data(DATA_PATH)

    X = vectorizer.transform(texts)
    probabilities = classifier.predict_proba(X)[:, 1]

    print("=" * 80)
    print("CHASM THRESHOLD TUNING")
    print("=" * 80)

    best_threshold = None
    best_f1 = -1

    for threshold in np.arange(0.20, 0.81, 0.025):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            labels,
            predictions,
        )

        precision = precision_score(
            labels,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            labels,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        print(
            f"threshold={threshold:.3f} "
            f"accuracy={accuracy:.4f} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"f1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("\n" + "=" * 80)
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()