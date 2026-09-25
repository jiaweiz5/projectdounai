import json
from pathlib import Path

import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
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

    return texts, labels


def main():
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)

    texts, labels = load_data(DATA_PATH)

    X = vectorizer.transform(texts)

    probabilities = classifier.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    print("=" * 60)
    print("CHASM VALIDATION EVALUATION")
    print("=" * 60)

    print("Examples:", len(texts))
    print(f"Accuracy:  {accuracy_score(labels, predictions):.4f}")
    print(
        f"Precision: "
        f"{precision_score(labels, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(labels, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1:        "
        f"{f1_score(labels, predictions, zero_division=0):.4f}"
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(labels, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print("\n" + "=" * 60)
    print("FALSE NEGATIVES — ads incorrectly called non-ads")
    print("=" * 60)

    count = 0

    for text, true_label, pred, prob in zip(
        texts,
        labels,
        predictions,
        probabilities,
    ):
        if true_label == 1 and pred == 0:
            count += 1
            print(f"\n#{count}")
            print(f"Probability: {prob:.4f}")
            print(text[:500])

    print("\n" + "=" * 60)
    print("FALSE POSITIVES — non-ads incorrectly called ads")
    print("=" * 60)

    count = 0

    for text, true_label, pred, prob in zip(
        texts,
        labels,
        predictions,
        probabilities,
    ):
        if true_label == 0 and pred == 1:
            count += 1
            print(f"\n#{count}")
            print(f"Probability: {prob:.4f}")
            print(text[:500])


if __name__ == "__main__":
    main()