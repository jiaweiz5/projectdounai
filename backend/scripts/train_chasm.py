import json
from pathlib import Path
from collections import Counter

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


TRAIN_PATH = Path("data/processed/chasm_train.jsonl")
VAL_PATH = Path("data/processed/chasm_validation.jsonl")

MODEL_DIR = Path("models")
VECTORIZER_PATH = MODEL_DIR / "chasm_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "chasm_classifier.joblib"


def load_jsonl(path):
    texts = []
    labels = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            texts.append(row["text"])
            labels.append(row["label"])

    return texts, labels


def main():
    print("=" * 60)
    print("TRAINING CHASM COVERT-AD DETECTOR")
    print("=" * 60)

    train_texts, train_labels = load_jsonl(TRAIN_PATH)
    val_texts, val_labels = load_jsonl(VAL_PATH)

    print("\nTraining examples:", len(train_texts))
    print("Validation examples:", len(val_texts))

    print("\nTraining label distribution:")
    print(Counter(train_labels))

    # Character-level TF-IDF works well for Chinese because
    # we do not need a Chinese word tokenizer.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
    )

    print("\nFitting TF-IDF...")

    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    print("Feature count:", X_train.shape[1])

    classifier = LogisticRegression(
        C=1,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    print("\nTraining Logistic Regression...")

    classifier.fit(X_train, train_labels)

    predictions = classifier.predict(X_val)

    accuracy = accuracy_score(val_labels, predictions)
    precision = precision_score(
        val_labels,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        val_labels,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        val_labels,
        predictions,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            val_labels,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH,
    )

    joblib.dump(
        classifier,
        CLASSIFIER_PATH,
    )

    print("\nSaved:")
    print(VECTORIZER_PATH)
    print(CLASSIFIER_PATH)

    print("\nStep 28 complete.")


if __name__ == "__main__":
    main()