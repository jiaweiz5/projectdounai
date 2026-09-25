import json
from pathlib import Path

import joblib

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parent

FIT_FILE = (
    BASE_DIR
    / "data"
    / "splits"
    / "fit.jsonl"
)

DEV_FILE = (
    BASE_DIR
    / "data"
    / "splits"
    / "dev.jsonl"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "authorship_baseline.joblib"
)


def read_jsonl(path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


fit_rows = read_jsonl(
    FIT_FILE
)

dev_rows = read_jsonl(
    DEV_FILE
)


X_train = [
    row["text"]
    for row in fit_rows
]

y_train = [
    row["label"]
    for row in fit_rows
]


X_dev = [
    row["text"]
    for row in dev_rows
]

y_dev = [
    row["label"]
    for row in dev_rows
]


print(
    "Training examples:",
    len(X_train)
)

print(
    "Development examples:",
    len(X_dev)
)


model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=2,
                max_features=120000,
                sublinear_tf=True,
            ),
        ),

        (
            "classifier",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)


print()
print(
    "Training model..."
)

model.fit(
    X_train,
    y_train
)


predictions = model.predict(
    X_dev
)

probabilities = (
    model.predict_proba(
        X_dev
    )[:, 1]
)


print()
print(
    "=" * 60
)

print(
    "DEVELOPMENT RESULTS"
)

print(
    "=" * 60
)


print(
    "Accuracy:",
    round(
        accuracy_score(
            y_dev,
            predictions
        ),
        4,
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_dev,
            predictions
        ),
        4,
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_dev,
            predictions
        ),
        4,
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_dev,
            predictions
        ),
        4,
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_dev,
            probabilities
        ),
        4,
    )
)


print()
print(
    "Confusion matrix:"
)

print(
    confusion_matrix(
        y_dev,
        predictions
    )
)


joblib.dump(
    model,
    MODEL_FILE
)


print()
print(
    "Saved model:"
)

print(
    MODEL_FILE
)