import json
from collections import defaultdict
from pathlib import Path

import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parent.parent

DEV_FILE = (
    BASE_DIR
    / "data"
    / "splits"
    / "dev.jsonl"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
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


model = joblib.load(
    MODEL_FILE
)

rows = read_jsonl(
    DEV_FILE
)

by_source = defaultdict(list)

for row in rows:
    by_source[
        row["source"]
    ].append(row)


for source, records in by_source.items():

    X = [
        r["text"]
        for r in records
    ]

    y = [
        r["label"]
        for r in records
    ]

    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    print()
    print("=" * 60)
    print(source)
    print("=" * 60)

    print(
        "Examples:",
        len(records)
    )

    print(
        "Accuracy:",
        round(
            accuracy_score(
                y,
                predictions
            ),
            4
        )
    )

    print(
        "Precision:",
        round(
            precision_score(
                y,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                y,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print(
        "F1:",
        round(
            f1_score(
                y,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    if len(set(y)) == 2:

        print(
            "ROC-AUC:",
            round(
                roc_auc_score(
                    y,
                    probabilities
                ),
                4
            )
        )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y,
            predictions
        )
    )