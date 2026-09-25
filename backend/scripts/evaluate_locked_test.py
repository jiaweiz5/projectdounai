import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = (
    BASE_DIR
    / "data"
    / "splits"
    / "test_locked.jsonl"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "authorship_calibrated.joblib"
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


rows = read_jsonl(
    TEST_FILE
)

X_test = [
    row["text"]
    for row in rows
]

y_test = np.array(
    [
        row["label"]
        for row in rows
    ]
)


bundle = joblib.load(
    MODEL_FILE
)

base_model = bundle[
    "base_model"
]

calibrator = bundle[
    "calibrator"
]


raw_scores = (
    base_model
    .decision_function(
        X_test
    )
    .reshape(-1, 1)
)


probabilities = (
    calibrator
    .predict_proba(
        raw_scores
    )[:, 1]
)


predictions = (
    probabilities >= 0.5
).astype(int)


print()
print("=" * 60)
print("LOCKED TEST RESULTS")
print("=" * 60)

print(
    "Examples:",
    len(y_test)
)

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            predictions
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
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
            y_test,
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
            y_test,
            predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            probabilities
        ),
        4
    )
)

print(
    "Brier score:",
    round(
        brier_score_loss(
            y_test,
            probabilities
        ),
        4
    )
)

print()
print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# Optional: source-by-source summary
by_source = defaultdict(list)

for index, row in enumerate(rows):
    by_source[
        row["source"]
    ].append(index)


print()
print("=" * 60)
print("LOCKED TEST BY SOURCE")
print("=" * 60)


for source, indices in by_source.items():

    y_source = y_test[
        indices
    ]

    probability_source = (
        probabilities[
            indices
        ]
    )

    prediction_source = (
        predictions[
            indices
        ]
    )

    print()
    print(source)

    print(
        "Examples:",
        len(indices)
    )

    print(
        "Accuracy:",
        round(
            accuracy_score(
                y_source,
                prediction_source
            ),
            4
        )
    )

    print(
        "F1:",
        round(
            f1_score(
                y_source,
                prediction_source,
                zero_division=0
            ),
            4
        )
    )

    if len(set(y_source)) == 2:
        print(
            "ROC-AUC:",
            round(
                roc_auc_score(
                    y_source,
                    probability_source
                ),
                4
            )
        )