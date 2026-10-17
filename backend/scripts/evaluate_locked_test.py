import json
import argparse
import hashlib
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


parser = argparse.ArgumentParser()
parser.add_argument("--test-file", type=Path, default=TEST_FILE,
                    help="Use only a reserved, never tuned independent JSONL set")
args = parser.parse_args()
rows = read_jsonl(args.test_file)
if not rows:
    raise ValueError("Test file is empty")
if args.test_file.resolve() != TEST_FILE.resolve():
    test_ids = {row["id"] for row in rows}
    test_groups = {row["source_group_id"] for row in rows}
    if len(test_ids) != len(rows):
        raise ValueError("Independent test has duplicate IDs")
    for row in rows:
        variant = row.get("variant")
        if variant in ("H", "L1", "L2", "L3", "L4"):
            expected = 0 if variant == "H" else 1
            if type(row["label"]) is not int or row["label"] != expected:
                raise ValueError(f"Invalid label for {row['id']}: {variant}")
    for name in ("fit", "dev", "calibration", "test_locked"):
        for prior in read_jsonl(BASE_DIR / "data/splits" / f"{name}.jsonl"):
            if prior.get("id") in test_ids or prior.get("source_group_id") in test_groups:
                raise ValueError(f"Independent test overlaps {name}: {prior.get('id')}")
print("Test file:", args.test_file)
print("Test SHA256:", hashlib.sha256(args.test_file.read_bytes()).hexdigest())
print("Model SHA256:", hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest())

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
    probabilities >= bundle.get("decision_threshold", 0.5)
).astype(int)


def report_group(name, indices):
    actual = y_test[indices]
    predicted = predictions[indices]
    human = actual == 0
    ai = actual == 1
    print(
        f"{name}: n={len(indices)}, human={int(human.sum())}, "
        f"human false positives={int((predicted[human] == 1).sum())}, "
        f"H FPR={((predicted[human] == 1).mean() if human.any() else float('nan')):.3f}, "
        f"AI detected={int((predicted[ai] == 1).sum())}, "
        f"AI recall={((predicted[ai] == 1).mean() if ai.any() else float('nan')):.3f}"
    )


def length_bucket(text):
    size = len(text)
    if size < 100:
        return "<100"
    if size < 300:
        return "100-299"
    return "300+"


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

if len(set(y_test)) == 2:
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

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

print("Decision threshold:", bundle.get("decision_threshold", 0.5))
report_group("ALL", np.arange(len(rows)))

for field, value_fn in (
    ("source", lambda row: row.get("source", "unknown")),
    ("variant", lambda row: row.get("variant", "unspecified")),
    ("human_subtype", lambda row: row.get("human_subtype", "unspecified")),
    ("length", lambda row: length_bucket(row["text"])),
):
    print(f"\nBY {field.upper()}")
    values = sorted({value_fn(row) for row in rows})
    for value in values:
        indices = np.array([i for i, row in enumerate(rows) if value_fn(row) == value])
        report_group(value, indices)


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
