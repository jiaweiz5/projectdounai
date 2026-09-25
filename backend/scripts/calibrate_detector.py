import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CALIBRATION_FILE = (
    BASE_DIR
    / "data"
    / "splits"
    / "calibration.jsonl"
)

BASE_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "authorship_baseline.joblib"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_MODEL_FILE = (
    MODEL_DIR
    / "authorship_calibrated.joblib"
)


# ============================================================
# READ JSONL
# ============================================================

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


# ============================================================
# LOAD CALIBRATION DATA
# ============================================================

rows = read_jsonl(
    CALIBRATION_FILE
)

X_calibration = [
    row["text"]
    for row in rows
]

y_calibration = np.array(
    [
        row["label"]
        for row in rows
    ]
)


print(
    "Calibration examples:",
    len(rows)
)


# ============================================================
# LOAD EXISTING LAYER 1 MODEL
# ============================================================

base_model = joblib.load(
    BASE_MODEL_FILE
)


# ============================================================
# GET RAW MODEL SCORES
# ============================================================

raw_scores = (
    base_model
    .decision_function(
        X_calibration
    )
    .reshape(-1, 1)
)


# ============================================================
# TRAIN CALIBRATOR
# ============================================================

calibrator = LogisticRegression(
    solver="lbfgs",
    random_state=42
)

calibrator.fit(
    raw_scores,
    y_calibration
)


# ============================================================
# COMPARE BEFORE VS AFTER CALIBRATION
# ============================================================

before_probabilities = (
    base_model
    .predict_proba(
        X_calibration
    )[:, 1]
)


after_probabilities = (
    calibrator
    .predict_proba(
        raw_scores
    )[:, 1]
)


before_brier = brier_score_loss(
    y_calibration,
    before_probabilities
)

after_brier = brier_score_loss(
    y_calibration,
    after_probabilities
)


print()
print("=" * 60)
print("CALIBRATION RESULTS")
print("=" * 60)

print(
    "Brier score before:",
    round(
        before_brier,
        4
    )
)

print(
    "Brier score after:",
    round(
        after_brier,
        4
    )
)


# ============================================================
# SAVE BASE MODEL + CALIBRATOR TOGETHER
# ============================================================

bundle = {
    "base_model":
        base_model,

    "calibrator":
        calibrator,

    "model_version":
        "tfidf-logreg-v1",

    "calibration_version":
        "sigmoid-v1",
}


joblib.dump(
    bundle,
    OUTPUT_MODEL_FILE
)


print()
print(
    "Saved calibrated model:"
)

print(
    OUTPUT_MODEL_FILE
)