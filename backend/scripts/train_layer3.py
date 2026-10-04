"""Train and validate the learned Layer 3 coordination classifier."""

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from layer3_features import LAYER3_MODEL_FEATURE_NAMES


DATA_PATH = BASE_DIR / "data" / "layer3_features.json"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "layer3_model.joblib"
FEATURE_NAMES = list(LAYER3_MODEL_FEATURE_NAMES)


def make_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def positive_family(item, positive_position):
    """Keep each of the five synthetic families in one fold."""

    item_id = str(item.get("id", ""))
    match = re.search(r"(\d+)$", item_id)
    item_number = int(match.group(1)) if match else positive_position
    family_number = (item_number - 1) % 5
    return f"positive_family_{family_number}"


def calculate_metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
    }


def select_threshold(labels, probabilities):
    """Select the decision threshold from out-of-fold F1."""

    best_threshold = 0.5
    best_metrics = None

    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        metrics = calculate_metrics(labels, predictions)

        candidate = (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        )
        incumbent = (
            best_metrics["f1"],
            best_metrics["precision"],
            best_metrics["recall"],
        ) if best_metrics else (-1.0, -1.0, -1.0)

        if candidate > incumbent:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    labels = [int(item["label"]) for item in data]

    if len(set(labels)) < 2:
        raise ValueError(
            "Layer 3 training requires both label 0 and label 1."
        )

    rows = []
    targets = []
    validation_groups = []
    positive_position = 0

    for index, item in enumerate(data):
        features = item["features"]
        missing = [name for name in FEATURE_NAMES if name not in features]

        if missing:
            raise ValueError(
                "Missing learned features. Run "
                "python scripts/build_layer3_features.py first: "
                + ", ".join(missing)
            )

        rows.append([features[name] for name in FEATURE_NAMES])

        label = int(item["label"])
        targets.append(label)

        if label == 1:
            positive_position += 1
            validation_groups.append(
                positive_family(item, positive_position)
            )
        else:
            item_id = item.get("id", f"negative_{index}")
            validation_groups.append(f"negative_{item_id}")

    X = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=int)
    groups = np.asarray(validation_groups, dtype=object)

    print("=" * 65)
    print("LAYER 3 LEARNED-SEMANTIC FAMILY-HELD-OUT VALIDATION")
    print("=" * 65)

    splitter = GroupKFold(n_splits=5)
    out_of_fold_probabilities = np.zeros(len(y), dtype=float)
    fold_indices = []

    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        fold_model = make_model()
        fold_model.fit(X[train_indices], y[train_indices])

        probabilities = fold_model.predict_proba(X[test_indices])[:, 1]
        out_of_fold_probabilities[test_indices] = probabilities
        fold_indices.append((fold_number, test_indices))

    threshold = select_threshold(y, out_of_fold_probabilities)
    out_of_fold_predictions = (
        out_of_fold_probabilities >= threshold
    ).astype(int)

    for fold_number, test_indices in fold_indices:
        fold_metrics = calculate_metrics(
            y[test_indices],
            out_of_fold_predictions[test_indices],
        )

        negative_count = int(np.sum(y[test_indices] == 0))
        positive_count = int(np.sum(y[test_indices] == 1))

        print(
            f"\nFold {fold_number}: "
            f"{negative_count} negatives, "
            f"{positive_count} positives"
        )
        print(f"Accuracy:  {fold_metrics['accuracy']:.4f}")
        print(f"Precision: {fold_metrics['precision']:.4f}")
        print(f"Recall:    {fold_metrics['recall']:.4f}")
        print(f"F1:        {fold_metrics['f1']:.4f}")

    metrics = calculate_metrics(y, out_of_fold_predictions)
    matrix = confusion_matrix(y, out_of_fold_predictions)

    negative_probabilities = out_of_fold_probabilities[y == 0]
    medium_threshold = float(
        np.percentile(negative_probabilities, 95)
    )
    medium_threshold = min(medium_threshold, threshold)

    print("\n" + "=" * 65)
    print("OVERALL OUT-OF-FOLD RESULTS")
    print("=" * 65)
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1:                 {metrics['f1']:.4f}")
    print("\nConfusion matrix:")
    print(matrix)

    final_model = make_model()
    final_model.fit(X, y)

    classifier = final_model.named_steps["classifier"]
    ranked_features = sorted(
        zip(FEATURE_NAMES, classifier.coef_[0]),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )

    print("\nMost influential learned features:")

    for feature_name, coefficient in ranked_features[:10]:
        direction = "coordinated" if coefficient > 0 else "normal"
        print(
            f"{feature_name:36s} "
            f"{coefficient: .4f} toward {direction}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": final_model,
            "feature_names": FEATURE_NAMES,
            "threshold": threshold,
            "risk_thresholds": {
                "medium": medium_threshold,
                "high": threshold,
            },
            "validation_method": (
                "5-fold family-held-out out-of-fold validation"
            ),
            "validation_metrics": {
                name: float(value)
                for name, value in metrics.items()
            },
            "confusion_matrix": matrix.tolist(),
        },
        MODEL_PATH,
    )

    print(f"\nFinal model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
