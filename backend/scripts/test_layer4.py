"""Smoke-test and evaluate Layer 4 image-text alignment.

Examples:

    # Quick test of one aligned and one mismatched case.
    python scripts/test_layer4.py

    # Test specific dataset cases.
    python scripts/test_layer4.py --case case_001 --case case_036

    # Estimate threshold generalization using five held-out folds.
    python scripts/test_layer4.py --cross-validate

    # Re-run every alignment image through the final detector. This reports
    # development/calibration-set performance, not independent test accuracy.
    python scripts/test_layer4.py --all-alignment
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from layer4_detector import analyze_layer4_alignment


DATASET_PATH = BASE_DIR / "data" / "layer4" / "layer4_dataset.json"
SCORES_PATH = (
    BASE_DIR
    / "data"
    / "layer4"
    / "layer4_alignment_scores.json"
)

ALIGNMENT_ISSUE_TYPES = {
    "none",
    "image_text_mismatch",
}


def load_json(path: Path):
    """Load a UTF-8 JSON file with a clear missing-file error."""

    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_dataset() -> list[dict]:
    """Load either a plain list or a {cases: [...]} dataset wrapper."""

    data = load_json(DATASET_PATH)
    records = data.get("cases", []) if isinstance(data, dict) else data

    if not isinstance(records, list):
        raise ValueError("Layer 4 dataset must contain a list of cases.")

    return records


def expected_alignment_label(record: dict) -> int | None:
    """Return the expected label only for the alignment task.

    Cases 061-085 and 086-100 represent other Layer 4 tasks. Returning None
    prevents this script from incorrectly counting those cases as alignment
    successes or failures.
    """

    issue_type = record.get("issue_type")

    if issue_type == "none":
        return 0

    if issue_type == "image_text_mismatch":
        return 1

    return None


def find_case(records: list[dict], case_id: str) -> dict:
    for record in records:
        if record.get("id") == case_id:
            return record

    raise ValueError(f"Dataset case not found: {case_id}")


def analyze_record(record: dict) -> tuple[dict, int | None]:
    image_path = BASE_DIR / record["image_path"]

    result = analyze_layer4_alignment(
        post_text=record["post_text"],
        image_path=image_path,
    )

    return result, expected_alignment_label(record)


def print_case_result(record: dict, result: dict, expected: int | None):
    print("=" * 65)
    print(f"Case:       {record['id']}")
    print(f"Issue type: {record['issue_type']}")
    print(f"Score:      {result['image_text_similarity']:.4f}")
    print(f"Threshold:  {result['threshold']:.4f}")
    print(f"Prediction: {result['prediction']}")
    print(f"Risk:       {result['risk']}")

    if expected is None:
        print(
            "Evaluation:  not scored (this case belongs to a different "
            "Layer 4 task)"
        )
        return

    print(f"Expected:   {'image_text_mismatch' if expected else 'aligned'}")
    print(
        "Evaluation:  "
        + ("PASS" if result["label"] == expected else "FAIL")
    )


def print_metrics(labels: list[int], predictions: list[int]):
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    print(f"Accuracy:  {accuracy_score(labels, predictions):.4f}")
    print(
        "Precision: "
        f"{precision_score(labels, predictions, zero_division=0):.4f}"
    )
    print(
        "Recall:    "
        f"{recall_score(labels, predictions, zero_division=0):.4f}"
    )
    print(
        "F1:        "
        f"{f1_score(labels, predictions, zero_division=0):.4f}"
    )
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Choose the best mismatch threshold using only one training fold."""

    unique_scores = sorted(set(float(score) for score in scores))
    thresholds = [
        (left + right) / 2
        for left, right in zip(unique_scores, unique_scores[1:])
    ]

    if not thresholds:
        raise ValueError("A fold has fewer than two unique scores.")

    best_threshold = thresholds[0]
    best_key = None

    for threshold in thresholds:
        predictions = (scores < threshold).astype(int)
        key = (
            f1_score(labels, predictions, zero_division=0),
            recall_score(labels, predictions, zero_division=0),
            precision_score(labels, predictions, zero_division=0),
        )

        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold

    return float(best_threshold)


def run_cross_validation():
    """Evaluate threshold selection with five held-out prediction folds.

    SigLIP2 itself is pretrained and is not fitted here. Each fold chooses its
    threshold using only the other four folds, then predicts the held-out fold.
    This is more honest than reporting the threshold's score on all 60 cases
    that were used to select that same threshold.
    """

    records = load_json(SCORES_PATH)
    scores = np.asarray(
        [record["image_text_similarity"] for record in records],
        dtype=float,
    )
    labels = np.asarray(
        [record["label"] for record in records],
        dtype=int,
    )

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    out_of_fold_predictions = np.zeros_like(labels)
    fold_thresholds = []

    print("=" * 65)
    print("LAYER 4 FIVE-FOLD THRESHOLD VALIDATION")
    print("=" * 65)

    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(scores, labels),
        start=1,
    ):
        threshold = choose_threshold(
            scores[train_indices],
            labels[train_indices],
        )
        predictions = (scores[test_indices] < threshold).astype(int)
        out_of_fold_predictions[test_indices] = predictions
        fold_thresholds.append(threshold)

        fold_accuracy = accuracy_score(
            labels[test_indices],
            predictions,
        )

        print(
            f"Fold {fold_number}: threshold={threshold:.4f}, "
            f"accuracy={fold_accuracy:.4f}, "
            f"cases={len(test_indices)}"
        )

    print("\nOverall out-of-fold results")
    print("-" * 65)
    print_metrics(labels.tolist(), out_of_fold_predictions.tolist())
    print(
        "Fold thresholds: "
        + ", ".join(f"{value:.4f}" for value in fold_thresholds)
    )
    print(
        "Mean fold threshold: "
        f"{float(np.mean(fold_thresholds)):.4f}"
    )


def run_all_alignment(records: list[dict]):
    """Run all 60 development alignment cases through the final detector."""

    alignment_records = [
        record
        for record in records
        if record.get("issue_type") in ALIGNMENT_ISSUE_TYPES
    ]

    labels = []
    predictions = []

    print("=" * 65)
    print("LAYER 4 DEVELOPMENT ALIGNMENT SET")
    print("=" * 65)

    for index, record in enumerate(alignment_records, start=1):
        result, expected = analyze_record(record)

        if expected is None:
            continue

        labels.append(expected)
        predictions.append(result["label"])

        print(
            f"[{index:02d}/{len(alignment_records)}] "
            f"{record['id']} score={result['image_text_similarity']:.4f} "
            f"expected={expected} predicted={result['label']}"
        )

    print("\nDevelopment-set results")
    print("-" * 65)
    print_metrics(labels, predictions)
    print(
        "Note: these are the same 60 cases used to calibrate the final "
        "threshold, so this is not independent test accuracy."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Layer 4 image-text alignment."
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Dataset case ID to test. May be supplied multiple times.",
    )
    parser.add_argument(
        "--all-alignment",
        action="store_true",
        help="Run all 60 development alignment cases through the detector.",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run five-fold validation using saved alignment scores.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_dataset()

    if args.cross_validate:
        run_cross_validation()

    if args.all_alignment:
        run_all_alignment(records)

    # With no flags, run one typical aligned case and one typical mismatch.
    if not args.case_ids and not args.cross_validate and not args.all_alignment:
        args.case_ids = ["case_001", "case_036"]

    for case_id in args.case_ids or []:
        record = find_case(records, case_id)
        result, expected = analyze_record(record)
        print_case_result(record, result, expected)


if __name__ == "__main__":
    main()
