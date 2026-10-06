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


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from layer4_features import calculate_image_text_similarity


DATASET_PATH = BASE_DIR / "data" / "layer4" / "layer4_dataset.json"
SCORES_PATH = BASE_DIR / "data" / "layer4" / "layer4_alignment_scores.json"
CONFIG_PATH = BASE_DIR / "data" / "layer4" / "layer4_config.json"


def choose_best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Choose the threshold that gives the best F1 score for mismatches.

    Label 1 means mismatch. A lower similarity score means higher mismatch risk.
    """

    unique_scores = sorted(set(float(score) for score in scores))

    thresholds = [
        (left + right) / 2
        for left, right in zip(unique_scores, unique_scores[1:])
    ]

    if not thresholds:
        raise ValueError("Cannot calibrate with fewer than two unique scores.")

    best = None

    for threshold in thresholds:
        predictions = (scores < threshold).astype(int)

        metrics = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(
                precision_score(labels, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(labels, predictions, zero_division=0)
            ),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(
                labels,
                predictions,
            ).tolist(),
        }

        if best is None or (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
        ) > (
            best["f1"],
            best["recall"],
            best["precision"],
        ):
            best = metrics

    return best


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "minimum": round(float(min(values)), 4),
        "maximum": round(float(max(values)), 4),
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
    }


def main():
    with DATASET_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    alignment_cases = [
        item
        for item in data
        if item["issue_type"] in {"none", "image_text_mismatch"}
    ]

    if len(alignment_cases) != 60:
        raise ValueError(
            f"Expected 60 alignment cases, found {len(alignment_cases)}."
        )

    results = []

    print("=" * 60)
    print("CALIBRATING LAYER 4 IMAGE-TEXT ALIGNMENT")
    print("=" * 60)

    for index, item in enumerate(alignment_cases, start=1):
        label = int(item["issue_type"] == "image_text_mismatch")

        result = calculate_image_text_similarity(
            BASE_DIR / item["image_path"],
            item["post_text"],
        )

        results.append(
            {
                "id": item["id"],
                "label": label,
                "issue_type": item["issue_type"],
                "image_text_similarity": result[
                    "image_text_similarity"
                ],
            }
        )

        print(
            f"[{index:02d}/60] {item['id']}  "
            f"label={label}  "
            f"score={result['image_text_similarity']:.4f}"
        )

    scores = np.asarray(
        [item["image_text_similarity"] for item in results],
        dtype=float,
    )
    labels = np.asarray(
        [item["label"] for item in results],
        dtype=int,
    )

    best = choose_best_threshold(scores, labels)

    normal_scores = [
        item["image_text_similarity"]
        for item in results
        if item["label"] == 0
    ]
    mismatch_scores = [
        item["image_text_similarity"]
        for item in results
        if item["label"] == 1
    ]

    SCORES_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config = {
        "alignment": {
            "model_name": "google/siglip2-base-patch16-224",
            "mismatch_when": "image_text_similarity < threshold",
            "similarity_threshold": round(best["threshold"], 4),
            "calibration_case_count": len(results),
            "calibration_note": (
                "Development calibration using controlled Layer 4 cases "
                "001–060; this is not an independent production test."
            ),
        }
    }

    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"Normal-score summary:   {summarize(normal_scores)}")
    print(f"Mismatch-score summary: {summarize(mismatch_scores)}")
    print(f"\nBest threshold: {best['threshold']:.4f}")
    print(f"Accuracy:  {best['accuracy']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall:    {best['recall']:.4f}")
    print(f"F1:        {best['f1']:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {best['confusion_matrix']}")
    print(f"\nScores saved to: {SCORES_PATH}")
    print(f"Config saved to: {CONFIG_PATH}")


if __name__ == "__main__":
    main()