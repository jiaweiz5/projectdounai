"""Calibrate short-caption and optional long-caption Layer 4 thresholds."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from layer4_features import calculate_image_text_similarity


DATASET_PATH = BASE_DIR / "data" / "layer4" / "layer4_dataset.json"
LONG_DATASET_PATH = BASE_DIR / "data" / "layer4" / "layer4_long_alignment_cases.json"
SCORES_PATH = BASE_DIR / "data" / "layer4" / "layer4_alignment_scores.json"
LONG_SCORES_PATH = BASE_DIR / "data" / "layer4" / "layer4_long_alignment_scores.json"
CONFIG_PATH = BASE_DIR / "data" / "layer4" / "layer4_config.json"


def choose_best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Choose the similarity cutoff with the best mismatch F1 score."""
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
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        }
        if best is None or (
            metrics["f1"], metrics["recall"], metrics["precision"]
        ) > (best["f1"], best["recall"], best["precision"]):
            best = metrics
    return best


def summarize(values: list[float]) -> dict:
    """Describe the observed scores for a single label."""
    return {
        "count": len(values),
        "minimum": round(float(min(values)), 4),
        "maximum": round(float(max(values)), 4),
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
    }


def score_cases(cases: list[dict], output_path: Path, name: str) -> tuple[list[dict], dict]:
    """Score labeled cases, save their scores, and select a development cutoff."""
    results = []
    for index, item in enumerate(cases, start=1):
        label = int(item.get("label", item.get("issue_type") == "image_text_mismatch"))
        if label not in (0, 1):
            raise ValueError(f"Invalid label in {item.get('id')}: {label}")
        result = calculate_image_text_similarity(
            BASE_DIR / item["image_path"], item["post_text"]
        )
        results.append({
            "id": item["id"],
            "label": label,
            "issue_type": item.get("issue_type"),
            "image_text_similarity": result["image_text_similarity"],
            "text_window_count": result["text_window_count"],
            "text_coverage_ratio": result["text_coverage_ratio"],
        })
        print(
            f"[{index:02d}/{len(cases)}] {item['id']} "
            f"label={label} score={result['image_text_similarity']:.4f} "
            f"windows={result['text_window_count']}"
        )
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    scores = np.asarray([item["image_text_similarity"] for item in results], dtype=float)
    labels = np.asarray([item["label"] for item in results], dtype=int)
    best = choose_best_threshold(scores, labels)
    print(f"{name} development threshold: {best['threshold']:.4f}")
    print(f"{name} development F1: {best['f1']:.4f}")
    print(f"{name} confusion matrix: {best['confusion_matrix']}")
    for label in (0, 1):
        print(f"{name} label {label} scores: {summarize(scores[labels == label].tolist())}")
    return results, best


def main():
    """Keep the original short calibration and add long cases if supplied."""
    with DATASET_PATH.open(encoding="utf-8") as file:
        data = json.load(file)
    short_cases = [
        item for item in data
        if item["issue_type"] in {"none", "image_text_mismatch"}
    ]
    if len(short_cases) != 60:
        raise ValueError(f"Expected 60 alignment cases, found {len(short_cases)}.")
    print("CALIBRATING SHORT-CAPTION IMAGE-TEXT ALIGNMENT")
    short_results, short_best = score_cases(
        short_cases, SCORES_PATH, "Short caption"
    )
    if any(item["text_window_count"] != 1 for item in short_results):
        raise ValueError("The original 60 calibration cases must remain short captions.")

    alignment_config = {
        "model_name": "google/siglip2-base-patch16-224",
        "mismatch_when": "image_text_similarity < threshold",
        "similarity_threshold": round(short_best["threshold"], 4),
        "calibration_case_count": len(short_results),
        "calibration_note": (
            "Development calibration using controlled Layer 4 cases "
            "001–060; this is not an independent production test."
        ),
    }
    if LONG_DATASET_PATH.is_file():
        with LONG_DATASET_PATH.open(encoding="utf-8") as file:
            long_cases = json.load(file)
        if not isinstance(long_cases, list) or len(long_cases) < 10:
            raise ValueError("Supply at least 10 reviewed long cases (5 per label).")
        labels = [int(item["label"]) for item in long_cases]
        if labels.count(0) < 5 or labels.count(1) < 5:
            raise ValueError("Supply at least five long cases for each label.")
        if len({item["id"] for item in long_cases}) != len(long_cases):
            raise ValueError("Long calibration IDs must be unique.")
        print("\nCALIBRATING LONG-CAPTION IMAGE-TEXT ALIGNMENT")
        long_results, long_best = score_cases(
            long_cases, LONG_SCORES_PATH, "Long caption"
        )
        if any(
            item["text_window_count"] < 2 or item["text_coverage_ratio"] < 1.0
            for item in long_results
        ):
            raise ValueError(
                "Long calibration cases must use at least two windows with full coverage."
            )
        alignment_config.update({
            "long_text_similarity_threshold": round(long_best["threshold"], 4),
            "long_text_calibration_case_count": len(long_results),
            "long_text_calibration_note": (
                "Development calibration on reviewed long posts; validate on "
                "separate unseen posts before claiming real-world performance."
            ),
        })
    else:
        print("\nNo long-caption cases found. Long results remain provisional.")

    # Write a new config only after every requested calibration has succeeded.
    CONFIG_PATH.write_text(
        json.dumps({"alignment": alignment_config}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
