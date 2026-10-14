"""Export Layer 2 CHASM test errors for manual review.

Run from the backend directory:
    python scripts/export_layer2_errors.py

The threshold is imported from evaluate_chasm_test.py so both scripts agree.
This script does not change the model, threshold, or evaluation data.
"""

import json
from pathlib import Path

import joblib
import numpy as np

from evaluate_chasm_test import THRESHOLD


BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data" / "processed" / "chasm_test.jsonl"
VECTORIZER_PATH = BASE_DIR / "models" / "chasm_vectorizer.joblib"
CLASSIFIER_PATH = BASE_DIR / "models" / "chasm_classifier.joblib"
OUTPUT_DIR = BASE_DIR / "data" / "layer2_error_review"


def write_jsonl(path, rows):
    """Write one reviewable mistake per line while preserving Chinese text."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    """Run the saved model and export its false negatives and false positives."""
    # The imported cutoff is the one currently set in evaluate_chasm_test.py.
    if not 0 <= THRESHOLD <= 1:
        raise ValueError(f"Invalid threshold: {THRESHOLD}")

    examples = []
    # Keep source_line so each error can be found again in the original test file.
    with TEST_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid label at line {line_number}: {label}")
            examples.append({
                "source_line": line_number,
                "text": row["text"],
                "true_label": label,
            })

    if not examples:
        raise ValueError(f"No examples found in {TEST_PATH}")

    # Use the same vectorizer and classifier files as the existing evaluator.
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    features = vectorizer.transform([item["text"] for item in examples])
    probabilities = classifier.predict_proba(features)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    # A false negative is a missed label-1 ad; a false positive is a flagged label-0 post.
    false_negatives = []
    false_positives = []
    for item, score, prediction in zip(examples, probabilities, predictions):
        if int(prediction) == item["true_label"]:
            continue
        item["score"] = round(float(score), 6)
        item["predicted_label"] = int(prediction)
        if item["true_label"] == 1:
            false_negatives.append(item)
        else:
            false_positives.append(item)

    # Save reviews separately so neither the model nor the labeled test file changes.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    misses_path = OUTPUT_DIR / "missed_ads.jsonl"
    alarms_path = OUTPUT_DIR / "false_alarms.jsonl"
    write_jsonl(misses_path, false_negatives)
    write_jsonl(alarms_path, false_positives)

    print(f"Examples: {len(examples)}")
    print(f"Threshold from evaluate_chasm_test.py: {THRESHOLD:.4f}")
    print(f"Missed label-1 posts: {len(false_negatives)} -> {misses_path}")
    print(f"False alarms on label-0 posts: {len(false_positives)} -> {alarms_path}")
    print("Review source_line, score, and text in each JSONL file.")


if __name__ == "__main__":
    main()
