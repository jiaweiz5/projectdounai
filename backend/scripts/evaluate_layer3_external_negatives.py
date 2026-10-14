"""Check Layer 3 false positives on reviewed external negative groups.

Place this script in backend/scripts/ and keep the data separate from training.
Run from backend with: python scripts/evaluate_layer3_external_negatives.py
"""

import argparse
import json
import math
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.layer3_detector import predict_layer3


DEFAULT_DATA_PATH = BASE_DIR / "data" / "layer3_kaggle_screened_negatives_30.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="JSON list of reviewed groups whose labels are all 0",
    )
    args = parser.parse_args()

    with args.data.expanduser().open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Expected a nonempty JSON list of comment groups.")

    seen_ids = set()
    false_positives = []
    scores = []
    saved_threshold = None

    print("LAYER 3 EXTERNAL NEGATIVE-CONTROL CHECK")
    print(f"Data: {args.data}")

    for index, example in enumerate(dataset, start=1):
        if not isinstance(example, dict):
            raise ValueError(f"Group {index} is not a JSON object.")
        if type(example.get("label")) is not int or example["label"] != 0:
            raise ValueError(f"Group {index} must have a reviewed label of 0.")

        group_id = example.get("id")
        comments = example.get("comments")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"Group {index} needs a nonempty string id.")
        if group_id in seen_ids:
            raise ValueError(f"Duplicate group id: {group_id}")
        if not isinstance(comments, list) or len(comments) < 5:
            raise ValueError(f"Group {group_id} needs at least five comments.")
        seen_ids.add(group_id)

        # This is the same inference function the API uses. No model fitting or
        # threshold selection occurs in this evaluation.
        result = predict_layer3({"id": group_id, "comments": comments})
        if not result.get("available"):
            raise RuntimeError(result.get("message", "Layer 3 model unavailable."))

        score = float(result["score"])
        threshold = float(result["threshold"])
        predicted = int(result["label"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid probability for {group_id}: {score}")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Invalid threshold for {group_id}: {threshold}")
        if predicted not in (0, 1) or predicted != int(score >= threshold):
            raise ValueError(f"Inconsistent prediction for {group_id}.")
        if saved_threshold is None:
            saved_threshold = threshold
        elif not math.isclose(threshold, saved_threshold):
            raise ValueError("Model threshold changed during evaluation.")

        scores.append(score)
        if predicted == 1:
            false_positives.append(group_id)

        print(
            f"{group_id}  true=0 predicted={predicted} "
            f"score={score:.4f} threshold={threshold:.4f} "
            f"risk={result.get('risk', 'unknown')}"
        )

    total = len(dataset)
    fp = len(false_positives)
    tn = total - fp
    print("\nRESULTS")
    print(f"Reviewed negative groups: {total}")
    print(f"Saved model threshold: {saved_threshold:.4f}")
    print(f"True negatives: {tn}")
    print(f"False positives: {fp}")
    print(f"False-positive rate: {fp / total:.1%}")
    print(f"Specificity: {tn / total:.1%}")
    print(f"Score range: {min(scores):.4f}–{max(scores):.4f}")
    if false_positives:
        print("False-positive group IDs:")
        for group_id in false_positives:
            print(f"  {group_id}")
    print(
        "NOTE: These provisional negative controls cannot measure recall, "
        "precision, F1, or overall accuracy. Do not tune the model on them "
        "and then report this same set as an independent test."
    )


if __name__ == "__main__":
    main()
