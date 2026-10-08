"""Test Layer 4 reference-image editing detection."""

import argparse
import json
import sys
from pathlib import Path

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

from layer4_detector import analyze_layer4_reference


DATASET_PATH = BASE_DIR / "data" / "layer4" / "layer4_dataset.json"


def load_reference_cases() -> list[dict]:
    with DATASET_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    records = data.get("cases", []) if isinstance(data, dict) else data
    return [record for record in records if record.get("reference_image_path")]


def find_case(records: list[dict], case_id: str) -> dict:
    for record in records:
        if record["id"] == case_id:
            return record

    raise ValueError(f"Reference case not found: {case_id}")


def print_result(title: str, result: dict, expected: int):
    print("=" * 65)
    print(title)
    print(f"Edit probability:    {result['edit_probability']:.4f}")
    print(f"Prediction:          {result['prediction']}")
    print(f"Reference similarity:{result['reference_similarity']:.4f}")
    print(f"Changed regions:     {result['changed_region_count']}")
    print(f"Expected label:      {expected}")
    print(
        "Evaluation:          "
        + ("PASS" if result["label"] == expected else "FAIL")
    )


def test_one_case(record: dict, include_clean_control: bool):
    target_path = BASE_DIR / record["image_path"]
    reference_path = BASE_DIR / record["reference_image_path"]

    edited_result = analyze_layer4_reference(
        target_path,
        reference_path,
    )
    print_result(
        f"{record['id']} purposeful edit",
        edited_result,
        expected=1,
    )

    if include_clean_control:
        clean_result = analyze_layer4_reference(
            reference_path,
            reference_path,
        )
        print_result(
            f"{record['id']} unchanged reference control",
            clean_result,
            expected=0,
        )

    return edited_result


def print_all_metrics(labels, predictions):
    print("\n" + "=" * 65)
    print("ALL PURPOSEFUL REFERENCE EDITS")
    print("=" * 65)
    print(f"Accuracy:  {accuracy_score(labels, predictions):.4f}")
    print(
        "Precision: "
        f"{precision_score(labels, predictions, zero_division=0):.4f}"
    )
    print(
        "Recall:    "
        f"{recall_score(labels, predictions, zero_division=0):.4f}"
    )
    print(f"F1:        {f1_score(labels, predictions, zero_division=0):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))
    print(
        "Note: all dataset reference cases are purposeful edits. The grouped "
        "cross-validation printed during calibration is the better balanced "
        "evaluation because it also includes no-edit controls."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Layer 4 reference-image editing detection."
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Reference case ID to test. May be repeated.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 25 purposeful edited/reference pairs.",
    )
    parser.add_argument(
        "--no-clean-control",
        action="store_true",
        help="Skip the unchanged-reference control in individual tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_reference_cases()

    if args.all:
        labels = []
        predictions = []

        for index, record in enumerate(records, start=1):
            result = analyze_layer4_reference(
                BASE_DIR / record["image_path"],
                BASE_DIR / record["reference_image_path"],
            )
            labels.append(1)
            predictions.append(result["label"])
            print(
                f"[{index:02d}/25] {record['id']} "
                f"probability={result['edit_probability']:.4f} "
                f"prediction={result['prediction']}"
            )

        print_all_metrics(labels, predictions)

    case_ids = args.case_ids

    if not case_ids and not args.all:
        case_ids = ["case_061"]

    for case_id in case_ids or []:
        test_one_case(
            find_case(records, case_id),
            include_clean_control=not args.no_clean_control,
        )


if __name__ == "__main__":
    main()
