import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from layer4_features import Layer4OCRError, extract_ocr_from_path


DATASET_PATH = BASE_DIR / "data" / "layer4" / "layer4_dataset.json"

CASE_IDS = [
    "case_001",
    "case_036",
    "case_061",
    "case_086",
]


def main():
    with DATASET_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    records = {item["id"]: item for item in data}
    failures = []

    print("=" * 60)
    print("LAYER 4 OCR SMOKE TEST")
    print("=" * 60)

    for case_id in CASE_IDS:
        record = records[case_id]

        try:
            result = extract_ocr_from_path(
                BASE_DIR / record["image_path"]
            )
        except Layer4OCRError as exc:
            failures.append(case_id)
            print(f"\n{case_id}: FAILED")
            print(f"Reason: {exc}")
            continue

        print(f"\n{case_id}")
        print(f"Issue type: {record['issue_type']}")
        print(f"OCR items:  {result['ocr_item_count']}")
        print(f"Confidence: {result['ocr_confidence']}")
        print("OCR text:")
        print(result["ocr_text"] or "(No readable text detected)")

    print("\n" + "=" * 60)

    if failures:
        print(f"OCR TEST FAILED: {', '.join(failures)}")
        sys.exit(1)

    print("OCR TEST PASSED")


if __name__ == "__main__":
    main()