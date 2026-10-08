import json
from collections import Counter
from pathlib import Path


SOURCE_PATH = Path("data/layer4/layer4_dataset.json")
OUTPUT_PATH = Path("data/layer4/layer4_dataset_prepared.json")


def case_number(case_id: str) -> int:
    return int(case_id.split("_")[-1])


def prepare_record(record: dict) -> dict:
    prepared = record.copy()
    number = case_number(prepared["id"])

    # Cases 001–035: image and caption match.
    if 1 <= number <= 35:
        prepared["task_type"] = "image_text_alignment"
        prepared["labels"] = {
            "image_text_mismatch": 0,
            "image_edited": None,
            "ai_generated": None,
        }
        prepared["source_type"] = "team_created_or_ai_remade"
        prepared["review_status"] = "reviewed_for_development"
        prepared["evaluation_eligible"] = True

    # Cases 036–060: image and caption intentionally do not match.
    elif 36 <= number <= 60:
        prepared["task_type"] = "image_text_alignment"
        prepared["labels"] = {
            "image_text_mismatch": 1,
            "image_edited": None,
            "ai_generated": None,
        }
        prepared["source_type"] = "team_created_or_ai_remade"
        prepared["review_status"] = "reviewed_for_development"
        prepared["evaluation_eligible"] = True

    # Cases 061–085: edited image with original reference image.
    elif 61 <= number <= 85:
        prepared["task_type"] = "reference_image_comparison"
        prepared["labels"] = {
            "image_text_mismatch": None,
            "image_edited": 1,
            "ai_generated": None,
        }
        prepared["source_type"] = "team_created_edited"
        prepared["review_status"] = "reviewed_for_development"
        prepared["evaluation_eligible"] = True

        # Conservative grouping: do not split these reference pairs across
        # future train/test sets until their original-source relationships
        # have been verified.
        prepared["source_group_id"] = "edited_reference_pool"

    # Cases 086–100: experimental AI-generated / visual-anomaly examples.
    elif 86 <= number <= 100:
        prepared["task_type"] = "visual_anomaly_experimental"
        prepared["labels"] = {
            "image_text_mismatch": None,
            "image_edited": None,
            "ai_generated": 1,
        }
        prepared["source_type"] = "team_ai_generated"
        prepared["review_status"] = "needs_manual_review"
        prepared["evaluation_eligible"] = False

    else:
        raise ValueError(f"Unexpected Layer 4 case ID: {prepared['id']}")

    # This data is for development and calibration, not production training.
    prepared["split"] = "development"

    # Normal and mismatch cases can use their own case ID as a temporary group.
    if "source_group_id" not in prepared:
        prepared["source_group_id"] = prepared["id"]

    return prepared


def main():
    with SOURCE_PATH.open(encoding="utf-8") as file:
        raw_data = json.load(file)

    prepared_data = [prepare_record(record) for record in raw_data]

    OUTPUT_PATH.write_text(
        json.dumps(prepared_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    task_counts = Counter(item["task_type"] for item in prepared_data)

    print("=" * 60)
    print("LAYER 4 DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Input cases:  {len(raw_data)}")
    print(f"Output file:  {OUTPUT_PATH}")
    print("\nTask types:")
    for task_type, count in sorted(task_counts.items()):
        print(f"- {task_type}: {count}")


if __name__ == "__main__":
    main()