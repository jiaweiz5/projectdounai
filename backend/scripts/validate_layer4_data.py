import json
import sys
from pathlib import Path


ROOT = Path("data/layer4")
DATASET_PATH = ROOT / "layer4_dataset.json"


def main():
    errors = []

    if not DATASET_PATH.is_file():
        print(f"ERROR: Dataset file not found: {DATASET_PATH}")
        sys.exit(1)

    with DATASET_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        print("ERROR: layer4_dataset.json must contain a JSON list.")
        sys.exit(1)

    ids = [item.get("id") for item in data]
    duplicate_ids = sorted(
        item_id for item_id in set(ids)
        if ids.count(item_id) > 1
    )

    if duplicate_ids:
        errors.append(f"Duplicate IDs: {duplicate_ids}")

    missing_images = []
    missing_references = []

    for item in data:
        case_id = item.get("id", "unknown")

        image_path = item.get("image_path")
        if not image_path:
            errors.append(f"{case_id}: missing image_path")
        elif not Path(image_path).is_file():
            missing_images.append(case_id)

        reference_path = item.get("reference_image_path")
        if reference_path and not Path(reference_path).is_file():
            missing_references.append(case_id)

    if missing_images:
        errors.append(
            f"Missing detection images for: {', '.join(missing_images)}"
        )

    if missing_references:
        errors.append(
            f"Missing reference images for: {', '.join(missing_references)}"
        )

    image_count = len(list((ROOT / "images").glob("*.jpg")))
    reference_count = len(list((ROOT / "references").glob("*.jpg")))

    print("=" * 60)
    print("LAYER 4 DATASET VALIDATION")
    print("=" * 60)
    print(f"Cases:            {len(data)}")
    print(f"Detection images: {image_count}")
    print(f"Reference images: {reference_count}")
    print(f"Unique IDs:       {len(set(ids))}")
    print(f"README included:  {(ROOT / 'README.md').exists()}")

    if errors:
        print("\nVALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    print("\nVALIDATION PASSED")
    print("All required Layer 4 paths and IDs are valid.")


if __name__ == "__main__":
    main()