import json
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from layer3_features import extract_layer3_features


DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "layer3_dataset.json"
OUTPUT_PATH = DATA_DIR / "layer3_features.json"


def main():
    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    output = []

    for group in data:
        features = extract_layer3_features(group)

        output.append(
            {
                "id": group["id"],
                "label": group["label"],
                "features": features,
            }
        )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 65)
    print("LAYER 3 FEATURE DATASET CREATED")
    print("=" * 65)

    print(f"Examples: {len(output)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()