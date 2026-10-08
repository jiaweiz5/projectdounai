import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

NEGATIVE_PATH = DATA_DIR / "layer3_real_negatives.json"
POSITIVE_PATH = DATA_DIR / "layer3_positive.json"

OUTPUT_PATH = DATA_DIR / "layer3_dataset.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    negatives = load_json(NEGATIVE_PATH)
    positives = load_json(POSITIVE_PATH)

    print("=" * 65)
    print("PREPARING LAYER 3 DATASET")
    print("=" * 65)

    print(f"Negatives: {len(negatives)}")
    print(f"Positives: {len(positives)}")

    combined = negatives + positives

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            combined,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nCombined examples: {len(combined)}")
    print(f"Saved to: {OUTPUT_PATH}")

    if len(positives) == 0:
        print(
            "\nWARNING: Positive dataset is empty."
        )
        print(
            "Dataset has been prepared, but DO NOT train Layer 3 yet."
        )


if __name__ == "__main__":
    main()