import json
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))

from layer3_features import extract_layer3_features


DATA_PATH = (
    BASE_DIR
    / "data"
    / "layer3_real_negatives.json"
)


def main():

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    print("=" * 65)
    print("LAYER 3 FEATURE EXTRACTION TEST")
    print("=" * 65)

    print(f"Groups loaded: {len(data)}")

    print("\nFirst 10 groups:")
    print("-" * 65)

    for group in data[:10]:

        features = extract_layer3_features(
            group
        )

        print(
            f"\n{group['id']}"
        )

        for name, value in features.items():
            print(
                f"  {name:24s}: {value}"
            )

    print("\n" + "=" * 65)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()