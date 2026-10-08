import json
from pathlib import Path
from statistics import mean, median

import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from layer3_features import extract_layer3_features


DATA_PATH = BASE_DIR / "data" / "layer3_real_negatives.json"


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    index = int((len(values) - 1) * p)

    return values[index]


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_features = []

    for group in data:
        features = extract_layer3_features(group)
        all_features.append(features)

    print("=" * 70)
    print("LAYER 3 NEGATIVE FEATURE ANALYSIS")
    print("=" * 70)

    print(f"Groups analyzed: {len(all_features)}")

    feature_names = list(all_features[0].keys())

    for feature_name in feature_names:
        values = [
            features[feature_name]
            for features in all_features
        ]

        print("\n" + "-" * 70)
        print(feature_name)
        print("-" * 70)

        print(f"Mean:       {mean(values):.4f}")
        print(f"Median:     {median(values):.4f}")
        print(f"Min:        {min(values):.4f}")
        print(f"Max:        {max(values):.4f}")
        print(f"25th pct:   {percentile(values, 0.25):.4f}")
        print(f"75th pct:   {percentile(values, 0.75):.4f}")
        print(f"90th pct:   {percentile(values, 0.90):.4f}")
        print(f"95th pct:   {percentile(values, 0.95):.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()