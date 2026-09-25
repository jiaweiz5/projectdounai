import json
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = Path("data/processed/chasm_ads.jsonl")
OUTPUT_PATH = Path("data/processed/chasm_ads_clean.jsonl")


def split_priority(split):
    """
    If the same exact text appears in multiple splits,
    keep the evaluation copy rather than the training copy.

    This prevents train -> validation/test leakage.
    """
    if split == "Test":
        return 3
    if split == "Validation":
        return 2
    if split in {"Train_1", "Train_2", "Train_3", "Train_4"}:
        return 1

    return 0


def main():
    rows = []

    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)

                # Normalize surrounding whitespace
                row["text"] = row["text"].strip()

                rows.append(row)

    print("=" * 60)
    print("CHASM CLEANING")
    print("=" * 60)

    print("Original rows:", len(rows))

    # ---------------------------------------------------------
    # Group rows by exact text
    # ---------------------------------------------------------

    groups = defaultdict(list)

    for row in rows:
        groups[row["text"]].append(row)

    print("Unique texts:", len(groups))

    # ---------------------------------------------------------
    # Check for conflicting labels
    # ---------------------------------------------------------

    conflicting_texts = set()

    for text, group in groups.items():
        labels = {row["label"] for row in group}

        if len(labels) > 1:
            conflicting_texts.add(text)

    print("Conflicting-label texts:", len(conflicting_texts))

    if conflicting_texts:
        print("\nWARNING: conflicting labels found.")

        for text in list(conflicting_texts)[:10]:
            print("\nTEXT:")
            print(text[:300])

            for row in groups[text]:
                print(
                    " ",
                    row["id"],
                    row["original_split"],
                    "label=",
                    row["label"],
                )

    # ---------------------------------------------------------
    # Clean duplicates
    # ---------------------------------------------------------

    cleaned = []

    duplicate_groups = 0
    cross_split_duplicates = 0
    removed_duplicates = 0
    removed_conflicts = 0

    for text, group in groups.items():

        # If identical text has contradictory labels,
        # remove all versions because ground truth is ambiguous.
        if text in conflicting_texts:
            removed_conflicts += len(group)
            continue

        if len(group) > 1:
            duplicate_groups += 1

        splits = {row["original_split"] for row in group}

        if len(splits) > 1:
            cross_split_duplicates += 1

        # Prefer Test, then Validation, then Train.
        # Within the same priority, retain the first copy.
        group = sorted(
            group,
            key=lambda row: split_priority(row["original_split"]),
            reverse=True,
        )

        cleaned.append(group[0])

        removed_duplicates += len(group) - 1

    # ---------------------------------------------------------
    # Stable sort for reproducibility
    # ---------------------------------------------------------

    cleaned.sort(
        key=lambda row: (
            row["original_split"],
            row["id"],
        )
    )

    # ---------------------------------------------------------
    # Save cleaned dataset
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in cleaned:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    label_counts = Counter(row["label"] for row in cleaned)
    split_counts = Counter(row["original_split"] for row in cleaned)

    print("\n" + "=" * 60)
    print("CLEANING RESULTS")
    print("=" * 60)

    print("Original rows:", len(rows))
    print("Clean rows:", len(cleaned))
    print("Duplicate groups:", duplicate_groups)
    print("Cross-split duplicate groups:", cross_split_duplicates)
    print("Removed duplicate rows:", removed_duplicates)
    print("Removed conflicting rows:", removed_conflicts)

    print("\nLabels:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print("\nSplits:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\nStep 26 complete.")


if __name__ == "__main__":
    main()