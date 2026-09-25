import json
from pathlib import Path
from collections import Counter


INPUT_PATH = Path("data/processed/chasm_ads_clean.jsonl")

TRAIN_PATH = Path("data/processed/chasm_train.jsonl")
VAL_PATH = Path("data/processed/chasm_validation.jsonl")
TEST_PATH = Path("data/processed/chasm_test.jsonl")


TRAIN_SPLITS = {
    "Train_1",
    "Train_2",
    "Train_3",
    "Train_4",
}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


def print_stats(name, rows):
    labels = Counter(row["label"] for row in rows)

    print(f"\n{name}")
    print("-" * 40)
    print("Rows:", len(rows))

    for label in sorted(labels):
        count = labels[label]
        pct = count / len(rows) * 100 if rows else 0

        print(
            f"Label {label}: "
            f"{count} ({pct:.2f}%)"
        )


def main():

    rows = []

    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    train_rows = []
    val_rows = []
    test_rows = []

    unknown = []

    for row in rows:

        split = row["original_split"]

        if split in TRAIN_SPLITS:
            train_rows.append(row)

        elif split == "Validation":
            val_rows.append(row)

        elif split == "Test":
            test_rows.append(row)

        else:
            unknown.append(row)

    if unknown:
        print("WARNING: unknown splits found:")

        for row in unknown[:10]:
            print(
                row["id"],
                row["original_split"]
            )

        raise ValueError(
            "Dataset contains unexpected split names."
        )

    # --------------------------------------------------
    # Leakage checks
    # --------------------------------------------------

    train_texts = {row["text"] for row in train_rows}
    val_texts = {row["text"] for row in val_rows}
    test_texts = {row["text"] for row in test_rows}

    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts

    print("=" * 60)
    print("CHASM SPLIT CREATION")
    print("=" * 60)

    print_stats("TRAIN", train_rows)
    print_stats("VALIDATION", val_rows)
    print_stats("TEST", test_rows)

    print("\n" + "=" * 60)
    print("LEAKAGE CHECK")
    print("=" * 60)

    print(
        "Train <-> Validation overlap:",
        len(train_val_overlap)
    )

    print(
        "Train <-> Test overlap:",
        len(train_test_overlap)
    )

    print(
        "Validation <-> Test overlap:",
        len(val_test_overlap)
    )

    assert not train_val_overlap
    assert not train_test_overlap
    assert not val_test_overlap

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(VAL_PATH, val_rows)
    write_jsonl(TEST_PATH, test_rows)

    print("\nSaved:")

    print(TRAIN_PATH)
    print(VAL_PATH)
    print(TEST_PATH)

    print("\nPASS: no exact-text leakage between splits.")
    print("Step 27 complete.")


if __name__ == "__main__":
    main()