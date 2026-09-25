import csv
import json
import os


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REVIEWED_CSV = os.path.join(
    BASE_DIR,
    "data",
    "layer3_reviewed.csv",
)

CANDIDATES_JSON = os.path.join(
    BASE_DIR,
    "data",
    "layer3_candidates.json",
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "layer3_real_negatives.json",
)


# --------------------------------------------------
# Load original candidate groups
# --------------------------------------------------

with open(
    CANDIDATES_JSON,
    "r",
    encoding="utf-8",
) as f:
    candidates = json.load(f)


candidate_map = {
    item["id"]: item
    for item in candidates
}


# --------------------------------------------------
# Load reviewed labels
# --------------------------------------------------

reviewed_rows = []

with open(
    REVIEWED_CSV,
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        reviewed_rows.append(row)


# --------------------------------------------------
# Build clean negative dataset
# --------------------------------------------------

real_negatives = []


for row in reviewed_rows:

    review_label = row["review_label"].strip()

    # Keep only confident normal examples
    if review_label != "0":
        continue

    candidate_id = row["candidate_id"]

    if candidate_id not in candidate_map:
        print(
            f"Warning: {candidate_id} not found "
            f"in layer3_candidates.json"
        )
        continue

    candidate = candidate_map[candidate_id].copy()

    candidate["label"] = 0

    candidate["ground_truth_source"] = (
        "human_reviewed_real"
    )

    candidate["review_notes"] = (
        row.get(
            "review_notes",
            "",
        )
    )

    # Remove fields that are no longer needed
    candidate.pop(
        "review_label",
        None,
    )

    real_negatives.append(candidate)


# --------------------------------------------------
# Save
# --------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        real_negatives,
        f,
        ensure_ascii=False,
        indent=2,
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("=" * 65)
print("LAYER 3 REAL NEGATIVES BUILT")
print("=" * 65)

print(
    f"Reviewed rows: "
    f"{len(reviewed_rows)}"
)

print(
    f"Confident real negatives: "
    f"{len(real_negatives)}"
)

print()

print(
    "Saved to:"
)

print(
    OUTPUT_PATH
)