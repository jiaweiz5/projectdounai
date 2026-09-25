import json
import os
import random

from datasets import load_dataset


DATASET_NAME = "Jingyi77/CHASM-Covert_Advertisement_on_RedNote"

SPLITS = [
    "Train_1",
    "Train_2",
    "Train_3",
    "Train_4",
    "Validation",
    "Test",
]

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "layer3_candidates.json",
)

TARGET_GROUPS = 200
MIN_COMMENTS = 5
MAX_COMMENTS = 30

random.seed(42)


def clean_comment(comment):
    """
    Convert CHASM comment formats into clean text.
    """

    if isinstance(comment, str):
        text = comment

    elif isinstance(comment, dict):
        text = (
            comment.get("text")
            or comment.get("content")
            or comment.get("comment")
            or ""
        )

    else:
        return ""

    text = str(text).strip()

    return text


def clean_comments(comments):
    """
    Clean comments but DO NOT remove duplicates.

    Duplicate comments are useful evidence for Layer 3,
    so we intentionally preserve them.
    """

    if not isinstance(comments, list):
        return []

    cleaned = []

    for comment in comments:

        text = clean_comment(comment)

        if len(text) < 2:
            continue

        cleaned.append(text)

    return cleaned


# --------------------------------------------------
# Reservoir sampling
# --------------------------------------------------

selected = []
eligible_seen = 0

print("=" * 65)
print("LAYER 3 RAW CHASM COMMENT EXTRACTION")
print("=" * 65)


for split in SPLITS:

    print()
    print("=" * 65)
    print(f"Reading split: {split}")
    print("=" * 65)

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
    )

    # Keep only fields potentially useful to Layer 3.
    # This helps avoid unnecessary image handling.
    keep_columns = {
        "id",
        "note_id",
        "post_id",
        "comments",
        "label",
        "title",
        "description",
    }

    existing_columns = set(dataset.column_names or [])

    remove_columns = [
        column
        for column in existing_columns
        if column not in keep_columns
    ]

    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    seen = 0
    eligible = 0

    for index, record in enumerate(dataset):

        seen += 1

        raw_comments = record.get("comments", [])

        comments = clean_comments(raw_comments)

        if len(comments) < MIN_COMMENTS:
            continue

        eligible += 1
        eligible_seen += 1

        # Don't allow giant comment sections to dominate.
        comments = comments[:MAX_COMMENTS]

        original_id = (
            record.get("id")
            or record.get("note_id")
            or record.get("post_id")
            or f"{split}_{index}"
        )

        candidate = {
            "id": None,
            "source": "CHASM",
            "original_split": split,
            "original_post_id": str(original_id),

            # IMPORTANT:
            # This is the original CHASM advertisement label.
            # It is NOT the Layer 3 coordination label.
            "original_ad_label": record.get("label"),

            # Human review later:
            #
            # 0 = normal / independent
            # 1 = coordinated
            # 2 = uncertain / exclude
            "review_label": None,

            "review_notes": "",

            "comments": [
                {
                    "id": f"c{i + 1:02d}",
                    "text": text,
                }
                for i, text in enumerate(comments)
            ],
        }

        # Reservoir sampling gives all eligible records
        # a chance of entering the final 200.
        if len(selected) < TARGET_GROUPS:

            selected.append(candidate)

        else:

            j = random.randint(
                0,
                eligible_seen - 1,
            )

            if j < TARGET_GROUPS:
                selected[j] = candidate

    print(f"Records seen: {seen}")
    print(f"Groups with >= {MIN_COMMENTS} comments: {eligible}")


# --------------------------------------------------
# Assign clean IDs
# --------------------------------------------------

for i, candidate in enumerate(selected):

    candidate["id"] = f"candidate_{i + 1:04d}"


# --------------------------------------------------
# Save
# --------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        selected,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 65)
print("LAYER 3 EXTRACTION COMPLETE")
print("=" * 65)

print(f"Total eligible groups seen: {eligible_seen}")
print(f"Groups selected: {len(selected)}")
print(f"Saved to: {OUTPUT_PATH}")