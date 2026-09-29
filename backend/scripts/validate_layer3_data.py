import json
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

NEGATIVE_PATH = DATA_DIR / "layer3_real_negatives.json"
POSITIVE_PATH = DATA_DIR / "layer3_positive.json"


def load_json(path):
    if not path.exists():
        print(f"[ERROR] File does not exist: {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {path.name}: {e}")
        return None


def extract_comment_text(comment):
    """
    Returns the textual content of a comment.

    Supports:
    - plain string comments
    - dictionary comments
    """

    if isinstance(comment, str):
        return comment

    if isinstance(comment, dict):
        # Try common text-field names
        possible_keys = [
            "text",
            "content",
            "comment",
            "comment_text",
            "body",
        ]

        for key in possible_keys:
            value = comment.get(key)

            if isinstance(value, str):
                return value

    return None


def validate_dataset(data, expected_label, name):
    print("\n" + "=" * 65)
    print(f"VALIDATING: {name}")
    print("=" * 65)

    if data is None:
        return False

    if not isinstance(data, list):
        print("[ERROR] Dataset must be a JSON list.")
        return False

    print(f"Examples: {len(data)}")

    if len(data) == 0:
        print("[INFO] Dataset is currently empty.")
        return True

    errors = []
    warnings = []

    ids = []
    labels = []
    comment_counts = []

    string_comment_count = 0
    dict_comment_count = 0

    for index, item in enumerate(data):

        location = f"item {index}"

        if not isinstance(item, dict):
            errors.append(
                f"{location}: must be an object/dictionary"
            )
            continue

        # ---------------------------------------------------------
        # ID validation
        # ---------------------------------------------------------

        item_id = item.get("id")

        if not item_id:
            errors.append(
                f"{location}: missing id"
            )

        elif not isinstance(item_id, str):
            errors.append(
                f"{location}: id must be a string"
            )

        else:
            ids.append(item_id)

        display_id = item_id or location

        # ---------------------------------------------------------
        # Label validation
        # ---------------------------------------------------------

        label = item.get("label")
        labels.append(label)

        if label not in (0, 1):
            errors.append(
                f"{display_id}: invalid label {label!r}; "
                "expected 0 or 1"
            )

        elif label != expected_label:
            errors.append(
                f"{display_id}: label is {label}, "
                f"but expected {expected_label}"
            )

        # ---------------------------------------------------------
        # Comments validation
        # ---------------------------------------------------------

        comments = item.get("comments")

        if comments is None:
            errors.append(
                f"{display_id}: missing comments"
            )
            continue

        if not isinstance(comments, list):
            errors.append(
                f"{display_id}: comments must be a list"
            )
            continue

        comment_counts.append(len(comments))

        if len(comments) == 0:
            errors.append(
                f"{display_id}: comments list is empty"
            )

        elif len(comments) < 3:
            warnings.append(
                f"{display_id}: only {len(comments)} comments"
            )

        # ---------------------------------------------------------
        # Individual comments
        # ---------------------------------------------------------

        seen_text = set()

        for comment_index, comment in enumerate(comments):

            if isinstance(comment, str):
                string_comment_count += 1

            elif isinstance(comment, dict):
                dict_comment_count += 1

            else:
                errors.append(
                    f"{display_id}: comment {comment_index} "
                    f"has unsupported type "
                    f"{type(comment).__name__}"
                )
                continue

            text = extract_comment_text(comment)

            if text is None:
                errors.append(
                    f"{display_id}: comment {comment_index} "
                    "has no recognizable text field"
                )
                continue

            stripped = text.strip()

            if not stripped:
                errors.append(
                    f"{display_id}: comment {comment_index} "
                    "has empty text"
                )
                continue

            if stripped in seen_text:
                warnings.append(
                    f"{display_id}: duplicate comment text detected"
                )

            seen_text.add(stripped)

    # -------------------------------------------------------------
    # Duplicate IDs
    # -------------------------------------------------------------

    id_counts = Counter(ids)

    duplicate_ids = [
        item_id
        for item_id, count in id_counts.items()
        if count > 1
    ]

    for duplicate_id in duplicate_ids:
        errors.append(
            f"Duplicate ID: {duplicate_id}"
        )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print("\nDataset summary")
    print("-" * 65)

    print(f"Unique IDs: {len(set(ids))}")

    if labels:
        print(
            f"Label distribution: {Counter(labels)}"
        )

    if comment_counts:
        print(
            f"Minimum comments: {min(comment_counts)}"
        )

        print(
            f"Maximum comments: {max(comment_counts)}"
        )

        print(
            "Average comments: "
            f"{sum(comment_counts) / len(comment_counts):.2f}"
        )

    print(
        f"String comments: {string_comment_count}"
    )

    print(
        f"Dictionary comments: {dict_comment_count}"
    )

    # -------------------------------------------------------------
    # Warnings
    # -------------------------------------------------------------

    print("\nWarnings")
    print("-" * 65)

    if warnings:

        for warning in warnings[:20]:
            print(
                f"[WARNING] {warning}"
            )

        if len(warnings) > 20:
            print(
                f"...and {len(warnings) - 20} more warnings"
            )

    else:
        print("None")

    # -------------------------------------------------------------
    # Errors
    # -------------------------------------------------------------

    print("\nErrors")
    print("-" * 65)

    if errors:

        for error in errors[:30]:
            print(
                f"[ERROR] {error}"
            )

        if len(errors) > 30:
            print(
                f"...and {len(errors) - 30} more errors"
            )

        print(
            f"\nFAILED: {len(errors)} error(s)"
        )

        return False

    print("\nPASSED")

    return True


def main():

    print("=" * 65)
    print("LAYER 3 DATA VALIDATION")
    print("=" * 65)

    negatives = load_json(
        NEGATIVE_PATH
    )

    positives = load_json(
        POSITIVE_PATH
    )

    negative_ok = validate_dataset(
        negatives,
        expected_label=0,
        name="REAL NEGATIVES",
    )

    positive_ok = validate_dataset(
        positives,
        expected_label=1,
        name="POSITIVE EXAMPLES",
    )

    print("\n" + "=" * 65)
    print("FINAL RESULT")
    print("=" * 65)

    if negative_ok and positive_ok:

        print(
            "Layer 3 datasets are valid."
        )

        if (
            positives is not None
            and len(positives) == 0
        ):

            print(
                "Positive dataset is empty, which is acceptable "
                "while waiting for positive data."
            )

        print("\nStatus:")

        print(
            "Negatives:",
            len(negatives)
            if negatives is not None
            else 0,
        )

        print(
            "Positives:",
            len(positives)
            if positives is not None
            else 0,
        )

    else:

        print(
            "Layer 3 dataset validation FAILED."
        )


if __name__ == "__main__":
    main()