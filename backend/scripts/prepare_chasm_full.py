import json
from pathlib import Path

from datasets import load_dataset

from scope_filter import (
    clean_text,
    analyze_scope,
)


BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "chasm_ads.jsonl"
)


# CHASM training shards
SPLITS = [
    "Train_1",
    "Train_2",
    "Train_3",
    "Train_4",
    "Validation",
    "Test",
]


def combine_post_text(example):
    """
    Combine title + description.

    We deliberately do not include images yet.
    Comments can be handled separately later.
    """

    title = (
        example.get("title")
        or ""
    )

    description = (
        example.get("description")
        or ""
    )

    text = f"{title}\n{description}"

    return clean_text(text)


def main():

    total_seen = 0
    total_saved = 0

    counts_by_split = {}

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output:

        for split in SPLITS:

            print()
            print("=" * 60)
            print(
                f"Processing CHASM split: {split}"
            )
            print("=" * 60)

            dataset = load_dataset(
                "Jingyi77/CHASM-Covert_Advertisement_on_RedNote",
                split=split,
                streaming=True,
            )

            split_seen = 0
            split_saved = 0

            for example in dataset:

                total_seen += 1
                split_seen += 1

                text = combine_post_text(
                    example
                )

                if not text:
                    continue

                # Keep Xiaohongshu recommendation-type content.
                scope = analyze_scope(
                    text
                )

                if not scope["accepted"]:
                    continue

                label = example.get(
                    "label"
                )

                if label is None:
                    continue

                record = {
                    "id":
                        str(
                            example.get(
                                "id",
                                ""
                            )
                        ),

                    "text":
                        text,

                    "label":
                        int(label),

                    "source":
                        "CHASM",

                    "original_split":
                        split,

                    "domain":
                        "xhs_zhongcao",

                    "task":
                        "covert_ad_detection",

                    "scope_score":
                        scope["score"],

                    "recommendation_terms":
                        scope[
                            "recommendation_terms"
                        ],

                    "commerce_terms":
                        scope[
                            "commerce_terms"
                        ],

                    "experience_terms":
                        scope[
                            "experience_terms"
                        ],
                }

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                total_saved += 1
                split_saved += 1

            counts_by_split[split] = {
                "seen":
                    split_seen,

                "saved":
                    split_saved,
            }

            print(
                f"Seen: {split_seen}"
            )

            print(
                f"Saved: {split_saved}"
            )


    print()
    print("=" * 60)
    print("CHASM PREPARATION COMPLETE")
    print("=" * 60)

    print(
        "Total seen:",
        total_seen
    )

    print(
        "Total saved:",
        total_saved
    )

    print()

    for split, counts in (
        counts_by_split.items()
    ):

        print(
            split,
            counts
        )

    print()
    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()