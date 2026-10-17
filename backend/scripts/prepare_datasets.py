
import sys
import json
import hashlib
import argparse
from pathlib import Path

from datasets import load_dataset


BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from scope_filter import (
    clean_text,
    analyze_scope,
)

REDNOTE_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "rednote_vibe"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



def make_id(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]


def write_jsonl(path, rows):

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

def read_jsonl(path):

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                yield json.loads(
                    line
                )

def prepare_rednote():

    human_file = (
        REDNOTE_DIR
        / "training_set_human.jsonl"
    )

    ai_file = (
        REDNOTE_DIR
        / "training_set_aigc.jsonl"
    )

    output = []

    configs = [
        (
            human_file,
            0,
            "human"
        ),

        (
            ai_file,
            1,
            "ai"
        ),
    ]

    for (
        path,
        label,
        authorship
    ) in configs:

        content_field = (
            "desc" if authorship == "human"
            else "note_content"
        )

        for index, row in enumerate(
            read_jsonl(path)
        ):

            if content_field not in row:
                raise KeyError(
                    f"{path.name} row {index}: missing {content_field}"
                )

            title = clean_text(
                row.get(
                    "note_title",
                    ""
                )
            )

            content = clean_text(
                row[content_field]
            )

            text = (
                title
                + "\n"
                + content
            ).strip()

            scope = (
                analyze_scope(text)
            )

            if not scope[
                "accepted"
            ]:
                continue

            record = {

                "id":
                    "rednote-"
                    + make_id(
                        f"{path.name}-{index}"
                    ),

                # One original post and its known derivatives must
                # always be assigned to the same split.
                "source_group_id": (
                    "rednote-" + str(row["note_id"])
                    if authorship == "human" and row.get("note_id")
                    else "rednote-" + make_id(f"{path.name}-{index}")
                ),

                "text":
                    text,

                "label":
                    label,

                "authorship":
                    authorship,

                "source":
                    "RedNote-Vibe",

                "domain":
                    "xhs_zhongcao",

                "original_domain":
                    row.get(
                        "domain"
                    ),

                "model_family":
                    row.get(
                        "model_family"
                    ),

                "model":
                    row.get(
                        "model"
                    ),

                "commerce_terms":
    scope[
        "commerce_terms"
    ],

"experience_terms":
    scope[
        "experience_terms"
    ],
                "recommendation_terms":
                    scope[
                        "recommendation_terms"
                    ],
            }

            output.append(
                record
            )

    return output

def prepare_anx():

    dataset = load_dataset(
        "AnxForever/chinese-ai-detection-dataset"
    )

    output = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        if split_name not in dataset:
            continue

        for index, row in enumerate(
            dataset[split_name]
        ):

            text = clean_text(
                row["text"]
            )

            #
            # CRITICAL
            #
            # Their mixed samples may have
            # artificial [SEP] boundary markers.
            #
            # Real users will not have these.
            #
            text = text.replace(
                "[SEP]",
                " "
            )

            scope = (
                analyze_scope(text)
            )

            if not scope[
                "accepted"
            ]:
                continue

            record = {

                "id":
                    f"anx-{split_name}-{index}",

                "source_group_id":
                    f"anx-{split_name}-{index}",

                "text":
                    text,

                "label":
                    int(
                        row["label"]
                    ),

                "authorship":
                    (
                        "human"
                        if int(
                            row["label"]
                        ) == 0
                        else "ai_involved"
                    ),

                "source":
                    "AnxForever",

                "domain":
                    "xhs_zhongcao",

                "category":
                    row.get(
                        "category"
                    ),

                "original_source":
                    row.get(
                        "source"
                    ),

                "original_split":
                    split_name,

                "commerce_terms":
    scope[
        "commerce_terms"
    ],

"experience_terms":
    scope[
        "experience_terms"
    ],

                "recommendation_terms":
                    scope[
                        "recommendation_terms"
                    ],
            }

            output.append(
                record
            )

    return output

def prepare_chasm():

    output = []

    for split_name in [
        "Example",
    ]:

        dataset = load_dataset(
            "Jingyi77/CHASM-Covert_Advertisement_on_RedNote",
            split=split_name,
            streaming=True,
        )

        for row in dataset:

            title = clean_text(
                row.get(
                    "title",
                    ""
                )
            )

            description = clean_text(
                row.get(
                    "description",
                    ""
                )
            )

            text = (
                title
                + "\n"
                + description
            ).strip()

            scope = (
                analyze_scope(text)
            )

            if not scope[
                "accepted"
            ]:
                continue

            record = {

                "id":
                    str(
                        row["id"]
                    ),

                "text":
                    text,

                "comments":
                    row.get(
                        "comments",
                        []
                    ),

                #
                # THIS IS NOT
                # AN AI LABEL.
                #
                "ad_label":
                    int(
                        row["label"]
                    ),

                "source":
                    "CHASM",

                "domain":
                    "xhs_zhongcao",

                "original_split":
                    split_name,

                "image_count":
                    row.get(
                        "image_count",
                        0
                    ),

                "commerce_terms":
    scope[
        "commerce_terms"
    ],

"experience_terms":
    scope[
        "experience_terms"
    ],
                "recommendation_terms":
                    scope[
                        "recommendation_terms"
                    ],
            }

            output.append(
                record
            )

    return output

def prepare_reviewed(path):
    """Import manually verified H/L examples with explicit origin links."""
    output = []
    seen_ids = set()
    groups = {}
    for index, row in enumerate(read_jsonl(path), 1):
        for field in ("id", "source_group_id", "variant", "text", "label"):
            if field not in row or row[field] in (None, ""):
                raise ValueError(f"{path}:{index}: missing {field}")
        if row["id"] in seen_ids:
            raise ValueError(f"{path}:{index}: duplicate id {row['id']}")
        seen_ids.add(row["id"])
        if row["variant"] not in ("H", "L1", "L2", "L3", "L4"):
            raise ValueError(f"{path}:{index}: unsupported variant")
        expected = 0 if row["variant"] == "H" else 1
        if type(row["label"]) is not int or row["label"] != expected:
            raise ValueError(f"{path}:{index}: label must be {expected} for {row['variant']}")
        group = groups.setdefault(str(row["source_group_id"]), set())
        if row["variant"] in group:
            raise ValueError(f"{path}:{index}: duplicate variant within original post")
        group.add(row["variant"])
        record = dict(row)
        record["text"] = clean_text(row["text"])
        record["source_group_id"] = "reviewed-" + str(row["source_group_id"])
        record["source"] = "reviewed-H-L"
        record["authorship"] = "human" if expected == 0 else "ai_involved"
        if not record["text"]:
            raise ValueError(f"{path}:{index}: empty text")
        output.append(record)
    for group_id, variants in groups.items():
        if variants not in ({"H"}, {"H", "L1", "L2", "L3", "L4"}):
            raise ValueError(f"{path}: incomplete H/L1-L4 group {group_id}: {sorted(variants)}")
    return output


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-data", type=Path,
                        help="Verified H/L1-L4 JSONL; keep final evaluation data separate")
    args = parser.parse_args()

    print(
        "Preparing RedNote-Vibe..."
    )

    rednote = prepare_rednote()

    print(
        "Accepted RedNote:",
        len(rednote)
    )


    print(
        "Preparing AnxForever..."
    )

    anx = prepare_anx()

    print(
        "Accepted Anx:",
        len(anx)
    )


    print(
        "Preparing CHASM..."
    )

    chasm = prepare_chasm()

    print(
        "Accepted CHASM:",
        len(chasm)
    )


    #
    # AI authorship training data
    #

    authorship = (
        rednote
        + anx
    )

    if args.reviewed_data:
        reviewed = prepare_reviewed(args.reviewed_data)
        authorship.extend(reviewed)
        print("Verified H/L examples:", len(reviewed),
              "groups:", len({row["source_group_id"] for row in reviewed}))


    write_jsonl(
        OUTPUT_DIR
        / "authorship.jsonl",

        authorship
    )


    #
    # CHASM stays separate
    #

    write_jsonl(
        OUTPUT_DIR
        / "chasm_domain.jsonl",

        chasm
    )


    print()
    print(
        "AI-authorship examples:",
        len(authorship)
    )

    print(
        "CHASM domain examples:",
        len(chasm)
    )


if __name__ == "__main__":
    main()
