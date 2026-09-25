import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "authorship.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# READ / WRITE
# ============================================================

def read_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


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


# ============================================================
# NORMALIZATION FOR DUPLICATE CHECKING
# ============================================================

def canonical_text(text):

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        "",
        text
    )

    text = re.sub(
        r"[^\w\u4e00-\u9fff]",
        "",
        text
    )

    return text


# ============================================================
# UNION FIND
# Used to keep near-duplicates in the same split
# ============================================================

class UnionFind:

    def __init__(self, n):

        self.parent = list(
            range(n)
        )

    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:

            self.parent[rb] = ra


# ============================================================
# LOAD
# ============================================================

rows = read_jsonl(
    INPUT_FILE
)

print(
    "Original rows:",
    len(rows)
)


# ============================================================
# EXACT DEDUPLICATION
# ============================================================

by_text = defaultdict(list)

for row in rows:

    key = canonical_text(
        row["text"]
    )

    by_text[key].append(
        row
    )


deduped = []
conflicting = 0

for key, group in by_text.items():

    labels = {
        row["label"]
        for row in group
    }

    # If the exact same text has conflicting labels,
    # don't let the model train on contradictory truth.
    if len(labels) > 1:

        conflicting += len(group)

        continue

    deduped.append(
        group[0]
    )


rows = deduped

print(
    "After exact deduplication:",
    len(rows)
)

print(
    "Dropped conflicting duplicates:",
    conflicting
)


# ============================================================
# NEAR-DUPLICATE GROUPING
# ============================================================

print()
print(
    "Finding near-duplicate groups..."
)

texts = [
    row["text"]
    for row in rows
]


vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
)

X = vectorizer.fit_transform(
    texts
)


neighbors = NearestNeighbors(
    n_neighbors=6,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1,
)

neighbors.fit(X)

distances, indices = (
    neighbors.kneighbors(X)
)


uf = UnionFind(
    len(rows)
)


# cosine distance <= 0.08
# roughly means similarity >= 0.92

for i in range(
    len(rows)
):

    for distance, j in zip(
        distances[i],
        indices[i]
    ):

        if i == j:
            continue

        if distance <= 0.08:

            uf.union(
                i,
                int(j)
            )


groups = defaultdict(list)

for i in range(
    len(rows)
):

    root = uf.find(i)

    groups[root].append(i)


print(
    "Near-duplicate groups:",
    len(groups)
)


# ============================================================
# CREATE GROUP-LEVEL LABEL
# ============================================================

group_ids = []
group_labels = []


for group_id, members in groups.items():

    labels = [
        rows[index]["label"]
        for index in members
    ]

    majority_label = (
        Counter(labels)
        .most_common(1)[0][0]
    )

    group_ids.append(
        group_id
    )

    group_labels.append(
        majority_label
    )


# ============================================================
# SPLIT GROUPS
#
# FIT          50%
# DEV          15%
# CALIBRATION  15%
# TEST         20%
# ============================================================

fit_groups, remainder_groups = (
    train_test_split(
        group_ids,
        test_size=0.50,
        random_state=42,
        stratify=group_labels,
    )
)


remainder_labels = []

group_label_map = dict(
    zip(
        group_ids,
        group_labels
    )
)

for group_id in remainder_groups:

    remainder_labels.append(
        group_label_map[group_id]
    )


dev_groups, temp_groups = (
    train_test_split(
        remainder_groups,
        test_size=0.70,
        random_state=42,
        stratify=remainder_labels,
    )
)


temp_labels = [
    group_label_map[g]
    for g in temp_groups
]


calibration_groups, test_groups = (
    train_test_split(
        temp_groups,
        test_size=4 / 7,
        random_state=42,
        stratify=temp_labels,
    )
)


# ============================================================
# CONVERT GROUP SPLITS BACK INTO ROWS
# ============================================================

def rows_for_groups(
    selected_groups
):

    selected_groups = set(
        selected_groups
    )

    output = []

    for group_id in selected_groups:

        for index in groups[
            group_id
        ]:

            output.append(
                rows[index]
            )

    return output


fit_rows = rows_for_groups(
    fit_groups
)

dev_rows = rows_for_groups(
    dev_groups
)

calibration_rows = rows_for_groups(
    calibration_groups
)

test_rows = rows_for_groups(
    test_groups
)


# ============================================================
# WRITE
# ============================================================

write_jsonl(
    OUTPUT_DIR / "fit.jsonl",
    fit_rows
)

write_jsonl(
    OUTPUT_DIR / "dev.jsonl",
    dev_rows
)

write_jsonl(
    OUTPUT_DIR / "calibration.jsonl",
    calibration_rows
)

write_jsonl(
    OUTPUT_DIR / "test_locked.jsonl",
    test_rows
)


# ============================================================
# SUMMARY
# ============================================================

def show_stats(
    name,
    data
):

    print()
    print(
        "=" * 60
    )

    print(name)

    print(
        "Rows:",
        len(data)
    )

    print(
        "Labels:",
        Counter(
            r["label"]
            for r in data
        )
    )

    print(
        "Sources:",
        Counter(
            r["source"]
            for r in data
        )
    )


show_stats(
    "FIT",
    fit_rows
)

show_stats(
    "DEV",
    dev_rows
)

show_stats(
    "CALIBRATION",
    calibration_rows
)

show_stats(
    "LOCKED TEST",
    test_rows
)


print()
print(
    "Saved to:",
    OUTPUT_DIR
)