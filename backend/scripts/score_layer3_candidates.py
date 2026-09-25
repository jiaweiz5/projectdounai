import csv
import json
import os
import sys


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

from comment_detector import analyze_comment_coordination


INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "layer3_candidates.json",
)

OUTPUT_JSON = os.path.join(
    BASE_DIR,
    "data",
    "layer3_scored_candidates.json",
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "data",
    "layer3_review.csv",
)


# --------------------------------------------------
# Load candidates
# --------------------------------------------------

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
) as f:

    candidates = json.load(f)


print("=" * 65)
print("SCORING LAYER 3 CHASM CANDIDATES")
print("=" * 65)

print(f"Candidates loaded: {len(candidates)}")


# --------------------------------------------------
# Score candidates
# --------------------------------------------------

scored = []


for candidate in candidates:

    result = analyze_comment_coordination(
        candidate["comments"]
    )

    scored_candidate = candidate.copy()

    scored_candidate["coordination_score"] = (
        result["coordination_score"]
    )

    scored_candidate["risk"] = (
        result["risk"]
    )

    scored_candidate["max_similarity"] = (
        result.get(
            "max_similarity",
            0.0,
        )
    )

    scored_candidate["exact_duplicate_groups"] = (
        result.get(
            "exact_duplicate_groups",
            [],
        )
    )

    scored_candidate["similar_pairs"] = (
        result.get(
            "similar_pairs",
            [],
        )
    )

    scored_candidate["similarity_clusters"] = (
        result.get(
            "similarity_clusters",
            [],
        )
    )

    scored_candidate["evidence"] = (
        result.get(
            "evidence",
            [],
        )
    )

    scored.append(
        scored_candidate
    )


# --------------------------------------------------
# Sort from most suspicious to least suspicious
# --------------------------------------------------

scored.sort(
    key=lambda x: x["coordination_score"],
    reverse=True,
)


# --------------------------------------------------
# Save full JSON
# --------------------------------------------------

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        scored,
        f,
        ensure_ascii=False,
        indent=2,
    )


# --------------------------------------------------
# Save review-friendly CSV
# --------------------------------------------------

fieldnames = [
    "candidate_id",
    "original_split",
    "original_post_id",
    "original_ad_label",
    "comment_count",
    "coordination_score",
    "risk",
    "max_similarity",
    "exact_duplicate_groups",
    "similarity_clusters",
    "similar_pairs",
    "comments",
    "review_label",
    "review_notes",
]


with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for candidate in scored:

        comments_text = "\n".join(
            f'{comment["id"]}: {comment["text"]}'
            for comment in candidate["comments"]
        )

        writer.writerow(
            {
                "candidate_id": candidate["id"],

                "original_split": candidate.get(
                    "original_split",
                    "",
                ),

                "original_post_id": candidate.get(
                    "original_post_id",
                    "",
                ),

                "original_ad_label": candidate.get(
                    "original_ad_label",
                    "",
                ),

                "comment_count": len(
                    candidate["comments"]
                ),

                "coordination_score": candidate[
                    "coordination_score"
                ],

                "risk": candidate["risk"],

                "max_similarity": candidate[
                    "max_similarity"
                ],

                "exact_duplicate_groups": json.dumps(
                    candidate[
                        "exact_duplicate_groups"
                    ],
                    ensure_ascii=False,
                ),

                "similarity_clusters": json.dumps(
                    candidate[
                        "similarity_clusters"
                    ],
                    ensure_ascii=False,
                ),

                "similar_pairs": json.dumps(
                    candidate[
                        "similar_pairs"
                    ],
                    ensure_ascii=False,
                ),

                "comments": comments_text,

                # Your team fills these later
                "review_label": "",
                "review_notes": "",
            }
        )


# --------------------------------------------------
# Summary statistics
# --------------------------------------------------

high = sum(
    1
    for x in scored
    if x["risk"] == "high"
)

medium = sum(
    1
    for x in scored
    if x["risk"] == "medium"
)

low = sum(
    1
    for x in scored
    if x["risk"] == "low"
)


print()
print("=" * 65)
print("SCORING COMPLETE")
print("=" * 65)

print(f"Groups scored: {len(scored)}")
print()
print(f"High risk:   {high}")
print(f"Medium risk: {medium}")
print(f"Low risk:    {low}")


print()
print("=" * 65)
print("TOP 20 HIGHEST SCORES")
print("=" * 65)


for candidate in scored[:20]:

    print(
        f'{candidate["id"]:16} '
        f'score={candidate["coordination_score"]:.4f} '
        f'risk={candidate["risk"]:6} '
        f'comments={len(candidate["comments"])}'
    )


print()
print("Full scored JSON:")
print(OUTPUT_JSON)

print()
print("Review CSV:")
print(OUTPUT_CSV)