import re
import unicodedata
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SIMILARITY_THRESHOLD = 0.78
MIN_COMMENT_LENGTH = 4


def normalize_comment(text: str) -> str:
    """
    Normalize a comment so trivial formatting differences
    do not prevent duplicate detection.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()

    # Remove whitespace
    text = re.sub(r"\s+", "", text)

    # Remove punctuation while preserving Chinese,
    # letters, numbers and underscores.
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)

    return text


def _build_clusters(similarity_matrix, threshold):
    """
    Build connected groups of comments whose similarity
    exceeds the chosen threshold.
    """
    n = len(similarity_matrix)
    graph = defaultdict(set)

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= threshold:
                graph[i].add(j)
                graph[j].add(i)

    visited = set()
    clusters = []

    for node in range(n):
        if node in visited or node not in graph:
            continue

        stack = [node]
        cluster = []

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            cluster.append(current)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

        if len(cluster) >= 2:
            clusters.append(sorted(cluster))

    return clusters


def analyze_comment_coordination(comments):
    """
    Analyze whether a group of comments contains signs
    of coordination, copying, or repeated templates.

    comments format:

    [
        {
            "id": "c01",
            "text": "这个真的太好用了！",
            "timestamp": None
        }
    ]
    """

    if not comments:
        return {
            "coordination_score": 0.0,
            "risk": "low",
            "comment_count": 0,
            "valid_comment_count": 0,
            "exact_duplicate_groups": [],
            "similarity_clusters": [],
            "max_similarity": 0.0,
            "evidence": [],
        }

    prepared = []

    for index, comment in enumerate(comments):
        text = comment.get("text", "")
        normalized = normalize_comment(text)

        if len(normalized) < MIN_COMMENT_LENGTH:
            continue

        prepared.append(
            {
                "original_index": index,
                "id": comment.get("id", f"comment_{index}"),
                "text": text,
                "normalized": normalized,
            }
        )

    n = len(prepared)

    if n < 2:
        return {
            "coordination_score": 0.0,
            "risk": "low",
            "comment_count": len(comments),
            "valid_comment_count": n,
            "exact_duplicate_groups": [],
            "similarity_clusters": [],
            "max_similarity": 0.0,
            "evidence": [
                "Not enough usable comments to evaluate coordination."
            ],
        }

    # --------------------------------------------------
    # 1. Exact duplicate detection
    # --------------------------------------------------

    exact_map = defaultdict(list)

    for item in prepared:
        exact_map[item["normalized"]].append(item["id"])

    exact_duplicate_groups = [
        ids for ids in exact_map.values()
        if len(ids) >= 2
    ]

    exact_duplicate_ids = set()

    for group in exact_duplicate_groups:
        exact_duplicate_ids.update(group)

    # --------------------------------------------------
    # 2. Character TF-IDF
    # --------------------------------------------------

    texts = [item["normalized"] for item in prepared]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(texts)

    similarities = cosine_similarity(matrix)

    # Ignore self-similarity
    np.fill_diagonal(similarities, 0.0)

    max_similarity = float(np.max(similarities))

    # --------------------------------------------------
    # 3. Near-duplicate pairs
    # --------------------------------------------------

    similar_pairs = []

    for i in range(n):
        for j in range(i + 1, n):

            score = float(similarities[i][j])

            if score >= SIMILARITY_THRESHOLD:
                similar_pairs.append(
                    {
                        "comment_1": prepared[i]["id"],
                        "comment_2": prepared[j]["id"],
                        "similarity": round(score, 4),
                    }
                )

    # --------------------------------------------------
    # 4. Similarity clusters
    # --------------------------------------------------

    raw_clusters = _build_clusters(
        similarities,
        SIMILARITY_THRESHOLD,
    )

    similarity_clusters = []

    coordinated_ids = set()

    for cluster in raw_clusters:

        ids = [prepared[i]["id"] for i in cluster]

        coordinated_ids.update(ids)

        similarity_clusters.append(ids)

    # --------------------------------------------------
    # 5. Calculate heuristic coordination score
    # --------------------------------------------------

    total_pairs = n * (n - 1) / 2

    exact_duplicate_ratio = (
        len(exact_duplicate_ids) / n
        if n else 0
    )

    similar_comment_ratio = (
        len(coordinated_ids) / n
        if n else 0
    )

    largest_cluster_ratio = (
        max(
            [len(cluster) for cluster in similarity_clusters],
            default=0,
        ) / n
    )

    similar_pair_ratio = (
        len(similar_pairs) / total_pairs
        if total_pairs else 0
    )

    coordination_score = (
        0.35 * exact_duplicate_ratio
        + 0.30 * similar_comment_ratio
        + 0.20 * largest_cluster_ratio
        + 0.15 * similar_pair_ratio
    )

    coordination_score = min(
        max(coordination_score, 0.0),
        1.0,
    )

    # These are initial heuristic thresholds.
    # We will calibrate them using Layer 3 evaluation data.
    if coordination_score >= 0.60:
        risk = "high"
    elif coordination_score >= 0.25:
        risk = "medium"
    else:
        risk = "low"

    # --------------------------------------------------
    # 6. Human-readable evidence
    # --------------------------------------------------

    evidence = []

    if exact_duplicate_groups:
        evidence.append(
            f"Found {len(exact_duplicate_groups)} exact duplicate comment group(s)."
        )

    if similar_pairs:
        evidence.append(
            f"Found {len(similar_pairs)} highly similar comment pair(s)."
        )

    if similarity_clusters:
        largest = max(len(x) for x in similarity_clusters)

        evidence.append(
            f"Largest suspicious similarity cluster contains {largest} comments."
        )

    if not evidence:
        evidence.append(
            "No strong comment coordination signals detected."
        )

    return {
        "coordination_score": round(coordination_score, 4),
        "risk": risk,
        "comment_count": len(comments),
        "valid_comment_count": n,
        "exact_duplicate_groups": exact_duplicate_groups,
        "similar_pairs": similar_pairs,
        "similarity_clusters": similarity_clusters,
        "max_similarity": round(max_similarity, 4),
        "evidence": evidence,
    }