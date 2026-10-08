"""Learned semantic features for Layer 3 comment coordination detection."""

import os
import re
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer


TEXT_KEYS = (
    "text",
    "content",
    "comment",
    "comment_text",
    "body",
)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# Every value returned by extract_layer3_features().
LAYER3_FEATURE_NAMES = [
    "comment_count",
    "duplicate_ratio",
    "max_exact_repeat_ratio",
    "semantic_pair_mean",
    "semantic_pair_std",
    "semantic_pair_p75",
    "semantic_pair_p90",
    "semantic_pair_max",
    "semantic_top_pair_mean",
    "semantic_nearest_mean",
    "semantic_nearest_std",
    "semantic_centroid_mean",
    "semantic_centroid_std",
    "semantic_component_concentration",
]

# Only these coordination-focused values are supplied to the classifier.
# Comment count is returned for reporting but deliberately excluded here.
LAYER3_MODEL_FEATURE_NAMES = [
    "duplicate_ratio",
    "max_exact_repeat_ratio",
    "semantic_pair_mean",
    "semantic_pair_std",
    "semantic_pair_p75",
    "semantic_pair_p90",
    "semantic_pair_max",
    "semantic_top_pair_mean",
    "semantic_nearest_mean",
    "semantic_nearest_std",
    "semantic_centroid_mean",
    "semantic_centroid_std",
    "semantic_component_concentration",
]


_embedding_model = None


def get_embedding_model():
    """Load the sentence-embedding model once per Python process."""

    global _embedding_model

    if _embedding_model is None:
        model_name = os.getenv(
            "LAYER3_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        )
        device = os.getenv("LAYER3_EMBEDDING_DEVICE") or None

        _embedding_model = SentenceTransformer(
            model_name,
            device=device,
        )

    return _embedding_model


def extract_comment_text(comment):
    """Extract text from a string or a supported comment dictionary."""

    if isinstance(comment, str):
        return comment.strip()

    if isinstance(comment, dict):
        for key in TEXT_KEYS:
            value = comment.get(key)

            if isinstance(value, str):
                return value.strip()

    return ""


def normalize_text(text):
    """Normalize only formatting noise for exact-duplicate comparison."""

    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(
        r"[，。！？、；：,.!?;:'\"“”‘’（）()\[\]{}~～]",
        "",
        text,
    )
    return text


def empty_features():
    """Return a complete all-zero feature dictionary."""

    return {
        name: 0.0
        for name in LAYER3_FEATURE_NAMES
    }


def _safe_round(value):
    return round(float(value), 6)


def _semantic_features(texts):
    """Create continuous semantic-coordination features from embeddings."""

    total = len(texts)

    if total < 2:
        return {
            "semantic_pair_mean": 0.0,
            "semantic_pair_std": 0.0,
            "semantic_pair_p75": 0.0,
            "semantic_pair_p90": 0.0,
            "semantic_pair_max": 0.0,
            "semantic_top_pair_mean": 0.0,
            "semantic_nearest_mean": 0.0,
            "semantic_nearest_std": 0.0,
            "semantic_centroid_mean": 0.0,
            "semantic_centroid_std": 0.0,
            "semantic_component_concentration": 0.0,
        }

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=min(32, total),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    similarity_matrix = embeddings @ embeddings.T
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

    upper_indices = np.triu_indices(total, k=1)
    pair_values = similarity_matrix[upper_indices]

    top_count = max(1, int(np.ceil(len(pair_values) * 0.20)))
    top_pair_values = np.partition(
        pair_values,
        len(pair_values) - top_count,
    )[-top_count:]

    nearest_matrix = similarity_matrix.copy()
    np.fill_diagonal(nearest_matrix, -np.inf)
    nearest_values = np.max(nearest_matrix, axis=1)

    centroid = embeddings.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)

    if centroid_norm > 0:
        centroid = centroid / centroid_norm
        centroid_values = embeddings @ centroid
    else:
        centroid_values = np.zeros(total, dtype=np.float32)

    singular_values = np.linalg.svd(
        embeddings,
        compute_uv=False,
    )
    singular_energy = np.square(singular_values)
    total_energy = float(np.sum(singular_energy))
    component_concentration = (
        float(singular_energy[0] / total_energy)
        if total_energy > 0
        else 0.0
    )

    return {
        "semantic_pair_mean": _safe_round(np.mean(pair_values)),
        "semantic_pair_std": _safe_round(np.std(pair_values)),
        "semantic_pair_p75": _safe_round(np.percentile(pair_values, 75)),
        "semantic_pair_p90": _safe_round(np.percentile(pair_values, 90)),
        "semantic_pair_max": _safe_round(np.max(pair_values)),
        "semantic_top_pair_mean": _safe_round(np.mean(top_pair_values)),
        "semantic_nearest_mean": _safe_round(np.mean(nearest_values)),
        "semantic_nearest_std": _safe_round(np.std(nearest_values)),
        "semantic_centroid_mean": _safe_round(np.mean(centroid_values)),
        "semantic_centroid_std": _safe_round(np.std(centroid_values)),
        "semantic_component_concentration": _safe_round(
            component_concentration
        ),
    }


def extract_layer3_features(group):
    """Extract learned group-level coordination features."""

    comments = group.get("comments", [])
    texts = []

    for comment in comments:
        text = extract_comment_text(comment)

        if text:
            texts.append(text)

    total_comments = len(texts)

    if total_comments == 0:
        return empty_features()

    normalized = [normalize_text(text) for text in texts]
    exact_counts = Counter(normalized)

    duplicate_count = sum(
        count - 1
        for count in exact_counts.values()
        if count > 1
    )

    duplicate_ratio = duplicate_count / total_comments
    max_exact_repeat_ratio = max(exact_counts.values()) / total_comments

    features = {
        "comment_count": total_comments,
        "duplicate_ratio": _safe_round(duplicate_ratio),
        "max_exact_repeat_ratio": _safe_round(max_exact_repeat_ratio),
    }
    features.update(_semantic_features(texts))

    return features
