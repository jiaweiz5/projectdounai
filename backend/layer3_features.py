import re
from collections import Counter
from statistics import mean


TEXT_KEYS = [
    "text",
    "content",
    "comment",
    "comment_text",
    "body",
]


def extract_comment_text(comment):
    """
    Extract comment text from either:
    - a plain string
    - a dictionary
    """

    if isinstance(comment, str):
        return comment.strip()

    if isinstance(comment, dict):
        for key in TEXT_KEYS:
            value = comment.get(key)

            if isinstance(value, str):
                return value.strip()

    return ""


def normalize_text(text):
    """
    Basic normalization used for duplicate/similarity features.
    """

    text = text.strip().lower()

    # Remove whitespace
    text = re.sub(r"\s+", "", text)

    # Remove common punctuation
    text = re.sub(
        r"[，。！？、；：,.!?;:'\"“”‘’（）()\[\]{}]",
        "",
        text,
    )

    return text


def extract_layer3_features(group):
    """
    Extract comment-section behavioral/text features
    from one Layer 3 group.

    Returns a dictionary of numerical features.
    """

    comments = group.get("comments", [])

    texts = []

    for comment in comments:
        text = extract_comment_text(comment)

        if text:
            texts.append(text)

    total_comments = len(texts)

    # Avoid divide-by-zero
    if total_comments == 0:
        return {
            "comment_count": 0,
            "unique_comment_ratio": 0.0,
            "duplicate_ratio": 0.0,
            "avg_comment_length": 0.0,
            "min_comment_length": 0,
            "max_comment_length": 0,
            "short_comment_ratio": 0.0,
            "emoji_like_ratio": 0.0,
            "exclamation_ratio": 0.0,
            "question_ratio": 0.0,
        }

    # ---------------------------------------------------------
    # Basic text statistics
    # ---------------------------------------------------------

    lengths = [len(text) for text in texts]

    avg_length = mean(lengths)
    min_length = min(lengths)
    max_length = max(lengths)

    # ---------------------------------------------------------
    # Duplicate / repeated comment behavior
    # ---------------------------------------------------------

    normalized = [
        normalize_text(text)
        for text in texts
        if normalize_text(text)
    ]

    counts = Counter(normalized)

    unique_count = len(counts)

    unique_ratio = (
        unique_count / len(normalized)
        if normalized
        else 0.0
    )

    duplicate_count = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    duplicate_ratio = (
        duplicate_count / len(normalized)
        if normalized
        else 0.0
    )

    # ---------------------------------------------------------
    # Short comments
    # ---------------------------------------------------------

    short_count = sum(
        1
        for text in texts
        if len(text) <= 5
    )

    short_comment_ratio = (
        short_count / total_comments
    )

    # ---------------------------------------------------------
    # Emoji-like comments
    #
    # Simple approximation:
    # comments containing relatively little Chinese/alphanumeric
    # content.
    # ---------------------------------------------------------

    emoji_like_count = 0

    for text in texts:

        meaningful_chars = re.findall(
            r"[\u4e00-\u9fffA-Za-z0-9]",
            text,
        )

        if len(meaningful_chars) <= 2:
            emoji_like_count += 1

    emoji_like_ratio = (
        emoji_like_count / total_comments
    )

    # ---------------------------------------------------------
    # Punctuation behavior
    # ---------------------------------------------------------

    exclamation_count = sum(
        1
        for text in texts
        if "!" in text or "！" in text
    )

    question_count = sum(
        1
        for text in texts
        if "?" in text or "？" in text
    )

    exclamation_ratio = (
        exclamation_count / total_comments
    )

    question_ratio = (
        question_count / total_comments
    )

    return {
        "comment_count": total_comments,
        "unique_comment_ratio": round(
            unique_ratio,
            4,
        ),
        "duplicate_ratio": round(
            duplicate_ratio,
            4,
        ),
        "avg_comment_length": round(
            avg_length,
            2,
        ),
        "min_comment_length": min_length,
        "max_comment_length": max_length,
        "short_comment_ratio": round(
            short_comment_ratio,
            4,
        ),
        "emoji_like_ratio": round(
            emoji_like_ratio,
            4,
        ),
        "exclamation_ratio": round(
            exclamation_ratio,
            4,
        ),
        "question_ratio": round(
            question_ratio,
            4,
        ),
    }