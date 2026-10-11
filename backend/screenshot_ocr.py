"""OCR utilities for extracting text lines from comment screenshots."""

from io import BytesIO
import re
from statistics import mean

import numpy as np
from PIL import Image, UnidentifiedImageError
from rapidocr import RapidOCR


MAX_IMAGE_PIXELS = 25_000_000


_ocr_engine = None


class ScreenshotOCRError(RuntimeError):
    pass


def get_ocr_engine():
    """Load the local Chinese/English OCR engine once."""

    global _ocr_engine

    if _ocr_engine is None:
        _ocr_engine = RapidOCR()

    return _ocr_engine


def _load_image(image_bytes):
    if not image_bytes:
        raise ScreenshotOCRError("The uploaded image is empty.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ScreenshotOCRError(
            "The uploaded file is not a readable image."
        ) from exc

    width, height = image.size

    if width * height > MAX_IMAGE_PIXELS:
        raise ScreenshotOCRError(
            "The screenshot is too large. Crop or resize it first."
        )

    return np.asarray(image.convert("RGB"))


def _box_bounds(box):
    points = np.asarray(box, dtype=float)
    xs = points[:, 0]
    ys = points[:, 1]

    return {
        "left": float(np.min(xs)),
        "top": float(np.min(ys)),
        "right": float(np.max(xs)),
        "bottom": float(np.max(ys)),
        "center_y": float(np.mean(ys)),
        "height": float(np.max(ys) - np.min(ys)),
    }


def _read_ocr_output(result):
    """Support the current RapidOCR object and older tuple output."""

    if hasattr(result, "txts"):
        boxes = result.boxes
        texts = result.txts
        scores = result.scores
    elif isinstance(result, (tuple, list)) and result:
        rows = result[0] if len(result) == 2 else result
        boxes = [row[0] for row in rows]
        texts = [row[1] for row in rows]
        scores = [row[2] for row in rows]
    else:
        return []

    if boxes is None or texts is None or scores is None:
        return []

    detections = []

    for box, text, score in zip(boxes, texts, scores):
        cleaned_text = str(text).strip()

        if not cleaned_text:
            continue

        bounds = _box_bounds(box)
        detections.append(
            {
                "text": cleaned_text,
                "confidence": float(score),
                **bounds,
            }
        )

    return detections


def _merge_visual_rows(detections):
    """Merge OCR fragments that occupy the same visual text row."""

    if not detections:
        return []

    ordered = sorted(
        detections,
        key=lambda item: (item["center_y"], item["left"]),
    )

    typical_height = max(
        6.0,
        float(np.median([item["height"] for item in ordered])),
    )
    row_tolerance = typical_height * 0.55

    rows = []

    for detection in ordered:
        matching_row = None

        for row in reversed(rows[-3:]):
            if abs(detection["center_y"] - row["center_y"]) <= row_tolerance:
                matching_row = row
                break

        if matching_row is None:
            rows.append(
                {
                    "parts": [detection],
                    "center_y": detection["center_y"],
                }
            )
            continue

        matching_row["parts"].append(detection)
        matching_row["center_y"] = mean(
            part["center_y"]
            for part in matching_row["parts"]
        )

    merged = []

    for row in rows:
        parts = sorted(row["parts"], key=lambda item: item["left"])
        text = " ".join(part["text"] for part in parts).strip()

        if not text:
            continue

        merged.append(
            {
                "text": text,
                "confidence": round(
                    mean(part["confidence"] for part in parts),
                    4,
                ),
                "box": {
                    "left": round(min(part["left"] for part in parts), 2),
                    "top": round(min(part["top"] for part in parts), 2),
                    "right": round(max(part["right"] for part in parts), 2),
                    "bottom": round(max(part["bottom"] for part in parts), 2),
                },
            }
        )

    return merged


def extract_comment_lines(ocr_lines):
    """Extract comment bodies from OCR rows using visual block structure.

    Xiaohongshu-style comment cards generally begin with a username row and
    place the comment body immediately below it at the same indentation. This
    parser uses only geometry and row spacing; it does not contain username,
    location, button, or Chinese phrase lists.
    """

    if not ocr_lines:
        return []

    ordered = sorted(
        ocr_lines,
        key=lambda item: (
            item["box"]["top"],
            item["box"]["left"],
        ),
    )

    heights = [
        max(1.0, item["box"]["bottom"] - item["box"]["top"])
        for item in ordered
    ]
    typical_height = max(18.0, float(np.median(heights)))

    # A new visual block begins after a gap notably larger than one text row.
#
# The multiplier is calculated from the screenshot's typical OCR row height,
# so it automatically scales for differently sized screenshots. A value of
# 1.75 allows normal username-to-comment spacing while still separating
# distinct comment cards.
    block_gap = typical_height * 1.75
    blocks = []
    current_block = []
    previous_top = None

    for line in ordered:
        current_top = line["box"]["top"]

        if (
            current_block
            and previous_top is not None
            and current_top - previous_top > block_gap
        ):
            blocks.append(current_block)
            current_block = []

        current_block.append(line)
        previous_top = current_top

    if current_block:
        blocks.append(current_block)

    def is_comment_meta(text):
        """Return true for the small date/action row below a comment."""

        normalized = " ".join(str(text).split()).lower()
        if not normalized:
            return False
        return bool(
            re.search(r"\b\d{1,2}\s*/\s*\d{1,2}\b", normalized)
            or re.search(r"\b\d{1,2}\s*(?:分钟前|小时前|天前)\b", normalized)
            or re.search(r"\b(?:刚刚|昨天|今天)\b", normalized)
            or "reply" in normalized
            or "translate" in normalized
            or re.search(r"\bvi[eo]w\s+\d", normalized)
        )

    def is_date_meta(text):
        normalized = " ".join(str(text).split()).lower()
        return bool(
            re.search(r"\b\d{1,2}\s*/\s*\d{1,2}\b", normalized)
            or re.search(r"\b\d{1,2}\s*(?:分钟前|小时前|天前)\b", normalized)
            or re.search(r"\b(?:刚刚|昨天|今天)\b", normalized)
        )

    # A date/action row is the most reliable boundary for wrapped comments.
    # Use it first when available; this also prevents surrounding page chrome
    # from being mistaken for the first username/comment block.
    date_indices = [index for index, line in enumerate(ordered) if is_date_meta(line["text"])]
    if date_indices:
        dated_comments = []
        previous_index = -1
        for date_index in date_indices:
            marker = ordered[date_index]
            candidates = [
                line for line in ordered[previous_index + 1 : date_index]
                if abs(line["box"]["left"] - marker["box"]["left"]) <= max(36.0, typical_height * 2.0)
                and marker["box"]["top"] - line["box"]["top"] <= max(90.0, typical_height * 4.0)
                and not is_comment_meta(line["text"])
            ]
            # Usernames are usually short; start at the first plausible body
            # line and keep every wrapped line until the date marker.
            body_start = next(
                (position for position, line in enumerate(candidates) if len(line["text"].strip()) >= 8),
                max(0, len(candidates) - 1),
            )
            body_rows = candidates[body_start:]
            text = "".join(row["text"] for row in body_rows).strip()
            if text:
                dated_comments.append({
                    "text": text,
                    "confidence": round(mean(row["confidence"] for row in body_rows), 4),
                    "box": {
                        "left": round(min(row["box"]["left"] for row in body_rows), 2),
                        "top": round(min(row["box"]["top"] for row in body_rows), 2),
                        "right": round(max(row["box"]["right"] for row in body_rows), 2),
                        "bottom": round(max(row["box"]["bottom"] for row in body_rows), 2),
                    },
                })
            previous_index = date_index
        if dated_comments:
            return dated_comments

    comments = []
    alignment_tolerance = typical_height * 0.75
    continuation_gap = typical_height * 1.20

    for block in blocks:
        # A one-row block is usually an interface control rather than a full
        # username/comment pair, so it is not sent to the classifier.
        if len(block) < 2:
            continue

        username_row = block[0]
        first_body_row = block[1]

        if (
            abs(
                first_body_row["box"]["left"]
                - username_row["box"]["left"]
            )
            > alignment_tolerance
        ):
            continue

        body_rows = [first_body_row]
        previous_body_row = first_body_row

        for candidate in block[2:]:
            # On Xiaohongshu the date/action row follows the complete body.
            # Treat it as an explicit boundary so wrapped comment text stays
            # together instead of being mistaken for a new comment.
            if is_comment_meta(candidate["text"]):
                break
            vertical_gap = (
                candidate["box"]["top"]
                - previous_body_row["box"]["top"]
            )
            alignment_gap = abs(
                candidate["box"]["left"]
                - first_body_row["box"]["left"]
            )

            if (
                vertical_gap <= continuation_gap * 1.8
                and alignment_gap <= alignment_tolerance
            ):
                body_rows.append(candidate)
                previous_body_row = candidate
            else:
                break

        text = "".join(row["text"] for row in body_rows).strip()

        if not text:
            continue

        comments.append(
            {
                "text": text,
                "confidence": round(
                    mean(row["confidence"] for row in body_rows),
                    4,
                ),
                "box": {
                    "left": round(
                        min(row["box"]["left"] for row in body_rows),
                        2,
                    ),
                    "top": round(
                        min(row["box"]["top"] for row in body_rows),
                        2,
                    ),
                    "right": round(
                        max(row["box"]["right"] for row in body_rows),
                        2,
                    ),
                    "bottom": round(
                        max(row["box"]["bottom"] for row in body_rows),
                        2,
                    ),
                },
            }
        )

    # If no comment candidates were found, return an empty list.
    if not comments:
        return []

    # Collect the horizontal starting position of every candidate.
    # Xiaohongshu top-level comments normally share one main left margin.
    candidate_lefts = [
        comment["box"]["left"]
        for comment in comments
    ]

    # Find the left margin with the largest group of nearby candidates.
    # This identifies the dominant top-level comment indentation.
    root_left = max(
        candidate_lefts,
        key=lambda candidate_left: (
            sum(
                abs(other_left - candidate_left)
                <= alignment_tolerance
                for other_left in candidate_lefts
            ),
            # If two groups are equally large, prefer the leftmost group,
            # since replies and image content are normally indented.
            -candidate_left,
        ),
    )

    # Keep only candidates aligned with the top-level comment margin.
    # This removes indented replies, reply controls, and text inside
    # attached reply images from the Layer 3 model input.
    comments = [
        comment
        for comment in comments
        if abs(comment["box"]["left"] - root_left)
        <= alignment_tolerance
    ]

    return comments

def extract_text_lines(image_bytes, minimum_confidence=0.50):
    """Run OCR and return reading-order text lines with confidence and boxes."""

    image = _load_image(image_bytes)
    engine = get_ocr_engine()

    try:
        result = engine(
            image,
            text_score=float(minimum_confidence),
        )
    except Exception as exc:
        raise ScreenshotOCRError(
            "OCR could not process this screenshot."
        ) from exc

    detections = [
        item
        for item in _read_ocr_output(result)
        if item["confidence"] >= minimum_confidence
    ]

    return _merge_visual_rows(detections)
