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
    """Group username/body/date cards; preserve duplicates and undated tails.

    Dates terminate a card, while geometry is the fallback for missing dates.
    Ambiguous rows remain available in the raw OCR response for user review.
    """
    if not ocr_lines:
        return []
    ordered = sorted(ocr_lines, key=lambda row: (row["box"]["top"], row["box"]["left"]))
    height = max(6.0, float(np.median([
        max(1.0, row["box"]["bottom"] - row["box"]["top"]) for row in ordered
    ])))
    tolerance = height * 0.75

    def metadata(text):
        value = " ".join(str(text).split()).lower()
        # Anchor dates to the beginning: a date mentioned within a comment
        # must not truncate the rest of that comment.
        return bool(
            re.match(r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2})(?:\s|$|(?=.*(?:回复|reply|translate)))", value)
            or re.match(r"^\d+\s*(?:分钟前|小时前|天前|周前|个月前)(?:\s|$)", value)
            or re.match(r"^(?:刚刚|昨天|今天|前天)(?:\s|$)", value)
            or re.match(r"^\d+\s*(?:m|h|d|minutes?|hours?|days?)\s+ago\b", value)
            or re.fullmatch(r"(?:like\s*(?:\d+)?\s*)?(?:reply|translate)(?:\s+translate)?", value)
            or re.fullmatch(r"(?:回复|翻译|赞)(?:\s+\d+)?", value)
            or re.match(r"^vi[eo]w\s+\d+\s+repl", value)
            or re.fullmatch(r"展开\s*\d*\s*条?回复", value)
        )

    blocks, current = [], []
    for row in ordered:
        if metadata(row["text"]):
            if current:
                blocks.append(current)
                current = []
            continue
        if current and (
            row["box"]["top"] - current[-1]["box"]["top"] > height * 1.9
            or abs(row["box"]["left"] - current[-1]["box"]["left"]) > tolerance
        ):
            blocks.append(current)
            current = []
        current.append(row)
    if current:
        blocks.append(current)

    comments = []
    for block in blocks:
        if len(block) < 2:
            continue
        # Comment cards start with the author row; length is not a reliable
        # way to distinguish a username from a short comment.
        body = block[1:]
        comments.append({
            "text": "".join(row["text"] for row in body).strip(),
            "confidence": round(mean(row["confidence"] for row in body), 4),
            "box": {
                "left": min(row["box"]["left"] for row in body),
                "top": min(row["box"]["top"] for row in body),
                "right": max(row["box"]["right"] for row in body),
                "bottom": max(row["box"]["bottom"] for row in body),
            },
        })
    if not comments:
        return []
    lefts = [row["box"]["left"] for row in comments]
    root = max(lefts, key=lambda left: (
        sum(abs(other - left) <= tolerance for other in lefts), -left
    ))
    return [row for row in comments if abs(row["box"]["left"] - root) <= tolerance]

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
