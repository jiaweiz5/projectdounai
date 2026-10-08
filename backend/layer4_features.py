"""Feature extraction utilities for Layer 4 image verification."""

from io import BytesIO
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from rapidocr import RapidOCR
from transformers import AutoModel, AutoProcessor


MAX_IMAGE_PIXELS = 25_000_000
MAX_TEXT_WINDOWS = 16
TEXT_WINDOW_OVERLAP = 8
ALIGNMENT_MODEL_NAME = "google/siglip2-base-patch16-224"

_ocr_engine = None
_alignment_model = None
_alignment_processor = None


class Layer4OCRError(RuntimeError):
    """Raised when Layer 4 cannot read or process an image."""


def get_ocr_engine():
    """Load the local Chinese/English OCR engine once."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _load_image(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes into an RGB NumPy array for RapidOCR."""
    if not image_bytes:
        raise Layer4OCRError("The uploaded image is empty.")
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise Layer4OCRError("The uploaded file is not a readable image.") from exc
    width, height = image.size
    if width * height > MAX_IMAGE_PIXELS:
        raise Layer4OCRError("The image is too large. Crop or resize it first.")
    return np.asarray(image.convert("RGB"))


def _box_bounds(box) -> dict:
    """Convert OCR polygon points into a simple rectangle."""
    points = np.asarray(box, dtype=float)
    xs = points[:, 0]
    ys = points[:, 1]
    return {
        "left": round(float(np.min(xs)), 2),
        "top": round(float(np.min(ys)), 2),
        "right": round(float(np.max(xs)), 2),
        "bottom": round(float(np.max(ys)), 2),
    }


def _read_ocr_output(result) -> list[dict]:
    """Support current RapidOCR output and older tuple output."""
    if hasattr(result, "txts"):
        boxes, texts, scores = result.boxes, result.txts, result.scores
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
        if cleaned_text:
            detections.append({
                "text": cleaned_text,
                "confidence": round(float(score), 4),
                "box": _box_bounds(box),
            })
    return sorted(detections, key=lambda item: (
        item["box"]["top"], item["box"]["left"]
    ))


def extract_ocr_from_bytes(image_bytes: bytes) -> dict:
    """Extract all readable visible text from uploaded image bytes."""
    image_array = _load_image(image_bytes)
    result = get_ocr_engine()(image_array)
    detections = _read_ocr_output(result)
    return {
        "ocr_text": "\n".join(item["text"] for item in detections),
        "ocr_confidence": round(mean(item["confidence"] for item in detections), 4)
        if detections else 0.0,
        "ocr_boxes": detections,
        "ocr_item_count": len(detections),
    }


def extract_ocr_from_path(image_path: str | Path) -> dict:
    """Extract OCR evidence from a local Layer 4 dataset image."""
    path = Path(image_path)
    if not path.is_file():
        raise Layer4OCRError(f"Image file not found: {path}")
    return extract_ocr_from_bytes(path.read_bytes())


def get_alignment_components():
    """Load the multilingual SigLIP2 image-text model once."""
    global _alignment_model, _alignment_processor
    if _alignment_model is None:
        # Publish both objects only when initialization has fully succeeded.
        model = AutoModel.from_pretrained(ALIGNMENT_MODEL_NAME)
        processor = AutoProcessor.from_pretrained(ALIGNMENT_MODEL_NAME)
        model.eval()
        _alignment_model, _alignment_processor = model, processor
    return _alignment_model, _alignment_processor


def make_text_windows(token_ids: list[int], content_limit: int) -> list[tuple[int, list[int]]]:
    """Cover tokenized text in overlapping windows, with bounded model work.

    Windows are spread across very long posts so the end is not always ignored.
    The response separately reports how many original tokens were covered.
    """
    if content_limit <= TEXT_WINDOW_OVERLAP:
        raise Layer4OCRError("Layer 4 text window is too small for overlap.")
    if not token_ids:
        raise Layer4OCRError("Post text contains no usable tokens.")
    step = content_limit - TEXT_WINDOW_OVERLAP
    last_start = max(0, len(token_ids) - content_limit)
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    if len(starts) > MAX_TEXT_WINDOWS:
        # Select evenly spaced windows, including the beginning and end.
        starts = [
            starts[round(index * (len(starts) - 1) / (MAX_TEXT_WINDOWS - 1))]
            for index in range(MAX_TEXT_WINDOWS)
        ]
    return [(start, token_ids[start:start + content_limit]) for start in starts]


def _covered_token_count(windows: list[tuple[int, list[int]]]) -> int:
    """Count each source token once even where adjacent windows overlap."""
    covered = 0
    end = 0
    for start, ids in windows:
        next_end = start + len(ids)
        covered += max(0, next_end - max(start, end))
        end = max(end, next_end)
    return covered


def _prepare_window_texts(tokenizer, windows, max_tokens):
    """Decode token windows and fit each one using the tokenizer's public API.

    The model checkpoint can expose GemmaTokenizer, which does not implement
    build_inputs_with_special_tokens. Retokenizing each decoded window lets
    the tokenizer itself insert its required special tokens. A few tokens may
    be removed at a boundary if decoding changes the token count.
    """
    texts = []
    adjusted_windows = []
    for start, ids in windows:
        size = len(ids)
        while size:
            text = tokenizer.decode(
                ids[:size], skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            encoded = tokenizer(
                text, add_special_tokens=True, truncation=False
            )["input_ids"]
            if text.strip() and len(encoded) <= max_tokens:
                texts.append(text)
                adjusted_windows.append((start, ids[:size]))
                break
            size -= 1
        else:
            raise Layer4OCRError("A Layer 4 text window could not be encoded.")
    return texts, adjusted_windows


def calculate_image_text_similarity(image_path: str | Path, post_text: str) -> dict:
    """Score image alignment using one caption or bounded text windows.

    Short captions keep the original one-pair scoring path. Longer captions
    use the highest matching window; this aggregation needs separate calibration.
    """
    path = Path(image_path)
    cleaned_text = post_text.strip()
    if not path.is_file():
        raise Layer4OCRError(f"Image file not found: {path}")
    if not cleaned_text:
        raise Layer4OCRError("Post text is empty.")
    try:
        with Image.open(path) as source_image:
            image = source_image.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise Layer4OCRError(
            "The image could not be loaded for similarity analysis."
        ) from exc

    model, processor = get_alignment_components()
    tokenizer = processor.tokenizer
    max_tokens = int(model.config.text_config.max_position_embeddings)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_limit = max_tokens - special_tokens
    token_ids = tokenizer(
        cleaned_text, add_special_tokens=False, truncation=False
    )["input_ids"]
    windows = make_text_windows(token_ids, content_limit)
    token_count = len(token_ids)

    if len(windows) == 1:
        # Preserve the original processor's result for short calibration cases.
        inputs = processor(
            text=[cleaned_text], images=[image], padding="max_length",
            truncation=True, max_length=max_tokens, return_tensors="pt",
        )
    else:
        # The processor adds the checkpoint's special tokens for every text.
        # Validate token length after decoding, then process all windows together.
        window_texts, windows = _prepare_window_texts(
            tokenizer, windows, max_tokens
        )
        inputs = processor(
            text=window_texts, images=[image], padding="max_length",
            truncation=True, max_length=max_tokens, return_tensors="pt",
        )
    covered = _covered_token_count(windows)
    inputs = {name: value.to(model.device) for name, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
    # One image compared with every window yields one logit per text window.
    logits = outputs.logits_per_image[0]
    if len(logits) != len(windows):
        raise Layer4OCRError("Layer 4 model returned an unexpected score shape.")
    best_index = int(torch.argmax(logits).item())
    raw_logit = float(logits[best_index].item())
    similarity = float(torch.sigmoid(logits[best_index]).item())
    return {
        "image_text_similarity": round(similarity, 4),
        "raw_logit": round(raw_logit, 4),
        "model_name": ALIGNMENT_MODEL_NAME,
        "status": "not_calibrated" if len(windows) == 1 else "long_text_provisional",
        "text_window_count": len(windows),
        "text_token_count": token_count,
        "text_tokens_covered": covered,
        "text_coverage_ratio": round(covered / token_count, 4),
        "best_window_index": best_index + 1,
        "text_scoring_method": "single_caption" if len(windows) == 1 else "max_window",
    }
