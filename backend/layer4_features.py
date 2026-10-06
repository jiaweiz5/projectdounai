"""Feature extraction utilities for Layer 4 image verification."""
import torch
from transformers import AutoModel, AutoProcessor
from io import BytesIO
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, UnidentifiedImageError
from rapidocr import RapidOCR


MAX_IMAGE_PIXELS = 25_000_000

_ocr_engine = None
ALIGNMENT_MODEL_NAME = "google/siglip2-base-patch16-224"
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
        raise Layer4OCRError(
            "The uploaded file is not a readable image."
        ) from exc

    width, height = image.size

    if width * height > MAX_IMAGE_PIXELS:
        raise Layer4OCRError(
            "The image is too large. Crop or resize it first."
        )

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

        detections.append(
            {
                "text": cleaned_text,
                "confidence": round(float(score), 4),
                "box": _box_bounds(box),
            }
        )

    return sorted(
        detections,
        key=lambda item: (
            item["box"]["top"],
            item["box"]["left"],
        ),
    )


def extract_ocr_from_bytes(image_bytes: bytes) -> dict:
    """Extract all readable visible text from uploaded image bytes."""

    image_array = _load_image(image_bytes)
    result = get_ocr_engine()(image_array)
    detections = _read_ocr_output(result)

    return {
        "ocr_text": "\n".join(item["text"] for item in detections),
        "ocr_confidence": round(
            mean(item["confidence"] for item in detections),
            4,
        ) if detections else 0.0,
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
        _alignment_model = AutoModel.from_pretrained(
            ALIGNMENT_MODEL_NAME
        )
        _alignment_model.eval()

        _alignment_processor = AutoProcessor.from_pretrained(
            ALIGNMENT_MODEL_NAME
        )

    return _alignment_model, _alignment_processor


def calculate_image_text_similarity(
    image_path: str | Path,
    post_text: str,
) -> dict:
    """Score how well an image and post caption match."""

    path = Path(image_path)
    cleaned_text = post_text.strip()

    if not path.is_file():
        raise Layer4OCRError(f"Image file not found: {path}")

    if not cleaned_text:
        raise Layer4OCRError("Post text is empty.")

    try:
        image = Image.open(path).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise Layer4OCRError(
            "The image could not be loaded for similarity analysis."
        ) from exc

    model, processor = get_alignment_components()

    inputs = processor(
        text=[cleaned_text],
        images=[image],
        padding="max_length",
        return_tensors="pt",
    )

    inputs = {
        name: value.to(model.device)
        for name, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    raw_logit = float(outputs.logits_per_image[0][0].item())
    similarity = float(
        torch.sigmoid(outputs.logits_per_image)[0][0].item()
    )

    return {
        "image_text_similarity": round(similarity, 4),
        "raw_logit": round(raw_logit, 4),
        "model_name": ALIGNMENT_MODEL_NAME,
        "status": "not_calibrated",
    }