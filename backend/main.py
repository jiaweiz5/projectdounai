"""FastAPI entry point for the XHS verifier."""

import base64
import binascii
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ad_detector import predict_covert_ad
from detector import predict_ai_involvement
from layer4_detector import (
    Layer4ConfigurationError,
    analyze_layer4_alignment,
    analyze_layer4_reference,
)
from layer4_features import Layer4OCRError, extract_ocr_from_bytes
from layer4_reference_features import Layer4ReferenceError
from screenshot_ocr import (
    ScreenshotOCRError,
    extract_comment_lines,
    extract_text_lines,
)
from scripts.layer3_detector import predict_layer3


app = FastAPI(title="XHS Verifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload limits protect the server from unexpectedly large requests.
MAX_SCREENSHOT_BYTES = 12 * 1024 * 1024
MAX_LAYER4_IMAGE_BYTES = 12 * 1024 * 1024
MAX_LAYER4_IMAGES = 2


class CommentInput(BaseModel):
    """Comment shape currently sent by the Next.js page."""

    text: str


class AnalyzeRequest(BaseModel):
    text: str

    # Accept both ["comment"] and [{"text": "comment"}]. This keeps the
    # backend compatible with terminal tests and the existing frontend.
    comments: list[str | CommentInput] = Field(default_factory=list)

    # The current frontend sends selected images as base64 data URLs.
    images: list[str] = Field(default_factory=list)


def clean_comment_texts(
    comments: list[str | CommentInput],
) -> list[str]:
    """Convert supported comment inputs into non-empty strings."""

    cleaned_comments: list[str] = []

    for comment in comments:
        if isinstance(comment, str):
            text = comment
        else:
            text = comment.text

        text = text.strip()

        if text:
            cleaned_comments.append(text)

    return cleaned_comments


def analyze_comment_coordination(
    comments: list[str | CommentInput],
) -> dict:
    """Run Layer 3 after normalizing frontend and API comment formats."""

    cleaned_comments = clean_comment_texts(comments)

    if len(cleaned_comments) < 5:
        return {
            "status": "insufficient_data",
            "available": True,
            "score": None,
            "threshold": None,
            "label": None,
            "risk": "unknown",
            "comment_count": len(cleaned_comments),
            "message": "Layer 3 requires at least 5 extracted comments.",
        }

    result = predict_layer3(
        {
            "id": "api_request",
            "comments": cleaned_comments,
        }
    )

    if not result.get("available", False):
        return {
            "status": "model_unavailable",
            "available": False,
            "score": None,
            "threshold": None,
            "label": None,
            "risk": "unknown",
            "comment_count": len(cleaned_comments),
            "message": result.get(
                "message",
                "Layer 3 model is unavailable.",
            ),
        }

    safe_features = {}

    for name, value in result.get("features", {}).items():
        if hasattr(value, "item"):
            value = value.item()

        safe_features[name] = value

    label = int(result["label"])

    return {
        "status": "completed",
        "available": True,
        "score": float(result["score"]),
        "threshold": float(result["threshold"]),
        "label": label,
        "prediction": "coordinated" if label == 1 else "normal",
        "risk": result["risk"],
        "comment_count": len(cleaned_comments),
        "features": safe_features,
    }


def decode_image_data_url(data_url: str) -> tuple[bytes, str]:
    """Decode one browser FileReader data URL into bytes and a suffix."""

    if not isinstance(data_url, str):
        raise ValueError("Each Layer 4 image must be a base64 data URL.")

    header, separator, encoded = data_url.partition(",")

    if (
        not separator
        or not header.startswith("data:image/")
        or ";base64" not in header
    ):
        raise ValueError(
            "Each Layer 4 image must use the data:image/...;base64 format."
        )

    media_type = header[5:].split(";", 1)[0].lower()

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("A Layer 4 image contains invalid base64 data.") from exc

    if not image_bytes:
        raise ValueError("A Layer 4 image is empty.")

    if len(image_bytes) > MAX_LAYER4_IMAGE_BYTES:
        raise ValueError("Each Layer 4 image must be 12 MB or smaller.")

    suffix_by_type = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/heic": ".heic",
    }
    suffix = suffix_by_type.get(media_type, ".img")

    return image_bytes, suffix


def analyze_layer4_images(post_text: str, images: list[str]) -> dict:
    """Run image-text alignment and OCR for frontend image data URLs."""

    if not images:
        return {
            "status": "insufficient_data",
            "available": True,
            "label": None,
            "prediction": None,
            "risk": "unknown",
            "image_count": 0,
            "message": "Upload at least one post image for Layer 4.",
            "images": [],
        }

    if len(images) > MAX_LAYER4_IMAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Layer 4 accepts at most {MAX_LAYER4_IMAGES} images.",
        )

    image_results = []

    try:
        with TemporaryDirectory(prefix="xhs_layer4_") as temporary_directory:
            temporary_path = Path(temporary_directory)

            for index, data_url in enumerate(images, start=1):
                image_bytes, suffix = decode_image_data_url(data_url)
                image_path = temporary_path / f"post_image_{index}{suffix}"
                image_path.write_bytes(image_bytes)

                # Alignment answers: does this image agree with the post text?
                alignment = analyze_layer4_alignment(
                    image_path=str(image_path),
                    post_text=post_text,
                )

                # OCR is included as supporting evidence for the UI/user.
                ocr = extract_ocr_from_bytes(image_bytes)

                image_results.append(
                    {
                        "index": index,
                        "byte_count": len(image_bytes),
                        "alignment": alignment,
                        "ocr": ocr,
                    }
                )
    except (
        ValueError,
        Layer4OCRError,
        Layer4ConfigurationError,
    ) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # One suspicious image is enough to flag the post for review.
    suspicious = any(
        int(item["alignment"].get("label", 0)) == 1
        for item in image_results
    )

    return {
        "status": "completed",
        "available": True,
        "label": int(suspicious),
        "prediction": (
            "image_text_mismatch" if suspicious else "aligned"
        ),
        "risk": "high" if suspicious else "low",
        "image_count": len(image_results),
        "images": image_results,
        "message": (
            "Layer 4 checks image-text alignment. It does not by itself "
            "prove that an image was intentionally deceptive."
        ),
    }


async def read_image_upload(file: UploadFile, name: str) -> bytes:
    """Validate and read one multipart image upload."""

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"{name} must be a PNG, JPEG, WEBP, or another image.",
        )

    image_bytes = await file.read(MAX_LAYER4_IMAGE_BYTES + 1)

    if len(image_bytes) > MAX_LAYER4_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{name} must be 12 MB or smaller.",
        )

    if not image_bytes:
        raise HTTPException(status_code=422, detail=f"{name} is empty.")

    return image_bytes


def upload_file_suffix(file: UploadFile) -> str:
    """Choose a safe extension so image libraries can decode the upload."""

    suffix = Path(file.filename or "").suffix.lower()
    allowed_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }

    if suffix in allowed_suffixes:
        return suffix

    suffix_by_type = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    return suffix_by_type.get(file.content_type or "", ".img")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.4.0",
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Run Layers 1–4 using the data already sent by the main page."""

    authorship_result = predict_ai_involvement(request.text)
    covert_ad_result = predict_covert_ad(request.text)
    layer3_result = analyze_comment_coordination(request.comments)
    layer4_result = analyze_layer4_images(request.text, request.images)

    return {
        "authorship": authorship_result,
        "covert_ad": covert_ad_result,
        "layer3": layer3_result,
        "layer4": layer4_result,
    }


@app.post("/analyze-screenshot")
async def analyze_screenshot(file: UploadFile):
    """Extract comments from a screenshot and run Layer 3."""

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Upload a PNG, JPEG, WEBP, or another image file.",
        )

    image_bytes = await file.read(MAX_SCREENSHOT_BYTES + 1)

    if len(image_bytes) > MAX_SCREENSHOT_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The screenshot must be 12 MB or smaller.",
        )

    try:
        ocr_lines = extract_text_lines(image_bytes)
    except ScreenshotOCRError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    comment_lines = extract_comment_lines(ocr_lines)
    comments = [line["text"] for line in comment_lines]
    layer3_result = analyze_comment_coordination(comments)

    return {
        "filename": file.filename,
        "ocr": {
            "line_count": len(ocr_lines),
            "lines": ocr_lines,
            "comment_count": len(comment_lines),
            "comments": comment_lines,
        },
        "layer3": layer3_result,
        "message": (
            "Review ocr.comments before relying on the Layer 3 score. "
            "For best results, crop the screenshot to the comment section."
        ),
    }


@app.post("/analyze-layer4-reference")
async def analyze_layer4_reference_uploads(
    file: UploadFile,
    reference_file: UploadFile,
):
    """Compare a suspected image with a trusted original/reference image."""

    image_bytes = await read_image_upload(file, "file")
    reference_bytes = await read_image_upload(
        reference_file,
        "reference_file",
    )

    try:
        with TemporaryDirectory(prefix="xhs_layer4_reference_") as directory:
            directory_path = Path(directory)
            image_path = directory_path / (
                "target_image" + upload_file_suffix(file)
            )
            reference_path = directory_path / (
                "reference_image" + upload_file_suffix(reference_file)
            )
            image_path.write_bytes(image_bytes)
            reference_path.write_bytes(reference_bytes)

            result = analyze_layer4_reference(
                image_path=str(image_path),
                reference_image_path=str(reference_path),
            )
    except (Layer4ReferenceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "filename": file.filename,
        "reference_filename": reference_file.filename,
        "layer4_reference": result,
        "message": (
            "Reference comparison requires a trustworthy original image. "
            "A possible-editing result is evidence for review, not proof of intent."
        ),
    }
