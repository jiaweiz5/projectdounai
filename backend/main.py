"""FastAPI entry point for the XHS verifier."""
import asyncio
import base64
import binascii
import logging
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
from qwen_agent import generate_final_report
from screenshot_ocr import (
    ScreenshotOCRError,
    extract_comment_lines,
    extract_text_lines,
)
from scripts.layer3_detector import predict_layer3
app = FastAPI(title="XHS Verifier")
logger = logging.getLogger(__name__)
# Allow the local Next.js development server to call the backend directly.
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
# Bound uploaded image sizes and the number of images analyzed per post.
MAX_SCREENSHOT_BYTES = 12 * 1024 * 1024
MAX_LAYER4_IMAGE_BYTES = 12 * 1024 * 1024
MAX_LAYER4_IMAGES = 2
class CommentInput(BaseModel):
    """Comment object currently sent by the Next.js page."""
    text: str
class AnalyzeRequest(BaseModel):
    """Input for the main four-layer analysis endpoint."""
    text: str
    # The frontend opts in; older API tests need no Qwen request.
    include_report: bool = False
    # Accept either plain strings or objects with a text field.
    comments: list[str | CommentInput] = Field(default_factory=list)
    # Browser-selected images arrive as base64 data URLs.
    images: list[str] = Field(default_factory=list)
def clean_comment_texts(comments: list[str | CommentInput]) -> list[str]:
    """Extract non-empty text from both supported comment shapes."""
    cleaned_comments: list[str] = []
    for comment in comments:
        text = comment if isinstance(comment, str) else comment.text
        text = text.strip()
        if text:
            cleaned_comments.append(text)
    return cleaned_comments
def analyze_comment_coordination(comments: list[str | CommentInput]) -> dict:
    """Normalize comments, run Layer 3, and produce JSON-safe values."""
    cleaned_comments = clean_comment_texts(comments)
    if len(cleaned_comments) < 5:
        # The model needs at least five comments to make a supported estimate.
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
    result = predict_layer3({"id": "api_request", "comments": cleaned_comments})
    if not result.get("available", False):
        return {
            "status": "model_unavailable",
            "available": False,
            "score": None,
            "threshold": None,
            "label": None,
            "risk": "unknown",
            "comment_count": len(cleaned_comments),
            "message": result.get("message", "Layer 3 model is unavailable."),
        }
    # Convert NumPy scalar feature values into native Python values for JSON.
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
    """Validate and decode one browser image data URL."""
    if not isinstance(data_url, str):
        raise ValueError("Each Layer 4 image must be a base64 data URL.")
    header, separator, encoded = data_url.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
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
    return image_bytes, suffix_by_type.get(media_type, ".img")
def analyze_layer4_images(post_text: str, images: list[str]) -> dict:
    """Run image checks independently so one model error cannot block Layers 1–3."""
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
    # Temporary files give existing image libraries a path to read.
    with TemporaryDirectory(prefix="xhs_layer4_") as temporary_directory:
        temporary_path = Path(temporary_directory)
        for index, data_url in enumerate(images, start=1):
            try:
                image_bytes, suffix = decode_image_data_url(data_url)
            except ValueError as exc:
                # Invalid uploads are input errors; name the image that needs fixing.
                raise HTTPException(
                    status_code=422, detail=f"Image {index}: {exc}"
                ) from exc
            image_path = temporary_path / f"post_image_{index}{suffix}"
            image_path.write_bytes(image_bytes)
            try:
                alignment = analyze_layer4_alignment(
                    image_path=str(image_path),
                    post_text=post_text,
                )
            except (Layer4OCRError, Layer4ConfigurationError, RuntimeError, ValueError):
                # Log the traceback for developers but keep the other layers' output.
                logger.exception("Layer 4 alignment failed for image %s", index)
                alignment = {
                    "status": "unavailable", "available": False,
                    "task": "image_text_alignment", "label": None,
                    "prediction": None, "risk": "unknown",
                    "message": "This image could not be compared with the post text.",
                }
            try:
                ocr = extract_ocr_from_bytes(image_bytes)
            except (Layer4OCRError, RuntimeError):
                logger.exception("Layer 4 OCR failed for image %s", index)
                ocr = {
                    "ocr_text": "", "ocr_confidence": 0.0,
                    "ocr_boxes": [], "ocr_item_count": 0,
                    "message": "Text could not be read from this image.",
                }
            image_results.append({
                "index": index,
                "byte_count": len(image_bytes),
                "alignment": alignment,
                "ocr": ocr,
            })

    # A provisional or missing image score must not be presented as a firm verdict.
    alignments = [item["alignment"] for item in image_results]
    usable = [item for item in alignments if item.get("label") is not None]
    provisional = any(item.get("status") == "provisional" for item in alignments)
    unavailable = any(item.get("status") == "unavailable" for item in alignments)
    suspicious = any(
        int(item["label"]) == 1 for item in usable
    )
    confirmed_suspicious = any(
        item.get("status") == "completed" and item.get("label") == 1
        for item in usable
    )
    status = (
        "provisional" if provisional else
        "unavailable" if not usable else
        "partial" if unavailable else "completed"
    )
    # Keep a clear user-facing notice when only part of the evidence was scored.
    message = (
        "Layer 4 checks image-text alignment. A mismatch does not prove "
        "deceptive intent."
    )
    if provisional:
        coverage = min(
            (item.get("text_coverage_ratio", 1.0) for item in alignments
             if item.get("status") == "provisional"),
            default=1.0,
        )
        message += (
            f" Long-text comparison is provisional; it covered at least "
            f"{coverage:.0%} of the caption's tokens. Review before relying "
            "on this result."
        )
    if unavailable:
        message += " Some images could not be checked."
    return {
        "status": status,
        "available": bool(usable or provisional),
        "label": int(suspicious) if usable else None,
        "prediction": (
            "image_text_mismatch" if suspicious else "aligned"
        ) if usable else None,
        "risk": "high" if confirmed_suspicious else (
            "unknown" if status != "completed" else "low"
        ),
        "image_count": len(image_results),
        "images": image_results,
        "message": message,
    }
async def read_image_upload(file: UploadFile, name: str) -> bytes:
    """Read a multipart upload while checking type, size, and emptiness."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"{name} must be a PNG, JPEG, WEBP, or another image.",
        )
    image_bytes = await file.read(MAX_LAYER4_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_LAYER4_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{name} must be 12 MB or smaller.")
    if not image_bytes:
        raise HTTPException(status_code=422, detail=f"{name} is empty.")
    return image_bytes
def upload_file_suffix(file: UploadFile) -> str:
    """Select a file extension recognized by the image-processing libraries."""
    suffix = Path(file.filename or "").suffix.lower()
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
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
    """Let local checks confirm that the API process is running."""
    return {"status": "ok", "version": "0.4.0"}
@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Run Layers 1–4 and optionally summarize their results with Qwen."""
    # The detectors are synchronous; run each in a worker thread so the
    # asynchronous API route remains responsive while they process inputs.
    authorship_result = await asyncio.to_thread(predict_ai_involvement, request.text)
    covert_ad_result = await asyncio.to_thread(predict_covert_ad, request.text)
    layer3_result = await asyncio.to_thread(
        analyze_comment_coordination, request.comments
    )
    layer4_result = await asyncio.to_thread(
        analyze_layer4_images, request.text, request.images
    )
    # Preserve the existing four-layer response contract.
    result = {
        "authorship": authorship_result,
        "covert_ad": covert_ad_result,
        "layer3": layer3_result,
        "layer4": layer4_result,
    }
    # Only requests that opt in make an external Qwen call.
    if request.include_report:
        result["final_report"] = await generate_final_report(
            request.text,
            result,
        )
    return result
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
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # OCR text is filtered into comment lines before Layer 3 sees it.
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
    """Compare a suspected image with a trusted original image."""
    image_bytes = await read_image_upload(file, "file")
    reference_bytes = await read_image_upload(reference_file, "reference_file")
    try:
        with TemporaryDirectory(prefix="xhs_layer4_reference_") as directory:
            directory_path = Path(directory)
            image_path = directory_path / ("target_image" + upload_file_suffix(file))
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
