"""FastAPI entry point for the XHS verifier."""

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ad_detector import predict_covert_ad
from detector import predict_ai_involvement
from screenshot_ocr import (
    ScreenshotOCRError,
    extract_comment_lines,
    extract_text_lines,
)
from scripts.layer3_detector import predict_layer3


app = FastAPI(title="XHS Verifier")

MAX_SCREENSHOT_BYTES = 12 * 1024 * 1024


class AnalyzeRequest(BaseModel):
    text: str
    comments: list[str] = Field(default_factory=list)


def analyze_comment_coordination(comments: list[str]) -> dict:
    cleaned_comments = [
        comment.strip()
        for comment in comments
        if isinstance(comment, str) and comment.strip()
    ]

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.3.0",
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    authorship_result = predict_ai_involvement(request.text)
    covert_ad_result = predict_covert_ad(request.text)
    layer3_result = analyze_comment_coordination(request.comments)

    return {
        "authorship": authorship_result,
        "covert_ad": covert_ad_result,
        "layer3": layer3_result,
    }


@app.post("/analyze-screenshot")
async def analyze_screenshot(file: UploadFile):
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
