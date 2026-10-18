"""Layer 1 AI-involvement detector using the fine-tuned Chinese V3 checkpoint."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(
    os.environ.get(
        "LAYER1_MODEL_DIR",
        str(BASE_DIR.parent / "model_runs" / "zhv3-run1"),
    )
).expanduser().resolve()

for filename in (
    "config.json",
    "model.safetensors",
    "training_metadata.json",
    "decision.json",
    "calibration.joblib",
):
    if not (MODEL_DIR / filename).is_file():
        raise RuntimeError(f"Missing Layer 1 model file: {MODEL_DIR / filename}")

metadata = json.loads(
    (MODEL_DIR / "training_metadata.json").read_text(encoding="utf-8")
)
decision = json.loads(
    (MODEL_DIR / "decision.json").read_text(encoding="utf-8")
)

MAX_LENGTH = int(metadata["max_length"])
DECISION_THRESHOLD = float(decision["threshold"])
MODEL_VERSION = f"zhv3-finetuned-epoch{metadata['epoch']}"
CALIBRATION_VERSION = (
    f"zhv3-calibration-{decision['calibration_sha256'][:8]}"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR, local_files_only=True, trust_remote_code=False
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, local_files_only=True, trust_remote_code=False
).to(DEVICE).eval()

if model.config.num_labels != 2:
    raise RuntimeError("Expected class 0=human and class 1=AI")

calibrator = joblib.load(MODEL_DIR / "calibration.joblib")


def predict_ai_involvement(text: str) -> dict:
    text = (text or "").strip()

    if len(text) < 20:
        return {
            "status": "insufficient_evidence",
            "risk_score": None,
            "probability_any_ai": None,
            "reason": "text_too_short",
            "character_count": len(text),
            "model_version": MODEL_VERSION,
            "calibration_version": CALIBRATION_VERSION,
        }

    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits
        raw_probability = float(
            logits.softmax(dim=-1)[0, 1].detach().cpu()
        )

    clipped = float(np.clip(raw_probability, 1e-7, 1 - 1e-7))
    log_odds = np.array(
        [[np.log(clipped / (1 - clipped))]], dtype=float
    )
    calibrated_probability = float(
        calibrator.predict_proba(log_odds)[0, 1]
    )

    classification = (
        "ai_involved"
        if calibrated_probability >= DECISION_THRESHOLD
        else "human_like"
    )

    return {
        "status": "supported",
        "risk_score": round(raw_probability, 4),
        "probability_any_ai": round(calibrated_probability, 4),
        "classification": classification,
        "character_count": len(text),
        "model_version": MODEL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
    }
