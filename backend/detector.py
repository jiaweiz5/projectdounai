"""Layer 1 AI-involvement detector using the fine-tuned Chinese V3 checkpoint."""

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(
    os.environ.get(
        "LAYER1_MODEL_DIR",
        str(BASE_DIR.parent / "model_runs" / "zhv3-run1"),
    )
).expanduser().resolve()

REQUIRED_V3_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.txt",
    "training_metadata.json",
    "decision.json",
    "calibration.joblib",
)
_missing_v3_files = [name for name in REQUIRED_V3_FILES if not (MODEL_DIR / name).is_file()]
_use_v3 = not _missing_v3_files

if _use_v3:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    metadata = json.loads(
        (MODEL_DIR / "training_metadata.json").read_text(encoding="utf-8")
    )
    decision = json.loads(
        (MODEL_DIR / "decision.json").read_text(encoding="utf-8")
    )

    MAX_LENGTH = int(metadata["max_length"])
    DECISION_THRESHOLD = float(decision["threshold"])
    MODEL_VERSION = f"zhv3-finetuned-epoch{metadata['epoch']}"
    CALIBRATION_VERSION = f"zhv3-calibration-{decision['calibration_sha256'][:8]}"

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
else:
    # An explicit path or partial checkpoint signals a deployment mistake.
    # Only a fresh checkout with no V3 artifact may use the committed baseline.
    if "LAYER1_MODEL_DIR" in os.environ or MODEL_DIR.exists():
        raise RuntimeError(
            f"Incomplete Layer 1 V3 checkpoint in {MODEL_DIR}: "
            + ", ".join(_missing_v3_files)
        )

    warnings.warn(
        f"Layer 1 V3 checkpoint absent at {MODEL_DIR}; serving the committed "
        "TF-IDF baseline. Set LAYER1_MODEL_DIR to a complete V3 artifact "
        "to enable the fine-tuned model.",
        RuntimeWarning,
        stacklevel=2,
    )
    bundle = joblib.load(BASE_DIR / "models" / "authorship_calibrated.joblib")
    base_model = bundle["base_model"]
    calibrator = bundle["calibrator"]
    DECISION_THRESHOLD = 0.5
    MODEL_VERSION = bundle.get("model_version", "tfidf-logreg-v1")
    CALIBRATION_VERSION = bundle.get("calibration_version", "sigmoid-v1")


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

    if _use_v3:
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
    else:
        raw_probability = float(base_model.predict_proba([text])[0, 1])
        raw_score = base_model.decision_function([text]).reshape(-1, 1)
        calibrated_probability = float(calibrator.predict_proba(raw_score)[0, 1])

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
