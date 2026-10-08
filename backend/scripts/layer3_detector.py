"""Inference wrapper for the trained Layer 3 model."""

import sys
from pathlib import Path

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from layer3_features import extract_layer3_features


MODEL_PATH = BASE_DIR / "models" / "layer3_model.joblib"


_model_bundle = None


def load_model():
    global _model_bundle

    if _model_bundle is None:
        if not MODEL_PATH.exists():
            return None

        _model_bundle = joblib.load(MODEL_PATH)

    return _model_bundle


def predict_layer3(group):
    bundle = load_model()

    if bundle is None:
        return {
            "available": False,
            "score": None,
            "label": None,
            "message": "Layer 3 model has not been trained yet.",
        }

    model = bundle["model"]
    feature_names = bundle["feature_names"]
    features = extract_layer3_features(group)

    missing = [
        name
        for name in feature_names
        if name not in features
    ]

    if missing:
        raise ValueError(
            "The saved Layer 3 model expects missing features: "
            + ", ".join(missing)
        )

    row = [features[name] for name in feature_names]
    X = np.asarray([row], dtype=float)

    probability = float(model.predict_proba(X)[0][1])
    threshold = float(bundle.get("threshold", 0.5))
    label = int(probability >= threshold)

    risk_thresholds = bundle.get("risk_thresholds", {})
    medium_threshold = float(
        risk_thresholds.get("medium", threshold)
    )
    high_threshold = float(
        risk_thresholds.get("high", threshold)
    )

    if probability >= high_threshold:
        risk = "high"
    elif probability >= medium_threshold:
        risk = "medium"
    else:
        risk = "low"

    return {
        "available": True,
        "score": probability,
        "threshold": threshold,
        "label": label,
        "risk": risk,
        "features": features,
    }
