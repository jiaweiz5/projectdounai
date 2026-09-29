from pathlib import Path

import joblib
import numpy as np

from layer3_features import extract_layer3_features


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "layer3_model.joblib"
)


_model_bundle = None


def load_model():
    global _model_bundle

    if _model_bundle is None:

        if not MODEL_PATH.exists():
            return None

        _model_bundle = joblib.load(
            MODEL_PATH
        )

    return _model_bundle


def predict_layer3(group):

    bundle = load_model()

    if bundle is None:
        return {
            "available": False,
            "score": None,
            "label": None,
            "message": (
                "Layer 3 model has not been trained yet."
            ),
        }

    model = bundle["model"]

    feature_names = bundle[
        "feature_names"
    ]

    features = extract_layer3_features(
        group
    )

    row = [
        features[name]
        for name in feature_names
    ]

    X = np.array(
        [row]
    )

    probability = model.predict_proba(
        X
    )[0][1]

    label = int(
        probability >= 0.5
    )

    return {
        "available": True,
        "score": float(probability),
        "label": label,
        "features": features,
    }