from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent

VECTORIZER_PATH = (
    BASE_DIR
    / "models"
    / "chasm_vectorizer.joblib"
)

CLASSIFIER_PATH = (
    BASE_DIR
    / "models"
    / "chasm_classifier.joblib"
)


# Frozen from validation tuning in Step 30
THRESHOLD = 0.55


vectorizer = joblib.load(
    VECTORIZER_PATH
)

classifier = joblib.load(
    CLASSIFIER_PATH
)


def predict_covert_ad(text: str):
    text = text.strip()

    if not text:
        return {
            "label": "unknown",
            "probability": 0.0,
            "threshold": THRESHOLD,
        }

    X = vectorizer.transform([text])

    probability = float(
        classifier.predict_proba(X)[0][1]
    )

    is_ad = probability >= THRESHOLD

    return {
        "label": (
            "likely_ad"
            if is_ad
            else "likely_non_ad"
        ),
        "probability": round(
            probability,
            4,
        ),
        "threshold": THRESHOLD,
    }