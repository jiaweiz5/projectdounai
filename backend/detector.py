from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "authorship_calibrated.joblib"
)


bundle = joblib.load(
    MODEL_FILE
)

base_model = bundle[
    "base_model"
]

calibrator = bundle[
    "calibrator"
]

MODEL_VERSION = bundle.get(
    "model_version",
    "tfidf-logreg-v1"
)

CALIBRATION_VERSION = bundle.get(
    "calibration_version",
    "sigmoid-v1"
)


def predict_ai_involvement(
    text: str
):
    """
    Analyze text for AI involvement.

    This does NOT estimate what percentage
    of characters were written by AI.

    It estimates whether the overall text
    resembles AI-involved examples from
    our training data.
    """

    text = (
        text or ""
    ).strip()


    # ----------------------------------------
    # Abstain on extremely short text
    # ----------------------------------------

    if len(text) < 20:

        return {
            "status":
                "insufficient_evidence",

            "risk_score":
                None,

            "probability_any_ai":
                None,

            "reason":
                "text_too_short",

            "character_count":
                len(text),

            "model_version":
                MODEL_VERSION,

            "calibration_version":
                CALIBRATION_VERSION,
        }


    # ----------------------------------------
    # Original model score
    # ----------------------------------------

    base_probability = float(
        base_model
        .predict_proba(
            [text]
        )[0, 1]
    )


    # ----------------------------------------
    # Raw decision score
    # ----------------------------------------

    raw_score = (
        base_model
        .decision_function(
            [text]
        )
        .reshape(-1, 1)
    )


    # ----------------------------------------
    # Calibrated probability
    # ----------------------------------------

    calibrated_probability = float(
        calibrator
        .predict_proba(
            raw_score
        )[0, 1]
    )


    # ----------------------------------------
    # Classification
    # ----------------------------------------

    if calibrated_probability >= 0.5:

        classification = (
            "ai_involved"
        )

    else:

        classification = (
            "human_like"
        )


    return {
        "status":
            "supported",

        # Original detector risk score.
        "risk_score":
            round(
                base_probability,
                4
            ),

        # Calibrated estimate.
        "probability_any_ai":
            round(
                calibrated_probability,
                4
            ),

        "classification":
            classification,

        "character_count":
            len(text),

        "model_version":
            MODEL_VERSION,

        "calibration_version":
            CALIBRATION_VERSION,
    }