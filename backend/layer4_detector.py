"""Config-driven Layer 4 image verification detector."""

import json
from pathlib import Path

import joblib
import numpy as np

from layer4_features import calculate_image_text_similarity
from layer4_reference_features import compare_reference_images


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "data" / "layer4" / "layer4_config.json"
REFERENCE_MODEL_PATH = BASE_DIR / "models" / "layer4_reference_model.joblib"

_config_cache = None
_reference_bundle_cache = None


class Layer4ConfigurationError(RuntimeError):
    """Raised when calibrated Layer 4 configuration is unavailable."""


def load_layer4_config() -> dict:
    """Read the calibration once and verify its image-text cutoff."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    if not CONFIG_PATH.is_file():
        raise Layer4ConfigurationError(
            "Layer 4 configuration was not found. Run "
            "python scripts/calibrate_layer4.py first."
        )
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Layer4ConfigurationError(
            "Layer 4 configuration could not be read."
        ) from exc
    alignment = config.get("alignment")
    if not isinstance(alignment, dict) or not isinstance(
        alignment.get("similarity_threshold"), (int, float)
    ):
        raise Layer4ConfigurationError(
            "Layer 4 alignment configuration is missing or invalid."
        )
    _config_cache = config
    return config


def load_reference_model():
    """Load the learned reference-editing model once per process."""
    global _reference_bundle_cache
    if _reference_bundle_cache is None:
        if not REFERENCE_MODEL_PATH.is_file():
            return None
        _reference_bundle_cache = joblib.load(REFERENCE_MODEL_PATH)
    return _reference_bundle_cache


def analyze_layer4_alignment(post_text: str, image_path: str | Path) -> dict:
    """Compare an image with the caption and mark long-text scores provisional."""
    alignment_config = load_layer4_config()["alignment"]
    feature_result = calculate_image_text_similarity(
        image_path=image_path, post_text=post_text
    )
    similarity = float(feature_result["image_text_similarity"])
    long_text = feature_result["text_window_count"] > 1
    full_coverage = feature_result["text_coverage_ratio"] >= 1.0
    long_threshold = alignment_config.get("long_text_similarity_threshold")
    long_calibrated = (
        long_text and full_coverage and isinstance(long_threshold, (int, float))
    )
    threshold = float(
        long_threshold if long_calibrated
        else alignment_config["similarity_threshold"]
    )
    candidate_label = int(similarity < threshold)
    provisional = long_text and not long_calibrated
    note = alignment_config.get(
        "long_text_calibration_note" if long_calibrated else "calibration_note"
    )
    if provisional:
        # A missing long-text cutoff or partial text coverage prevents a firm verdict.
        note = (
            f"{note or ''} Long-text result is provisional: the highest score "
            "among text windows uses the short-caption threshold and needs "
            f"separate calibration; {feature_result['text_tokens_covered']} of "
            f"{feature_result['text_token_count']} caption tokens were covered."
        ).strip()

    return {
        "status": "provisional" if provisional else "completed",
        "available": True,
        "task": "image_text_alignment",
        "image_text_similarity": similarity,
        "threshold": threshold,
        "decision_margin": round(similarity - threshold, 4),
        # Keep the comparison visible, but do not issue a binary verdict
        # from a cutoff that was tuned on single-caption scores.
        "label": None if provisional else candidate_label,
        "prediction": None if provisional else (
            "image_text_mismatch" if candidate_label else "aligned"
        ),
        "candidate_label": candidate_label if provisional else None,
        "risk": "unknown" if provisional else (
            "high" if candidate_label else "low"
        ),
        "model_name": feature_result["model_name"],
        "raw_logit": feature_result["raw_logit"],
        "calibration_case_count": None if provisional else alignment_config.get(
            "long_text_calibration_case_count" if long_calibrated
            else "calibration_case_count"
        ),
        "calibration_note": note,
        "text_window_count": feature_result["text_window_count"],
        "text_token_count": feature_result["text_token_count"],
        "text_tokens_covered": feature_result["text_tokens_covered"],
        "text_coverage_ratio": feature_result["text_coverage_ratio"],
        "best_window_index": feature_result["best_window_index"],
        "text_scoring_method": feature_result["text_scoring_method"],
    }


def analyze_layer4_reference(
    image_path: str | Path, reference_image_path: str | Path
) -> dict:
    """Classify possible editing and return localized changed regions."""
    bundle = load_reference_model()
    if bundle is None:
        return {
            "status": "model_unavailable", "available": False,
            "task": "reference_image_editing", "label": None,
            "prediction": None, "risk": "unknown",
            "message": (
                "Reference model is unavailable. Run "
                "python scripts/calibrate_layer4_reference.py first."
            ),
        }
    comparison = compare_reference_images(image_path, reference_image_path)
    feature_names = bundle["feature_names"]
    row = [comparison["features"][name] for name in feature_names]
    probability = float(bundle["model"].predict_proba(
        np.asarray([row], dtype=float)
    )[0][1])
    threshold = float(bundle.get("decision_threshold", 0.5))
    label = int(probability >= threshold)
    return {
        "status": "completed", "available": True,
        "task": "reference_image_editing",
        "edit_probability": probability,
        "threshold": threshold,
        "label": label,
        "prediction": "possible_editing" if label else "no_edit_detected",
        "risk": "high" if label else "low",
        "reference_similarity": comparison["reference_similarity"],
        "features": comparison["features"],
        "changed_regions": comparison["changed_regions"],
        "changed_region_count": comparison["changed_region_count"],
        "target_size": comparison["target_size"],
        "reference_size": comparison["reference_size"],
        "training_note": bundle.get("training_note"),
    }


def analyze_layer4(
    post_text: str,
    image_path: str | Path,
    reference_image_path: str | Path | None = None,
) -> dict:
    """Run available Layer 4 tasks and combine their binary evidence."""
    alignment = analyze_layer4_alignment(post_text, image_path)
    reference = None
    if reference_image_path is not None:
        reference = analyze_layer4_reference(image_path, reference_image_path)
    suspicious_labels = [alignment["label"]] if alignment["label"] is not None else []
    if reference and reference.get("label") is not None:
        suspicious_labels.append(int(reference["label"]))
    overall_label = int(any(suspicious_labels))
    provisional = alignment["status"] == "provisional"
    confirmed_reference_edit = bool(reference and reference.get("label") == 1)
    return {
        "status": "provisional" if provisional else "completed",
        "available": True,
        "label": overall_label if not provisional or confirmed_reference_edit else None,
        "prediction": (
            "suspicious" if overall_label else "normal"
        ) if not provisional or confirmed_reference_edit else None,
        "risk": "high" if confirmed_reference_edit else (
            "unknown" if provisional else ("high" if overall_label else "low")
        ),
        "alignment": alignment,
        "reference_comparison": reference,
    }


__all__ = [
    "analyze_layer4",
    "analyze_layer4_alignment",
    "analyze_layer4_reference",
    "load_layer4_config",
    "load_reference_model",
]
