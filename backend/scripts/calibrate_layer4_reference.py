"""Train and validate Layer 4 reference-image editing detection."""

import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from layer4_reference_features import (
    REFERENCE_FEATURE_NAMES,
    compare_reference_arrays,
    compare_reference_images,
    load_cv_image,
)


DATASET_PATH = BASE_DIR / "data" / "layer4" / "layer4_dataset.json"
FEATURES_PATH = (
    BASE_DIR / "data" / "layer4" / "layer4_reference_features.json"
)
MODEL_PATH = BASE_DIR / "models" / "layer4_reference_model.joblib"


def load_dataset() -> list[dict]:
    with DATASET_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    return data.get("cases", []) if isinstance(data, dict) else data


def make_no_edit_control(reference_image: np.ndarray) -> np.ndarray:
    """Simulate normal platform recompression without purposeful editing."""

    success, encoded = cv2.imencode(
        ".jpg",
        reference_image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        raise RuntimeError("Could not create the no-edit JPEG control.")

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if decoded is None:
        raise RuntimeError("Could not decode the no-edit JPEG control.")

    return decoded


def feature_row(result: dict) -> list[float]:
    return [
        float(result["features"][name])
        for name in REFERENCE_FEATURE_NAMES
    ]


def build_model() -> Pipeline:
    """Scale continuous image features and learn an editing probability."""

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def print_metrics(labels, predictions):
    print(f"Accuracy:  {accuracy_score(labels, predictions):.4f}")
    print(
        "Precision: "
        f"{precision_score(labels, predictions, zero_division=0):.4f}"
    )
    print(
        "Recall:    "
        f"{recall_score(labels, predictions, zero_division=0):.4f}"
    )
    print(f"F1:        {f1_score(labels, predictions, zero_division=0):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))


def main():
    records = load_dataset()
    reference_cases = [
        record
        for record in records
        if record.get("reference_image_path")
    ]

    if len(reference_cases) != 25:
        raise ValueError(
            f"Expected 25 reference cases, found {len(reference_cases)}."
        )

    X = []
    y = []
    groups = []
    saved_features = []

    print("=" * 65)
    print("BUILDING LAYER 4 REFERENCE FEATURES")
    print("=" * 65)

    for index, record in enumerate(reference_cases, start=1):
        target_path = BASE_DIR / record["image_path"]
        reference_path = BASE_DIR / record["reference_image_path"]

        # Positive example: the team's purposefully edited image compared with
        # its preserved original reference.
        edited_result = compare_reference_images(
            target_path,
            reference_path,
        )

        # Negative control: the reference image after ordinary JPEG
        # recompression compared with the original. This teaches the model not
        # to call routine codec noise an edit.
        reference_image = load_cv_image(reference_path)
        control_image = make_no_edit_control(reference_image)
        control_result = compare_reference_arrays(
            control_image,
            reference_image,
        )

        for example_name, label, result in [
            ("edited", 1, edited_result),
            ("recompressed_control", 0, control_result),
        ]:
            X.append(feature_row(result))
            y.append(label)
            groups.append(record["id"])
            saved_features.append(
                {
                    "id": record["id"],
                    "example_type": example_name,
                    "label": label,
                    "features": result["features"],
                    "reference_similarity": result[
                        "reference_similarity"
                    ],
                    "changed_region_count": result[
                        "changed_region_count"
                    ],
                }
            )

        print(
            f"[{index:02d}/25] {record['id']} "
            f"edited_similarity={edited_result['reference_similarity']:.4f} "
            f"edited_ratio={edited_result['features']['changed_pixel_ratio']:.4f} "
            f"control_similarity={control_result['reference_similarity']:.4f}"
        )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    # Keep the edited example and its control in the same fold. This prevents
    # one version of a source image from leaking into the other side.
    splitter = GroupKFold(n_splits=5)
    out_of_fold_predictions = np.zeros_like(y)
    out_of_fold_probabilities = np.zeros(len(y), dtype=float)

    print("\n" + "=" * 65)
    print("GROUP-HELD-OUT REFERENCE VALIDATION")
    print("=" * 65)

    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        model = build_model()
        model.fit(X[train_indices], y[train_indices])
        predictions = model.predict(X[test_indices])
        probabilities = model.predict_proba(X[test_indices])[:, 1]

        out_of_fold_predictions[test_indices] = predictions
        out_of_fold_probabilities[test_indices] = probabilities

        print(
            f"Fold {fold_number}: "
            f"accuracy={accuracy_score(y[test_indices], predictions):.4f}, "
            f"source_cases={len(set(groups[test_indices]))}"
        )

    print("\nOverall out-of-fold results")
    print("-" * 65)
    print_metrics(y, out_of_fold_predictions)

    for index, probability in enumerate(out_of_fold_probabilities):
        saved_features[index]["out_of_fold_edit_probability"] = round(
            float(probability),
            6,
        )
        saved_features[index]["out_of_fold_prediction"] = int(
            out_of_fold_predictions[index]
        )

    final_model = build_model()
    final_model.fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_names": REFERENCE_FEATURE_NAMES,
            "decision_threshold": 0.2782,
            "training_case_count": len(reference_cases),
            "training_example_count": len(y),
            "training_note": (
                "Development model trained on 25 purposeful edits and 25 "
                "JPEG-recompressed controls. It requires a known reference "
                "image and is not a general AI-image detector."
            ),
        },
        MODEL_PATH,
    )

    FEATURES_PATH.write_text(
        json.dumps(saved_features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nFeatures saved to: {FEATURES_PATH}")
    print(f"Final model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
