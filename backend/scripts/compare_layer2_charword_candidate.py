"""Compare a Layer 2 text feature experiment on validation data only.

Copy to backend/scripts/ and run from backend:
    python scripts/compare_layer2_charword_candidate.py

The production model, API threshold, and test data are never written or used.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, make_pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "chasm_train.jsonl"
VAL_PATH = BASE_DIR / "data" / "processed" / "chasm_validation.jsonl"
VECTORIZER_PATH = BASE_DIR / "models" / "chasm_vectorizer.joblib"
CLASSIFIER_PATH = BASE_DIR / "models" / "chasm_classifier.joblib"


def load_jsonl(path):
    """Read each post's text and label from one training or validation file."""
    texts, labels = [], []
    # Each nonblank line is a separate JSON record; preserve its original order.
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                texts.append(row["text"])
                labels.append(int(row["label"]))
    # A two-class comparison needs at least one ad and one non-ad post.
    if not texts or set(labels) != {0, 1}:
        raise ValueError(f"Expected both labels in nonempty file: {path}")
    return texts, np.asarray(labels)


def metrics(labels, scores, threshold):
    """Turn scores into decisions and count correct and incorrect examples."""
    # The score is the model's estimate for label 1 (covert advertisement).
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return (
        precision_score(labels, predictions, zero_division=0),
        recall_score(labels, predictions, zero_division=0),
        f1_score(labels, predictions, zero_division=0),
        int(tn), int(fp), int(fn), int(tp),
    )


def show_result(name, labels, scores, threshold):
    """Print one model's precision, recall, F1, and confusion-matrix counts."""
    precision, recall, f1, tn, fp, fn, tp = metrics(labels, scores, threshold)
    print(
        f"{name:<18} {threshold:>5.2f}  "
        f"{precision:>7.4f}  {recall:>6.4f}  {f1:>6.4f}  "
        f"{tn:>3} {fp:>3} {fn:>3} {tp:>3}"
    )


def main():
    """Train an experimental model and compare it to the saved API model."""
    # Fit only on the existing train split; use validation only for comparison.
    train_texts, train_labels = load_jsonl(TRAIN_PATH)
    val_texts, val_labels = load_jsonl(VAL_PATH)

    # Load the deployed baseline without fitting or modifying it.
    baseline_vectorizer = joblib.load(VECTORIZER_PATH)
    baseline_classifier = joblib.load(CLASSIFIER_PATH)
    # Column 1 is the probability for label 1, matching the production model.
    baseline_scores = baseline_classifier.predict_proba(
        baseline_vectorizer.transform(val_texts)
    )[:, 1]

    # Keep the existing character vectorizer and classifier settings; add
    # token features for spaces/punctuation and Latin brand names. This word
    # analyzer does not perform Chinese word segmentation.
    candidate_features = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), min_df=2,
            max_features=100000, sublinear_tf=True,
        )),
        ("token", TfidfVectorizer(
            analyzer="word", token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2), min_df=2,
            max_features=30000, sublinear_tf=True,
        )),
    ])
    # A pipeline ensures both vectorizers learn vocabulary from training data only.
    candidate = make_pipeline(
        candidate_features,
        LogisticRegression(
            C=1, max_iter=2000, class_weight="balanced", random_state=42
        ),
    )
    # The candidate stays in memory: this never overwrites models/*.joblib.
    candidate.fit(train_texts, train_labels)
    candidate_scores = candidate.predict_proba(val_texts)[:, 1]

    print("LAYER 2 VALIDATION-ONLY FEATURE EXPERIMENT")
    print(f"Training: {len(train_texts)} | Validation: {len(val_texts)}")
    print("No production model or API files have been changed.")
    print("model              cutoff   prec.   recall      F1   TN  FP  FN  TP")
    # Compare both models at the same cutoffs to expose false-alarm tradeoffs.
    for name, scores in (
        ("Current char model", baseline_scores),
        ("Char + token trial", candidate_scores),
    ):
        for threshold in (0.45, 0.50, 0.55, 0.60):
            show_result(name, val_labels, scores, threshold)
        print()
    print("The deployed API currently uses the char model at threshold 0.55.")
    print("Choose a candidate using validation data; reserve new unseen posts for final testing.")


if __name__ == "__main__":
    main()
