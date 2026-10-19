"""Compare the serving Layer 2 model with a ModernBert candidate on CHASM validation.

Run from backend/ after fine_tune_layer2_modernbert.py has completed.
Reads neither the CHASM test split nor serving configuration; writes only a
private JSONL review file containing validation errors and disagreements.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from fine_tune_layer2_modernbert import BASE, load_rows, score, sha256


def confusion(rows, probabilities, threshold):
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    predicted = np.asarray(probabilities) >= threshold
    return {
        "tn": int(np.sum((labels == 0) & ~predicted)),
        "fp": int(np.sum((labels == 0) & predicted)),
        "fn": int(np.sum((labels == 1) & ~predicted)),
        "tp": int(np.sum((labels == 1) & predicted)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        default=BASE / "data/private/layer2_validation_comparison.jsonl")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    metadata = json.loads((args.model_dir / "training_metadata.json").read_text(encoding="utf-8"))
    validation_path = BASE / "data/processed/chasm_validation.jsonl"
    if sha256(validation_path) != metadata["validation_sha256"]:
        raise ValueError("Validation data have changed since candidate training")
    rows = load_rows(validation_path)

    # Serving threshold from ad_detector.py. Do not import that module: importing
    # would load its global classifier, and a copied constant is easier to audit.
    old_threshold = 0.55
    vectorizer = joblib.load(BASE / "models/chasm_vectorizer.joblib")
    classifier = joblib.load(BASE / "models/chasm_classifier.joblib")
    old_scores = classifier.predict_proba(vectorizer.transform(
        [row["text"] for row in rows]))[:, 1]

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint = args.model_dir / "best"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                              trust_remote_code=False)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
        attn_implementation="eager").to(device).eval()
    new_scores = score(model, tokenizer, rows, device, args.batch_size,
                       metadata["max_length"])
    new_threshold = metadata["validation_at_selected_threshold"]["threshold_raw"]

    old = confusion(rows, old_scores, old_threshold)
    new = confusion(rows, new_scores, new_threshold)
    review_rows = []
    newly_wrong = newly_right = 0
    for row, old_score, new_score in zip(rows, old_scores, new_scores):
        old_pred = int(old_score >= old_threshold)
        new_pred = int(new_score >= new_threshold)
        old_wrong = old_pred != row["label"]
        new_wrong = new_pred != row["label"]
        if not (old_wrong or new_wrong):
            continue
        if new_wrong and not old_wrong:
            newly_wrong += 1
        if old_wrong and not new_wrong:
            newly_right += 1
        token_count = len(tokenizer(row["text"], truncation=False)["input_ids"])
        review_rows.append({
            "id": row["id"], "label": row["label"],
            "text": row["text"], "original_split": row.get("original_split"),
            "old_score": float(old_score), "old_threshold": old_threshold,
            "old_prediction": old_pred, "old_error": old_wrong,
            "candidate_score": float(new_score), "candidate_threshold": new_threshold,
            "candidate_prediction": new_pred, "candidate_error": new_wrong,
            "candidate_error_type": ("false_alarm" if row["label"] == 0 else
                                     "missed_promotion") if new_wrong else None,
            "token_count": token_count,
            "truncated": token_count > metadata["max_length"],
        })

    # Put candidate false alarms first, then misses, then errors fixed by it.
    review_rows.sort(key=lambda row: (
        0 if row["candidate_error_type"] == "false_alarm" else
        1 if row["candidate_error_type"] == "missed_promotion" else 2,
        row["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                   for row in review_rows), encoding="utf-8")
    print(json.dumps({
        "validation_rows": len(rows), "validation_sha256": sha256(validation_path),
        "selected_epoch": metadata["selected_epoch"],
        "serving_tfidf_at_0_55": old,
        "modernbert_candidate": {"threshold_raw": new_threshold, **new},
        "candidate_wrong_baseline_right": newly_wrong,
        "candidate_right_baseline_wrong": newly_right,
        "review_rows": len(review_rows), "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
