"""Choose a Layer 1 threshold on dev only, under a human FPR limit."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

BASE = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-human-fpr", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.max_human_fpr < 1:
        parser.error("--max-human-fpr must be between 0 and 1")

    dev_path = BASE / "data/splits/dev.jsonl"
    model_path = BASE / "models/authorship_calibrated.joblib"
    rows = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Dev needs both verified human and AI examples")
    bundle = joblib.load(model_path)
    raw = bundle["base_model"].decision_function([row["text"] for row in rows]).reshape(-1, 1)
    scores = bundle["calibrator"].predict_proba(raw)[:, 1]
    labels = np.array([row["label"] for row in rows])
    # Larger threshold means fewer false alarms. The value immediately
    # above the largest score can deliberately abstain from all AI calls.
    candidates = sorted(set([0.0, 1.0, np.nextafter(float(scores.max()), np.inf)] + scores.tolist()))
    feasible = []
    for threshold in candidates:
        predicted = scores >= threshold
        fpr = float(predicted[labels == 0].mean())
        recall = float(predicted[labels == 1].mean())
        if fpr <= args.max_human_fpr:
            feasible.append((recall, -fpr, threshold))
    recall, neg_fpr, threshold = max(feasible)
    bundle["decision_threshold"] = float(threshold)
    bundle["threshold_selection"] = {
        "split": "dev", "max_human_fpr": args.max_human_fpr,
        "observed_human_fpr": -neg_fpr, "observed_ai_recall": recall,
        "human_count": int((labels == 0).sum()), "ai_count": int((labels == 1).sum()),
    }
    joblib.dump(bundle, model_path)
    print(json.dumps({"threshold": threshold, **bundle["threshold_selection"]}, indent=2))


if __name__ == "__main__":
    main()
