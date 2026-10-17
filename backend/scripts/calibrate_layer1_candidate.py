"""Calibrate a fine-tuned detector on calibration; select threshold on dev."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from fine_tune_layer1 import BASE, read_rows, score_model, operating_point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-human-fpr", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.max_human_fpr < 1 or args.batch_size < 1:
        parser.error("invalid false-positive target or batch size")
    if (args.model_dir / "calibration.joblib").exists():
        parser.error("already calibrated; use a new model directory for a new run")
    metadata = json.loads((args.model_dir / "training_metadata.json").read_text(encoding="utf-8"))
    paths = {name: BASE / "data/splits" / f"{name}.jsonl"
             for name in ("fit", "dev", "calibration", "test_locked")}
    if metadata["fit_sha256"] != hashlib.sha256(paths["fit"].read_bytes()).hexdigest():
        raise ValueError("Fit split changed since fine-tuning")
    if metadata["dev_sha256"] != hashlib.sha256(paths["dev"].read_bytes()).hexdigest():
        raise ValueError("Dev split changed since fine-tuning")
    dev = read_rows(paths["dev"])
    calibration = read_rows(paths["calibration"])
    if {r["source_group_id"] for r in dev} & {r["source_group_id"] for r in calibration}:
        raise ValueError("Dev and calibration share original posts")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(device).eval()
    max_length = metadata["max_length"]
    cal_scores = score_model(model, tokenizer, calibration, device, args.batch_size, max_length)
    dev_scores = score_model(model, tokenizer, dev, device, args.batch_size, max_length)

    def logits(probability):
        p = np.clip(probability, 1e-7, 1-1e-7)
        return np.log(p / (1-p)).reshape(-1, 1)

    calibrator = LogisticRegression(solver="lbfgs", random_state=42)
    calibrator.fit(logits(cal_scores), [r["label"] for r in calibration])
    calibrated_cal = calibrator.predict_proba(logits(cal_scores))[:, 1]
    calibrated_dev = calibrator.predict_proba(logits(dev_scores))[:, 1]
    result = operating_point(dev, calibrated_dev, args.max_human_fpr)
    decision = {
        "threshold": result["dev_threshold"],
        "max_human_fpr": args.max_human_fpr,
        "calibration_brier": float(brier_score_loss([r["label"] for r in calibration], calibrated_cal)),
        "calibration_sha256": hashlib.sha256(paths["calibration"].read_bytes()).hexdigest(),
        "dev_sha256": metadata["dev_sha256"],
        **result,
    }
    joblib.dump(calibrator, args.model_dir / "calibration.joblib")
    (args.model_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
