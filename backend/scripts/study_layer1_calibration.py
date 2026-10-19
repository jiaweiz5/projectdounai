"""Compare Layer 1 calibration methods without changing the deployed detector.

Uses only the existing calibration and development splits. Do not interpret the
development comparison as an untouched test or replace the deployed files here.
Run from backend/ with the Layer 1 virtual environment activated.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from fine_tune_layer1 import BASE, operating_point, read_rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_odds(scores):
    p = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


def score_margin(model, tokenizer, rows, device, batch_size, max_length):
    import torch

    model.eval()
    raw, margin = [], []
    for start in range(0, len(rows), batch_size):
        encoded = tokenizer([row["text"] for row in rows[start:start + batch_size]],
                            padding=True, truncation=True, max_length=max_length,
                            return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        raw.extend(logits.softmax(-1)[:, 1].detach().cpu().tolist())
        margin.extend((logits[:, 1] - logits[:, 0]).detach().cpu().tolist())
        if (start // batch_size + 1) % 500 == 0:
            print(f"scored {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return np.asarray(raw, dtype=np.float64), np.asarray(margin, dtype=np.float64)


def metrics(rows, scores, max_fpr):
    labels = np.asarray([row["label"] for row in rows])
    scores = np.clip(scores, 1e-7, 1 - 1e-7)
    point = operating_point(rows, scores, max_fpr)
    threshold = point["dev_threshold"]
    predicted = scores >= threshold
    human, ai = labels == 0, labels == 1
    return {
        "brier": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "auc": float(roc_auc_score(labels, scores)),
        "threshold_at_max_fpr": threshold,
        "human_fp": int(np.sum(predicted & human)),
        "human_n": int(np.sum(human)),
        "human_fpr": float(np.mean(predicted[human])),
        "ai_detected": int(np.sum(predicted & ai)),
        "ai_n": int(np.sum(ai)),
        "ai_recall": float(np.mean(predicted[ai])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-human-fpr", type=float, default=.05)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.max_human_fpr < 1:
        parser.error("invalid batch size or false-positive limit")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty; do not overwrite previous experiments")
    metadata = json.loads((args.model_dir / "training_metadata.json").read_text(encoding="utf-8"))
    decision = json.loads((args.model_dir / "decision.json").read_text(encoding="utf-8"))
    paths = {name: BASE / "data/splits" / f"{name}.jsonl"
             for name in ("fit", "dev", "calibration")}
    for name in ("fit", "dev"):
        if sha256(paths[name]) != metadata[f"{name}_sha256"]:
            raise ValueError(f"{name} changed since fine-tuning")
    if sha256(paths["calibration"]) != decision["calibration_sha256"]:
        raise ValueError("calibration changed since original calibration")
    cal, dev = read_rows(paths["calibration"]), read_rows(paths["dev"])
    if {r["source_group_id"] for r in cal} & {r["source_group_id"] for r in dev}:
        raise ValueError("calibration and development share source groups")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True).to(device).eval()
    if model.config.num_labels != 2:
        raise ValueError("Expected a binary authorship classifier")
    max_length = metadata["max_length"]
    print(f"Scoring calibration ({len(cal)}) and development ({len(dev)}) on {device}", flush=True)
    cal_raw, cal_margin = score_margin(model, tokenizer, cal, device, args.batch_size, max_length)
    dev_raw, dev_margin = score_margin(model, tokenizer, dev, device, args.batch_size, max_length)

    original = joblib.load(args.model_dir / "calibration.joblib")
    old_dev = original.predict_proba(log_odds(dev_raw))[:, 1]
    platt = LogisticRegression(solver="lbfgs", random_state=42)
    platt.fit(cal_margin.reshape(-1, 1), [row["label"] for row in cal])
    platt_dev = platt.predict_proba(dev_margin.reshape(-1, 1))[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0., y_max=1.)
    isotonic.fit(cal_margin, [row["label"] for row in cal])
    isotonic_dev = isotonic.predict(dev_margin)

    labels_cal = np.asarray([row["label"] for row in cal])
    def temp_loss(log_temp):
        probabilities = np.clip(expit(cal_margin / np.exp(log_temp)), 1e-7, 1 - 1e-7)
        return log_loss(labels_cal, probabilities, labels=[0, 1])
    fit_temp = minimize_scalar(temp_loss, bounds=(np.log(.05), np.log(100)), method="bounded")
    temperature = float(np.exp(fit_temp.x))
    temperature_dev = expit(dev_margin / temperature)

    scores = {"current_platt": old_dev, "margin_platt": platt_dev,
              "margin_isotonic": isotonic_dev, "margin_temperature": temperature_dev,
              "raw_softmax_diagnostic_only": dev_raw}
    report = {
        "model_dir": str(args.model_dir),
        "calibration_sha256": sha256(paths["calibration"]),
        "dev_sha256": sha256(paths["dev"]),
        "checkpoint_sha256": sha256(args.model_dir / "model.safetensors"),
        "calibration_n": len(cal), "dev_n": len(dev), "temperature": temperature,
        "raw_rounded_to_1_on_dev": int(np.sum(dev_raw == 1)),
        "raw_rounded_to_0_on_dev": int(np.sum(dev_raw == 0)),
        "max_human_fpr": args.max_human_fpr,
        "methods": {name: metrics(dev, values, args.max_human_fpr)
                    for name, values in scores.items()},
        "note": "Exploratory development comparison; dev selected the original epoch and threshold. No locked test was read.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "dev_scores.npz", ids=np.array([r["id"] for r in dev]),
                        labels=np.array([r["label"] for r in dev]),
                        raw=dev_raw, margin=dev_margin,
                        **{k: v for k, v in scores.items() if k != "raw_softmax_diagnostic_only"})
    joblib.dump({"margin_platt": platt, "margin_isotonic": isotonic,
                 "temperature": temperature}, args.output_dir / "candidate_calibrators.joblib")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
