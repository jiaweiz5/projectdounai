"""Read-only grouped evaluation of a fine-tuned calibrated checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from benchmark_layer1_candidates import summarize
from fine_tune_layer1 import BASE, score_model


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=BASE / "data/splits/test_locked.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    test = read_jsonl(args.test_file)
    if not test or any(r.get("label") not in (0, 1) or not r.get("source_group_id") for r in test):
        raise ValueError("Need verified binary labels and source_group_id")
    for row in test:
        variant = row.get("variant")
        if variant in ("H", "L1", "L2", "L3", "L4"):
            if row["label"] != (0 if variant == "H" else 1):
                raise ValueError(f"Variant/label mismatch: {row['id']}")
    test_ids = {r["id"] for r in test}
    test_groups = {r["source_group_id"] for r in test}
    if len(test_ids) != len(test):
        raise ValueError("Duplicate test IDs")
    for name in ("fit", "dev", "calibration"):
        for row in read_jsonl(BASE / "data/splits" / f"{name}.jsonl"):
            if row["id"] in test_ids or row["source_group_id"] in test_groups:
                raise ValueError(f"Test overlaps {name}: {row['id']}")
    if args.test_file.resolve() != (BASE / "data/splits/test_locked.jsonl").resolve():
        for row in read_jsonl(BASE / "data/splits/test_locked.jsonl"):
            if row["id"] in test_ids or row["source_group_id"] in test_groups:
                raise ValueError(f"External test overlaps old locked test: {row['id']}")
    metadata = json.loads((args.model_dir / "training_metadata.json").read_text(encoding="utf-8"))
    decision = json.loads((args.model_dir / "decision.json").read_text(encoding="utf-8"))
    for name in ("fit", "dev", "calibration"):
        expected = metadata.get(f"{name}_sha256", decision.get(f"{name}_sha256"))
        actual = hashlib.sha256((BASE / "data/splits" / f"{name}.jsonl").read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"{name} split changed after training/calibration")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(device).eval()
    raw = score_model(model, tokenizer, test, device, args.batch_size, metadata["max_length"])
    p = np.clip(raw, 1e-7, 1 - 1e-7)
    logits = np.log(p / (1-p)).reshape(-1, 1)
    calibrated = joblib.load(args.model_dir / "calibration.joblib").predict_proba(logits)[:, 1]
    truncated = np.array([len(tokenizer(r["text"], truncation=False)["input_ids"]) > metadata["max_length"]
                          for r in test])
    print(json.dumps({
        "test_file": str(args.test_file),
        "test_sha256": hashlib.sha256(args.test_file.read_bytes()).hexdigest(),
        "base_model": metadata["base_model"], "selected_epoch": metadata["epoch"],
        "threshold": decision["threshold"],
        "metrics": summarize(test, calibrated, decision["threshold"], truncated),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
