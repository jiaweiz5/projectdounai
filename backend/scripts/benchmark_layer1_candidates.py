"""Read-only comparison of current Layer 1 and Chinese V3 checkpoints.

Requires torch and transformers plus access to Hugging Face or cached weights.
Does not change the API model, calibrate scores, or train on the input file.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

BASE = Path(__file__).resolve().parent.parent
MODEL_IDS = {
    "zhv3": "yuchuantian/AIGC_detector_zhv3",
    "zhv3short": "yuchuantian/AIGC_detector_zhv3short",
}


def summarize(rows, scores, threshold, truncated=None):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups["ALL"].append(i)
        groups[f"variant={row.get('variant', 'unspecified')}"].append(i)
        if row["label"] == 0:
            groups[f"human_subtype={row.get('human_subtype', 'unspecified')}"].append(i)
        groups[f"length={'<100' if len(row['text']) < 100 else '100-299' if len(row['text']) < 300 else '300+'}"].append(i)
    results = {}
    for name, indices in sorted(groups.items()):
        labels = np.array([rows[i]["label"] for i in indices])
        predictions = scores[indices] >= threshold
        human = labels == 0
        ai = labels == 1
        result = {
            "n": len(indices), "human_n": int(human.sum()), "ai_n": int(ai.sum()),
            "human_false_positives": int(predictions[human].sum()),
            "human_fpr": float(predictions[human].mean()) if human.any() else None,
            "ai_detected": int(predictions[ai].sum()),
            "ai_recall": float(predictions[ai].mean()) if ai.any() else None,
        }
        if truncated is not None:
            result["truncated_n"] = int(truncated[indices].sum())
        results[name] = result
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Verified JSONL; use diagnostic/dev data for model choice, not a final locked test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--models", nargs="*", choices=MODEL_IDS, default=list(MODEL_IDS),
                        help="Choose V3 variants; pass --models alone to run current model only")
    parser.add_argument("--output", type=Path, help="Optional JSON report (no post text)")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Need nonempty data with unique IDs")
    for row in rows:
        if not row.get("text") or row.get("label") not in (0, 1):
            raise ValueError(f"Missing text or binary label: {row.get('id')}")
        variant = row.get("variant")
        if variant in ("H", "L1", "L2", "L3", "L4"):
            if row["label"] != (0 if variant == "H" else 1):
                raise ValueError(f"Variant/label mismatch: {row['id']}")

    texts = [row["text"] for row in rows]
    report = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "n": len(rows), "models": {}}
    bundle = joblib.load(BASE / "models/authorship_calibrated.joblib")
    logits = bundle["base_model"].decision_function(texts).reshape(-1, 1)
    baseline = bundle["calibrator"].predict_proba(logits)[:, 1]
    report["models"]["current"] = {
        "threshold": bundle.get("decision_threshold", 0.5),
        "metrics": summarize(rows, baseline, bundle.get("decision_threshold", 0.5)),
    }

    if args.models:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    for name in args.models:
        model_id = MODEL_IDS[name]
        print("Loading", model_id, "on", device, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=False
        ).to(device).eval()
        if model.config.num_labels != 2:
            raise ValueError(f"{model_id}: expected two classes; inspect label mapping")
        # The authors' public demo maps output 0 to human, 1 to AI.
        scores = []
        truncated = []
        for offset in range(0, len(texts), args.batch_size):
            batch = texts[offset:offset + args.batch_size]
            lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in batch]
            truncated.extend(length > 512 for length in lengths)
            inputs = tokenizer(batch, max_length=512, truncation=True,
                               padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model(**inputs).logits.softmax(dim=-1)[:, 1]
            scores.extend(output.detach().cpu().tolist())
        report["models"][name] = {
            "model_id": model_id, "threshold": 0.5,
            "checkpoint_commit": getattr(model.config, "_commit_hash", None),
            "metrics": summarize(rows, np.array(scores), 0.5, np.array(truncated)),
        }
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
