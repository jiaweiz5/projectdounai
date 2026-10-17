"""Fine-tune a Chinese V3 classifier using fit only; select epoch on dev.

Run from backend/. Keeps the deployed Layer 1 model untouched.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from benchmark_layer1_candidates import MODEL_IDS

BASE = Path(__file__).resolve().parent.parent


def read_rows(path):
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows or {row["label"] for row in rows} != {0, 1}:
        raise ValueError(f"{path} must have both verified binary labels")
    if any(not row.get("source_group_id") or not row.get("text") for row in rows):
        raise ValueError(f"{path} needs source_group_id and complete text")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path} has duplicate IDs")
    return rows


def score_model(model, tokenizer, rows, device, batch_size, max_length):
    import torch

    model.eval()
    scores = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        encoded = tokenizer([r["text"] for r in batch], padding=True,
                            truncation=True, max_length=max_length,
                            return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        scores.extend(logits.softmax(-1)[:, 1].detach().cpu().tolist())
    return np.array(scores)


def operating_point(rows, scores, max_fpr):
    labels = np.array([r["label"] for r in rows])
    feasible = []
    for threshold in sorted(set(scores.tolist() + [float(np.nextafter(scores.max(), np.inf))])):
        predicted = scores >= threshold
        fpr = float(predicted[labels == 0].mean())
        if fpr <= max_fpr:
            recall = float(predicted[labels == 1].mean())
            feasible.append((recall, -fpr, threshold))
    recall, negative_fpr, threshold = max(feasible)
    return {"dev_ai_recall": recall, "dev_human_fpr": -negative_fpr,
            "dev_threshold": threshold, "dev_auc": float(roc_auc_score(labels, scores))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=MODEL_IDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-human-fpr", type=float, default=0.05)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 1 <= args.max_length <= 512:
        parser.error("epochs/batch-size must be positive; max-length must be 1..512")
    if not 0 <= args.max_human_fpr < 1:
        parser.error("max-human-fpr must be between 0 and 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty; never overwrite a previous run")

    fit_path, dev_path = (BASE / "data/splits" / f"{name}.jsonl" for name in ("fit", "dev"))
    fit, dev = read_rows(fit_path), read_rows(dev_path)
    fit_groups = {r["source_group_id"] for r in fit}
    if fit_groups & {r["source_group_id"] for r in dev}:
        raise ValueError("Fit and dev share original posts")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    name = MODEL_IDS[args.base]
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=False).to(device)
    if model.config.num_labels != 2:
        raise ValueError("Expected class 0=human, class 1=AI")

    def collate(batch):
        encoded = tokenizer([r["text"] for r in batch], padding=True,
                            truncation=True, max_length=args.max_length,
                            return_tensors="pt")
        encoded["labels"] = torch.tensor([r["label"] for r in batch], dtype=torch.long)
        return encoded

    loader = DataLoader(fit, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate)
    counts = Counter(r["label"] for r in fit)
    weights = torch.tensor([len(fit)/(2 * counts[0]), len(fit)/(2 * counts[1])],
                           dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        scores = score_model(model, tokenizer, dev, device, args.batch_size, args.max_length)
        metrics = operating_point(dev, scores, args.max_human_fpr)
        quality = (metrics["dev_ai_recall"], -metrics["dev_human_fpr"], metrics["dev_auc"])
        print(json.dumps({"epoch": epoch, "mean_fit_loss": total_loss / len(loader), **metrics}, indent=2), flush=True)
        if best is None or quality > best:
            best = quality
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
            metadata = {
                "base_model": name, "checkpoint_commit": getattr(model.config, "_commit_hash", None),
                "epoch": epoch, "max_length": args.max_length, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "seed": 42,
                "fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
                "dev_sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
                "fit_count": len(fit), "dev_count": len(dev), **metrics,
            }
            (args.output_dir / "training_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Saved best checkpoint to", args.output_dir)


if __name__ == "__main__":
    main()
