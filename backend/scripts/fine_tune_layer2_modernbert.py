"""Train an isolated ChineseModernBert candidate for text-only covert promotion.

Run from any directory. Does not read CHASM test or replace the serving model.
The published CHASM validation split selects the epoch and operating point.
An internal, stratified portion of CHASM train is reserved for later calibration.
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


BASE = Path(__file__).resolve().parent.parent
MODEL_ID = "qihoo360/Zhinao-ChineseModernBert"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or {row.get("label") for row in rows} != {0, 1}:
        raise ValueError(f"Expected both binary labels in {path}")
    if any(not isinstance(row.get("text"), str) or not row["text"].strip() or
           not row.get("id") for row in rows):
        raise ValueError(f"Expected a text and ID for every row in {path}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def normalized(row):
    return re.sub(r"\s+", "", row["text"]).casefold()


def select_threshold(rows, scores, max_fpr):
    labels = np.asarray([r["label"] for r in rows])
    # Each unique score is a possible decision boundary; include a zero-positive
    # prediction point. With 128 validation negatives, 5% allows at most 6 FP.
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    best = None
    for threshold in candidates:
        predicted = scores >= threshold
        fp = int(np.sum(predicted & (labels == 0)))
        tp = int(np.sum(predicted & (labels == 1)))
        fpr = fp / int(np.sum(labels == 0))
        if fpr <= max_fpr:
            key = (tp, -fp, float(threshold))
            if best is None or key > best[0]:
                best = (key, {"threshold_raw": float(threshold), "fp": fp,
                              "tn": int(np.sum(labels == 0)) - fp,
                              "tp": tp, "fn": int(np.sum(labels == 1)) - tp,
                              "non_covert_fpr": fpr,
                              "covert_recall": tp / int(np.sum(labels == 1))})
    return best[1]


def score(model, tokenizer, rows, device, batch_size, max_length):
    import torch

    model.eval()
    values = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        inputs = tokenizer([r["text"] for r in batch], padding=True,
                           truncation=True, max_length=max_length,
                           return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        values.extend(logits.float().softmax(dim=-1)[:, 1].cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-non-covert-fpr", type=float, default=0.05)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="One training batch and one validation batch; write no artifacts")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("epochs, batch-size, and gradient-accumulation must be positive")
    if not 1 <= args.max_length <= 1536 or args.learning_rate <= 0:
        parser.error("max-length must be 1..1536 and learning-rate must be positive")
    if not 0 <= args.max_non_covert_fpr < 1 or not 0 < args.calibration_fraction < .5:
        parser.error("invalid FPR limit or calibration fraction")
    if not args.smoke_test and args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty so previous runs cannot be overwritten")

    train_path = BASE / "data/processed/chasm_train.jsonl"
    val_path = BASE / "data/processed/chasm_validation.jsonl"
    train, val = load_rows(train_path), load_rows(val_path)
    if {r["id"] for r in train} & {r["id"] for r in val}:
        raise ValueError("Train/validation IDs overlap")
    if {normalized(r) for r in train} & {normalized(r) for r in val}:
        raise ValueError("Train/validation texts overlap after whitespace normalization")

    train_idx, cal_idx = train_test_split(
        np.arange(len(train)), test_size=args.calibration_fraction,
        stratify=[r["label"] for r in train], random_state=args.seed)
    fit = [train[i] for i in train_idx]
    cal = [train[i] for i in cal_idx]
    if {normalized(r) for r in fit} & {normalized(r) for r in cal}:
        raise ValueError("Fit/calibration texts overlap")

    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # This specific model's published installation notes require Transformers 4.56.2+.
    from packaging.version import Version
    if Version(transformers.__version__) < Version("4.56.2"):
        raise RuntimeError("Install transformers>=4.56.2,<5 before training")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model={MODEL_ID} device={device} torch={torch.__version__} "
          f"transformers={transformers.__version__}", flush=True)
    print(f"fit={len(fit)} {dict(Counter(r['label'] for r in fit))} "
          f"calibration_reserved={len(cal)} {dict(Counter(r['label'] for r in cal))} "
          f"validation={len(val)} {dict(Counter(r['label'] for r in val))}", flush=True)
    print("CHASM rows have no parent group IDs; near-duplicate overlap is not proven absent.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, trust_remote_code=False,
        attn_implementation="eager").to(device)
    # A randomly initialized classifier head is expected for an MLM checkpoint.
    if model.config.num_labels != 2:
        raise RuntimeError("Expected binary classifier")

    def collate(batch):
        encoded = tokenizer([r["text"] for r in batch], padding=True,
                            truncation=True, max_length=args.max_length,
                            return_tensors="pt")
        encoded["labels"] = torch.tensor([r["label"] for r in batch], dtype=torch.long)
        return encoded

    loader = DataLoader(fit, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate)
    counts = Counter(r["label"] for r in fit)
    weights = torch.tensor([len(fit) / (2 * counts[i]) for i in (0, 1)],
                           dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=0.01)
    if args.smoke_test:
        model.train()
        batch = next(iter(loader))
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        loss.backward()
        sample = score(model, tokenizer, val[:args.batch_size], device,
                       args.batch_size, args.max_length)
        print(json.dumps({"smoke_test": "passed", "loss": float(loss.detach().cpu()),
                          "sample_scores": sample.tolist()}), flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration_reserved.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cal), encoding="utf-8")
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for step, batch in enumerate(loader, 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
            (loss / args.gradient_accumulation).backward()
            loss_total += float(loss.detach().cpu())
            if step % args.gradient_accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0 or step == len(loader):
                print(f"epoch={epoch} batch={step}/{len(loader)}", flush=True)
        scores = score(model, tokenizer, val, device, args.batch_size, args.max_length)
        metrics = select_threshold(val, scores, args.max_non_covert_fpr)
        auc = float(roc_auc_score([r["label"] for r in val], scores))
        report = {"epoch": epoch, "mean_fit_loss": loss_total / len(loader),
                  "validation_auc": auc, **metrics}
        print(json.dumps(report, ensure_ascii=False), flush=True)
        with (args.output_dir / "epoch_metrics.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(report) + "\n")
        quality = (metrics["tp"], -metrics["fp"], auc)
        if best is None or quality > best:
            best = quality
            checkpoint = args.output_dir / "best"
            checkpoint.mkdir(exist_ok=True)
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            metadata = {"base_model": MODEL_ID, "selected_epoch": epoch,
                        "device": str(device), "torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "train_sha256": sha256(train_path), "validation_sha256": sha256(val_path),
                        "fit_count": len(fit), "reserved_calibration_count": len(cal),
                        "validation_count": len(val), "max_length": args.max_length,
                        "batch_size": args.batch_size,
                        "gradient_accumulation": args.gradient_accumulation,
                        "learning_rate": args.learning_rate, "seed": args.seed,
                        "max_non_covert_fpr": args.max_non_covert_fpr,
                        "validation_at_selected_threshold": report}
            (args.output_dir / "training_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved selected checkpoint at epoch {epoch}: {checkpoint}", flush=True)
    print(f"Finished {args.epochs} epochs. Serving model untouched.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Training stopped: {exc}", file=sys.stderr, flush=True)
        raise
