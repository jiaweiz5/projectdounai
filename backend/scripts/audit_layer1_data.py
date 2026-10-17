"""Read-only provenance, label, length and group-overlap audit for Layer 1."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FILES = [
    BASE / "data/raw/rednote_vibe/training_set_human.jsonl",
    BASE / "data/raw/rednote_vibe/training_set_aigc.jsonl",
    BASE / "data/processed/authorship.jsonl",
    *(BASE / f"data/splits/{name}.jsonl" for name in
      ("fit", "dev", "calibration", "test_locked")),
    BASE / "models/authorship_baseline.joblib",
    BASE / "models/authorship_calibrated.joblib",
    BASE / "models/authorship_provenance.json",
]


def bucket(text):
    if len(text) < 100:
        return "<100"
    if len(text) < 300:
        return "100-299"
    return "300+"


def main():
    origins = defaultdict(set)
    ids = defaultdict(set)
    missing_origin = Counter()
    for path in FILES:
        if not path.exists():
            print("MISSING", path.relative_to(BASE))
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"\n{path.relative_to(BASE)} sha256={digest}")
        if path.suffix != ".jsonl":
            if path.name == "authorship_provenance.json":
                print(path.read_text(encoding="utf-8"))
            continue
        counts = Counter()
        lengths = defaultdict(list)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row.get("source"), row.get("label"),
                       row.get("variant", "unspecified"))
                counts[key] += 1
                if "text" in row:
                    lengths[key].append(len(row["text"]))
                if path.parent.name == "splits":
                    origin = row.get("source_group_id")
                    if origin is None:
                        missing_origin[path.stem] += 1
                    else:
                        origins[origin].add(path.stem)
                    ids[row.get("id")].add(path.stem)
        for key, n in sorted(counts.items(), key=str):
            size = lengths[key]
            if size:
                ordered = sorted(size)
                buckets = Counter(bucket("x" * value) for value in size)
                print(key, "n=", n, "median length=", ordered[len(size)//2],
                      "length buckets=", dict(buckets))
            else:
                print(key, "n=", n)
    leaked = {key: sorted(names) for key, names in origins.items() if len(names) > 1}
    repeated = {key: sorted(names) for key, names in ids.items() if len(names) > 1}
    print("\nSource groups crossing splits:", len(leaked))
    print("IDs crossing splits:", len(repeated))
    print("Rows missing source_group_id:", dict(missing_origin))
    if leaked or repeated:
        print("First overlaps:", list(leaked.items())[:5], list(repeated.items())[:5])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
