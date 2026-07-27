#!/usr/bin/env python
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from daph_latent_memory.benchmarks.dataset import generate_dataset, save_jsonl


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    a = generate_dataset(args.n, args.seed)
    b = generate_dataset(args.n, args.seed)
    if [x.model_dump() for x in a] != [x.model_dump() for x in b]:
        raise SystemExit("FAIL: dataset generation is not deterministic")

    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "a.jsonl"
        p2 = Path(tmp) / "b.jsonl"
        save_jsonl(a, p1)
        save_jsonl(b, p2)
        result = {
            "examples": len(a),
            "seed": args.seed,
            "sha256_a": sha256(p1),
            "sha256_b": sha256(p2),
            "deterministic": sha256(p1) == sha256(p2),
        }
    print(json.dumps(result, indent=2))
    if not result["deterministic"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
