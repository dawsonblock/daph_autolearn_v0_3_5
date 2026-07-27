#!/usr/bin/env python
from __future__ import annotations
import argparse
from collections import Counter
from daph_latent_memory.benchmarks.dataset import generate_dataset, save_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    data = generate_dataset(args.n, args.seed)
    save_jsonl(data, args.out)
    print("wrote", len(data), "examples to", args.out)
    print("splits:", Counter(x.split for x in data))
    print("templates:", Counter(x.template for x in data))


if __name__ == "__main__":
    main()
