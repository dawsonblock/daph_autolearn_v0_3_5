#!/usr/bin/env python
"""Generate the v0.5.2 dataset with all OOD splits."""
from __future__ import annotations
import argparse
from daph_latent_memory.benchmarks.dataset import generate_dataset,save_jsonl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=4000)
    ap.add_argument("--seed",type=int,default=1337)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    examples=generate_dataset(args.n,args.seed)
    save_jsonl(examples,args.output)
    from collections import Counter
    counts=Counter(e.split for e in examples)
    print(f"Generated {len(examples)} examples:")
    for split,count in sorted(counts.items()): print(f"  {split}: {count}")

if __name__=="__main__": main()
