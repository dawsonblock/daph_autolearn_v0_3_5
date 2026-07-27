#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

from daph_latent_memory.config import load_config
from daph_latent_memory.runtime import load_frozen_model, primary_device
from daph_latent_memory.benchmarks.dataset import load_jsonl
from daph_latent_memory.teacher.capture import capture_teacher_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_frozen_model(cfg)
    device = primary_device(model)
    data = load_jsonl(args.dataset)[:args.limit]
    records = []
    for ex in data:
        h = capture_teacher_state(
            model, tokenizer, ex.question, ex.state.process_text(),
            cfg["training"]["teacher_layer_fraction"], device
        )
        records.append({"example_id": ex.example_id, "shape": list(h.shape)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print("captured", len(records), "teacher states")


if __name__ == "__main__":
    main()
