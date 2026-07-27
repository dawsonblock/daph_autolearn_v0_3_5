#!/usr/bin/env python
"""Calibrate the functional-mismatch loss margin m (v0.5.2 §3 Task 3.2).

The margin should be set to half the gap between the matched-latent NLL and
the shuffled-latent NLL on a small probe set, evaluated with an *untrained*
instance encoder (random latents). This gives a data-driven initial margin
in nats:

    functional_margin = 0.5 * (r_shuffled_mean - r_matched_mean)

A negative or near-zero margin means the model has no causal-specificity
signal even at initialization; in that case the script warns and writes a
small positive floor (0.5 nats) so the hinge is not degenerate.

Usage:
    PYTHONPATH=src:scripts python scripts/calibrate_margin.py \
        --config configs/qwen25_1_5b.yaml \
        --dataset data/v052_benchmark_5000.jsonl \
        --probe-size 100 \
        --output runs/margin_calib_1_5b.json

The output JSON is consumed by train_instance_encoder.py via
--margin-override <path> (or --calibrate-margin to run inline).
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from daph_latent_memory.config import load_config
from daph_latent_memory.runtime import load_frozen_model, primary_device
from daph_latent_memory.benchmarks.dataset import load_jsonl
from daph_latent_memory.latent.skill_bank import SkillBank
from daph_latent_memory.latent.instance_state import InstanceEncoder
from daph_latent_memory.latent.composer import make_composer
from daph_latent_memory.latent.injection import LatentInjector
from daph_latent_memory.training.collate import tokenize_batch
from daph_latent_memory.training.losses import compute_R

SKILL_IDS = {"add": 0, "subtract": 1, "multiply": 2, "divide": 3, "verify": 4, "decompose": 5}
DEFAULT_FLOOR_MARGIN = 0.5  # nats; used when the calibrated margin is <= 0


def calibrate(
    cfg: dict,
    dataset_path: str,
    probe_size: int = 100,
    seed: int | None = None,
) -> dict:
    """Run the calibration probe and return a report dict.

    Returns:
        {
          "functional_margin": float,
          "r_matched_mean": float,
          "r_shuffled_mean": float,
          "gap": float,
          "n_probe": int,
          "used_floor": bool,
          "seed": int,
        }
    """
    if seed is None:
        seed = cfg.get("seed", 1337)
    random.seed(seed)
    torch.manual_seed(seed)

    model, tokenizer = load_frozen_model(cfg)
    device = primary_device(model)
    hidden = model.get_input_embeddings().embedding_dim
    mem = cfg["memory"]

    # Untrained components — random initialization gives a neutral baseline.
    skill_bank = SkillBank(mem["num_skills"], mem["latent_tokens"], mem["skill_dim"]).to(device)
    instance_encoder = InstanceEncoder(
        len(tokenizer),
        min(512, hidden),
        mem["encoder_width"],
        mem["latent_tokens"],
        mem["instance_dim"],
        mem["encoder_layers"],
        mem["dropout"],
    ).to(device)
    composer = make_composer(
        mem.get("composition", "gated"),
        mem["skill_dim"],
        mem["instance_dim"],
        hidden,
        mem["latent_tokens"],
    ).to(device)
    injector = LatentInjector(model)

    skill_bank.eval()
    instance_encoder.eval()
    composer.eval()

    # Probe set: IID examples (not train) so the calibration reflects
    # generalization, not memorization.
    all_data = load_jsonl(dataset_path)
    probe_pool = [x for x in all_data if x.split == "iid"]
    if len(probe_pool) < probe_size:
        # Fall back to train examples if IID is too small.
        probe_pool = [x for x in all_data if x.split == "train"]
    if len(probe_pool) < 2:
        raise ValueError(
            f"calibration needs >= 2 probe examples, got {len(probe_pool)}"
        )
    rng = random.Random(seed)
    rng.shuffle(probe_pool)
    probe = probe_pool[:probe_size]

    tcfg = cfg["training"]
    r_matched_list: list[float] = []
    r_shuffled_list: list[float] = []

    with torch.inference_mode():
        for ex in probe:
            p, s, a = tokenize_batch(
                [ex], tokenizer,
                tcfg["max_prompt_tokens"],
                tcfg["max_answer_tokens"],
            )
            p = {k: v.to(device) for k, v in p.items()}
            s = {k: v.to(device) for k, v in s.items()}
            a = {k: v.to(device) for k, v in a.items()}

            skill_id = torch.tensor(
                [SKILL_IDS.get(ex.skill_label, 0)], device=device
            )
            z_skill = skill_bank.get_skill_batch(skill_id)
            z_instance = instance_encoder(s["input_ids"], s["attention_mask"])
            z_matched = composer(z_skill, z_instance)

            r_matched = compute_R(
                model, injector,
                p["input_ids"], p["attention_mask"],
                a["input_ids"], a["attention_mask"],
                z_matched,
            )
            r_matched_list.append(float(r_matched.item()))

            # Shuffled latent: a different example's composed latent.
            other = rng.choice(probe)
            if other.example_id == ex.example_id:
                # avoid self-match
                other = rng.choice(probe)
            sp = tokenizer(
                other.state.process_text(),
                return_tensors="pt",
                truncation=True,
                max_length=mem["max_state_tokens"],
            )
            sp = {k: v.to(device) for k, v in sp.items()}
            other_skill_id = torch.tensor(
                [SKILL_IDS.get(other.skill_label, 0)], device=device
            )
            z_other_skill = skill_bank.get_skill_batch(other_skill_id)
            z_other_inst = instance_encoder(sp["input_ids"], sp["attention_mask"])
            z_shuffled = composer(z_other_skill, z_other_inst)

            r_shuffled = compute_R(
                model, injector,
                p["input_ids"], p["attention_mask"],
                a["input_ids"], a["attention_mask"],
                z_shuffled,
            )
            r_shuffled_list.append(float(r_shuffled.item()))

    r_matched_mean = sum(r_matched_list) / len(r_matched_list)
    r_shuffled_mean = sum(r_shuffled_list) / len(r_shuffled_list)
    gap = r_shuffled_mean - r_matched_mean
    raw_margin = 0.5 * gap
    used_floor = False
    if raw_margin <= 0:
        used_floor = True
        functional_margin = DEFAULT_FLOOR_MARGIN
    else:
        functional_margin = raw_margin

    return {
        "functional_margin": functional_margin,
        "r_matched_mean": r_matched_mean,
        "r_shuffled_mean": r_shuffled_mean,
        "gap": gap,
        "n_probe": len(probe),
        "used_floor": used_floor,
        "seed": seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate L_functional margin m")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--probe-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None,
                    help="Override config seed (default: use config seed)")
    ap.add_argument("--output", required=True,
                    help="Path to write the calibration JSON report")
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = calibrate(cfg, args.dataset, probe_size=args.probe_size, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("FUNCTIONAL MARGIN CALIBRATION REPORT")
    print("=" * 60)
    print(f"  probe size:      {report['n_probe']}")
    print(f"  seed:            {report['seed']}")
    print(f"  r_matched_mean:  {report['r_matched_mean']:.4f} nats")
    print(f"  r_shuffled_mean: {report['r_shuffled_mean']:.4f} nats")
    print(f"  gap:             {report['gap']:.4f} nats")
    print(f"  raw margin:      {0.5 * report['gap']:.4f} nats")
    print(f"  used floor:      {report['used_floor']}")
    print(f"  FUNCTIONAL MARGIN m = {report['functional_margin']:.4f} nats")
    print("=" * 60)
    print(f"  Report written to: {out_path}")
    print("  Use with: train_instance_encoder.py --margin-override " + str(out_path))


if __name__ == "__main__":
    main()
