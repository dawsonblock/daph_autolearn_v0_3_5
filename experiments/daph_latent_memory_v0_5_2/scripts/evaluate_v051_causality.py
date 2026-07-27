#!/usr/bin/env python
"""Phase 1 diagnostic: Causal evaluation of v0.5.1 latents.

Loads the persisted v0.5.1 latents from Phase 0 and runs the full causal
evaluation framework on them. This is expected to demonstrate the v0.5.1
failure mode: the hidden_alignment_loss (cosine alignment to teacher
hidden states) produces latents that do NOT carry causally specific
information. We expect:
  - matched accuracy ≈ shuffled accuracy (no causal specificity)
  - CSS near 0, gate fails
  - WMP near 0 (wrong-class memory doesn't hurt because latents are generic)

This baseline is what v0.5.2's redesigned losses (L_functional,
L_disentangle, L_invariance) are meant to fix.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from daph_latent_memory.config import load_config
from daph_latent_memory.runtime import load_frozen_model, primary_device
from daph_latent_memory.latent.injection import LatentInjector
from daph_latent_memory.evaluation.controls import (
    random_latents_like, random_norm_matched_latents_like,
    shuffled_global, shuffled_same_class, shuffled_wrong_class,
    nearest_neighbor_wrong, farthest_neighbor,
    normalized_latents, scaled_latents, sign_flipped_latents,
    mean_latent, centroid_latent, pca_low_rank, pca_residual,
    corrupt_latent_vector,
)
from daph_latent_memory.evaluation.metrics import (
    exact_match, causal_specificity_score, wrong_memory_penalty,
    css_null_distribution, css_threshold,
)
from daph_latent_memory.evaluation.statistics import (
    mcnemar_test, cochran_q_test, wilcoxon_signed_rank, paired_bootstrap_ci,
)


def _generate_with_latents(model, tokenizer, prompt_ids, prompt_mask, latents, device, max_new):
    """Generate text with latent tokens prepended after the prompt."""
    embed = model.get_input_embeddings()
    prompt_emb = embed(prompt_ids)
    latents = latents.to(device=prompt_emb.device, dtype=prompt_emb.dtype)
    inputs_embeds = torch.cat([prompt_emb, latents], dim=1)
    mask = torch.cat([
        prompt_mask,
        torch.ones(prompt_mask.size(0), latents.size(1), dtype=prompt_mask.dtype, device=device)
    ], dim=1)
    out = model.generate(
        inputs_embeds=inputs_embeds, attention_mask=mask,
        max_new_tokens=max_new, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def _base_generate(model, tokenizer, prompts, device, max_new):
    """Generate text without any latent injection (baseline)."""
    tok = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model.generate(
        **tok, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    prefix = tok["input_ids"].size(1)
    if out.size(1) >= prefix:
        out = out[:, prefix:]
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description="Phase 1: Causal eval of v0.5.1 latents")
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--latents-dir", required=True, help="Path to latents_v0_5_1/ from Phase 0")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    eval_cfg = load_config(args.eval_config)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_frozen_model(cfg)
    device = primary_device(model)
    mem = cfg["memory"]
    max_new = cfg["evaluation"]["generation_max_new_tokens"]
    seed = eval_cfg.get("seed", cfg["seed"])

    # Load persisted v0.5.1 latents and metadata
    latents_dir = Path(args.latents_dir)
    with open(latents_dir / "examples.json") as f:
        examples = json.load(f)

    # Load matched latents (the encoded state latents)
    all_z = torch.load(latents_dir / "matched.pt", map_location="cpu", weights_only=False)
    constant_z = torch.load(latents_dir / "constant.pt", map_location="cpu", weights_only=False)

    # Limit examples
    n = min(args.limit, len(examples), all_z.size(0))
    examples = examples[:n]
    all_z = all_z[:n]
    labels = [ex.get("skill_label") or ex.get("operation") for ex in examples]

    print(f"Phase 1: Causal evaluation of v0.5.1 latents")
    print(f"  Examples: {n}")
    print(f"  Latent shape: {all_z.shape}")
    print(f"  Labels: {set(labels)}")
    print()

    # Define all conditions using the persisted/computed latents
    condition_fns = {
        "matched": lambda i: all_z[i:i+1],
        "zero": lambda i: torch.zeros_like(all_z[i:i+1]),
        "constant": lambda i: constant_z,
        "random_gaussian": lambda i: random_latents_like(all_z[i:i+1], seed=seed+i),
        "random_norm_matched": lambda i: random_norm_matched_latents_like(all_z[i:i+1], seed=seed+i),
        "shuffled_global": lambda i: shuffled_global(all_z, seed=seed)[i:i+1],
        "shuffled_same_class": lambda i: shuffled_same_class(all_z, labels, seed=seed)[i:i+1],
        "shuffled_wrong_class": lambda i: shuffled_wrong_class(all_z, labels, seed=seed)[i:i+1],
        "nearest_neighbor_wrong": lambda i: nearest_neighbor_wrong(all_z, labels)[i:i+1],
        "farthest_neighbor": lambda i: farthest_neighbor(all_z)[i:i+1],
        "normalized": lambda i: normalized_latents(all_z[i:i+1]),
        "scaled": lambda i: scaled_latents(all_z[i:i+1], c=2.0),
        "sign_flipped": lambda i: sign_flipped_latents(all_z[i:i+1]),
        "mean_latent": lambda i: mean_latent(all_z)[i:i+1],
        "centroid_latent": lambda i: centroid_latent(all_z, labels)[i:i+1],
        "pca_low_rank": lambda i: pca_low_rank(all_z, k=8)[i:i+1],
        "pca_residual": lambda i: pca_residual(all_z, k=8)[i:i+1],
    }

    results = {"conditions": {}, "corruption_sweep": {}, "statistics": {}, "metadata": {
        "n_examples": n,
        "latent_shape": list(all_z.shape),
        "latents_dir": str(latents_dir),
        "config": args.config,
        "eval_config": args.eval_config,
        "phase": "v0.5.1 baseline (Phase 0 latents)",
    }}

    # Generate under each condition
    for cond_name, fn in condition_fns.items():
        correct_list = []
        for i, ex in enumerate(tqdm(examples, desc=f"  {cond_name}")):
            prompt = f"Question:\n{ex['question']}\nAnswer:"
            p = tokenizer(prompt, return_tensors="pt", truncation=True)
            p = {k: v.to(device) for k, v in p.items()}
            z = fn(i)
            texts = _generate_with_latents(model, tokenizer, p["input_ids"], p["attention_mask"], z, device, max_new)
            correct_list.append(exact_match(texts[0], ex["answer"]))
        accuracy = sum(correct_list) / len(correct_list)
        results["conditions"][cond_name] = {"accuracy": accuracy, "correct": correct_list}
        print(f"  {cond_name}: A={accuracy:.3f}")

    # Base condition (no latent injection)
    base_correct = []
    for ex in tqdm(examples, desc="  base"):
        prompt = f"Question:\n{ex['question']}\nAnswer:"
        texts = _base_generate(model, tokenizer, [prompt], device, max_new)
        base_correct.append(exact_match(texts[0], ex["answer"]))
    results["conditions"]["base"] = {"accuracy": sum(base_correct) / len(base_correct), "correct": base_correct}
    print(f"  base: A={sum(base_correct)/len(base_correct):.3f}")

    # Corruption sweep
    alphas = eval_cfg.get("corruption", {}).get("alphas", [0.0, 0.05, 0.1, 0.5, 1.0, 2.0])
    per_alpha_seeds = eval_cfg.get("corruption", {}).get("per_alpha_seeds", 3)
    corruption_results = {}
    for alpha in alphas:
        alpha_correct_all = []
        for s in range(per_alpha_seeds):
            correct_list = []
            for i, ex in enumerate(examples):
                prompt = f"Question:\n{ex['question']}\nAnswer:"
                p = tokenizer(prompt, return_tensors="pt", truncation=True)
                p = {k: v.to(device) for k, v in p.items()}
                z = corrupt_latent_vector(all_z[i:i+1], alpha, seed=seed + s * 1000 + i)
                texts = _generate_with_latents(model, tokenizer, p["input_ids"], p["attention_mask"], z, device, max_new)
                correct_list.append(exact_match(texts[0], ex["answer"]))
            alpha_correct_all.extend(correct_list)
        accuracy = sum(alpha_correct_all) / len(alpha_correct_all)
        corruption_results[str(alpha)] = {"accuracy": accuracy, "correct": alpha_correct_all}
        print(f"  corruption α={alpha}: A={accuracy:.3f}")
    results["corruption_sweep"] = corruption_results

    # Statistics
    matched_c = results["conditions"]["matched"]["correct"]
    shuffled_c = results["conditions"]["shuffled_global"]["correct"]
    wrong_c = results["conditions"]["shuffled_wrong_class"]["correct"]
    results["statistics"]["mcnemar_matched_vs_shuffled"] = mcnemar_test(matched_c, shuffled_c)
    results["statistics"]["mcnemar_matched_vs_wrong"] = mcnemar_test(matched_c, wrong_c)

    # CSS null distribution
    null_css = css_null_distribution(
        matched_c, shuffled_c,
        n_permutations=eval_cfg.get("statistics", {}).get("null_permutations", 200),
        seed=seed,
    )
    threshold = css_threshold(
        null_css,
        percentile=eval_cfg.get("statistics", {}).get("null_percentile", 99),
        floor_pp=eval_cfg.get("statistics", {}).get("css_floor_pp", 5.0),
    )
    results["statistics"]["css_null_distribution"] = null_css
    results["statistics"]["css_threshold"] = threshold

    # Cochran's Q for corruption sweep
    corruption_conditions = [corruption_results[a]["correct"] for a in sorted(corruption_results.keys(), key=float)]
    if len(corruption_conditions) >= 3:
        results["statistics"]["cochran_q_corruption"] = cochran_q_test(corruption_conditions)

    # Headline metrics
    a_matched = results["conditions"]["matched"]["accuracy"]
    a_shuffled = results["conditions"]["shuffled_global"]["accuracy"]
    a_wrong = results["conditions"]["shuffled_wrong_class"]["accuracy"]
    a_base = results["conditions"]["base"]["accuracy"]
    results["headline"] = {
        "css": causal_specificity_score(a_matched, a_shuffled),
        "wmp": wrong_memory_penalty(a_matched, a_wrong),
        "lug": (a_matched - a_base) / (mem["latent_tokens"] + 1),
        "a_matched": a_matched,
        "a_shuffled": a_shuffled,
        "a_wrong": a_wrong,
        "a_base": a_base,
        "css_passes_gate": (
            causal_specificity_score(a_matched, a_shuffled) >= threshold
            and a_matched > a_base
        ),
        "expected_failure": True,  # v0.5.1 is expected to fail the gate
    }

    with open(outdir / "v051_causal_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase 1 (v0.5.1 baseline) Results:")
    print(f"  A_matched:  {a_matched:.3f}")
    print(f"  A_shuffled: {a_shuffled:.3f}")
    print(f"  A_wrong:    {a_wrong:.3f}")
    print(f"  A_base:     {a_base:.3f}")
    print(f"  CSS:        {results['headline']['css']:.1f}pp (threshold: {threshold:.1f}pp)")
    print(f"  WMP:        {results['headline']['wmp']:.1f}pp")
    print(f"  LUG:        {results['headline']['lug']:.4f}")
    print(f"  Gate:       {'PASS' if results['headline']['css_passes_gate'] else 'FAIL'}")
    print(f"  Expected:   FAIL (v0.5.1 alignment loss doesn't produce causal specificity)")
    print(f"{'='*60}")
    print(f"\nResults saved to {outdir / 'v051_causal_eval.json'}")


if __name__ == "__main__":
    main()
