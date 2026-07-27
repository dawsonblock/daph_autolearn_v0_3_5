#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from daph_latent_memory.config import load_config
from daph_latent_memory.runtime import load_frozen_model, primary_device
from daph_latent_memory.benchmarks.dataset import load_jsonl
from daph_latent_memory.latent.encoder import StateTokenEncoder
from daph_latent_memory.latent.constant import ConstantLatent
from daph_latent_memory.evaluation.controls import (
    random_latents_like,
    deranged_states,
    corrupted_states,
)
from daph_latent_memory.evaluation.metrics import exact_match, bootstrap_ci, utility
from daph_latent_memory.evaluation.leakage import leakage_probe


def _decode_tokens(tokenizer, token_ids: torch.Tensor) -> tuple[list[str], list[int]]:
    """Decode a [batch, seq] tensor of token IDs into text and per-row token counts."""
    if token_ids.ndim != 2:
        raise ValueError("Expected [batch, sequence] generated token IDs")
    texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    counts = [int(row.numel()) for row in token_ids]
    return texts, counts


def generate_from_embeds(
    model,
    tokenizer,
    prompt_ids,
    prompt_mask,
    latents,
    max_new,
    *,
    embeds_return_includes_prefix: bool = False,
):
    """Generate from inputs_embeds (prompt embeddings + latent tokens).

    HF Transformers `generate(inputs_embeds=...)` returns ONLY the newly generated
    token IDs in transformers>=4.51: it cannot prepend the embeds prefix because
    there are no input IDs to echo back. Therefore the entire output tensor is the
    continuation by default.

    Do NOT infer prefix semantics from sequence length. The previous heuristic
    `if sequences.size(1) > prefix_len: sequences[:, prefix_len:]` was unsafe: when
    the model generated more tokens than the prefix length (e.g. prefix=28,
    generated=40), it would incorrectly strip the first 28 generated tokens,
    corrupting prediction text, answer extraction, generated-token counts,
    accuracy, and utility.

    If a specific transformers version is verified to prepend the prefix as IDs,
    set `embeds_return_includes_prefix=True` (or via config
    `evaluation.embeds_generate_includes_prefix`).
    """
    embed = model.get_input_embeddings()
    prompt_emb = embed(prompt_ids)
    latents = latents.to(device=prompt_emb.device, dtype=prompt_emb.dtype)
    inputs_embeds = torch.cat([prompt_emb, latents], dim=1)
    latent_mask = torch.ones(
        (prompt_mask.size(0), latents.size(1)),
        dtype=prompt_mask.dtype,
        device=prompt_mask.device,
    )
    mask = torch.cat([prompt_mask, latent_mask], dim=1)
    out = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=mask,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if embeds_return_includes_prefix:
        prefix_len = inputs_embeds.size(1)
        if out.size(1) >= prefix_len:
            out = out[:, prefix_len:]
        # If output is shorter than the prefix (unexpected when prefix is
        # included), leave it intact to avoid an empty decode.
    return _decode_tokens(tokenizer, out)


def base_generate(model, tokenizer, texts, device, max_new):
    """Generate from input_ids.

    With input_ids, `generate` always returns the full sequence (prompt prefix +
    continuation), so strip the prompt prefix explicitly by length.
    """
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model.generate(
        **tok,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prefix_len = tok["input_ids"].size(1)
    if out.size(1) >= prefix_len:
        out = out[:, prefix_len:]
    return _decode_tokens(tokenizer, out)


def encode_states(encoder, tokenizer, states, device, max_state_tokens):
    texts = [s.process_text() for s in states]
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_state_tokens,
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    with torch.no_grad():
        return encoder(tok["input_ids"], tok["attention_mask"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit-per-split", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_frozen_model(cfg)
    device = primary_device(model)
    hidden = model.get_input_embeddings().embedding_dim
    mem = cfg["memory"]

    encoder = StateTokenEncoder(
        vocab_size=len(tokenizer),
        input_embed_dim=min(512, hidden),
        hidden_dim=mem["encoder_width"],
        latent_tokens=mem["latent_tokens"],
        output_dim=hidden,
        layers=mem["encoder_layers"],
        dropout=mem["dropout"],
    ).to(device)
    constant = ConstantLatent(mem["latent_tokens"], hidden).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    if ckpt.get("constant") is None:
        raise ValueError("Checkpoint lacks trained constant-latent baseline; retrain with v0.5.1")
    constant.load_state_dict(ckpt["constant"])
    encoder.eval()
    constant.eval()

    data = load_jsonl(args.dataset)
    max_new = cfg["evaluation"]["generation_max_new_tokens"]
    embeds_includes_prefix = bool(
        cfg["evaluation"].get("embeds_generate_includes_prefix", False)
    )
    rows = []
    leakage_features = []
    leakage_labels = []

    for split in ("iid", "composition", "ood"):
        exs = [x for x in data if x.split == split][: args.limit_per_split]
        if len(exs) < 2:
            raise ValueError(f"Need at least two examples in split {split} for shuffled control")
        states = [x.state for x in exs]
        shuffled = deranged_states(states, seed=cfg["seed"] + len(split))
        corrupted = corrupted_states(states, seed=cfg["seed"] + 100 * len(split))

        for i, ex in enumerate(exs):
            prompt = f"Question:\n{ex.question}\nAnswer:"
            p = tokenizer(prompt, return_tensors="pt", truncation=True)
            p = {k: v.to(device) for k, v in p.items()}
            real = encode_states(
                encoder, tokenizer, [states[i]], device, mem["max_state_tokens"]
            )

            conditions: dict[str, tuple[str, int, float]] = {}

            start = time.perf_counter()
            texts, counts = base_generate(model, tokenizer, [prompt], device, max_new)
            conditions["base"] = (texts[0], counts[0], time.perf_counter() - start)

            text_prompt = (
                f"Question:\n{ex.question}\n\nKnown process state:\n{states[i].process_text()}\n\n"
                "Continue solving the problem.\nAnswer:"
            )
            start = time.perf_counter()
            texts, counts = base_generate(model, tokenizer, [text_prompt], device, max_new)
            conditions["text_state"] = (texts[0], counts[0], time.perf_counter() - start)

            latent_variants = {
                "process_latent": real,
                "random_latent": random_latents_like(real, seed=cfg["seed"] + i),
                "constant_latent": constant(1),
                "shuffled_latent": encode_states(
                    encoder, tokenizer, [shuffled[i]], device, mem["max_state_tokens"]
                ),
                "corrupted_latent": encode_states(
                    encoder, tokenizer, [corrupted[i]], device, mem["max_state_tokens"]
                ),
            }

            for name, z in latent_variants.items():
                start = time.perf_counter()
                texts, counts = generate_from_embeds(
                    model, tokenizer,
                    p["input_ids"], p["attention_mask"], z, max_new,
                    embeds_return_includes_prefix=embeds_includes_prefix,
                )
                conditions[name] = (texts[0], counts[0], time.perf_counter() - start)

            leakage_features.append(real.mean(dim=1).squeeze(0).float().cpu().numpy())
            leakage_labels.append(int(ex.answer) % 5)

            for name, (pred, token_count, latency) in conditions.items():
                rows.append(
                    {
                        "example_id": ex.example_id,
                        "split": split,
                        "condition": name,
                        "gold": ex.answer,
                        "prediction": pred,
                        "exact_match": exact_match(pred, ex.answer),
                        "generated_tokens": token_count,
                        "latency_s": latency,
                    }
                )

    if not rows:
        raise ValueError("No evaluation rows were generated")

    with open(outdir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for split in ("iid", "composition", "ood"):
        summary[split] = {}
        for cond in sorted(set(r["condition"] for r in rows)):
            subset = [r for r in rows if r["split"] == split and r["condition"] == cond]
            if not subset:
                continue
            vals = [r["exact_match"] for r in subset]
            lats = [r["latency_s"] for r in subset]
            toks = [r["generated_tokens"] for r in subset]
            ci = bootstrap_ci(vals, cfg["evaluation"]["bootstrap_samples"], cfg["seed"])
            avg_lat = sum(lats) / len(lats)
            avg_tokens = sum(toks) / len(toks)
            summary[split][cond] = {
                "accuracy": ci["mean"],
                "accuracy_ci95": [ci["lo"], ci["hi"]],
                "latency_s": avg_lat,
                "generated_tokens": avg_tokens,
                "utility": utility(ci["mean"], avg_lat, int(round(avg_tokens)), cfg["utility"]),
            }

    leak = leakage_probe(
        np.stack(leakage_features), np.asarray(leakage_labels), seed=cfg["seed"]
    )
    summary["leakage_probe"] = leak
    summary["leakage_pass"] = (
        None
        if leak.get("accuracy") is None
        else leak["accuracy"] <= cfg["evaluation"]["leakage_max_accuracy"]
    )

    with open(outdir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(outdir / "ablation_table.md", "w", encoding="utf-8") as f:
        f.write("| Split | Condition | Accuracy | 95% CI | Tokens | Latency s | Utility |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for split in ("iid", "composition", "ood"):
            for cond, m in summary[split].items():
                f.write(
                    f"| {split} | {cond} | {m['accuracy']:.4f} | "
                    f"[{m['accuracy_ci95'][0]:.4f}, {m['accuracy_ci95'][1]:.4f}] | "
                    f"{m['generated_tokens']:.1f} | {m['latency_s']:.4f} | {m['utility']:.4f} |\n"
                )

    print(json.dumps(summary, indent=2))
    print("wrote evaluation to", outdir)


if __name__ == "__main__":
    main()
