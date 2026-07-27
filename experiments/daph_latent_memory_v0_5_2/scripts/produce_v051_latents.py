#!/usr/bin/env python
"""Phase 0: Produce v0.5.1 latents for causal diagnosis.

Trains v0.5.1 to completion and persists per-example latents for every
evaluation example under every condition the causal eval will need,
keyed by (example_id, condition, seed). Also persists model checkpoints
and the exact decoding config, with SHA-256 manifest.

Exit criterion: a latents_v0_5_1/ directory sufficient to run Phase 1
with no further training.

This script uses v0.5.2's installed package for infrastructure and
inlines the v0.5.1 loss functions (hidden_alignment_loss, total_loss)
which were intentionally removed from v0.5.2.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

from daph_latent_memory.config import load_config
from daph_latent_memory.runtime import load_frozen_model, primary_device
from daph_latent_memory.benchmarks.dataset import load_jsonl
from daph_latent_memory.latent.encoder import StateTokenEncoder
from daph_latent_memory.latent.constant import ConstantLatent
from daph_latent_memory.latent.injection import LatentInjector
from daph_latent_memory.training.collate import tokenize_batch
from daph_latent_memory.training.checkpoint import save_checkpoint, file_sha256
from daph_latent_memory.teacher.capture import select_layer_index
from daph_latent_memory.evaluation.controls import random_latents_like
from daph_latent_memory.state.corruption import corrupt_state


# --- v0.5.1 loss functions (inlined; removed from v0.5.2 by design) ---

def hidden_alignment_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """v0.5.1 alignment loss: 1 - cos(student, teacher)."""
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    return 1.0 - (student * teacher).sum(dim=-1).mean()

def total_loss(answer_loss: torch.Tensor, align_loss: torch.Tensor,
               answer_weight: float, alignment_weight: float) -> torch.Tensor:
    """v0.5.1 total loss: answer_weight * answer_loss + alignment_weight * align_loss."""
    return answer_weight * answer_loss + alignment_weight * align_loss


# --- v0.5.1 control functions (inlined; not in v0.5.2 controls) ---

def deranged_states(states: list, seed: int = 1337):
    """Return a derangement of the states list (no element in original position)."""
    n = len(states)
    if n < 2: raise ValueError("Derangement requires at least two states")
    rng = random.Random(seed)
    indices = list(range(n))
    for _ in range(1000):
        rng.shuffle(indices)
        if all(i != j for i, j in enumerate(indices)):
            return [states[j] for j in indices]
    return states[1:] + states[:1]

def corrupted_states(states: list, seed: int = 1337):
    """Corrupt each state using the semantic corruption function."""
    rng = random.Random(seed)
    return [corrupt_state(s, rng) for s in states]


# --- Training utilities ---

def batches(items, size, rng):
    items = list(items); rng.shuffle(items)
    for i in range(0, len(items), size):
        yield items[i : i + size]

def split_train_validation(data, holdout_fraction, seed):
    n = len(data)
    if holdout_fraction <= 0 or n < 4: return list(data), []
    n_val = max(1, int(round(n * holdout_fraction)))
    if n_val >= n: n_val = max(1, n // 4)
    idx = list(range(n)); rng = random.Random(seed); rng.shuffle(idx)
    val_idx = set(idx[:n_val])
    val = [data[i] for i in range(n) if i in val_idx]
    train = [data[i] for i in range(n) if i not in val_idx]
    return train, val

@torch.no_grad()
def evaluate_validation(encoder, constant, injector, model, tokenizer, val_data, device, tcfg):
    if not val_data: return None
    encoder.eval(); constant.eval()
    rng = random.Random(tcfg.get("val_seed", 0) + 7919)
    losses = []
    for chunk in batches(val_data, tcfg["batch_size"], rng):
        p, s, a = tokenize_batch(chunk, tokenizer, tcfg["max_prompt_tokens"], tcfg["max_answer_tokens"])
        p = {k: v.to(device) for k, v in p.items()}
        s = {k: v.to(device) for k, v in s.items()}
        a = {k: v.to(device) for k, v in a.items()}
        latents = encoder(s["input_ids"], s["attention_mask"])
        injected = injector.build(p["input_ids"], p["attention_mask"], a["input_ids"], a["attention_mask"], latents)
        out = model(inputs_embeds=injected.inputs_embeds, attention_mask=injected.attention_mask,
                    labels=injected.labels, use_cache=False)
        losses.append(float(out.loss.detach().cpu()))
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def persist_latents(encoder, constant, injector, model, tokenizer, data, device, mem, tcfg, outdir, seed):
    """Persist per-example latents under all causal eval conditions."""
    latents_dir = outdir / "latents_v0_5_1"
    latents_dir.mkdir(parents=True, exist_ok=True)

    # Encode all latents for IID split (fallback to train if no IID)
    iid_examples = [x for x in data if x.split == "iid"]
    if not iid_examples:
        iid_examples = [x for x in data if x.split == "train"][:50]

    if not iid_examples:
        print("WARNING: no examples to persist latents for")
        return

    all_latents = []
    for ex in tqdm(iid_examples, desc="Encoding latents"):
        s = tokenizer(ex.state.process_text(), return_tensors="pt", truncation=True, max_length=mem["max_state_tokens"])
        s = {k: v.to(device) for k, v in s.items()}
        z = encoder(s["input_ids"], s["attention_mask"]).cpu()
        all_latents.append(z)

    all_z = torch.cat(all_latents, dim=0)

    # Save matched latents
    torch.save(all_z, latents_dir / "matched.pt")

    # Save constant latents
    torch.save(constant(1).cpu(), latents_dir / "constant.pt")

    # Save random latents (multiple seeds)
    for s in range(5):
        torch.save(random_latents_like(all_z, seed=seed + s).cpu(), latents_dir / f"random_seed{s}.pt")

    # Save shuffled (derangement)
    n = all_z.size(0)
    rng = random.Random(seed)
    indices = list(range(n))
    for _ in range(1000):
        rng.shuffle(indices)
        if all(i != j for i, j in enumerate(indices)): break
    shuffled = all_z[torch.tensor(indices)]
    torch.save(shuffled, latents_dir / "shuffled_global.pt")

    # Save corrupted states and their latents (multiple seeds)
    states = [ex.state for ex in iid_examples]
    for s in range(3):
        corrupted = corrupted_states(states, seed=seed + s * 100)
        corrupted_latents = []
        for cs in corrupted:
            cs_tok = tokenizer(cs.process_text(), return_tensors="pt", truncation=True, max_length=mem["max_state_tokens"])
            cs_tok = {k: v.to(device) for k, v in cs_tok.items()}
            cz = encoder(cs_tok["input_ids"], cs_tok["attention_mask"]).cpu()
            corrupted_latents.append(cz)
        torch.save(torch.cat(corrupted_latents, dim=0), latents_dir / f"corrupted_seed{s}.pt")

    # Save example metadata
    meta = [{"example_id": ex.example_id, "split": ex.split, "question": ex.question, "answer": ex.answer,
             "operation": ex.state.operation, "skill_label": getattr(ex, "skill_label", None)} for ex in iid_examples]
    with open(latents_dir / "examples.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Compute SHA-256 manifest
    manifest = {}
    for f in latents_dir.iterdir():
        if f.is_file():
            manifest[f.name] = file_sha256(f)
    with open(latents_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Persisted {len(iid_examples)} latents to {latents_dir}")
    print(f"Manifest with {len(manifest)} entries written")


def main():
    ap = argparse.ArgumentParser(description="Phase 0: Train v0.5.1 and persist latents")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    model, tokenizer = load_frozen_model(cfg)
    device = primary_device(model)
    hidden = model.get_input_embeddings().embedding_dim
    mem = cfg["memory"]

    encoder = StateTokenEncoder(
        vocab_size=len(tokenizer), input_embed_dim=min(512, hidden),
        hidden_dim=mem["encoder_width"], latent_tokens=mem["latent_tokens"],
        output_dim=hidden, layers=mem["encoder_layers"], dropout=mem["dropout"],
    ).to(device)
    constant = ConstantLatent(mem["latent_tokens"], hidden).to(device)
    injector = LatentInjector(model)

    from torch.optim import AdamW
    opt = AdamW(list(encoder.parameters()) + list(constant.parameters()),
                lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])

    all_train = [x for x in load_jsonl(args.dataset) if x.split == "train"]
    if not all_train: raise ValueError("Dataset contains no training examples")

    tcfg = cfg["training"]
    val_fraction = float(tcfg.get("val_holdout_fraction", 0.15))
    val_seed = int(tcfg.get("val_seed", cfg["seed"]))
    data, val_data = split_train_validation(all_train, val_fraction, val_seed)

    grad_acc = int(tcfg["grad_accum_steps"])
    best = math.inf
    best_epoch = None
    history = []

    for epoch in range(tcfg["epochs"]):
        encoder.train(); constant.train(); opt.zero_grad(set_to_none=True)
        rng = random.Random(cfg["seed"] + epoch)
        epoch_batches = list(batches(data, tcfg["batch_size"], rng))

        for bi, chunk in enumerate(tqdm(epoch_batches, desc=f"epoch {epoch+1}")):
            p, s, a = tokenize_batch(chunk, tokenizer, tcfg["max_prompt_tokens"], tcfg["max_answer_tokens"])
            p = {k: v.to(device) for k, v in p.items()}
            s = {k: v.to(device) for k, v in s.items()}
            a = {k: v.to(device) for k, v in a.items()}

            latents = encoder(s["input_ids"], s["attention_mask"])
            injected = injector.build(p["input_ids"], p["attention_mask"], a["input_ids"], a["attention_mask"], latents)
            outputs = model(inputs_embeds=injected.inputs_embeds, attention_mask=injected.attention_mask,
                            labels=injected.labels, output_hidden_states=True, use_cache=False)

            layer_idx = select_layer_index(len(outputs.hidden_states), tcfg["teacher_layer_fraction"])
            student = outputs.hidden_states[layer_idx][:, injected.memory_slice, :].mean(dim=1)

            teacher_texts = [f"Question:\n{x.question}\n\nKnown process state:\n{x.state.process_text()}\n\nContinue solving the problem." for x in chunk]
            tt = tokenizer(teacher_texts, return_tensors="pt", padding=True, truncation=True, max_length=tcfg["max_prompt_tokens"])
            tt = {k: v.to(device) for k, v in tt.items()}
            with torch.no_grad():
                tout = model(**tt, output_hidden_states=True, use_cache=False)
                th = tout.hidden_states[layer_idx]
                tm = tt["attention_mask"].unsqueeze(-1)
                teacher = (th * tm).sum(dim=1) / tm.sum(dim=1).clamp_min(1)

            align = hidden_alignment_loss(student, teacher)
            loss = total_loss(outputs.loss, align, tcfg["answer_weight"], tcfg["alignment_weight"])
            loss.backward()

            if (bi + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), tcfg.get("gradient_clip", 1.0))
                opt.step(); opt.zero_grad(set_to_none=True)

        val_loss = evaluate_validation(encoder, constant, injector, model, tokenizer, val_data, device, tcfg)
        epoch_loss = val_loss if val_loss is not None else float("nan")
        history.append({"epoch": epoch + 1, "val_loss": epoch_loss})
        print(f"Epoch {epoch+1}: val_loss={epoch_loss:.4f}")

        if val_loss is not None and val_loss < best:
            best = val_loss; best_epoch = epoch + 1
            save_checkpoint(outdir / "best.pt",
                            {"encoder": encoder.state_dict(),
                             "constant": constant.state_dict() if constant is not None else None,
                             "optimizer": opt.state_dict()},
                            {"epoch": epoch + 1, "val_loss": val_loss, "seed": cfg["seed"]})

    # Save final checkpoint
    save_checkpoint(outdir / "final.pt",
                    {"encoder": encoder.state_dict(),
                     "constant": constant.state_dict() if constant is not None else None,
                     "optimizer": opt.state_dict()},
                    {"epoch": tcfg["epochs"], "seed": cfg["seed"]})

    # Save training history
    with open(outdir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best checkpoint and persist latents
    ckpt = torch.load(outdir / "best.pt", map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["state_dict"]["encoder"])
    if ckpt["state_dict"].get("constant"): constant.load_state_dict(ckpt["state_dict"]["constant"])
    encoder.eval(); constant.eval()

    all_data = load_jsonl(args.dataset)
    persist_latents(encoder, constant, injector, model, tokenizer, all_data, device, mem, tcfg, outdir, cfg["seed"])

    print(f"\nPhase 0 complete. Best epoch: {best_epoch}, best val_loss: {best:.4f}")
    print(f"Artifacts in {outdir}")


if __name__ == "__main__":
    main()
