#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from daph_latent_memory.config import load_config
from daph_latent_memory.runtime import load_frozen_model, primary_device
from daph_latent_memory.benchmarks.dataset import load_jsonl
from daph_latent_memory.latent.encoder import StateTokenEncoder
from daph_latent_memory.latent.constant import ConstantLatent
from daph_latent_memory.latent.injection import LatentInjector
from daph_latent_memory.training.collate import tokenize_batch
from daph_latent_memory.training.losses import hidden_alignment_loss, total_loss
from daph_latent_memory.training.checkpoint import save_checkpoint, file_sha256
from daph_latent_memory.teacher.capture import select_layer_index


def batches(items, size, rng):
    items = list(items)
    rng.shuffle(items)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main():
    ap = argparse.ArgumentParser()
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
        vocab_size=len(tokenizer),
        input_embed_dim=min(512, hidden),
        hidden_dim=mem["encoder_width"],
        latent_tokens=mem["latent_tokens"],
        output_dim=hidden,
        layers=mem["encoder_layers"],
        dropout=mem["dropout"],
    ).to(device)
    constant = ConstantLatent(mem["latent_tokens"], hidden).to(device)

    injector = LatentInjector(model)
    opt = AdamW(
        list(encoder.parameters()) + list(constant.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    data = [x for x in load_jsonl(args.dataset) if x.split == "train"]
    if not data:
        raise ValueError("Dataset contains no training examples")

    tcfg = cfg["training"]
    grad_acc = int(tcfg["grad_accum_steps"])
    best = math.inf
    global_step = 0
    history = []

    for epoch in range(tcfg["epochs"]):
        encoder.train()
        constant.train()
        opt.zero_grad(set_to_none=True)
        running_process = []
        running_constant = []
        rng = random.Random(cfg["seed"] + epoch)
        epoch_batches = list(batches(data, tcfg["batch_size"], rng))

        for bi, chunk in enumerate(tqdm(epoch_batches, desc=f"epoch {epoch + 1}")):
            p, s, a = tokenize_batch(
                chunk, tokenizer, tcfg["max_prompt_tokens"], tcfg["max_answer_tokens"]
            )
            p = {k: v.to(device) for k, v in p.items()}
            s = {k: v.to(device) for k, v in s.items()}
            a = {k: v.to(device) for k, v in a.items()}

            latents = encoder(s["input_ids"], s["attention_mask"])
            injected = injector.build(
                p["input_ids"], p["attention_mask"], a["input_ids"], a["attention_mask"], latents
            )
            outputs = model(
                inputs_embeds=injected.inputs_embeds,
                attention_mask=injected.attention_mask,
                labels=injected.labels,
                output_hidden_states=True,
                use_cache=False,
            )

            layer_idx = select_layer_index(
                len(outputs.hidden_states), tcfg["teacher_layer_fraction"]
            )
            student = outputs.hidden_states[layer_idx][:, injected.memory_slice, :].mean(dim=1)

            teacher_texts = [
                f"Question:\n{x.question}\n\nKnown process state:\n{x.state.process_text()}\n\n"
                "Continue solving the problem."
                for x in chunk
            ]
            tt = tokenizer(
                teacher_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tcfg["max_prompt_tokens"],
            )
            tt = {k: v.to(device) for k, v in tt.items()}
            with torch.no_grad():
                tout = model(**tt, output_hidden_states=True, use_cache=False)
                th = tout.hidden_states[layer_idx]
                tm = tt["attention_mask"].unsqueeze(-1)
                teacher = (th * tm).sum(dim=1) / tm.sum(dim=1).clamp_min(1)

            align = hidden_alignment_loss(student, teacher)
            process_loss = total_loss(
                outputs.loss, align, tcfg["answer_weight"], tcfg["alignment_weight"]
            )

            constant_tokens = constant(len(chunk))
            constant_injected = injector.build(
                p["input_ids"],
                p["attention_mask"],
                a["input_ids"],
                a["attention_mask"],
                constant_tokens,
            )
            constant_out = model(
                inputs_embeds=constant_injected.inputs_embeds,
                attention_mask=constant_injected.attention_mask,
                labels=constant_injected.labels,
                use_cache=False,
            )
            constant_loss = constant_out.loss

            combined = (process_loss + constant_loss) / grad_acc
            combined.backward()
            running_process.append(float(process_loss.detach().cpu()))
            running_constant.append(float(constant_loss.detach().cpu()))

            is_boundary = (bi + 1) % grad_acc == 0
            is_last = bi + 1 == len(epoch_batches)
            if is_boundary or is_last:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(constant.parameters()),
                    tcfg["gradient_clip"],
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

        epoch_process = sum(running_process) / max(1, len(running_process))
        epoch_constant = sum(running_constant) / max(1, len(running_constant))
        selection_loss = epoch_process
        history.append(
            {"epoch": epoch + 1, "process_loss": epoch_process, "constant_loss": epoch_constant}
        )
        meta = {
            "version": "0.5.1",
            "model": cfg["model"]["name"],
            "dataset_sha256": file_sha256(args.dataset),
            "config": cfg,
            "history": history,
            "global_step": global_step,
            "note": "Base transformer frozen; process encoder and constant baseline trained.",
        }
        save_checkpoint(outdir / "last.pt", encoder, opt, meta, constant=constant)
        if selection_loss < best:
            best = selection_loss
            save_checkpoint(outdir / "best.pt", encoder, opt, meta, constant=constant)
        print(json.dumps(history[-1], indent=2))

    print("best process training loss:", best)
    print("checkpoint:", outdir / "best.pt")


if __name__ == "__main__":
    main()
