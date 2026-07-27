#!/usr/bin/env python
"""Phase A: Train reusable skill vectors using groups of tasks.

Objective: L_task + λ_inv * L_invariance
The same skill latent should remain useful when operands change.
"""
from __future__ import annotations
import argparse,math,random
from pathlib import Path
import torch
from torch.optim import AdamW
from tqdm import tqdm
from daph_latent_memory.config import load_config
from daph_latent_memory.runtime import load_frozen_model,primary_device
from daph_latent_memory.benchmarks.dataset import load_jsonl
from daph_latent_memory.latent.skill_bank import SkillBank
from daph_latent_memory.latent.injection import LatentInjector
from daph_latent_memory.training.collate import tokenize_batch
from daph_latent_memory.training.losses import task_loss,invariance_loss
from daph_latent_memory.training.checkpoint import save_checkpoint
from daph_latent_memory.training.negatives import sample_same_skill_pairs

SKILL_IDS={"add":0,"subtract":1,"multiply":2,"divide":3,"verify":4,"decompose":5}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    cfg=load_config(args.config);outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
    model,tokenizer=load_frozen_model(cfg);device=primary_device(model)
    hidden=model.get_input_embeddings().embedding_dim;mem=cfg["memory"]
    num_skills=mem["num_skills"];latent_tokens=mem["latent_tokens"];skill_dim=mem["skill_dim"]
    skill_bank=SkillBank(num_skills,latent_tokens,skill_dim).to(device)
    skill_proj=torch.nn.Linear(skill_dim,hidden,bias=False).to(device)
    injector=LatentInjector(model)
    # v0.3.8 DEF-02: wire the latent injection safety clamp if configured.
    relative_norm_limit=float(cfg["training"].get("latent_relative_norm_limit",0.0))
    if relative_norm_limit > 0:
        injector.set_relative_norm_limit(relative_norm_limit)
    opt=AdamW(list(skill_bank.parameters())+list(skill_proj.parameters()),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    tcfg=cfg["training"];all_train=[x for x in load_jsonl(args.dataset) if x.split=="train"]
    inv_weight=tcfg.get("invariance_weight",0.2)
    for epoch in range(tcfg["epochs"]):
        skill_bank.train();opt.zero_grad(set_to_none=True)
        rng=random.Random(cfg["seed"]+epoch)
        epoch_batches=list(_batches(all_train,tcfg["batch_size"],rng))
        for chunk in (pbar:=tqdm(epoch_batches,desc=f"Phase A epoch {epoch+1}")):
            p,s,a=tokenize_batch(chunk,tokenizer,tcfg["max_prompt_tokens"],tcfg["max_answer_tokens"])
            p={k:v.to(device) for k,v in p.items()};s={k:v.to(device) for k,v in s.items()};a={k:v.to(device) for k,v in a.items()}
            skill_ids=torch.tensor([SKILL_IDS.get(x.skill_label,0) for x in chunk],device=device)
            z_skill=skill_bank.get_skill_batch(skill_ids)  # [batch, latent_tokens, skill_dim]
            # Project skill to hidden dim for injection
            # For Phase A, we use a simple linear projection (no composer yet)
            z_inject=skill_proj(z_skill)
            injected=injector.build(p["input_ids"],p["attention_mask"],a["input_ids"],a["attention_mask"],z_inject)
            out=model(inputs_embeds=injected.inputs_embeds,attention_mask=injected.attention_mask,labels=injected.labels,use_cache=False)
            loss=task_loss(out)
            # Invariance loss
            pairs=sample_same_skill_pairs(chunk,seed=cfg["seed"]+epoch)
            if pairs:
                inv_losses=[]
                for ex_a,ex_b in pairs:
                    sid_a=SKILL_IDS.get(ex_a.skill_label,0);sid_b=SKILL_IDS.get(ex_b.skill_label,0)
                    if sid_a!=sid_b: continue
                    za=skill_bank.get_skill(sid_a);zb=skill_bank.get_skill(sid_b)
                    inv_losses.append(invariance_loss(za.unsqueeze(0),zb.unsqueeze(0)))
                if inv_losses: loss=loss+inv_weight*torch.stack(inv_losses).mean()
            loss.backward()
            if (pbar.n+1)%tcfg.get("grad_accum_steps",1)==0:
                torch.nn.utils.clip_grad_norm_(list(skill_bank.parameters())+list(skill_proj.parameters()),tcfg.get("gradient_clip",1.0))
                opt.step();opt.zero_grad(set_to_none=True)
    save_checkpoint(outdir/"skill_bank.pt",{"skill_bank":skill_bank.state_dict(),"skill_proj":skill_proj.state_dict()},{"phase":"A","seed":cfg["seed"],"epoch":tcfg["epochs"]})
    print(f"Phase A complete. Skill bank saved to {outdir/'skill_bank.pt'}")

def _batches(items,size,rng):
    items=list(items);rng.shuffle(items)
    for i in range(0,len(items),size): yield items[i:i+size]

if __name__=="__main__": main()
