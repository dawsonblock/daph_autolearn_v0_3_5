#!/usr/bin/env python
"""Phase B: Train instance encoder on exact problem state.

Freeze skill bank. Train instance encoder.
Desired: z_instance,a ≠ z_instance,b even when both are multiplication.
"""
from __future__ import annotations
import argparse,json,math,random
from pathlib import Path
import torch
from torch.optim import AdamW
from tqdm import tqdm
from daph_latent_memory.config import load_config
from daph_latent_memory.runtime import load_frozen_model,primary_device
from daph_latent_memory.benchmarks.dataset import load_jsonl
from daph_latent_memory.latent.skill_bank import SkillBank
from daph_latent_memory.latent.instance_state import InstanceEncoder
from daph_latent_memory.latent.composer import make_composer
from daph_latent_memory.latent.injection import LatentInjector
from daph_latent_memory.training.collate import tokenize_batch
from daph_latent_memory.training.losses import task_loss,disentangle_loss,compute_R,functional_mismatch_loss,total_loss_v052
from daph_latent_memory.training.checkpoint import load_checkpoint,save_checkpoint
from daph_latent_memory.training.negatives import sample_negatives

SKILL_IDS={"add":0,"subtract":1,"multiply":2,"divide":3,"verify":4,"decompose":5}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--skill-checkpoint",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--calibrate-margin",action="store_true",
                    help="v0.3.8 Task 3.2: run the margin calibration probe before training. "
                         "Computes m = 0.5 * (r_shuffled_mean - r_matched_mean) on a 100-example "
                         "probe set with the frozen skill bank + untrained instance encoder, "
                         "then overrides functional_margin in-memory.")
    ap.add_argument("--margin-override",default=None,
                    help="Path to a calibration JSON (from calibrate_margin.py) to override "
                         "functional_margin. Takes precedence over --calibrate-margin and the "
                         "config value.")
    ap.add_argument("--probe-size",type=int,default=100,
                    help="Probe set size for --calibrate-margin (default 100).")
    args=ap.parse_args()
    cfg=load_config(args.config);outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
    model,tokenizer=load_frozen_model(cfg);device=primary_device(model)
    hidden=model.get_input_embeddings().embedding_dim;mem=cfg["memory"]
    tcfg=cfg["training"]
    # Load frozen skill bank
    skill_bank=SkillBank(mem["num_skills"],mem["latent_tokens"],mem["skill_dim"]).to(device)
    ckpt=load_checkpoint(args.skill_checkpoint)
    skill_bank.load_state_dict(ckpt["state_dict"]["skill_bank"])
    skill_bank.freeze()
    skill_proj=torch.nn.Linear(mem["skill_dim"],hidden,bias=False).to(device)
    skill_proj.load_state_dict(ckpt["state_dict"]["skill_proj"])
    skill_proj.requires_grad_(False)
    # Instance encoder
    instance_encoder=InstanceEncoder(len(tokenizer),min(512,hidden),mem["encoder_width"],mem["latent_tokens"],mem["instance_dim"],mem["encoder_layers"],mem["dropout"]).to(device)
    # Composer
    composer=make_composer(mem.get("composition","gated"),mem["skill_dim"],mem["instance_dim"],hidden,mem["latent_tokens"]).to(device)
    injector=LatentInjector(model)
    # v0.3.8 DEF-02: wire the latent injection safety clamp if configured.
    relative_norm_limit=float(tcfg.get("latent_relative_norm_limit",0.0))
    if relative_norm_limit > 0:
        injector.set_relative_norm_limit(relative_norm_limit)
    params=list(instance_encoder.parameters())+list(composer.parameters())
    opt=AdamW(params,lr=tcfg["learning_rate"],weight_decay=tcfg["weight_decay"])
    all_train=[x for x in load_jsonl(args.dataset) if x.split=="train"]
    func_weight=tcfg.get("functional_weight",0.5);dis_weight=tcfg.get("disentangle_weight",0.1)
    func_margin=tcfg.get("functional_margin",0.0);func_negs=tcfg.get("functional_negatives",3)

    # v0.3.8 Task 3.2: resolve the functional margin. Precedence:
    #   1. --margin-override <path>  (explicit calibration file)
    #   2. --calibrate-margin         (run the probe inline now)
    #   3. config training.functional_margin
    if args.margin_override is not None:
        import json as _json
        with open(args.margin_override) as _f:
            calib=_json.load(_f)
        func_margin=float(calib["functional_margin"])
        print(f"[margin] loaded from {args.margin_override}: m={func_margin:.4f} nats")
    elif args.calibrate_margin:
        from scripts.calibrate_margin import calibrate as _calibrate
        calib=_calibrate(cfg,args.dataset,probe_size=args.probe_size)
        func_margin=float(calib["functional_margin"])
        print(f"[margin] calibrated inline: m={func_margin:.4f} nats "
              f"(r_matched={calib['r_matched_mean']:.4f}, "
              f"r_shuffled={calib['r_shuffled_mean']:.4f}, "
              f"gap={calib['gap']:.4f}, used_floor={calib['used_floor']})")
        # Persist the calibration alongside the checkpoint for provenance.
        with open(outdir/"margin_calibration.json","w") as _f:
            _json.dump(calib,_f,indent=2)
    else:
        print(f"[margin] from config: m={func_margin:.4f} nats")
    for epoch in range(tcfg["epochs"]):
        instance_encoder.train();composer.train();opt.zero_grad(set_to_none=True)
        rng=random.Random(cfg["seed"]+epoch)
        epoch_batches=list(_batches(all_train,tcfg["batch_size"],rng))
        for chunk in tqdm(epoch_batches,desc=f"Phase B epoch {epoch+1}"):
            p,s,a=tokenize_batch(chunk,tokenizer,tcfg["max_prompt_tokens"],tcfg["max_answer_tokens"])
            p={k:v.to(device) for k,v in p.items()};s={k:v.to(device) for k,v in s.items()};a={k:v.to(device) for k,v in a.items()}
            skill_ids=torch.tensor([SKILL_IDS.get(x.skill_label,0) for x in chunk],device=device)
            z_skill=skill_bank.get_skill_batch(skill_ids)
            z_instance=instance_encoder(s["input_ids"],s["attention_mask"])
            z=composer(z_skill,z_instance)
            injected=injector.build(p["input_ids"],p["attention_mask"],a["input_ids"],a["attention_mask"],z)
            out=model(inputs_embeds=injected.inputs_embeds,attention_mask=injected.attention_mask,labels=injected.labels,use_cache=False)
            tl=task_loss(out)
            # Functional mismatch loss
            func_loss=None
            if func_weight>0 and func_negs>0:
                r_pos=compute_R(model,injector,p["input_ids"],p["attention_mask"],a["input_ids"],a["attention_mask"],z)
                r_negs_list=[]
                for ex in chunk:
                    negs=sample_negatives(ex,all_train,k=func_negs,seed=cfg["seed"]+epoch+hash(ex.example_id)%1000)
                    for neg in negs:
                        ns=tokenizer(neg.negative.state.process_text(),return_tensors="pt",padding=True,truncation=True,max_length=mem["max_state_tokens"])
                        ns={k:v.to(device) for k,v in ns.items()}
                        z_neg_inst=instance_encoder(ns["input_ids"],ns["attention_mask"])
                        neg_skill_id=torch.tensor([SKILL_IDS.get(neg.negative.skill_label,0)],device=device)
                        z_neg_skill=skill_bank.get_skill_batch(neg_skill_id)
                        z_neg=composer(z_neg_skill,z_neg_inst)
                        r_neg=compute_R(model,injector,p["input_ids"][:1],p["attention_mask"][:1],a["input_ids"][:1],a["attention_mask"][:1],z_neg[:1])
                        r_negs_list.append(r_neg)
                if r_negs_list:
                    r_negs=torch.stack(r_negs_list).reshape(len(chunk),-1)
                    func_loss=functional_mismatch_loss(r_pos,r_negs,func_margin)
            dis_loss=disentangle_loss(z_skill,z_instance)
            loss=total_loss_v052(tl,func_loss,dis_loss,tcfg["answer_weight"],func_weight,dis_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,tcfg.get("gradient_clip",1.0))
            opt.step();opt.zero_grad(set_to_none=True)
    save_checkpoint(outdir/"phase_b.pt",{"instance_encoder":instance_encoder.state_dict(),"composer":composer.state_dict()},{"phase":"B","seed":cfg["seed"]})
    print(f"Phase B complete. Saved to {outdir/'phase_b.pt'}")

def _batches(items,size,rng):
    items=list(items);rng.shuffle(items)
    for i in range(0,len(items),size): yield items[i:i+size]

if __name__=="__main__": main()
