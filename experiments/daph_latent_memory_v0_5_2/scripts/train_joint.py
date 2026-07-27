#!/usr/bin/env python
"""Phase C: Joint composition with freeze anneal.

Unfreeze skill bank with an anneal schedule:
  freeze_ratio(t) = max(0, 1 - t / T_unfreeze)

T_unfreeze = 30% of total Phase C steps.
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
from daph_latent_memory.latent.instance_state import InstanceEncoder
from daph_latent_memory.latent.composer import make_composer
from daph_latent_memory.latent.injection import LatentInjector
from daph_latent_memory.router.router import LatentRouter
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
    ap.add_argument("--instance-checkpoint",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    cfg=load_config(args.config);outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
    model,tokenizer=load_frozen_model(cfg);device=primary_device(model)
    hidden=model.get_input_embeddings().embedding_dim;mem=cfg["memory"];tcfg=cfg["training"]
    # Load skill bank and instance encoder
    skill_bank=SkillBank(mem["num_skills"],mem["latent_tokens"],mem["skill_dim"]).to(device)
    instance_encoder=InstanceEncoder(len(tokenizer),min(512,hidden),mem["encoder_width"],mem["latent_tokens"],mem["instance_dim"],mem["encoder_layers"],mem["dropout"]).to(device)
    composer=make_composer(mem.get("composition","gated"),mem["skill_dim"],mem["instance_dim"],hidden,mem["latent_tokens"]).to(device)
    router=LatentRouter(hidden,mem["num_skills"],router_topk=0).to(device)
    sk_ckpt=load_checkpoint(args.skill_checkpoint);skill_bank.load_state_dict(sk_ckpt["state_dict"]["skill_bank"])
    inst_ckpt=load_checkpoint(args.instance_checkpoint);instance_encoder.load_state_dict(inst_ckpt["state_dict"]["instance_encoder"]);composer.load_state_dict(inst_ckpt["state_dict"]["composer"])
    injector=LatentInjector(model)
    # v0.3.8 DEF-02: wire the latent injection safety clamp if configured.
    relative_norm_limit=float(tcfg.get("latent_relative_norm_limit",0.0))
    if relative_norm_limit > 0:
        injector.set_relative_norm_limit(relative_norm_limit)
    params=list(skill_bank.parameters())+list(instance_encoder.parameters())+list(composer.parameters())+list(router.parameters())
    opt=AdamW(params,lr=tcfg["learning_rate"],weight_decay=tcfg["weight_decay"])
    all_train=[x for x in load_jsonl(args.dataset) if x.split=="train"]
    func_weight=tcfg.get("functional_weight",0.5);dis_weight=tcfg.get("disentangle_weight",0.1)
    func_margin=tcfg.get("functional_margin",0.0);func_negs=tcfg.get("functional_negatives",3)
    unfreeze_frac=tcfg.get("unfreeze_fraction",0.3)
    # Compute total steps for anneal
    total_steps=sum(1 for _ in _batches(all_train,tcfg["batch_size"],random.Random(0)))*tcfg["epochs"]
    t_unfreeze=max(1,int(total_steps*unfreeze_frac))
    global_step=0
    for epoch in range(tcfg["epochs"]):
        skill_bank.train();instance_encoder.train();composer.train();router.train()
        opt.zero_grad(set_to_none=True)
        rng=random.Random(cfg["seed"]+epoch)
        epoch_batches=list(_batches(all_train,tcfg["batch_size"],rng))
        for chunk in tqdm(epoch_batches,desc=f"Phase C epoch {epoch+1}"):
            # Freeze anneal
            freeze_ratio=max(0.0,1.0-global_step/t_unfreeze)
            skill_bank.partial_freeze(freeze_ratio)
            p,s,a=tokenize_batch(chunk,tokenizer,tcfg["max_prompt_tokens"],tcfg["max_answer_tokens"])
            p={k:v.to(device) for k,v in p.items()};s={k:v.to(device) for k,v in s.items()};a={k:v.to(device) for k,v in a.items()}
            # Router-based skill selection
            prompt_emb=model.get_input_embeddings()(p["input_ids"])
            h_q=(prompt_emb*p["attention_mask"].unsqueeze(-1)).sum(dim=1)/p["attention_mask"].sum(dim=1).clamp_min(1)
            z_skill=router.route(h_q,skill_bank)
            z_instance=instance_encoder(s["input_ids"],s["attention_mask"])
            z=composer(z_skill,z_instance)
            injected=injector.build(p["input_ids"],p["attention_mask"],a["input_ids"],a["attention_mask"],z)
            out=model(inputs_embeds=injected.inputs_embeds,attention_mask=injected.attention_mask,labels=injected.labels,use_cache=False)
            tl=task_loss(out)
            # Supervised routing loss
            skill_ids=torch.tensor([SKILL_IDS.get(x.skill_label,0) for x in chunk],device=device)
            route_loss=router.supervised_loss(h_q,skill_ids)
            # Functional loss (simplified — skip negatives if too expensive)
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
                        z_neg_skill=router.route(h_q[:1],skill_bank)
                        z_neg=composer(z_neg_skill,z_neg_inst)
                        r_neg=compute_R(model,injector,p["input_ids"][:1],p["attention_mask"][:1],a["input_ids"][:1],a["attention_mask"][:1],z_neg[:1])
                        r_negs_list.append(r_neg)
                if r_negs_list:
                    r_negs=torch.stack(r_negs_list).reshape(len(chunk),-1)
                    func_loss=functional_mismatch_loss(r_pos,r_negs,func_margin)
            dis_loss=disentangle_loss(z_skill,z_instance)
            loss=total_loss_v052(tl,func_loss,dis_loss,tcfg["answer_weight"],func_weight,dis_weight)+0.1*route_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,tcfg.get("gradient_clip",1.0))
            opt.step();opt.zero_grad(set_to_none=True);global_step+=1
    save_checkpoint(outdir/"phase_c.pt",{"skill_bank":skill_bank.state_dict(),"instance_encoder":instance_encoder.state_dict(),"composer":composer.state_dict(),"router":router.state_dict()},{"phase":"C","seed":cfg["seed"],"global_step":global_step})
    print(f"Phase C complete. Saved to {outdir/'phase_c.pt'}")

def _batches(items,size,rng):
    items=list(items);rng.shuffle(items)
    for i in range(0,len(items),size): yield items[i:i+size]

if __name__=="__main__": main()
