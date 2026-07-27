#!/usr/bin/env python
"""Latent geometry analysis — PCA/SVD, probe suite, subspace analysis.

Run PCA/SVD on successful latent states per layer.
Test how much performance survives using k = 1,2,4,8,16,32,64,128 components.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from daph_latent_memory.config import load_config
from daph_latent_memory.runtime import load_frozen_model,primary_device
from daph_latent_memory.benchmarks.dataset import load_jsonl
from daph_latent_memory.latent.skill_bank import SkillBank
from daph_latent_memory.latent.instance_state import InstanceEncoder
from daph_latent_memory.latent.composer import make_composer
from daph_latent_memory.router.router import LatentRouter
from daph_latent_memory.evaluation.metrics import exact_match
from daph_latent_memory.evaluation.controls import pca_low_rank
from daph_latent_memory.training.checkpoint import load_checkpoint

def _generate(model,tokenizer,prompt_ids,prompt_mask,latents,device,max_new):
    embed=model.get_input_embeddings()
    prompt_emb=embed(prompt_ids);latents=latents.to(device=prompt_emb.device,dtype=prompt_emb.dtype)
    inputs_embeds=torch.cat([prompt_emb,latents],dim=1)
    mask=torch.cat([prompt_mask,torch.ones(prompt_mask.size(0),latents.size(1),dtype=prompt_mask.dtype,device=device)],dim=1)
    out=model.generate(inputs_embeds=inputs_embeds,attention_mask=mask,max_new_tokens=max_new,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    return tokenizer.batch_decode(out,skip_special_tokens=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True);ap.add_argument("--dataset",required=True)
    ap.add_argument("--checkpoint",required=True);ap.add_argument("--output",required=True)
    ap.add_argument("--limit",type=int,default=100)
    args=ap.parse_args()
    cfg=load_config(args.config);outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    model,tokenizer=load_frozen_model(cfg);device=primary_device(model)
    hidden=model.get_input_embeddings().embedding_dim;mem=cfg["memory"]
    skill_bank=SkillBank(mem["num_skills"],mem["latent_tokens"],mem["skill_dim"]).to(device)
    instance_encoder=InstanceEncoder(len(tokenizer),min(512,hidden),mem["encoder_width"],mem["latent_tokens"],mem["instance_dim"],mem["encoder_layers"],mem["dropout"]).to(device)
    composer=make_composer(mem.get("composition","gated"),mem["skill_dim"],mem["instance_dim"],hidden,mem["latent_tokens"]).to(device)
    router=LatentRouter(hidden,mem["num_skills"]).to(device)
    ckpt=load_checkpoint(args.checkpoint)
    for k,v in ckpt["state_dict"].items():
        if k=="skill_bank": skill_bank.load_state_dict(v)
        elif k=="instance_encoder": instance_encoder.load_state_dict(v)
        elif k=="composer": composer.load_state_dict(v)
        elif k=="router": router.load_state_dict(v)
    skill_bank.eval();instance_encoder.eval();composer.eval();router.eval()
    data=load_jsonl(args.dataset);max_new=cfg["evaluation"]["generation_max_new_tokens"]
    exs=[x for x in data if x.split=="iid"][:args.limit]
    if len(exs)<2: raise ValueError("Need at least 2 examples")
    # Encode all latents
    all_z=torch.cat([_encode(instance_encoder,composer,router,skill_bank,tokenizer,ex,device,mem,hidden) for ex in exs],dim=0)
    # PCA analysis
    flat=all_z.reshape(all_z.size(0),-1).float()
    mean=flat.mean(dim=0,keepdim=True);centered=flat-mean
    U,S,Vh=torch.linalg.svd(centered,full_matrices=False)
    # Explained variance ratio
    var_ratio=(S**2/(S**2).sum()).numpy().tolist()
    # Cumulative variance
    cumvar=np.cumsum(var_ratio).tolist()
    results={"svd":{"singular_values":S.numpy().tolist(),"variance_ratio":var_ratio,"cumulative_variance":cumvar}}
    # PCA ablation: test performance with k components
    k_values=[1,2,4,8,16,32,64,128]
    k_results={}
    for k in k_values:
        if k>flat.size(1): continue
        z_pca=pca_low_rank(all_z,k=k)
        correct=0
        for i,ex in enumerate(exs):
            prompt=f"Question:\n{ex.question}\nAnswer:"
            p=tokenizer(prompt,return_tensors="pt",truncation=True);p={k:v.to(device) for k,v in p.items()}
            texts=_generate(model,tokenizer,p["input_ids"],p["attention_mask"],z_pca[i:i+1],device,max_new)
            if exact_match(texts[0],ex.answer): correct+=1
        acc=correct/len(exs)
        k_results[str(k)]={"accuracy":acc}
        print(f"  PCA k={k}: A={acc:.3f}")
    results["pca_ablation"]=k_results
    # Probe: can latents predict operation type?
    from daph_latent_memory.evaluation.leakage import leakage_probe
    labels=np.array([int(hashlib.md5((x.skill_label or x.state.operation).encode()).hexdigest(),16)%1000 for x in exs])
    probe_result=leakage_probe(flat.numpy(),labels,seed=cfg["seed"])
    results["probe_operation_type"]=probe_result
    print(f"  Operation probe accuracy: {probe_result.get('accuracy')}")
    with open(outdir/"geometry.json","w") as f: json.dump(results,f,indent=2)
    print(f"Geometry analysis saved to {outdir/'geometry.json'}")

def _encode(instance_encoder,composer,router,skill_bank,tokenizer,ex,device,mem,hidden_dim):
    s=tokenizer(ex.state.process_text(),return_tensors="pt",truncation=True,max_length=mem["max_state_tokens"]);s={k:v.to(device) for k,v in s.items()}
    z_inst=instance_encoder(s["input_ids"],s["attention_mask"])
    h_q=torch.zeros(1,hidden_dim,dtype=z_inst.dtype,device=device)
    z_skill=router.route(h_q,skill_bank)
    return composer(z_skill,z_inst)

if __name__=="__main__": main()
