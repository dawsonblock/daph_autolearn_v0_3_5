#!/usr/bin/env python
"""Causal evaluation framework — all conditions from the corrected plan.

For every evaluation example i, generates:
  Baselines: base, matched, zero, constant, random_gaussian, random_norm_matched
  Query-binding: shuffled_global, shuffled_same_class, shuffled_wrong_class,
                 nearest_neighbor_wrong, farthest_neighbor
  Geometry: normalized, scaled, sign_flipped, mean_latent, centroid_latent,
            pca_low_rank, pca_residual
  Corruption sweep: alpha in {0.00, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00}

All conditions are reproducible from (example_id, condition, seed).
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from daph_latent_memory.config import load_config
from daph_latent_memory.runtime import load_frozen_model,primary_device
from daph_latent_memory.benchmarks.dataset import load_jsonl
from daph_latent_memory.latent.skill_bank import SkillBank
from daph_latent_memory.latent.instance_state import InstanceEncoder
from daph_latent_memory.latent.composer import make_composer
from daph_latent_memory.latent.constant import ConstantLatent
from daph_latent_memory.latent.injection import LatentInjector
from daph_latent_memory.router.router import LatentRouter
from daph_latent_memory.evaluation.controls import *
from daph_latent_memory.evaluation.metrics import exact_match,causal_specificity_score,wrong_memory_penalty,css_null_distribution,css_threshold
from daph_latent_memory.evaluation.statistics import mcnemar_test,cochran_q_test,wilcoxon_signed_rank,paired_bootstrap_ci
from daph_latent_memory.training.checkpoint import load_checkpoint

def _generate(model,tokenizer,prompt_ids,prompt_mask,latents,device,max_new):
    embed=model.get_input_embeddings()
    prompt_emb=embed(prompt_ids);latents=latents.to(device=prompt_emb.device,dtype=prompt_emb.dtype)
    inputs_embeds=torch.cat([prompt_emb,latents],dim=1)
    mask=torch.cat([prompt_mask,torch.ones(prompt_mask.size(0),latents.size(1),dtype=prompt_mask.dtype,device=device)],dim=1)
    out=model.generate(inputs_embeds=inputs_embeds,attention_mask=mask,max_new_tokens=max_new,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    return tokenizer.batch_decode(out,skip_special_tokens=True)

def _base_generate(model,tokenizer,prompts,device,max_new):
    tok=tokenizer(prompts,return_tensors="pt",padding=True,truncation=True);tok={k:v.to(device) for k,v in tok.items()}
    out=model.generate(**tok,max_new_tokens=max_new,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    prefix=tok["input_ids"].size(1)
    if out.size(1)>=prefix: out=out[:,prefix:]
    return tokenizer.batch_decode(out,skip_special_tokens=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True);ap.add_argument("--eval-config",required=True)
    ap.add_argument("--dataset",required=True);ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--output",required=True);ap.add_argument("--limit",type=int,default=100)
    args=ap.parse_args()
    cfg=load_config(args.config);eval_cfg=load_config(args.eval_config)
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    model,tokenizer=load_frozen_model(cfg);device=primary_device(model)
    hidden=model.get_input_embeddings().embedding_dim;mem=cfg["memory"]
    skill_bank=SkillBank(mem["num_skills"],mem["latent_tokens"],mem["skill_dim"]).to(device)
    instance_encoder=InstanceEncoder(len(tokenizer),min(512,hidden),mem["encoder_width"],mem["latent_tokens"],mem["instance_dim"],mem["encoder_layers"],mem["dropout"]).to(device)
    composer=make_composer(mem.get("composition","gated"),mem["skill_dim"],mem["instance_dim"],hidden,mem["latent_tokens"]).to(device)
    router=LatentRouter(hidden,mem["num_skills"]).to(device)
    constant=ConstantLatent(mem["latent_tokens"],hidden).to(device)
    ckpt=load_checkpoint(args.checkpoint)
    for k,v in ckpt["state_dict"].items():
        if k=="skill_bank": skill_bank.load_state_dict(v)
        elif k=="instance_encoder": instance_encoder.load_state_dict(v)
        elif k=="composer": composer.load_state_dict(v)
        elif k=="router": router.load_state_dict(v)
    skill_bank.eval();instance_encoder.eval();composer.eval();router.eval();constant.eval()
    data=load_jsonl(args.dataset);max_new=cfg["evaluation"]["generation_max_new_tokens"]
    seed=eval_cfg.get("seed",cfg["seed"])
    # Use IID split for causal eval
    exs=[x for x in data if x.split=="iid"][:args.limit]
    if len(exs)<2: raise ValueError("Need at least 2 examples for causal eval")
    labels=[x.skill_label or x.state.operation for x in exs]
    # Encode all latents
    all_z=torch.cat([_encode(instance_encoder,composer,router,skill_bank,tokenizer,ex,device,mem,hidden,model) for ex in exs],dim=0)
    # Generate all conditions
    results={"conditions":{},"corruption_sweep":{},"statistics":{}}
    condition_fns={
        "matched":lambda i:all_z[i:i+1],
        "zero":lambda i:torch.zeros_like(all_z[i:i+1]),
        "constant":lambda i:constant(1),
        "random_gaussian":lambda i:random_latents_like(all_z[i:i+1],seed=seed+i),
        "random_norm_matched":lambda i:random_norm_matched_latents_like(all_z[i:i+1],seed=seed+i),
        "shuffled_global":lambda i:shuffled_global(all_z,seed=seed)[i:i+1],
        "shuffled_same_class":lambda i:shuffled_same_class(all_z,labels,seed=seed)[i:i+1],
        "shuffled_wrong_class":lambda i:shuffled_wrong_class(all_z,labels,seed=seed)[i:i+1],
        "nearest_neighbor_wrong":lambda i:nearest_neighbor_wrong(all_z,labels)[i:i+1],
        "farthest_neighbor":lambda i:farthest_neighbor(all_z)[i:i+1],
        "normalized":lambda i:normalized_latents(all_z[i:i+1]),
        "scaled":lambda i:scaled_latents(all_z[i:i+1],c=2.0),
        "sign_flipped":lambda i:sign_flipped_latents(all_z[i:i+1]),
        "mean_latent":lambda i:mean_latent(all_z)[i:i+1],
        "centroid_latent":lambda i:centroid_latent(all_z,labels)[i:i+1],
        "pca_low_rank":lambda i:pca_low_rank(all_z,k=8)[i:i+1],
        "pca_residual":lambda i:pca_residual(all_z,k=8)[i:i+1],
    }
    for cond_name,fn in condition_fns.items():
        correct_list=[]
        for i,ex in enumerate(exs):
            prompt=f"Question:\n{ex.question}\nAnswer:"
            p=tokenizer(prompt,return_tensors="pt",truncation=True);p={k:v.to(device) for k,v in p.items()}
            z=fn(i)
            texts=_generate(model,tokenizer,p["input_ids"],p["attention_mask"],z,device,max_new)
            correct_list.append(exact_match(texts[0],ex.answer))
        accuracy=sum(correct_list)/len(correct_list)
        results["conditions"][cond_name]={"accuracy":accuracy,"correct":correct_list}
        print(f"  {cond_name}: A={accuracy:.3f}")
    # Base condition
    base_correct=[]
    for ex in exs:
        prompt=f"Question:\n{ex.question}\nAnswer:"
        texts=_base_generate(model,tokenizer,[prompt],device,max_new)
        base_correct.append(exact_match(texts[0],ex.answer))
    results["conditions"]["base"]={"accuracy":sum(base_correct)/len(base_correct),"correct":base_correct}
    print(f"  base: A={sum(base_correct)/len(base_correct):.3f}")
    # Corruption sweep
    alphas=eval_cfg.get("corruption",{}).get("alphas",[0.0,0.05,0.1,0.5,1.0,2.0])
    per_alpha_seeds=eval_cfg.get("corruption",{}).get("per_alpha_seeds",3)
    corruption_results={}
    for alpha in alphas:
        alpha_correct_all=[]
        for s in range(per_alpha_seeds):
            correct_list=[]
            for i,ex in enumerate(exs):
                prompt=f"Question:\n{ex.question}\nAnswer:"
                p=tokenizer(prompt,return_tensors="pt",truncation=True);p={k:v.to(device) for k,v in p.items()}
                z=corrupt_latent_vector(all_z[i:i+1],alpha,seed=seed+s*1000+i)
                texts=_generate(model,tokenizer,p["input_ids"],p["attention_mask"],z,device,max_new)
                correct_list.append(exact_match(texts[0],ex.answer))
            alpha_correct_all.extend(correct_list)
        accuracy=sum(alpha_correct_all)/len(alpha_correct_all)
        corruption_results[str(alpha)]={"accuracy":accuracy,"correct":alpha_correct_all}
        print(f"  corruption α={alpha}: A={accuracy:.3f}")
    results["corruption_sweep"]=corruption_results
    # Statistics
    matched_c=results["conditions"]["matched"]["correct"]
    shuffled_c=results["conditions"]["shuffled_global"]["correct"]
    wrong_c=results["conditions"]["shuffled_wrong_class"]["correct"]
    results["statistics"]["mcnemar_matched_vs_shuffled"]=mcnemar_test(matched_c,shuffled_c)
    results["statistics"]["mcnemar_matched_vs_wrong"]=mcnemar_test(matched_c,wrong_c)
    # CSS null distribution
    null_css=css_null_distribution(matched_c,shuffled_c,n_permutations=eval_cfg.get("statistics",{}).get("null_permutations",200),seed=seed)
    threshold=css_threshold(null_css,percentile=eval_cfg.get("statistics",{}).get("null_percentile",99),floor_pp=eval_cfg.get("statistics",{}).get("css_floor_pp",5.0))
    results["statistics"]["css_null_distribution"]=null_css
    results["statistics"]["css_threshold"]=threshold
    # Cochran's Q for corruption sweep
    corruption_conditions=[corruption_results[a]["correct"] for a in sorted(corruption_results.keys(),key=float)]
    if len(corruption_conditions)>=3:
        results["statistics"]["cochran_q_corruption"]=cochran_q_test(corruption_conditions)
    # Headline metrics
    a_matched=results["conditions"]["matched"]["accuracy"]
    a_shuffled=results["conditions"]["shuffled_global"]["accuracy"]
    a_wrong=results["conditions"]["shuffled_wrong_class"]["accuracy"]
    a_base=results["conditions"]["base"]["accuracy"]
    results["headline"]={
        "css":causal_specificity_score(a_matched,a_shuffled),
        "wmp":wrong_memory_penalty(a_matched,a_wrong),
        "lug":(a_matched-a_base)/(mem["latent_tokens"]+1),
        "css_passes_gate":causal_specificity_score(a_matched,a_shuffled)>=threshold and a_matched>a_base,
    }
    with open(outdir/"causal_eval.json","w") as f: json.dump(results,f,indent=2)
    print(f"\nHeadline: CSS={results['headline']['css']:.1f}pp WMP={results['headline']['wmp']:.1f}pp threshold={threshold:.1f}pp pass={results['headline']['css_passes_gate']}")

def _encode(instance_encoder,composer,router,skill_bank,tokenizer,ex,device,mem,hidden_dim,model):
    s=tokenizer(ex.state.process_text(),return_tensors="pt",truncation=True,max_length=mem["max_state_tokens"]);s={k:v.to(device) for k,v in s.items()}
    z_inst=instance_encoder(s["input_ids"],s["attention_mask"])
    # Use actual prompt hidden states for routing instead of zeros
    prompt=f"Question:\n{ex.question}\nAnswer:"
    pt=tokenizer(prompt,return_tensors="pt",truncation=True);pt={k:v.to(device) for k,v in pt.items()}
    with torch.no_grad():
        pout=model(**pt,output_hidden_states=True,use_cache=False)
        h_q=pout.hidden_states[-1].mean(dim=1).to(z_inst.dtype)
    z_skill=router.route(h_q,skill_bank)
    return composer(z_skill,z_inst)

if __name__=="__main__": main()
