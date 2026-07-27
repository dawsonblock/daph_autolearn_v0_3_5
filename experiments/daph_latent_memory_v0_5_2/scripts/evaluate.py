#!/usr/bin/env python
"""Full evaluation with all v0.5.2 conditions and metrics."""
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
from daph_latent_memory.evaluation.metrics import exact_match,bootstrap_ci,causal_specificity_score,wrong_memory_penalty,latent_utility_gain,generalization_gain,css_null_distribution,css_threshold
from daph_latent_memory.evaluation.statistics import mcnemar_test,paired_bootstrap_ci
from daph_latent_memory.training.checkpoint import load_checkpoint

SKILL_IDS={"add":0,"subtract":1,"multiply":2,"divide":3,"verify":4,"decompose":5}

def _decode(tokenizer,ids):
    return tokenizer.batch_decode(ids,skip_special_tokens=True)

def _generate(model,tokenizer,prompt_ids,prompt_mask,latents,device,max_new):
    embed=model.get_input_embeddings()
    prompt_emb=embed(prompt_ids);latents=latents.to(device=prompt_emb.device,dtype=prompt_emb.dtype)
    inputs_embeds=torch.cat([prompt_emb,latents],dim=1)
    mask=torch.cat([prompt_mask,torch.ones(prompt_mask.size(0),latents.size(1),dtype=prompt_mask.dtype,device=device)],dim=1)
    out=model.generate(inputs_embeds=inputs_embeds,attention_mask=mask,max_new_tokens=max_new,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    return _decode(tokenizer,out)

def _base_generate(model,tokenizer,prompts,device,max_new):
    tok=tokenizer(prompts,return_tensors="pt",padding=True,truncation=True);tok={k:v.to(device) for k,v in tok.items()}
    out=model.generate(**tok,max_new_tokens=max_new,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    prefix=tok["input_ids"].size(1)
    if out.size(1)>=prefix: out=out[:,prefix:]
    return _decode(tokenizer,out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True);ap.add_argument("--dataset",required=True)
    ap.add_argument("--checkpoint",required=True);ap.add_argument("--output",required=True)
    ap.add_argument("--limit-per-split",type=int,default=200)
    args=ap.parse_args()
    cfg=load_config(args.config);outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
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
    injector=LatentInjector(model)
    data=load_jsonl(args.dataset);max_new=cfg["evaluation"]["generation_max_new_tokens"]
    results={}
    for split in ["iid","composition","ood","operand_ood","structural_ood","operator_ood","distractor_ood","counterfactual","adversarial"]:
        exs=[x for x in data if x.split==split][:args.limit_per_split]
        if len(exs)<2: continue
        split_results={"per_example":[]}
        matched_correct=[];base_correct=[];shuffled_correct=[];wrong_class_correct=[]
        for i,ex in enumerate(exs):
            prompt=f"Question:\n{ex.question}\nAnswer:"
            p=tokenizer(prompt,return_tensors="pt",truncation=True);p={k:v.to(device) for k,v in p.items()}
            # Base
            base_texts=_base_generate(model,tokenizer,[prompt],device,max_new)
            bc=exact_match(base_texts[0],ex.answer);base_correct.append(bc)
            # Matched latent
            s=tokenizer(ex.state.process_text(),return_tensors="pt",truncation=True,max_length=mem["max_state_tokens"]);s={k:v.to(device) for k,v in s.items()}
            z_inst=instance_encoder(s["input_ids"],s["attention_mask"])
            prompt_emb=model.get_input_embeddings()(p["input_ids"])
            h_q=(prompt_emb*p["attention_mask"].unsqueeze(-1)).sum(dim=1)/p["attention_mask"].sum(dim=1).clamp_min(1)
            z_skill=router.route(h_q,skill_bank)
            z_matched=composer(z_skill,z_inst)
            matched_texts=_generate(model,tokenizer,p["input_ids"],p["attention_mask"],z_matched,device,max_new)
            mc=exact_match(matched_texts[0],ex.answer);matched_correct.append(mc)
            # Shuffled latent (global derangement)
            if i==0:
                all_z=torch.cat([_encode_latent(instance_encoder,composer,router,skill_bank,model,tokenizer,ex2.question,ex2.state.process_text(),device,mem,hidden) for ex2 in exs],dim=0)
            z_shuffled=shuffled_global(all_z,seed=cfg["seed"]+len(split))
            shuffled_texts=_generate(model,tokenizer,p["input_ids"][:1],p["attention_mask"][:1],z_shuffled[i:i+1],device,max_new)
            sc=exact_match(shuffled_texts[0],ex.answer);shuffled_correct.append(sc)
            # Wrong-class shuffled
            labels=[x.skill_label or x.state.operation for x in exs]
            z_wrong=shuffled_wrong_class(all_z,labels,seed=cfg["seed"]+len(split))
            wrong_texts=_generate(model,tokenizer,p["input_ids"][:1],p["attention_mask"][:1],z_wrong[i:i+1],device,max_new)
            wc=exact_match(wrong_texts[0],ex.answer);wrong_class_correct.append(wc)
            split_results["per_example"].append({"example_id":ex.example_id,"base":bc,"matched":mc,"shuffled":sc,"wrong_class":wc})
        a_matched=sum(matched_correct)/len(matched_correct);a_base=sum(base_correct)/len(base_correct)
        a_shuffled=sum(shuffled_correct)/len(shuffled_correct);a_wrong=sum(wrong_class_correct)/len(wrong_class_correct)
        css=causal_specificity_score(a_matched,a_shuffled);wmp=wrong_memory_penalty(a_matched,a_wrong)
        lug=latent_utility_gain(a_matched,a_base,mem["latent_tokens"])
        null_css=css_null_distribution(matched_correct,shuffled_correct,n_permutations=cfg["evaluation"].get("null_permutations",200),seed=cfg["seed"])
        threshold=css_threshold(null_css,percentile=cfg["evaluation"].get("null_percentile",99),floor_pp=cfg["evaluation"].get("css_floor_pp",5.0))
        mc=mcnemar_test(matched_correct,shuffled_correct)
        split_results.update({"a_matched":a_matched,"a_base":a_base,"a_shuffled":a_shuffled,"a_wrong_class":a_wrong,"css":css,"wmp":wmp,"lug":lug,"css_threshold":threshold,"mcnemar_matched_vs_shuffled":mc})
        results[split]=split_results
        print(f"{split}: A_matched={a_matched:.3f} A_base={a_base:.3f} A_shuffled={a_shuffled:.3f} CSS={css:.1f}pp WMP={wmp:.1f}pp threshold={threshold:.1f}pp")
    with open(outdir/"results.json","w") as f: json.dump(results,f,indent=2)
    print(f"Results saved to {outdir/'results.json'}")

def _encode_latent(instance_encoder,composer,router,skill_bank,model,tokenizer,question,state_text,device,mem,hidden_dim):
    s=tokenizer(state_text,return_tensors="pt",truncation=True,max_length=mem["max_state_tokens"]);s={k:v.to(device) for k,v in s.items()}
    z_inst=instance_encoder(s["input_ids"],s["attention_mask"])
    prompt=f"Question:\n{question}\nAnswer:"
    p=tokenizer(prompt,return_tensors="pt",truncation=True);p={k:v.to(device) for k,v in p.items()}
    prompt_emb=model.get_input_embeddings()(p["input_ids"])
    h_q=(prompt_emb*p["attention_mask"].unsqueeze(-1)).sum(dim=1)/p["attention_mask"].sum(dim=1).clamp_min(1)
    z_skill=router.route(h_q,skill_bank)
    z=composer(z_skill,z_inst)
    return z

if __name__=="__main__": main()
