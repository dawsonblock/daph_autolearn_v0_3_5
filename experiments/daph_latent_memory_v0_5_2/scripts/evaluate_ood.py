#!/usr/bin/env python
"""OOD evaluation — measure generalization gain across OOD splits."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from daph_latent_memory.config import load_config
from daph_latent_memory.runtime import load_frozen_model,primary_device
from daph_latent_memory.benchmarks.dataset import load_jsonl
from daph_latent_memory.latent.skill_bank import SkillBank
from daph_latent_memory.latent.instance_state import InstanceEncoder
from daph_latent_memory.latent.composer import make_composer
from daph_latent_memory.router.router import LatentRouter
from daph_latent_memory.evaluation.metrics import exact_match,generalization_gain
from daph_latent_memory.evaluation.statistics import mcnemar_test
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
    ood_splits=["operand_ood","structural_ood","operator_ood","distractor_ood","counterfactual","adversarial"]
    results={}
    for split in ood_splits:
        exs=[x for x in data if x.split==split][:args.limit]
        if len(exs)<2: continue
        base_correct=[];latent_correct=[]
        for ex in exs:
            prompt=f"Question:\n{ex.question}\nAnswer:"
            base_texts=_base_generate(model,tokenizer,[prompt],device,max_new)
            base_correct.append(exact_match(base_texts[0],ex.answer))
            p=tokenizer(prompt,return_tensors="pt",truncation=True);p={k:v.to(device) for k,v in p.items()}
            s=tokenizer(ex.state.process_text(),return_tensors="pt",truncation=True,max_length=mem["max_state_tokens"]);s={k:v.to(device) for k,v in s.items()}
            z_inst=instance_encoder(s["input_ids"],s["attention_mask"])
            h_q=torch.zeros(1,hidden,dtype=z_inst.dtype,device=device)
            z_skill=router.route(h_q,skill_bank)
            z=composer(z_skill,z_inst)
            latent_texts=_generate(model,tokenizer,p["input_ids"],p["attention_mask"],z,device,max_new)
            latent_correct.append(exact_match(latent_texts[0],ex.answer))
        a_base=sum(base_correct)/len(base_correct);a_latent=sum(latent_correct)/len(latent_correct)
        gg=generalization_gain(a_latent,a_base)
        mc=mcnemar_test(latent_correct,base_correct)
        results[split]={"a_base":a_base,"a_latent":a_latent,"gg":gg,"mcnemar":mc}
        print(f"{split}: A_base={a_base:.3f} A_latent={a_latent:.3f} GG={gg:.3f} p={mc['p_value']:.4f}")
    with open(outdir/"ood_results.json","w") as f: json.dump(results,f,indent=2)
    print(f"OOD results saved to {outdir/'ood_results.json'}")

if __name__=="__main__": main()
