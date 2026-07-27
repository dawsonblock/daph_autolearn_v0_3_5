from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def resolve_dtype(name:str):
    table={"float32":torch.float32,"float16":torch.float16,"bfloat16":torch.bfloat16}
    if name not in table: raise ValueError(f"Unsupported dtype: {name}")
    return table[name]

def load_frozen_model(cfg:dict):
    mcfg=cfg["model"]
    tokenizer=AutoTokenizer.from_pretrained(mcfg["name"],trust_remote_code=mcfg.get("trust_remote_code",True))
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(mcfg["name"],torch_dtype=resolve_dtype(mcfg.get("dtype","bfloat16")),trust_remote_code=mcfg.get("trust_remote_code",True),device_map=mcfg.get("device_map","auto"))
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,tokenizer

def primary_device(model)->torch.device:
    return next(model.parameters()).device
