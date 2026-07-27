from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch

def file_sha256(path:str|Path)->str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def save_checkpoint(path:str|Path,state_dict:dict,meta:dict)->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={"state_dict":state_dict,"meta":meta}
    torch.save(payload,path)
    with open(path.with_suffix(".json"),"w",encoding="utf-8") as f:json.dump(meta,f,indent=2,sort_keys=True)

def load_checkpoint(path:str|Path,map_location:str="cpu")->dict:
    return torch.load(path,map_location=map_location,weights_only=False)
