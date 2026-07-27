from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch

def file_sha256(path:str|Path)->str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def save_checkpoint(path:str|Path,encoder,optimizer,meta:dict,constant=None)->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);payload={"encoder":encoder.state_dict(),"constant":constant.state_dict() if constant is not None else None,"optimizer":optimizer.state_dict() if optimizer is not None else None,"meta":meta};torch.save(payload,path)
    with open(path.with_suffix(".json"),"w",encoding="utf-8") as f:json.dump(meta,f,indent=2,sort_keys=True)
