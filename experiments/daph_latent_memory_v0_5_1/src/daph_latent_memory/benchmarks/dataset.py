from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
from ..state.schema import MathExample
from .algebra import generate_example

def generate_dataset(n:int,seed:int=1337)->list[MathExample]:
    if n<20: raise ValueError("n must be >= 20")
    counts={"train":int(n*.70),"iid":int(n*.10),"composition":int(n*.10)}; counts["ood"]=n-sum(counts.values())
    out=[]
    for split,count in counts.items(): out.extend(generate_example(split,i,seed) for i in range(count))
    return out

def save_jsonl(examples:Iterable[MathExample],path:str|Path)->None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for ex in examples: f.write(ex.model_dump_json()+"\n")

def load_jsonl(path:str|Path)->list[MathExample]:
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(MathExample.model_validate(json.loads(line)))
    return out
