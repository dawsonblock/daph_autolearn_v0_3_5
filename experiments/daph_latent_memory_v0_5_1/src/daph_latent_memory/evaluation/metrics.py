from __future__ import annotations
import math,random,re,statistics

def normalize_answer(text:str)->str:
    text=text.strip();nums=re.findall(r"[-+]?\d+(?:\.\d+)?",text);return nums[-1] if nums else text.lower().strip()

def exact_match(pred:str,gold:str)->int:return int(normalize_answer(pred)==normalize_answer(gold))

def bootstrap_ci(values:list[float],samples:int=1000,seed:int=1337):
    if not values:return {"mean":math.nan,"lo":math.nan,"hi":math.nan}
    rng=random.Random(seed);means=[];n=len(values)
    for _ in range(samples):means.append(statistics.fmean([values[rng.randrange(n)] for _ in range(n)]))
    means.sort();return {"mean":statistics.fmean(values),"lo":means[int(.025*(samples-1))],"hi":means[int(.975*(samples-1))]}

def utility(accuracy:float,latency_s:float,tokens:int,cfg:dict)->float:return cfg["accuracy_weight"]*accuracy-cfg["latency_weight"]*latency_s-cfg["token_weight"]*tokens
