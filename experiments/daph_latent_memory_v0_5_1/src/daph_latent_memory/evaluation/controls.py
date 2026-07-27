from __future__ import annotations
import random
import torch
from ..state.corruption import corrupt_state

def random_latents_like(latents:torch.Tensor,seed:int|None=None)->torch.Tensor:
    if seed is None:return torch.randn_like(latents)
    generator=torch.Generator(device=latents.device);generator.manual_seed(seed)
    return torch.randn(latents.shape,dtype=latents.dtype,device=latents.device,generator=generator)

def deranged_states(states:list,seed:int=1337):
    n=len(states)
    if n<2: raise ValueError("Derangement requires at least two states")
    rng=random.Random(seed);indices=list(range(n))
    for _ in range(1000):
        rng.shuffle(indices)
        if all(i!=j for i,j in enumerate(indices)): return [states[j] for j in indices]
    return states[1:]+states[:1]

def corrupted_states(states:list,seed:int=1337):
    rng=random.Random(seed);return [corrupt_state(s,rng) for s in states]
