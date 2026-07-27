from __future__ import annotations
import copy,random,re
from .schema import MathWorkingState

_INT_RE=re.compile(r"(?<![A-Za-z])[-+]?\d+")

def _perturb_first_integer(text:str,rng:random.Random)->str:
    match=_INT_RE.search(text)
    if not match: return text+" [incorrect operation: multiply instead of invert]"
    value=int(match.group(0));delta=rng.choice([-3,-2,-1,1,2,3]);wrong=value+delta
    return text[:match.start()]+str(wrong)+text[match.end():]

def corrupt_state(state:MathWorkingState,rng:random.Random)->MathWorkingState:
    """Create a semantically wrong state rather than merely labeling content as corrupted."""
    out=copy.deepcopy(state.model_dump())
    if out["intermediate_steps"]:
        idx=rng.randrange(len(out["intermediate_steps"]));out["intermediate_steps"][idx]=_perturb_first_integer(out["intermediate_steps"][idx],rng)
    elif out["unresolved_subgoals"]:
        idx=rng.randrange(len(out["unresolved_subgoals"]));original=out["unresolved_subgoals"][idx]
        if "inverse" in original.lower() or "remove" in original.lower(): out["unresolved_subgoals"][idx]="apply the same operation again instead of its inverse"
        elif "divide" in original.lower(): out["unresolved_subgoals"][idx]=original.lower().replace("divide","multiply")
        else: out["unresolved_subgoals"][idx]=_perturb_first_integer(original,rng)
    elif out["constraints"]:
        idx=rng.randrange(len(out["constraints"]));out["constraints"][idx]=_perturb_first_integer(out["constraints"][idx],rng)
    else:
        out["unresolved_subgoals"].append("apply an invalid inverse operation")
    return MathWorkingState(**out)

def corrupt_latent_vector(z:object,alpha:float,seed:int|None=None)->object:
    """Add gaussian noise to a latent tensor: z' = z + alpha * eps.

    eps is normalized to match the per-element scale of z so that alpha has
    a consistent meaning across latents of different norms.
    """
    import torch
    if not isinstance(z,torch.Tensor): raise TypeError("corrupt_latent_vector requires a torch.Tensor")
    if alpha<=0: return z.clone()
    gen=torch.Generator(device=z.device)
    if seed is not None: gen.manual_seed(seed)
    eps=torch.randn(z.shape,dtype=z.dtype,device=z.device,generator=gen)
    # Normalize eps to match per-element scale of z
    z_std=z.std().clamp_min(1e-8)
    eps=eps*(z_std/eps.std().clamp_min(1e-8))
    return z+alpha*eps
