from __future__ import annotations
import random
import torch
import numpy as np
from ..state.corruption import corrupt_latent_vector

# =============================================================================
# v0.5.2 Causal Evaluation Controls
#
# All conditions are reproducible from (example_id, condition, seed).
# The single "corrupted" control from v0.5.1 is replaced by a full α-sweep.
# =============================================================================

def random_latents_like(latents:torch.Tensor,seed:int|None=None)->torch.Tensor:
    if seed is None: return torch.randn_like(latents)
    generator=torch.Generator(device=latents.device);generator.manual_seed(seed)
    return torch.randn(latents.shape,dtype=latents.dtype,device=latents.device,generator=generator)

def random_norm_matched_latents_like(latents:torch.Tensor,seed:int|None=None)->torch.Tensor:
    """Gaussian random rescaled to match the ||z_i|| distribution."""
    z=random_latents_like(latents,seed)
    target_norm=latents.norm(dim=-1,keepdim=True)
    z_norm=z.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    return z*(target_norm/z_norm)

def deranged_indices(n:int,seed:int=1337)->list[int]:
    """Return a derangement (permutation with no fixed points) of range(n)."""
    if n<2: raise ValueError("Derangement requires at least two elements")
    rng=random.Random(seed);indices=list(range(n))
    for _ in range(1000):
        rng.shuffle(indices)
        if all(i!=j for i,j in enumerate(indices)): return indices
    return indices[1:]+indices[:1]

def shuffled_global(latents:torch.Tensor,seed:int=1337)->torch.Tensor:
    """Shuffle latents across all examples (derangement)."""
    n=latents.size(0)
    perm=deranged_indices(n,seed)
    return latents[torch.tensor(perm,device=latents.device)]

def shuffled_same_class(latents:torch.Tensor,labels:list,seed:int=1337)->torch.Tensor:
    """Shuffle latents within same-class groups (derangement per class)."""
    out=latents.clone()
    by_class:dict[list[int],list[int]]={}
    for i,label in enumerate(labels): by_class.setdefault(label,[]).append(i)
    rng=random.Random(seed)
    for label,indices in by_class.items():
        if len(indices)<2: continue
        for _ in range(1000):
            shuffled=list(indices);rng.shuffle(shuffled)
            if all(a!=b for a,b in zip(indices,shuffled)):
                for orig,new in zip(indices,shuffled): out[orig]=latents[new]
                break
    return out

def shuffled_wrong_class(latents:torch.Tensor,labels:list,seed:int=1337)->torch.Tensor:
    """Shuffle latents across different-class examples only."""
    n=latents.size(0);out=latents.clone();rng=random.Random(seed)
    for i in range(n):
        candidates=[j for j in range(n) if labels[j]!=labels[i]]
        if candidates:
            out[i]=latents[rng.choice(candidates)]
        else:
            # No wrong-class examples; use derangement
            perm=deranged_indices(n,seed+i);out[i]=latents[perm[i]]
    return out

def _flatten_latents(latents:torch.Tensor)->torch.Tensor:
    """Flatten [n, tokens, dim] -> [n, tokens*dim] for similarity computation."""
    return latents.float().reshape(latents.size(0),-1)

def nearest_neighbor_wrong(latents:torch.Tensor,labels:list)->torch.Tensor:
    """For each example, find the nearest latent by cosine among wrong-class examples."""
    out=latents.clone()
    flat=_flatten_latents(latents)
    normalized=torch.nn.functional.normalize(flat,dim=-1)
    for i in range(latents.size(0)):
        wrong_mask=torch.tensor([labels[j]!=labels[i] for j in range(latents.size(0))],dtype=torch.bool,device=latents.device)
        if wrong_mask.sum()==0: continue
        sims=(normalized[i].unsqueeze(0)*normalized[wrong_mask]).sum(dim=-1)
        best=sims.argmax();out[i]=latents[wrong_mask.nonzero(as_tuple=True)[0][best]]
    return out

def farthest_neighbor(latents:torch.Tensor)->torch.Tensor:
    """For each example, find the farthest latent by cosine distance."""
    out=latents.clone()
    flat=_flatten_latents(latents)
    normalized=torch.nn.functional.normalize(flat,dim=-1)
    for i in range(latents.size(0)):
        sims=(normalized[i].unsqueeze(0)*normalized).sum(dim=-1)
        sims[i]=1.0  # exclude self
        worst=sims.argmin();out[i]=latents[worst]
    return out

def normalized_latents(latents:torch.Tensor)->torch.Tensor:
    return torch.nn.functional.normalize(latents.float(),dim=-1).to(latents.dtype)

def scaled_latents(latents:torch.Tensor,c:float=2.0)->torch.Tensor:
    return latents*c

def sign_flipped_latents(latents:torch.Tensor)->torch.Tensor:
    return -latents

def mean_latent(latents:torch.Tensor)->torch.Tensor:
    mean=latents.mean(dim=0,keepdim=True)
    return mean.expand(latents.size(0),-1,-1)

def centroid_latent(latents:torch.Tensor,labels:list)->torch.Tensor:
    """Per-class centroid latent."""
    out=latents.clone()
    by_class:dict[list[int],list[int]]={}
    for i,label in enumerate(labels): by_class.setdefault(label,[]).append(i)
    for label,indices in by_class.items():
        centroid=latents[indices].mean(dim=0)
        for idx in indices: out[idx]=centroid
    return out

def pca_low_rank(latents:torch.Tensor,k:int=8)->torch.Tensor:
    """Project latents onto top-k principal components."""
    flat=latents.reshape(latents.size(0),-1).float()
    mean=flat.mean(dim=0,keepdim=True);centered=flat-mean
    U,S,Vh=torch.linalg.svd(centered,full_matrices=False)
    proj=U[:,:k]@torch.diag(S[:k])@Vh[:k,:]
    proj=proj+mean
    return proj.reshape_as(latents).to(latents.dtype)

def pca_residual(latents:torch.Tensor,k:int=8)->torch.Tensor:
    """Project latents onto the residual subspace (remove top-k components)."""
    flat=latents.reshape(latents.size(0),-1).float()
    mean=flat.mean(dim=0,keepdim=True);centered=flat-mean
    U,S,Vh=torch.linalg.svd(centered,full_matrices=False)
    # Reconstruct with top-k removed
    residual=U[:,k:]@torch.diag(S[k:])@Vh[k:,:]
    residual=residual+mean
    return residual.reshape_as(latents).to(latents.dtype)

def corruption_sweep(latents:torch.Tensor,alphas:list[float],seeds:list[int])->dict[float,dict[int,torch.Tensor]]:
    """Generate corrupted latents for each (alpha, seed) pair.

    Returns: {alpha: {seed: corrupted_tensor}}
    """
    out:dict[float,dict[int,torch.Tensor]]={}
    for alpha in alphas:
        out[alpha]={}
        for seed in seeds:
            out[alpha][seed]=corrupt_latent_vector(latents,alpha,seed=seed)
    return out
