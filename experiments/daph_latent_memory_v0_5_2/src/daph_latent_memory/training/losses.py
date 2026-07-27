from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

# =============================================================================
# v0.5.2 Corrected Losses
#
# The v0.5.1 alignment loss (1 - cos(student_h, teacher_h)) is REMOVED.
# It rewarded the student for matching the teacher's activation regardless of
# which latent was injected, which caused A_corrupted > A_matched.
#
# L_contrast (representation similarity) is also DROPPED. It optimizes
# geometric closeness, which is in tension with L_functional (downstream
# usefulness). Only L_functional is retained.
#
# New objective:
#   L = L_task + λ_f * L_functional + λ_d * L_disentangle
#
# Phase A adds: + λ_inv * L_invariance
# =============================================================================

def task_loss(model_output)->torch.Tensor:
    """L_task: answer NLL from the model's forward pass with labels."""
    return model_output.loss

def functional_mismatch_loss(
    r_positive:torch.Tensor,
    r_negatives:torch.Tensor,
    margin:float,
)->torch.Tensor:
    """L_functional = mean_ij [ max(0, m - r_i^+ + r_ij^-) ]

    Args:
        r_positive: [batch] — R(q_i, z_i), negative NLL with matched latent (lower=better)
        r_negatives: [batch, K] — R(q_i, z_j), negative NLL with K hard negative latents
        margin: float — margin m in nats, calibrated from matched-vs-shuffled gap

    Returns:
        scalar loss tensor

    r_positive is the matched loss (we WANT this low).
    r_negatives are mismatched losses (we WANT these high relative to matched).
    The hinge pushes r_negatives to be at least `margin` nats higher than r_positive.
    """
    # Expand r_positive to [batch, K] for broadcasting
    r_pos_expanded=r_positive.unsqueeze(1).expand_as(r_negatives)
    # We want r_neg - r_pos >= margin, i.e., hinge = max(0, margin - (r_neg - r_pos))
    hinge=torch.clamp(margin-r_negatives+r_pos_expanded,min=0.0)
    return hinge.mean()

def disentangle_loss(
    z_skill:torch.Tensor,
    z_instance:torch.Tensor,
    W_s:nn.Module|None=None,
    W_i:nn.Module|None=None,
)->torch.Tensor:
    """L_disentangle: encourage skill and instance subspaces to be orthogonal.

    Penalizes cosine similarity between the projected skill and instance
    representations. This encourages them to encode different information.
    """
    # Flatten to [batch * latent_tokens, dim] and compute per-token cosine sim
    s=z_skill.reshape(-1,z_skill.size(-1))
    i=z_instance.reshape(-1,z_instance.size(-1))
    s=F.normalize(s.float(),dim=-1)
    i=F.normalize(i.float(),dim=-1)
    # Per-token cosine similarity
    cos_sim=(s*i).sum(dim=-1)
    # Penalize absolute cosine similarity (we want orthogonality, not anti-correlation)
    return cos_sim.abs().mean()

def invariance_loss(
    z_skill_a:torch.Tensor,
    z_skill_b:torch.Tensor,
)->torch.Tensor:
    """L_invariance: skill vectors for same-skill instances should be similar.

    Used in Phase A only. For pairs (a, b) where skill(a) == skill(b):
    L = mean [1 - cos(z_skill_a, z_skill_b)]

    Without this, the skill bank absorbs operand-range information and the
    "reusable" property is asserted but not enforced.
    """
    s_a=F.normalize(z_skill_a.float(),dim=-1)
    s_b=F.normalize(z_skill_b.float(),dim=-1)
    # Mean over latent_tokens dimension, then over batch
    cos=(s_a*s_b).sum(dim=-1).mean()
    return 1.0-cos

def compute_R(
    model,
    injector,
    prompt_ids:torch.Tensor,
    prompt_mask:torch.Tensor,
    answer_ids:torch.Tensor,
    answer_mask:torch.Tensor,
    latent_tokens:torch.Tensor,
)->torch.Tensor:
    """R(q, z) = negative token NLL of the decoder on query q with latent z injected.

    This is the differentiable functional score. Lower R = better.
    Returns per-example NLL: [batch]
    """
    injected=injector.build(prompt_ids,prompt_mask,answer_ids,answer_mask,latent_tokens)
    out=model(inputs_embeds=injected.inputs_embeds,attention_mask=injected.attention_mask,labels=injected.labels,use_cache=False)
    # out.loss is mean NLL over all non-ignored tokens in the batch.
    # We need per-example NLL for the functional loss.
    # Recompute per-example loss from logits.
    logits=out.logits[:,injected.memory_slice.stop:,:]  # only answer tokens
    labels=injected.labels[:,injected.memory_slice.stop:]
    # Shift for next-token prediction
    shift_logits=logits[:,:-1,:].contiguous()
    shift_labels=labels[:,1:].contiguous()
    loss_per_token=F.cross_entropy(shift_logits.view(-1,shift_logits.size(-1)),shift_labels.view(-1),ignore_index=-100,reduction="none")
    loss_per_token=loss_per_token.view(shift_labels.shape)
    # Sum over tokens, divide by non-ignored count per example
    mask=(shift_labels!=-100).float()
    per_example_nll=(loss_per_token*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
    return per_example_nll

def total_loss_v052(
    task:torch.Tensor,
    functional:torch.Tensor|None,
    disentangle:torch.Tensor|None,
    answer_weight:float,
    functional_weight:float,
    disentangle_weight:float,
)->torch.Tensor:
    """L = answer_weight * L_task + functional_weight * L_functional + disentangle_weight * L_disentangle"""
    loss=answer_weight*task
    if functional is not None: loss=loss+functional_weight*functional
    if disentangle is not None: loss=loss+disentangle_weight*disentangle
    return loss
