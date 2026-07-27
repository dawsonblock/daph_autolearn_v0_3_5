from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class InjectedBatch:
    inputs_embeds:torch.Tensor
    attention_mask:torch.Tensor
    labels:torch.Tensor
    memory_slice:slice

class LatentInjector(nn.Module):
    """Insert latent memory between prompt embeddings and answer teacher-forcing tokens.

    v0.3.8 DEF-02: optional relative-norm safety clamp. When
    ``relative_norm_limit`` > 0 (set via :meth:`set_relative_norm_limit`),
    :meth:`build` scales down any latent whose mean L2 norm exceeds
    ``relative_norm_limit * mean(||prompt_emb||_2)``. This extends the
    steering safety guarantee (R_pert ≤ 0.65) from the residual-hook path
    to the latent-injection path, preventing the unbounded-latent drift
    that can destabilize training on shallow or non-instruct models.

    The clamp is applied per-example in the batch: each row's latent is
    scaled by ``min(1, limit * ||prompt_emb_i|| / ||latent_i||)``. This
    preserves relative latent geometry within a row while capping the
    perturbation magnitude. The scaling factor is recorded in
    :attr:`last_clamp_multipliers` for telemetry.
    """
    def __init__(self,model:nn.Module)->None:
        super().__init__();self.model=model
        self._relative_norm_limit:float=0.0
        self.last_clamp_multipliers:torch.Tensor|None=None

    def set_relative_norm_limit(self,limit:float)->None:
        """Enable (limit>0) or disable (limit=0) the latent norm safety clamp."""
        if limit<0: raise ValueError(f"relative_norm_limit must be >= 0, got {limit}")
        self._relative_norm_limit=float(limit)

    @property
    def relative_norm_limit(self)->float:
        return self._relative_norm_limit

    def _clamp_latent_norm(self,prompt_emb:torch.Tensor,latent_tokens:torch.Tensor)->torch.Tensor:
        """Scale latent_tokens per-row so ||latent_i|| <= limit * ||prompt_emb_i||.

        Returns the (possibly scaled) latent tensor and records the per-row
        clamp multipliers in :attr:`last_clamp_multipliers`.
        """
        if self._relative_norm_limit<=0:
            self.last_clamp_multipliers=None
            return latent_tokens
        # Per-row L2 norms. prompt_emb: [batch, seq, hidden]; latent: [batch, k, hidden].
        prompt_norms=prompt_emb.float().norm(dim=-1).mean(dim=-1)  # [batch]
        latent_norms=latent_tokens.float().norm(dim=-1).mean(dim=-1)  # [batch]
        # allowed = limit * prompt_norm
        allowed=self._relative_norm_limit*prompt_norms
        # multiplier = min(1, allowed / latent_norm)
        multipliers=torch.ones_like(latent_norms)
        over=latent_norms>allowed
        safe_latent=latent_norms.clamp_min(1e-6)
        multipliers=torch.where(over,allowed/safe_latent,multipliers)
        self.last_clamp_multipliers=multipliers.detach()
        # Broadcast multiplier to [batch, 1, 1] for elementwise scaling.
        scaled=latent_tokens*multipliers.view(-1,1,1).to(latent_tokens.dtype)
        return scaled

    def build(self,prompt_ids:torch.Tensor,prompt_mask:torch.Tensor,answer_ids:torch.Tensor,answer_mask:torch.Tensor,latent_tokens:torch.Tensor)->InjectedBatch:
        embed=self.model.get_input_embeddings()
        prompt_emb=embed(prompt_ids);answer_emb=embed(answer_ids)
        latent_tokens=latent_tokens.to(device=prompt_emb.device,dtype=prompt_emb.dtype)
        answer_emb=answer_emb.to(device=prompt_emb.device,dtype=prompt_emb.dtype)
        # v0.3.8 DEF-02: safety clamp on the latent norm relative to the
        # prompt embedding norm. No-op when relative_norm_limit == 0.
        latent_tokens=self._clamp_latent_norm(prompt_emb,latent_tokens)
        b,k,_=latent_tokens.shape
        full=torch.cat([prompt_emb,latent_tokens,answer_emb],dim=1)
        latent_mask=torch.ones((b,k),device=prompt_mask.device,dtype=prompt_mask.dtype)
        mask=torch.cat([prompt_mask,latent_mask,answer_mask],dim=1)
        ignore_prompt=torch.full_like(prompt_ids,-100)
        ignore_latent=torch.full((b,k),-100,device=answer_ids.device,dtype=answer_ids.dtype)
        safe_answer_labels=answer_ids.masked_fill(answer_mask==0,-100)
        labels=torch.cat([ignore_prompt,ignore_latent,safe_answer_labels],dim=1)
        start=prompt_ids.size(1)
        return InjectedBatch(full,mask,labels,slice(start,start+k))
