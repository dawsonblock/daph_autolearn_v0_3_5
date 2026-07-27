from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class LatentRouter(nn.Module):
    """Routes queries to skill primitives.

    Input: h_q (query hidden state)
    Output: alpha_1 ... alpha_K (soft routing weights over K skills)

    In v0.5.2, trained supervised on operation labels.
    RL training is deferred to the follow-up domain.
    """
    def __init__(self,hidden_dim:int,num_skills:int,router_topk:int=0)->None:
        super().__init__()
        self.num_skills=num_skills
        self.router_topk=router_topk  # 0 = soft (use all), >0 = TopK sparse
        self.proj=nn.Linear(hidden_dim,num_skills)

    def forward(self,h_q:torch.Tensor)->torch.Tensor:
        """h_q: [batch, hidden_dim] -> alpha: [batch, num_skills]"""
        logits=self.proj(h_q)
        if self.router_topk>0 and self.router_topk<self.num_skills:
            # Sparse TopK routing
            topk_vals,topk_idx=logits.topk(self.router_topk,dim=-1)
            mask=torch.zeros_like(logits).scatter_(-1,topk_idx,1.0)
            # Softmax over only the top-k entries
            masked_logits=logits.masked_fill(mask==0,float("-inf"))
            alpha=F.softmax(masked_logits,dim=-1)
        else:
            alpha=F.softmax(logits,dim=-1)
        return alpha

    def route(self,h_q:torch.Tensor,skill_bank)->torch.Tensor:
        """Full routing: compute alpha and produce weighted skill composition.

        Returns: [batch, latent_tokens, skill_dim]
        """
        alpha=self.forward(h_q)  # [batch, num_skills]
        skills=skill_bank.get_all_skills()  # [num_skills, latent_tokens, skill_dim]
        # Weighted sum: [batch, latent_tokens, skill_dim]
        z_skill=torch.einsum("bk,ktj->btj",alpha,skills)
        return z_skill

    def supervised_loss(self,h_q:torch.Tensor,skill_ids:torch.Tensor)->torch.Tensor:
        """Cross-entropy loss for supervised routing training."""
        logits=self.proj(h_q)
        return F.cross_entropy(logits,skill_ids)
