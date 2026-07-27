from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class MemorySelector(nn.Module):
    """Learned selector that decides whether to inject a retrieved memory.

    Two-stage training (corrected plan section 10.2):
    1. Heuristic warm-start (first 50%): distill "inject iff sim > tau"
    2. REINFORCE fine-tuning (second 50%): reward = ΔR = R(skip) - R(inject)

    The distinction "retrieval relevance ≠ downstream usefulness" (ElasticMem)
    is handled by stage 2.
    """
    def __init__(self,key_dim:int,hidden_dim:int=128)->None:
        super().__init__()
        self.selector_net=nn.Sequential(
            nn.Linear(key_dim*2,hidden_dim),nn.GELU(),
            nn.Linear(hidden_dim,1),nn.Sigmoid(),
        )
        self.baseline:dict[str,float]={}  # per-capability moving average

    def forward(self,query_key:torch.Tensor,candidate_key:torch.Tensor)->torch.Tensor:
        """Returns inject probability in [0, 1]."""
        x=torch.cat([query_key,candidate_key],dim=-1).float()
        return self.selector_net(x).squeeze(-1)

    def heuristic_decision(self,sim:float,tau:float=0.5)->int:
        """Stage 1 heuristic: inject iff sim > tau."""
        return int(sim>tau)

    def reinforce_loss(
        self,
        query_key:torch.Tensor,
        candidate_key:torch.Tensor,
        delta_r:float,
        capability_id:str,
        gamma:float=0.9,
    )->torch.Tensor:
        """Stage 2 REINFORCE loss with per-capability baseline.

        delta_r = R(skip) - R(inject)  (positive = injecting helped)
        """
        prob=self.forward(query_key.unsqueeze(0),candidate_key.unsqueeze(0))
        baseline=self.baseline.get(capability_id,0.0)
        advantage=delta_r-baseline
        # Update baseline (moving average)
        self.baseline[capability_id]=gamma*baseline+(1-gamma)*delta_r
        # REINFORCE: encourage the action that was taken if advantage > 0
        action=prob.detach().round()
        loss=-action*advantage*torch.log(prob.clamp_min(1e-8))-(1-action)*(-advantage)*torch.log((1-prob).clamp_min(1e-8))
        return loss
