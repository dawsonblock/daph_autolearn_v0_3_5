from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class ElasticBudgetAllocator(nn.Module):
    """Elastic memory budget allocator with Gumbel-softmax.

    Predicts B_i in {0, 1, 2, 4, 8, 16} latent tokens per problem.

    Uses Gumbel-softmax with a straight-through estimator during training,
    annealed to hard one-hot over the first 50% of training.

    R_total = R_accuracy - beta * B

    beta is in units of accuracy per token; calibrated per dataset.
    """
    def __init__(self,hidden_dim:int,budget_options:list[int]=None,tau_init:float=2.0,tau_min:float=0.1)->None:
        super().__init__()
        self.budget_options=budget_options or [0,1,2,4,8,16]
        self.num_options=len(self.budget_options)
        self.tau_init=tau_init;self.tau_min=tau_min
        self.proj=nn.Linear(hidden_dim,self.num_options)

    def forward(self,h_q:torch.Tensor,hard:bool=False,tau:float|None=None)->tuple[torch.Tensor,torch.Tensor]:
        """Allocate latent budget.

        Args:
            h_q: [batch, hidden_dim] query hidden state
            hard: if True, use hard one-hot (straight-through)
            tau: Gumbel-softmax temperature (annealed during training)

        Returns:
            (budget_logits, budget_values)
            budget_logits: [batch, num_options] soft probabilities
            budget_values: [batch] selected budget (int if hard, float if soft)
        """
        if tau is None: tau=self.tau_init
        logits=self.proj(h_q)
        if hard:
            # Hard Gumbel-softmax with straight-through estimator
            budget_logits=F.gumbel_softmax(logits,tau=tau,hard=True)
        else:
            budget_logits=F.gumbel_softmax(logits,tau=tau,hard=False)
        # Compute budget value
        budget_values=(budget_logits*torch.tensor(self.budget_options,dtype=budget_logits.dtype,device=budget_logits.device)).sum(dim=-1)
        return budget_logits,budget_values

    def get_budget(self,h_q:torch.Tensor,hard:bool=True)->torch.Tensor:
        """Get the allocated budget as integer tokens."""
        _,budget_values=self.forward(h_q,hard=hard)
        return budget_values.round().clamp(min=0)

    def budget_penalty(self,budget_values:torch.Tensor,beta:float)->torch.Tensor:
        """R_total = R_accuracy - beta * B. Returns the penalty term."""
        return beta*budget_values.mean()

    def anneal_tau(self,step:int,total_steps:int)->float:
        """Anneal Gumbel-softmax temperature from tau_init to tau_min."""
        frac=min(1.0,step/max(1,total_steps))
        return self.tau_init+(self.tau_min-self.tau_init)*frac

    def calibrate_beta(self,accuracy_per_token_gain:float)->float:
        """Calibrate beta such that marginal accuracy gain of one token equals beta.

        beta is in units of accuracy per token.
        """
        return accuracy_per_token_gain
