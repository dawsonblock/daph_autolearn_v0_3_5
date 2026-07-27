from __future__ import annotations
import torch
from torch import nn

class ConstantLatent(nn.Module):
    """A task-independent trainable soft-memory baseline."""
    def __init__(self,latent_tokens:int,hidden_dim:int)->None:
        super().__init__();self.tokens=nn.Parameter(torch.randn(latent_tokens,hidden_dim)*.02)
    def forward(self,batch_size:int)->torch.Tensor:return self.tokens.unsqueeze(0).expand(batch_size,-1,-1)
