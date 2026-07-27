from __future__ import annotations
import torch
from torch import nn
from .skill_bank import SkillBank
from .instance_state import InstanceEncoder

class LinearComposer(nn.Module):
    """Linear composition: z = W_s @ z_skill + W_i @ z_instance."""
    def __init__(self,skill_dim:int,instance_dim:int,output_dim:int,latent_tokens:int)->None:
        super().__init__()
        self.W_s=nn.Linear(skill_dim,output_dim,bias=False)
        self.W_i=nn.Linear(instance_dim,output_dim,bias=False)
        self.norm=nn.LayerNorm(output_dim)

    def forward(self,z_skill:torch.Tensor,z_instance:torch.Tensor)->torch.Tensor:
        # z_skill: [batch, latent_tokens, skill_dim]
        # z_instance: [batch, latent_tokens, instance_dim]
        return self.norm(self.W_s(z_skill)+self.W_i(z_instance))

class GatedComposer(nn.Module):
    """Gated composition: z = g_s(q) * z_skill + g_i(q) * z_instance.

    Gates are computed from the instance representation (which serves as a
    proxy for the query encoding when a separate query encoder is not wired).
    """
    def __init__(self,skill_dim:int,instance_dim:int,output_dim:int,latent_tokens:int)->None:
        super().__init__()
        self.gate_s=nn.Linear(instance_dim,output_dim)
        self.gate_i=nn.Linear(instance_dim,output_dim)
        self.W_s=nn.Linear(skill_dim,output_dim,bias=False)
        self.W_i=nn.Linear(instance_dim,output_dim,bias=False)
        self.norm=nn.LayerNorm(output_dim)

    def forward(self,z_skill:torch.Tensor,z_instance:torch.Tensor)->torch.Tensor:
        g_s=torch.sigmoid(self.gate_s(z_instance));g_i=torch.sigmoid(self.gate_i(z_instance))
        return self.norm(g_s*self.W_s(z_skill)+g_i*self.W_i(z_instance))

def make_composer(composition_type:str,skill_dim:int,instance_dim:int,output_dim:int,latent_tokens:int)->nn.Module:
    if composition_type=="linear": return LinearComposer(skill_dim,instance_dim,output_dim,latent_tokens)
    if composition_type=="gated": return GatedComposer(skill_dim,instance_dim,output_dim,latent_tokens)
    raise ValueError(f"Unknown composition type: {composition_type}")
