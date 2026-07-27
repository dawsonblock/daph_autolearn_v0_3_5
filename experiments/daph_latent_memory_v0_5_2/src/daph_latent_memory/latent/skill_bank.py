from __future__ import annotations
import torch
from torch import nn

class SkillBank(nn.Module):
    """Holds K reusable skill vectors. Each skill is a single latent token.

    Skills are indexed by integer ID. On arithmetic, skills correspond to
    operators (add, subtract, multiply, divide, verify, decompose) but the
    bank is domain-agnostic — it just holds K learned vectors.
    """
    def __init__(self,num_skills:int,latent_tokens:int,skill_dim:int)->None:
        super().__init__()
        self.num_skills=num_skills;self.latent_tokens=latent_tokens;self.skill_dim=skill_dim
        # Each skill is [latent_tokens, skill_dim]
        self.skills=nn.Parameter(torch.randn(num_skills,latent_tokens,skill_dim)*.02)

    def get_skill(self,skill_id:int)->torch.Tensor:
        """Return a single skill vector [latent_tokens, skill_dim]."""
        return self.skills[skill_id]

    def get_skill_batch(self,skill_ids:torch.Tensor)->torch.Tensor:
        """Return skill vectors for a batch of IDs: [batch, latent_tokens, skill_dim]."""
        return self.skills[skill_ids]

    def get_all_skills(self)->torch.Tensor:
        """Return all skill vectors: [num_skills, latent_tokens, skill_dim]."""
        return self.skills

    def freeze(self)->None:
        for p in self.parameters(): p.requires_grad_(False)

    def unfreeze(self)->None:
        for p in self.parameters(): p.requires_grad_(True)

    def partial_freeze(self,ratio:float)->None:
        """Freeze a fraction of skills (rounded). ratio=1.0 freezes all, 0.0 unfreezes all.

        Frozen skills have their gradients zeroed via a backward hook so they
        are not updated by the optimizer while still participating in forward passes.
        """
        ratio=max(0.0,min(1.0,ratio))
        n_frozen=int(round(self.num_skills*ratio))
        self.skills.requires_grad_(True)
        if not hasattr(self,"_freeze_mask"):
            self.register_buffer("_freeze_mask",torch.zeros(self.num_skills,dtype=torch.bool))
        self._freeze_mask[:]=False
        if n_frozen>0:
            self._freeze_mask[:n_frozen]=True
        # Remove any previous hook
        if hasattr(self,"_freeze_hook") and self._freeze_hook is not None:
            self._freeze_hook.remove()
        # Register a gradient hook that zeros gradients for frozen skills
        mask=self._freeze_mask
        def _hook(grad:torch.Tensor)->torch.Tensor:
            return grad*(~mask).float().view(-1,*([1]*(grad.dim()-1)))
        self._freeze_hook=self.skills.register_hook(_hook)

    def get_freeze_mask(self)->torch.Tensor|None:
        return getattr(self,"_freeze_mask",None)
