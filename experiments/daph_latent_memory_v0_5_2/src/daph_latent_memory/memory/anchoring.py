from __future__ import annotations
import torch

class AntiAnchoringControl:
    """Anti-anchoring control (SteeM-inspired).

    Gives DAPH a memory reliance coefficient rho in [0, 1]:
      0.0  -> ignore long-term memory
      0.25 -> weak influence
      0.5  -> balanced
      0.75 -> strong memory dependence
      1.0  -> maximum historical fidelity

    Eventually the router can learn rho automatically. In v0.5.2, rho is
    a configurable parameter that scales the influence of retrieved memory
    on the composed latent.
    """
    def __init__(self,rho:float=0.5)->None:
        self.rho=max(0.0,min(1.0,rho))

    def scale_latent(self,z_memory:torch.Tensor,z_base:torch.Tensor)->torch.Tensor:
        """Scale memory influence by rho.

        z = rho * z_memory + (1 - rho) * z_base

        When rho=0, memory is completely ignored.
        When rho=1, memory is fully trusted.
        """
        return self.rho*z_memory+(1.0-self.rho)*z_base

    def scale_retrieved(self,retrieved_latents:torch.Tensor)->torch.Tensor:
        """Scale retrieved latents by rho before injection."""
        return self.rho*retrieved_latents

    def set_rho(self,rho:float)->None:
        """Set the memory reliance coefficient."""
        self.rho=max(0.0,min(1.0,rho))

    def get_rho(self)->float:
        return self.rho

    def should_retrieve(self,retrieval_score:float,threshold:float=0.5)->bool:
        """Decide whether to retrieve memory at all.

        When rho is high, the threshold is lowered (more willing to retrieve).
        When rho is very low, even high-relevance retrievals are skipped.
        """
        effective_threshold=threshold*(1.0-self.rho*0.5)  # lower threshold when rho is high
        return retrieval_score>=effective_threshold and self.rho>0.01
