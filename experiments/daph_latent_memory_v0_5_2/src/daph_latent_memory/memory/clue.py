from __future__ import annotations
import torch
from torch import nn

class CLUEVerifier:
    """Hidden-state success verification (CLUE).

    Maintains success and failure centroids from training trajectories.
    Computes a confidence score C = d_f - d_s where:
      d_s = ||Δh - c_success||
      d_f = ||Δh - c_failure||

    C > 0 means closer to success centroid (higher confidence).

    OOD caveat: centroids are training-distribution-biased. C is used as a
    confidence signal on IID only, not on OOD gates. On OOD splits, C is
    logged for analysis but not used as a gate or commit signal.
    """
    def __init__(self, hidden_dim:int)->None:
        self.hidden_dim=hidden_dim
        self.c_success:torch.Tensor|None=None
        self.c_failure:torch.Tensor|None=None
        self._success_accum=torch.zeros(hidden_dim)
        self._failure_accum=torch.zeros(hidden_dim)
        self._success_count=0
        self._failure_count=0

    def update(self,delta_h:torch.Tensor,success:bool)->None:
        """Accumulate a trajectory's hidden-state delta."""
        dh=delta_h.float().reshape(-1)[:self.hidden_dim]
        if dh.size(0)<self.hidden_dim:
            dh=torch.nn.functional.pad(dh,(0,self.hidden_dim-dh.size(0)))
        if success:
            self._success_accum+=dh;self._success_count+=1
            self.c_success=self._success_accum/self._success_count
        else:
            self._failure_accum+=dh;self._failure_count+=1
            self.c_failure=self._failure_accum/self._failure_count

    def compute_delta(self,h_start:torch.Tensor,h_end:torch.Tensor)->torch.Tensor:
        """Δh = h_end - h_start."""
        return (h_end-h_start).float()

    def confidence(self,delta_h:torch.Tensor)->float:
        """C = d_f - d_s. Positive = closer to success."""
        if self.c_success is None or self.c_failure is None:
            return 0.0
        dh=delta_h.float().reshape(-1)[:self.hidden_dim]
        if dh.size(0)<self.hidden_dim:
            dh=torch.nn.functional.pad(dh,(0,self.hidden_dim-dh.size(0)))
        d_s=torch.norm(dh-self.c_success.to(dh.device)).item()
        d_f=torch.norm(dh-self.c_failure.to(dh.device)).item()
        return d_f-d_s

    def is_calibrated(self)->bool:
        """Check if both centroids have been populated."""
        return self.c_success is not None and self.c_failure is not None

    def reset(self)->None:
        """Reset accumulated centroids."""
        self.c_success=None;self.c_failure=None
        self._success_accum=torch.zeros(self.hidden_dim)
        self._failure_accum=torch.zeros(self.hidden_dim)
        self._success_count=0;self._failure_count=0

    def save(self,path:str)->None:
        """Save centroids to file."""
        torch.save({
            "c_success":self.c_success,"c_failure":self.c_failure,
            "success_count":self._success_count,"failure_count":self._failure_count,
            "hidden_dim":self.hidden_dim,
        },path)

    def load(self,path:str)->None:
        """Load centroids from file."""
        data=torch.load(path,map_location="cpu",weights_only=False)
        self.c_success=data["c_success"];self.c_failure=data["c_failure"]
        self._success_count=data["success_count"];self._failure_count=data["failure_count"]
        self.hidden_dim=data["hidden_dim"]
