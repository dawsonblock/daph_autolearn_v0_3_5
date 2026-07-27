from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class LatentVerifier:
    """Verifies candidate latent states before committing to memory.

    Architecture:
        Model generates candidate latent
                |
                v
        Latent verifier
                |
         +------+------+
        pass          fail
         |              |
         v              v
       commit      repair / discard

    For arithmetic, uses deterministic verification (no LLM judge).
    SATQuest and DeepVerifier motivate the architecture.
    """
    def __init__(self,latent_dim:int,hidden_dim:int=256)->None:
        self.latent_dim=latent_dim
        # Lightweight MLP verifier: predicts whether a latent will produce
        # a correct answer for a given query embedding
        self.verifier_net=nn.Sequential(
            nn.Linear(latent_dim*2,hidden_dim),nn.GELU(),
            nn.Linear(hidden_dim,hidden_dim),nn.GELU(),
            nn.Linear(hidden_dim,1),nn.Sigmoid(),
        )

    def verify(self,query_embedding:torch.Tensor,latent:torch.Tensor)->tuple[bool,float]:
        """Verify a (query, latent) pair.

        Returns (passed, confidence_score).
        """
        # Concatenate query embedding and latent
        q=query_embedding.reshape(-1)[:self.latent_dim].float()
        z=latent.reshape(-1)[:self.latent_dim].float()
        if q.size(0)<self.latent_dim: q=F.pad(q,(0,self.latent_dim-q.size(0)))
        if z.size(0)<self.latent_dim: z=F.pad(z,(0,self.latent_dim-z.size(0)))
        x=torch.cat([q,z],dim=-1).unsqueeze(0)
        with torch.no_grad():
            score=float(self.verifier_net(x).item())
        return score>=0.5,score

    def verify_arithmetic(self,question:str,predicted_answer:str,expected_answer:str)->bool:
        """Deterministic verification for arithmetic (no LLM judge)."""
        from ..evaluation.metrics import normalize_answer
        return normalize_answer(predicted_answer)==normalize_answer(expected_answer)

    def train_verifier(self,query_embeddings:torch.Tensor,latents:torch.Tensor,labels:torch.Tensor,epochs:int=10,lr:float=1e-3)->list[float]:
        """Train the verifier network on (query, latent, success) triples."""
        opt=torch.optim.Adam(self.verifier_net.parameters(),lr=lr)
        losses:list[float]=[]
        for _ in range(epochs):
            opt.zero_grad()
            x=torch.cat([query_embeddings,latents],dim=-1)
            preds=self.verifier_net(x).squeeze(-1)
            loss=F.binary_cross_entropy(preds,labels.float())
            loss.backward();opt.step()
            losses.append(float(loss.item()))
        return losses
