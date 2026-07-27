from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .bank import MemoryBank,MemoryEntry,MemoryAction,MemoryState

class MemoryManager:
    """Manages memory write operations: ADD, UPDATE, MERGE, DELETE, NOOP.

    Per Memory-R1, learned operations beat heuristic append-only.
    MERGE is added because learned skill latents produce near-duplicates.
    """
    def __init__(self,bank:MemoryBank,similarity_threshold:float=0.95)->None:
        self.bank=bank
        self.similarity_threshold=similarity_threshold

    def _cosine_sim(self,a:torch.Tensor,b:torch.Tensor)->float:
        a=F.normalize(a.float().unsqueeze(0),dim=-1)
        b=F.normalize(b.float().unsqueeze(0),dim=-1)
        return float((a*b).sum().item())

    def propose_action(self,new_entry:MemoryEntry)->MemoryAction:
        """Decide which memory action to take for a new candidate entry.

        Heuristic policy (v0.5.2):
        - If no similar entry exists -> ADD
        - If a similar entry exists with lower confidence -> UPDATE
        - If a similar entry exists with similar confidence -> MERGE
        - If a contradicted entry exists -> DELETE (the old one)
        - Otherwise -> NOOP
        """
        retrieved=self.bank.retrieve(new_entry.key,top_k=1)
        if not retrieved:
            return MemoryAction.ADD
        best,sim=retrieved[0]
        if sim<self.similarity_threshold:
            return MemoryAction.ADD
        if best.state==MemoryState.CONTRADICTED:
            return MemoryAction.DELETE
        if best.confidence<new_entry.confidence-0.1:
            return MemoryAction.UPDATE
        if abs(best.confidence-new_entry.confidence)<=0.1:
            return MemoryAction.MERGE
        return MemoryAction.NOOP

    def execute(self,action:MemoryAction,new_entry:MemoryEntry)->bool:
        """Execute a memory action. Returns True if the bank was modified."""
        if action==MemoryAction.ADD:
            self.bank.add(new_entry);return True
        elif action==MemoryAction.UPDATE:
            retrieved=self.bank.retrieve(new_entry.key,top_k=1)
            if retrieved:
                best,sim=retrieved[0]
                if sim>=self.similarity_threshold:
                    idx=self.bank.entries.index(best)
                    new_entry.state=MemoryState.VERIFIED
                    self.bank.entries[idx]=new_entry;return True
            self.bank.add(new_entry);return True
        elif action==MemoryAction.MERGE:
            retrieved=self.bank.retrieve(new_entry.key,top_k=1)
            if retrieved:
                best,sim=retrieved[0]
                if sim>=self.similarity_threshold:
                    # Merge by averaging keys and latents
                    best.key=(best.key+new_entry.key)/2
                    best.latent=(best.latent+new_entry.latent)/2
                    best.confidence=max(best.confidence,new_entry.confidence)
                    best.verifier_score=max(best.verifier_score,new_entry.verifier_score)
                    return True
            self.bank.add(new_entry);return True
        elif action==MemoryAction.DELETE:
            retrieved=self.bank.retrieve(new_entry.key,top_k=1)
            if retrieved:
                best,sim=retrieved[0]
                if sim>=self.similarity_threshold and best.state==MemoryState.CONTRADICTED:
                    self.bank.entries.remove(best);return True
            return False
        elif action==MemoryAction.NOOP:
            return False
        return False

    def conflict_check(self,new_entry:MemoryEntry,verifier_result:bool)->bool:
        """Check if a new memory conflicts with existing verified memories.

        Returns True if the new entry is conflicting (should be rejected or
        marked CONTRADICTED).
        """
        if not verifier_result: return True
        retrieved=self.bank.retrieve(new_entry.key,top_k=3)
        for entry,sim in retrieved:
            if sim>=self.similarity_threshold and entry.state==MemoryState.VERIFIED:
                # Check if the latents point in different directions
                latent_sim=self._cosine_sim(entry.latent,new_entry.latent)
                if latent_sim<0.0:  # Opposite direction
                    return True
        return False
