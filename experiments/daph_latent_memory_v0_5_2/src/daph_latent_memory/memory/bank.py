from __future__ import annotations
import time
from dataclasses import dataclass,field
from enum import Enum
import torch
from torch import nn

class MemoryState(str,Enum):
    VERIFIED="verified"
    UNVERIFIED="unverified"
    DEPRECATED="deprecated"
    CONTRADICTED="contradicted"
    SUPERSEDED="superseded"

class MemoryAction(str,Enum):
    ADD="add"
    UPDATE="update"
    MERGE="merge"
    DELETE="delete"
    NOOP="noop"

@dataclass
class MemoryEntry:
    key:torch.Tensor
    latent:torch.Tensor
    capability_id:str
    source_task:str
    timestamp:float=field(default_factory=time.time)
    confidence:float=0.0
    verifier_score:float=0.0
    lineage:dict=field(default_factory=dict)
    state:MemoryState=MemoryState.UNVERIFIED

class MemoryBank:
    """Persistent latent memory bank with retrieval.

    Memory is never a flat immutable cache. Entries have states
    (verified, unverified, deprecated, contradicted, superseded).
    """
    def __init__(self,key_dim:int,latent_dim:int,max_size:int=10000)->None:
        self.key_dim=key_dim;self.latent_dim=latent_dim;self.max_size=max_size
        self.entries:list[MemoryEntry]=[]

    def __len__(self)->int:
        return len(self.entries)

    def add(self,entry:MemoryEntry)->None:
        if len(self.entries)>=self.max_size:
            # Evict oldest deprecated/unverified entry
            for i,e in enumerate(self.entries):
                if e.state in (MemoryState.DEPRECATED,MemoryState.UNVERIFIED):
                    self.entries.pop(i);break
            else:
                self.entries.pop(0)  # FIFO fallback
        self.entries.append(entry)

    def keys(self)->torch.Tensor|None:
        if not self.entries: return None
        return torch.stack([e.key for e in self.entries])

    def retrieve(self,query_key:torch.Tensor,top_k:int=5)->list[tuple[MemoryEntry,float]]:
        """Retrieve top-k memories by cosine similarity to query key."""
        if not self.entries: return []
        keys=self.keys().to(query_key.device)
        q=torch.nn.functional.normalize(query_key.unsqueeze(0).float(),dim=-1)
        k=torch.nn.functional.normalize(keys.float(),dim=-1)
        sims=(k*q).sum(dim=-1)
        topk_vals,topk_idx=sims.topk(min(top_k,len(self.entries)))
        return [(self.entries[int(idx)],float(sim)) for sim,idx in zip(topk_vals,topk_idx)]

    def get_verified(self)->list[MemoryEntry]:
        return [e for e in self.entries if e.state==MemoryState.VERIFIED]

    def get_by_capability(self,capability_id:str)->list[MemoryEntry]:
        return [e for e in self.entries if e.capability_id==capability_id]
