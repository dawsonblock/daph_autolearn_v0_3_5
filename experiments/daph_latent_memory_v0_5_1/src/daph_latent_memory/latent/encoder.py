from __future__ import annotations
import torch
from torch import nn

class StateTokenEncoder(nn.Module):
    """Encodes tokenized structured state into K latent vectors."""
    def __init__(self,vocab_size:int,input_embed_dim:int,hidden_dim:int,latent_tokens:int,output_dim:int,layers:int=2,dropout:float=0.0)->None:
        super().__init__();self.latent_tokens=latent_tokens;self.embedding=nn.Embedding(vocab_size,input_embed_dim);blocks=[];dim=input_embed_dim
        for _ in range(max(1,layers)):blocks += [nn.Linear(dim,hidden_dim),nn.GELU(),nn.Dropout(dropout)];dim=hidden_dim
        self.mlp=nn.Sequential(*blocks);self.attn_queries=nn.Parameter(torch.randn(latent_tokens,hidden_dim)*.02);self.project=nn.Linear(hidden_dim,output_dim);self.norm=nn.LayerNorm(output_dim)
    def forward(self,input_ids:torch.Tensor,attention_mask:torch.Tensor)->torch.Tensor:
        x=self.mlp(self.embedding(input_ids));q=self.attn_queries.unsqueeze(0).expand(x.size(0),-1,-1);scores=torch.einsum("bkh,bth->bkt",q,x)/(x.size(-1)**.5);mask=attention_mask[:,None,:].bool();scores=scores.masked_fill(~mask,torch.finfo(scores.dtype).min);weights=scores.softmax(dim=-1);pooled=torch.einsum("bkt,bth->bkh",weights,x);return self.norm(self.project(pooled))
