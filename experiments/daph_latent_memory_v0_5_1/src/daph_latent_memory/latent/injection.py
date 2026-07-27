from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class InjectedBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    memory_slice: slice


class LatentInjector(nn.Module):
    """Insert latent memory between prompt embeddings and answer teacher-forcing tokens."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def build(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
        latent_tokens: torch.Tensor,
    ) -> InjectedBatch:
        embed = self.model.get_input_embeddings()
        prompt_emb = embed(prompt_ids)
        answer_emb = embed(answer_ids)
        latent_tokens = latent_tokens.to(device=prompt_emb.device, dtype=prompt_emb.dtype)
        answer_emb = answer_emb.to(device=prompt_emb.device, dtype=prompt_emb.dtype)

        b, k, _ = latent_tokens.shape
        full = torch.cat([prompt_emb, latent_tokens, answer_emb], dim=1)

        latent_mask = torch.ones((b, k), device=prompt_mask.device, dtype=prompt_mask.dtype)
        mask = torch.cat([prompt_mask, latent_mask, answer_mask], dim=1)

        ignore_prompt = torch.full_like(prompt_ids, -100)
        ignore_latent = torch.full((b, k), -100, device=answer_ids.device, dtype=answer_ids.dtype)
        safe_answer_labels = answer_ids.masked_fill(answer_mask == 0, -100)
        labels = torch.cat([ignore_prompt, ignore_latent, safe_answer_labels], dim=1)

        start = prompt_ids.size(1)
        return InjectedBatch(full, mask, labels, slice(start, start + k))
