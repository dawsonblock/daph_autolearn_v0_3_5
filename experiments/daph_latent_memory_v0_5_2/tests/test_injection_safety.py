"""Tests for v0.3.8 DEF-02 LatentInjector relative-norm safety clamp.

Verifies that:
- The clamp is disabled by default (relative_norm_limit == 0) and build()
  preserves the v0.5.2 baseline behavior exactly.
- When enabled, build() scales down latents whose mean L2 norm exceeds
  limit * mean(||prompt_emb||_2), recording the per-row multipliers.
- The clamp preserves relative latent geometry within a row (only scales).
- set_relative_norm_limit rejects negative values.
- The clamp is per-example: a batch with mixed norms clamps only the rows
  that exceed the limit.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from daph_latent_memory.latent.injection import LatentInjector, InjectedBatch


class _FakeEmbed(nn.Module):
    """Stand-in for a model's get_input_embeddings() — returns a fixed
    embedding table so prompt/answer embedding norms are deterministic."""

    def __init__(self, vocab=20, hidden=8):
        super().__init__()
        self.embedding_dim = hidden
        self.weight = nn.Parameter(torch.randn(vocab, hidden))

    def forward(self, ids):
        return self.weight[ids]


class _FakeModel(nn.Module):
    """Minimal model exposing get_input_embeddings() for the injector."""

    def __init__(self, vocab=20, hidden=8):
        super().__init__()
        self._embed = _FakeEmbed(vocab, hidden)

    def get_input_embeddings(self):
        return self._embed


def _make_batch(hidden=8, batch=2, prompt_len=4, latent_tokens=3, answer_len=2):
    """Build deterministic prompt/answer/latent tensors."""
    prompt_ids = torch.randint(1, 20, (batch, prompt_len))
    prompt_mask = torch.ones_like(prompt_ids)
    answer_ids = torch.randint(1, 20, (batch, answer_len))
    answer_mask = torch.ones_like(answer_ids)
    latent = torch.randn(batch, latent_tokens, hidden)
    return prompt_ids, prompt_mask, answer_ids, answer_mask, latent


def test_injector_default_has_no_clamp():
    model = _FakeModel()
    inj = LatentInjector(model)
    assert inj.relative_norm_limit == 0.0
    assert inj.last_clamp_multipliers is None


def test_injector_set_relative_norm_limit_enables():
    model = _FakeModel()
    inj = LatentInjector(model)
    inj.set_relative_norm_limit(0.65)
    assert inj.relative_norm_limit == 0.65


def test_injector_set_relative_norm_limit_rejects_negative():
    model = _FakeModel()
    inj = LatentInjector(model)
    with pytest.raises(ValueError, match=">= 0"):
        inj.set_relative_norm_limit(-0.1)


def test_injector_set_relative_norm_limit_zero_disables():
    model = _FakeModel()
    inj = LatentInjector(model)
    inj.set_relative_norm_limit(0.65)
    inj.set_relative_norm_limit(0.0)
    assert inj.relative_norm_limit == 0.0


def test_injector_build_without_clamp_preserves_latent_norms():
    """With relative_norm_limit=0, the latent tokens in inputs_embeds must
    have the same L2 norms as the input latent tensor (no scaling)."""
    model = _FakeModel(hidden=8)
    inj = LatentInjector(model)
    p_ids, p_mask, a_ids, a_mask, latent = _make_batch(hidden=8)
    out = inj.build(p_ids, p_mask, a_ids, a_mask, latent)
    assert isinstance(out, InjectedBatch)
    # The latent slice is [prompt_len : prompt_len + latent_tokens]
    extracted = out.inputs_embeds[:, p_ids.size(1):p_ids.size(1) + latent.size(1), :]
    input_norms = latent.float().norm(dim=-1)
    extracted_norms = extracted.float().norm(dim=-1)
    assert torch.allclose(input_norms, extracted_norms, atol=1e-5)
    assert inj.last_clamp_multipliers is None


def test_injector_build_with_clamp_reduces_oversized_latent():
    """With a tiny limit and a huge latent, the injected latent norm must
    be <= limit * mean(||prompt_emb||) per row."""
    model = _FakeModel(hidden=8)
    inj = LatentInjector(model)
    inj.set_relative_norm_limit(0.1)  # very strict
    p_ids, p_mask, a_ids, a_mask, latent = _make_batch(hidden=8)
    # Make the latent huge so it will definitely be clamped.
    latent = latent * 100.0
    out = inj.build(p_ids, p_mask, a_ids, a_mask, latent)
    extracted = out.inputs_embeds[:, p_ids.size(1):p_ids.size(1) + latent.size(1), :]
    prompt_emb = model.get_input_embeddings()(p_ids)
    prompt_norms = prompt_emb.float().norm(dim=-1).mean(dim=-1)
    allowed = 0.1 * prompt_norms
    extracted_norms = extracted.float().norm(dim=-1).mean(dim=-1)
    # Each row's mean latent norm must be <= allowed + small tolerance.
    assert torch.all(extracted_norms <= allowed + 1e-3), (
        f"extracted norms {extracted_norms} exceed allowed {allowed}"
    )
    # Multipliers must be recorded and < 1 (clamping happened).
    assert inj.last_clamp_multipliers is not None
    assert torch.all(inj.last_clamp_multipliers < 1.0)


def test_injector_build_with_clamp_preserves_small_latent():
    """A latent that is already within the limit must NOT be scaled
    (multiplier == 1.0)."""
    model = _FakeModel(hidden=8)
    inj = LatentInjector(model)
    inj.set_relative_norm_limit(0.65)
    p_ids, p_mask, a_ids, a_mask, latent = _make_batch(hidden=8)
    # Make the latent tiny so it is well within the limit.
    latent = latent * 0.001
    out = inj.build(p_ids, p_mask, a_ids, a_mask, latent)
    extracted = out.inputs_embeds[:, p_ids.size(1):p_ids.size(1) + latent.size(1), :]
    input_norms = latent.float().norm(dim=-1)
    extracted_norms = extracted.float().norm(dim=-1)
    assert torch.allclose(input_norms, extracted_norms, atol=1e-5)
    assert inj.last_clamp_multipliers is not None
    assert torch.allclose(inj.last_clamp_multipliers, torch.ones(2), atol=1e-5)


def test_injector_build_with_clamp_is_per_row():
    """A batch where row 0 is oversized and row 1 is within limits must
    clamp only row 0."""
    model = _FakeModel(hidden=8)
    inj = LatentInjector(model)
    inj.set_relative_norm_limit(0.5)
    p_ids, p_mask, a_ids, a_mask, latent = _make_batch(hidden=8, batch=2)
    # Row 0 huge, row 1 tiny.
    latent[0] = latent[0] * 100.0
    latent[1] = latent[1] * 0.001
    out = inj.build(p_ids, p_mask, a_ids, a_mask, latent)
    assert inj.last_clamp_multipliers is not None
    mults = inj.last_clamp_multipliers
    assert mults[0] < 1.0, f"row 0 should be clamped, got {mults[0]}"
    assert abs(mults[1].item() - 1.0) < 1e-5, f"row 1 should not be clamped, got {mults[1]}"


def test_injector_build_with_clamp_preserves_relative_geometry():
    """Clamping scales the whole latent row by one scalar, so the relative
    direction (cosine) between latent tokens within a row is preserved."""
    model = _FakeModel(hidden=8)
    inj = LatentInjector(model)
    inj.set_relative_norm_limit(0.1)
    p_ids, p_mask, a_ids, a_mask, latent = _make_batch(hidden=8, latent_tokens=3)
    latent = latent * 100.0
    # Cosine between token 0 and token 1 in row 0, before clamping.
    cos_before = torch.nn.functional.cosine_similarity(
        latent[0, 0:1], latent[0, 1:2], dim=-1
    ).item()
    out = inj.build(p_ids, p_mask, a_ids, a_mask, latent)
    extracted = out.inputs_embeds[:, p_ids.size(1):p_ids.size(1) + latent.size(1), :]
    cos_after = torch.nn.functional.cosine_similarity(
        extracted[0, 0:1], extracted[0, 1:2], dim=-1
    ).item()
    assert abs(cos_before - cos_after) < 1e-5, (
        f"relative geometry changed: {cos_before} -> {cos_after}"
    )


def test_injector_build_labels_and_mask_unchanged_by_clamp():
    """The clamp only touches the latent embedding slice; labels and
    attention mask must be identical to the no-clamp path."""
    model = _FakeModel(hidden=8)
    p_ids, p_mask, a_ids, a_mask, latent = _make_batch(hidden=8)

    inj_no_clamp = LatentInjector(model)
    out_no = inj_no_clamp.build(p_ids, p_mask, a_ids, a_mask, latent)

    inj_clamp = LatentInjector(model)
    inj_clamp.set_relative_norm_limit(0.65)
    out_clamp = inj_clamp.build(p_ids, p_mask, a_ids, a_mask, latent)

    assert torch.equal(out_no.labels, out_clamp.labels)
    assert torch.equal(out_no.attention_mask, out_clamp.attention_mask)
    assert out_no.memory_slice == out_clamp.memory_slice
