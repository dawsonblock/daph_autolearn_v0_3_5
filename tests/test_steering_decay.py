"""Tests for v0.3.6 Phase 3.2 reasoning-steering decay.

See daph_learning.steering.hooks:
- decay_multiplier: pure per-step alpha multiplier
- residual_addition_hook: decay_schedule / decay_steps / decay_min_multiplier
- multi_layer_residual_addition_hook: per-config decay forwarding

The decay function is pure and tested exhaustively. The hook integration
is tested with a tiny fake model that records the effective alpha at
each forward pass.
"""
from __future__ import annotations

import math

import pytest

from daph_learning.steering.hooks import (
    decay_multiplier,
    multi_layer_residual_addition_hook,
    residual_addition_hook,
    resolve_transformer_layers,
)


# --- decay_multiplier: pure function tests ---

def test_decay_none_always_one():
    for step in range(20):
        assert decay_multiplier(step, schedule="none") == 1.0


def test_decay_none_ignores_decay_steps():
    # decay_steps=0 is legal for schedule="none"
    assert decay_multiplier(5, schedule="none", decay_steps=0) == 1.0


def test_decay_linear_ramps_to_min():
    assert decay_multiplier(0, schedule="linear", decay_steps=10) == 1.0
    assert decay_multiplier(5, schedule="linear", decay_steps=10) == pytest.approx(0.5)
    assert decay_multiplier(10, schedule="linear", decay_steps=10) == 0.0
    # clamps at min beyond decay_steps
    assert decay_multiplier(15, schedule="linear", decay_steps=10) == 0.0


def test_decay_linear_with_min_multiplier():
    assert decay_multiplier(0, schedule="linear", decay_steps=10, min_multiplier=0.2) == 1.0
    assert decay_multiplier(5, schedule="linear", decay_steps=10, min_multiplier=0.2) == pytest.approx(0.6)
    assert decay_multiplier(10, schedule="linear", decay_steps=10, min_multiplier=0.2) == 0.2
    assert decay_multiplier(20, schedule="linear", decay_steps=10, min_multiplier=0.2) == 0.2


def test_decay_exponential_approaches_min():
    assert decay_multiplier(0, schedule="exponential", decay_steps=10) == 1.0
    # exp(-1) ≈ 0.368
    assert decay_multiplier(10, schedule="exponential", decay_steps=10) == pytest.approx(math.exp(-1))
    # Never reaches min exactly, but approaches it
    val = decay_multiplier(1000, schedule="exponential", decay_steps=10)
    assert 0.0 <= val < 0.01


def test_decay_cosine_smooth_endpoints():
    assert decay_multiplier(0, schedule="cosine", decay_steps=10) == 1.0
    assert decay_multiplier(10, schedule="cosine", decay_steps=10) == 0.0
    # midpoint should be 0.5
    assert decay_multiplier(5, schedule="cosine", decay_steps=10) == pytest.approx(0.5)
    # clamps beyond decay_steps
    assert decay_multiplier(15, schedule="cosine", decay_steps=10) == 0.0


def test_decay_cosine_with_min_multiplier():
    assert decay_multiplier(0, schedule="cosine", decay_steps=10, min_multiplier=0.3) == 1.0
    assert decay_multiplier(10, schedule="cosine", decay_steps=10, min_multiplier=0.3) == 0.3
    mid = decay_multiplier(5, schedule="cosine", decay_steps=10, min_multiplier=0.3)
    assert mid == pytest.approx(0.65)


def test_decay_unknown_schedule_raises():
    with pytest.raises(ValueError, match="unknown decay schedule"):
        decay_multiplier(0, schedule="bogus", decay_steps=10)


def test_decay_zero_decay_steps_raises_for_non_none():
    for sched in ("linear", "exponential", "cosine"):
        with pytest.raises(ValueError, match="decay_steps must be >= 1"):
            decay_multiplier(0, schedule=sched, decay_steps=0)


def test_decay_negative_step_returns_one():
    """Negative steps (shouldn't happen in practice) clamp to step 0."""
    assert decay_multiplier(-1, schedule="linear", decay_steps=10) == 1.0


def test_decay_min_multiplier_out_of_range_raises():
    with pytest.raises(ValueError, match="min_multiplier"):
        decay_multiplier(1, schedule="linear", decay_steps=10, min_multiplier=-0.1)
    with pytest.raises(ValueError, match="min_multiplier"):
        decay_multiplier(1, schedule="linear", decay_steps=10, min_multiplier=1.5)


def test_decay_monotonic_non_increasing():
    """All non-none schedules must be monotonically non-increasing in step."""
    for sched in ("linear", "exponential", "cosine"):
        prev = 2.0
        for step in range(0, 30):
            val = decay_multiplier(step, schedule=sched, decay_steps=10)
            assert val <= prev + 1e-12, f"{sched} not monotonic at step {step}"
            prev = val


def test_decay_within_bounds():
    """All multipliers must lie in [min_multiplier, 1.0]."""
    for sched in ("linear", "exponential", "cosine"):
        for step in range(0, 30):
            val = decay_multiplier(
                step, schedule=sched, decay_steps=10, min_multiplier=0.2,
            )
            assert 0.2 - 1e-9 <= val <= 1.0 + 1e-9


# --- Hook integration tests with a fake model ---

def _make_fake_model(hidden=4, num_layers=2):
    import torch
    import torch.nn as nn

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden, hidden)
            self.linear.weight.data = torch.eye(hidden)
            self.linear.bias.data = torch.zeros(hidden)
        def forward(self, x):
            return (self.linear(x),)

    class FakeInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer() for _ in range(num_layers)])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, hidden)
            inner = FakeInner()
            self.model = inner
            self.lm_head = nn.Linear(hidden, 100, bias=False)
            class _Cfg:
                hidden_size = hidden
                num_hidden_layers = num_layers
                _name_or_path = "fake"
            self.config = _Cfg()
        def get_input_embeddings(self):
            return self.embed
        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)[0]
            return type("O", (), {"logits": self.lm_head(x)})()

    return FakeModel()


def test_residual_hook_decay_attenuates_over_decode_steps():
    """With token_scope='all' and a linear decay, the effective alpha
    must decrease across cached decode steps. We verify by inspecting
    the telemetry's av_norm_mean, which is proportional to the
    effective alpha."""
    import torch
    import numpy as np

    model = _make_fake_model()
    vec = np.ones(4, dtype=np.float32)
    telemetry: list = []

    # Simulate: prefill (seq_len=3) then 3 single-token decode steps.
    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=10.0,
        token_scope="all",
        telemetry_sink=telemetry,
        decay_schedule="linear",
        decay_steps=3,
        decay_min_multiplier=0.0,
    ):
        model(torch.tensor([[1, 2, 3]]))   # prefill, step 0
        model(torch.tensor([[4]]))         # decode step 1
        model(torch.tensor([[5]]))         # decode step 2
        model(torch.tensor([[6]]))         # decode step 3 (clamped to min)

    assert len(telemetry) == 4
    # decay_step field must be present and increasing
    steps = [t["decay_step"] for t in telemetry]
    assert steps == [0, 1, 2, 3]
    multipliers = [t["decay_multiplier"] for t in telemetry]
    assert multipliers[0] == 1.0
    assert multipliers[1] == pytest.approx(2 / 3)
    assert multipliers[2] == pytest.approx(1 / 3)
    assert multipliers[3] == 0.0
    # av_norm_mean (proportional to |alpha * v|) must decrease
    norms = [t["av_norm_mean"] for t in telemetry]
    assert norms[0] > norms[1] > norms[2] > norms[3]


def test_residual_hook_decay_none_keeps_constant_alpha():
    """With decay_schedule='none' (default), the effective alpha must
    stay constant across decode steps. This is the v0.3.5 behavior."""
    import torch
    import numpy as np

    model = _make_fake_model()
    vec = np.ones(4, dtype=np.float32)
    telemetry: list = []

    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=10.0,
        token_scope="all",
        telemetry_sink=telemetry,
    ):
        model(torch.tensor([[1, 2, 3]]))
        model(torch.tensor([[4]]))
        model(torch.tensor([[5]]))

    assert len(telemetry) == 3
    # No decay_* fields when decay is inactive
    assert "decay_step" not in telemetry[0]
    assert "decay_multiplier" not in telemetry[0]
    # av_norm_mean constant
    norms = [t["av_norm_mean"] for t in telemetry]
    assert norms[0] == pytest.approx(norms[1])
    assert norms[1] == pytest.approx(norms[2])


def test_residual_hook_decay_ignored_for_last_scope():
    """token_scope='last' is a one-shot intervention; decay parameters
    must be ignored (the hook applies once regardless)."""
    import torch
    import numpy as np

    model = _make_fake_model()
    vec = np.ones(4, dtype=np.float32)
    telemetry: list = []

    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=10.0,
        token_scope="last",
        target_token_index=-1,
        telemetry_sink=telemetry,
        decay_schedule="linear",
        decay_steps=3,
    ):
        model(torch.tensor([[1, 2, 3]]))   # prefill: one-shot applies
        model(torch.tensor([[4]]))         # decode: one-shot guard blocks

    # Only one telemetry entry (the prefill application)
    assert len(telemetry) == 1
    assert "decay_step" not in telemetry[0]


def test_multi_layer_hook_forwards_per_config_decay():
    """multi_layer_residual_addition_hook must forward per-config decay
    parameters. A reasoning vector with decay + a tool vector without
    decay must produce different telemetry signatures."""
    import torch
    import numpy as np

    model = _make_fake_model(num_layers=2)
    vec = np.ones(4, dtype=np.float32)
    telemetry: list = []

    configs = [
        {
            "layer_index": 0,
            "vector": vec,
            "alpha": 10.0,
            "token_scope": "all",
            "decay_schedule": "linear",
            "decay_steps": 2,
        },
        {
            "layer_index": 1,
            "vector": vec,
            "alpha": 5.0,
            "token_scope": "all",
            # no decay: defaults to "none"
        },
    ]
    with torch.inference_mode(), multi_layer_residual_addition_hook(
        model, configs, telemetry_sink=telemetry,
    ):
        model(torch.tensor([[1, 2, 3]]))   # prefill
        model(torch.tensor([[4]]))         # decode 1
        model(torch.tensor([[5]]))         # decode 2

    # 3 forward passes × 2 layers = 6 telemetry entries
    assert len(telemetry) == 6
    layer0_entries = [t for t in telemetry if t["layer_index"] == 0]
    layer1_entries = [t for t in telemetry if t["layer_index"] == 1]
    assert len(layer0_entries) == 3
    assert len(layer1_entries) == 3
    # Layer 0 has decay telemetry; layer 1 does not
    assert "decay_multiplier" in layer0_entries[0]
    assert "decay_multiplier" not in layer1_entries[0]
    # Layer 0 multipliers decrease
    assert layer0_entries[0]["decay_multiplier"] == 1.0
    assert layer0_entries[2]["decay_multiplier"] == 0.0
