"""Tests for v0.3.8 DEF-02 steering safety limits and auto-clamping.

Verifies that ``residual_addition_hook`` with a ``SafetyLimits`` config:
- auto-clamps the effective alpha so ``R_pert = |αv|/|h|`` does not exceed
  ``max_relative_perturbation`` (default 0.65)
- records ``safety_clamp_multiplier`` and ``safety_breaches`` in telemetry
- raises ``SteeringApplicationError`` when ``fail_closed=True`` and a limit
  is breached
- leaves steering unchanged when the perturbation is within bounds
"""
from __future__ import annotations

import pytest

from daph_learning.routing.errors import SteeringApplicationError
from daph_learning.steering.hooks import (
    SafetyLimits,
    residual_addition_hook,
)


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

        def forward(self, input_ids=None, **kwargs):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)[0]
            return type("O", (), {"logits": self.lm_head(x)})()

    return FakeModel()


def test_safety_limits_default_values():
    lim = SafetyLimits()
    assert lim.max_relative_perturbation == 0.65
    assert lim.max_cosine_shift == 0.25
    assert lim.max_kl_drift == 0.50
    assert lim.fail_closed is False


def test_safety_clamp_activates_when_r_pert_exceeds_limit():
    """With a huge alpha and a small hidden state, R_pert >> 0.65 so the
    clamp must scale effective_alpha down. The telemetry must record
    safety_clamp_multiplier < 1.0 and 'relative_perturbation' in breaches."""
    import torch
    import numpy as np

    model = _make_fake_model()
    vec = np.ones(4, dtype=np.float32)  # |v| = 2.0
    telemetry: list = []
    limits = SafetyLimits(max_relative_perturbation=0.65)

    # alpha=100 → |αv| = 200. Hidden norms from embed are small (~1-2),
    # so R_pert will be huge and must be clamped.
    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=100.0,
        token_scope="all",
        telemetry_sink=telemetry,
        safety_limits=limits,
    ):
        model(torch.tensor([[1, 2, 3]]))

    assert len(telemetry) >= 1
    rec = telemetry[0]
    assert "safety_clamp_multiplier" in rec
    assert rec["safety_clamp_multiplier"] < 1.0
    assert "relative_perturbation" in rec["safety_breaches"]
    # After clamping, the recorded relative_perturbation must be ≤ 0.65 + tol
    assert rec["relative_perturbation"] <= 0.65 + 1e-3


def test_safety_clamp_inactive_when_within_bounds():
    """With a tiny alpha, R_pert < 0.65 so no clamp: multiplier == 1.0 and
    no breaches."""
    import torch
    import numpy as np

    model = _make_fake_model()
    vec = np.ones(4, dtype=np.float32)
    telemetry: list = []
    limits = SafetyLimits()

    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=0.001,
        token_scope="all",
        telemetry_sink=telemetry,
        safety_limits=limits,
    ):
        model(torch.tensor([[1, 2, 3]]))

    rec = telemetry[0]
    assert rec["safety_clamp_multiplier"] == 1.0
    assert rec["safety_breaches"] == []


def test_safety_clamp_fail_closed_raises():
    import torch
    import numpy as np

    model = _make_fake_model()
    vec = np.ones(4, dtype=np.float32)
    limits = SafetyLimits(max_relative_perturbation=0.65, fail_closed=True)

    with pytest.raises(SteeringApplicationError, match="relative_perturbation"):
        with torch.inference_mode(), residual_addition_hook(
            model,
            layer_index=0,
            vector=vec,
            alpha=100.0,
            token_scope="all",
            safety_limits=limits,
        ):
            model(torch.tensor([[1, 2, 3]]))


def test_safety_clamp_custom_limit():
    """A stricter limit (0.3) must clamp more aggressively than the default."""
    import torch
    import numpy as np

    model = _make_fake_model()
    vec = np.ones(4, dtype=np.float32)
    telemetry_default: list = []
    telemetry_strict: list = []

    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=10.0,
        token_scope="all",
        telemetry_sink=telemetry_default,
        safety_limits=SafetyLimits(max_relative_perturbation=0.65),
    ):
        model(torch.tensor([[1, 2, 3]]))

    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=10.0,
        token_scope="all",
        telemetry_sink=telemetry_strict,
        safety_limits=SafetyLimits(max_relative_perturbation=0.3),
    ):
        model(torch.tensor([[1, 2, 3]]))

    rec_def = telemetry_default[0]
    rec_strict = telemetry_strict[0]
    # Both must be clamped (alpha=10 is large), but the stricter limit must
    # produce a smaller clamp multiplier and a lower post-clamp R_pert.
    assert rec_def["safety_clamp_multiplier"] < 1.0
    assert rec_strict["safety_clamp_multiplier"] < 1.0
    assert rec_strict["safety_clamp_multiplier"] < rec_def["safety_clamp_multiplier"]
    assert rec_strict["relative_perturbation"] <= 0.3 + 1e-3


def test_safety_clamp_last_scope():
    """The 'last' scope must also apply the safety clamp at the target
    position."""
    import torch
    import numpy as np

    model = _make_fake_model()
    vec = np.ones(4, dtype=np.float32)
    telemetry: list = []
    limits = SafetyLimits(max_relative_perturbation=0.65)

    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=100.0,
        token_scope="last",
        target_token_index=-1,
        telemetry_sink=telemetry,
        safety_limits=limits,
    ):
        model(torch.tensor([[1, 2, 3]]))

    assert len(telemetry) == 1
    rec = telemetry[0]
    assert rec["safety_clamp_multiplier"] < 1.0
    assert "relative_perturbation" in rec["safety_breaches"]
    assert rec["relative_perturbation"] <= 0.65 + 1e-3


def test_no_safety_limits_preserves_v036_behavior():
    """When safety_limits is None (default), no clamp fields appear in
    telemetry and the hook behaves exactly as before v0.3.8."""
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

    rec = telemetry[0]
    assert "safety_clamp_multiplier" not in rec
    assert "safety_breaches" not in rec
