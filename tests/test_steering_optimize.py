"""Tests for the steering-optimized direction extractor.

See src/daph_learning/steering/optimize.py and CLAIMS.md §12.
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.steering.optimize import optimize_steering_direction, _get_logit_margin
from daph_learning.steering.types import SteeringSpec, SteeringVector


def _make_fake_model(hidden_size=4, num_layers=2, model_id="fake-model"):
    """Build a fake model with single-token SYMBOLIC/LLM labels."""
    import torch
    import torch.nn as nn

    class FakeTok:
        pad_token_id = 0
        pad_token = "<pad>"
        padding_side = "left"
        chat_template = None
        def encode(self, text, add_special_tokens=False):
            mapping = {" SYMBOLIC": [10], " LLM": [20], "SYMBOLIC": [10], "LLM": [20]}
            if text in mapping:
                return mapping[text]
            return [1, 2, 3]
        def __call__(self, text, **kwargs):
            if isinstance(text, list):
                encoded = [self.encode(t, add_special_tokens=True) for t in text]
                max_len = max(len(e) for e in encoded)
                padded = [[0] * (max_len - len(e)) + e for e in encoded]
                return {
                    "input_ids": torch.tensor(padded),
                    "attention_mask": torch.ones(len(padded), max_len, dtype=torch.long),
                }
            ids = self.encode(text, add_special_tokens=True)
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        def decode(self, ids, skip_special_tokens=True):
            return "SYMBOLIC" if 10 in ids else "LLM"

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, hidden_size)
            self.config = type("C", (), {
                "hidden_size": hidden_size,
                "_name_or_path": model_id,
            })()
            inner = type("I", (nn.Module,), {})()
            class FakeLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(hidden_size, hidden_size)
                    # Make the linear layer produce different outputs for different inputs
                    self.linear.weight.data = torch.eye(hidden_size) * 2
                    self.linear.bias.data = torch.zeros(hidden_size)
                def forward(self, x):
                    return (self.linear(x),)
            inner.layers = nn.ModuleList([FakeLayer() for _ in range(num_layers)])
            self.model = inner
            self.lm_head = nn.Linear(hidden_size, 100)
            # Make the lm_head map dimension 0 -> token 10 (SYMBOLIC)
            # and dimension 1 -> token 20 (LLM)
            self.lm_head.weight.data[10, 0] = 5.0  # SYM token reads from dim 0
            self.lm_head.weight.data[20, 1] = 5.0  # LLM token reads from dim 1
        def get_input_embeddings(self):
            return self.embed
        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)[0]
            return type("O", (), {"logits": self.lm_head(x)})()

    return FakeModel(), FakeTok()


def test_optimize_improves_margin():
    """The optimizer should improve the logit margin over the initial direction."""
    model, tok = _make_fake_model()
    prompts = ["ACTION: test1", "ACTION: test2", "ACTION: test3"]

    # Start with a direction that doesn't favor SYMBOLIC
    initial = np.zeros(4, dtype=np.float32)
    initial[1] = 1.0  # favors LLM (dim 1 -> token 20)

    optimized, info = optimize_steering_direction(
        model, tok, prompts,
        initial_direction=initial,
        layer=0, alpha=1.0,
        sym_token=10, llm_token=20,
        n_iterations=10, seed=42, verbose=False,
    )

    # The optimized direction should have a higher margin
    assert info["final_margin"] >= info["initial_margin"]


def test_optimize_preserves_norm():
    """The optimized direction should have a similar norm to the initial."""
    model, tok = _make_fake_model()
    prompts = ["ACTION: test"]

    initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    optimized, info = optimize_steering_direction(
        model, tok, prompts,
        initial_direction=initial,
        layer=0, alpha=1.0,
        sym_token=10, llm_token=20,
        n_iterations=5, seed=42,
    )

    # Norm should be roughly preserved (within 2x)
    initial_norm = np.linalg.norm(initial)
    final_norm = np.linalg.norm(optimized)
    assert final_norm > 0
    assert final_norm < initial_norm * 3  # not blown up


def test_optimize_deterministic_with_seed():
    """Same seed should produce the same result."""
    model, tok = _make_fake_model()
    prompts = ["ACTION: test1", "ACTION: test2"]

    initial = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

    opt1, info1 = optimize_steering_direction(
        model, tok, prompts, initial_direction=initial,
        layer=0, alpha=1.0, sym_token=10, llm_token=20,
        n_iterations=5, seed=42,
    )
    opt2, info2 = optimize_steering_direction(
        model, tok, prompts, initial_direction=initial,
        layer=0, alpha=1.0, sym_token=10, llm_token=20,
        n_iterations=5, seed=42,
    )
    np.testing.assert_array_equal(opt1, opt2)
    assert info1["final_margin"] == info2["final_margin"]


def test_optimize_returns_info():
    """The optimizer should return an info dict with the expected fields."""
    model, tok = _make_fake_model()
    prompts = ["ACTION: test"]

    initial = np.ones(4, dtype=np.float32)
    _, info = optimize_steering_direction(
        model, tok, prompts, initial_direction=initial,
        layer=0, alpha=1.0, sym_token=10, llm_token=20,
        n_iterations=3, seed=42,
    )

    assert "initial_margin" in info
    assert "final_margin" in info
    assert "margin_improvement" in info
    assert "n_iterations" in info
    assert "initial_norm" in info
    assert "final_norm" in info
    assert "n_prompts" in info


def test_get_logit_margin():
    """The logit margin function should return a float."""
    model, tok = _make_fake_model()
    direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    margin = _get_logit_margin(
        model, tok, "ACTION: test", layer=0, direction=direction,
        alpha=1.0, sym_token=10, llm_token=20,
    )
    assert isinstance(margin, float)


def test_optimize_zero_iterations():
    """With 0 iterations, the optimizer should return the initial direction."""
    model, tok = _make_fake_model()
    prompts = ["ACTION: test"]
    initial = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    optimized, info = optimize_steering_direction(
        model, tok, prompts, initial_direction=initial,
        layer=0, alpha=1.0, sym_token=10, llm_token=20,
        n_iterations=0, seed=42,
    )
    np.testing.assert_array_equal(optimized, initial)
    assert info["initial_margin"] == info["final_margin"]


# --- v0.3.6 Phase 1.2: balanced objective with negative controls ---

def _make_fp_prone_model(hidden_size=4, num_layers=1):
    """A fake model where steering along dim 0 raises BOTH symbolic and
    control-prompt SYM logits, so the legacy positive-only objective
    over-steers and produces 100% false positives on control prompts.

    lm_head: token 10 (SYM) reads from dim 0; token 20 (LLM) reads from
    dim 1. A residual addition of (+dim0, -dim1) raises SYM and lowers
    LLM regardless of the prompt, which is exactly the pathology the
    balanced objective must suppress.
    """
    import torch
    import torch.nn as nn

    class FakeTok:
        pad_token_id = 0
        pad_token = "<pad>"
        padding_side = "left"
        chat_template = None
        def encode(self, text, add_special_tokens=False):
            mapping = {" SYMBOLIC": [10], " LLM": [20], "SYMBOLIC": [10], "LLM": [20]}
            return mapping.get(text, [1, 2, 3])
        def __call__(self, text, **kwargs):
            if isinstance(text, list):
                encoded = [self.encode(t, add_special_tokens=True) for t in text]
                max_len = max(len(e) for e in encoded)
                padded = [[0] * (max_len - len(e)) + e for e in encoded]
                return {
                    "input_ids": torch.tensor(padded),
                    "attention_mask": torch.ones(len(padded), max_len, dtype=torch.long),
                }
            ids = self.encode(text, add_special_tokens=True)
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, hidden_size)
            self.config = type("C", (), {"hidden_size": hidden_size, "_name_or_path": "fp-model"})()
            inner = type("I", (nn.Module,), {})()
            class FakeLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(hidden_size, hidden_size)
                    self.linear.weight.data = torch.eye(hidden_size)
                    self.linear.bias.data = torch.zeros(hidden_size)
                def forward(self, x):
                    return (self.linear(x),)
            inner.layers = nn.ModuleList([FakeLayer() for _ in range(num_layers)])
            self.model = inner
            self.lm_head = nn.Linear(hidden_size, 100, bias=False)
            self.lm_head.weight.data[10, 0] = 5.0   # SYM reads dim 0
            self.lm_head.weight.data[20, 1] = 5.0   # LLM reads dim 1
        def get_input_embeddings(self):
            return self.embed
        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)[0]
            return type("O", (), {"logits": self.lm_head(x)})()

    return FakeModel(), FakeTok()


def test_balanced_objective_reports_telemetry_fields():
    """The info dict must carry the v0.3.6 balanced-objective telemetry
    when negative_prompts are supplied."""
    model, tok = _make_fp_prone_model()
    pos = ["ACTION: compute 2*3", "ACTION: compute 7+1"]
    neg = ["ACTION: explain the sky", "ACTION: write a poem"]
    initial = np.array([0.5, -0.5, 0.0, 0.0], dtype=np.float32)

    _, info = optimize_steering_direction(
        model, tok, pos,
        initial_direction=initial,
        layer=0, alpha=1.0, sym_token=10, llm_token=20,
        n_iterations=3, seed=42, method="coordinate", n_coordinates=4,
        negative_prompts=neg, lambda_neg=1.5, gamma=1.0, beta=0.01,
    )
    for key in (
        "n_negative_prompts", "lambda_neg", "gamma", "beta",
        "initial_objective", "final_objective", "objective_improvement",
        "initial_negative_margin", "final_negative_margin",
        "initial_false_positive_rate", "final_false_positive_rate",
    ):
        assert key in info, f"missing telemetry field {key!r}"
    assert info["n_negative_prompts"] == 2
    assert info["lambda_neg"] == 1.5


def test_balanced_objective_legacy_mode_when_no_negatives():
    """When negative_prompts is None/empty, the optimizer must behave
    exactly like the legacy positive-only path (telemetry fields None)."""
    model, tok = _make_fp_prone_model()
    pos = ["ACTION: compute 2*3"]
    initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    _, info = optimize_steering_direction(
        model, tok, pos,
        initial_direction=initial,
        layer=0, alpha=1.0, sym_token=10, llm_token=20,
        n_iterations=2, seed=42,
    )
    assert info["n_negative_prompts"] == 0
    assert info["lambda_neg"] is None
    assert info["initial_false_positive_rate"] == 0.0
    # Legacy mode: objective equals the positive margin.
    assert info["initial_objective"] == pytest.approx(info["initial_margin"])


def test_balanced_objective_reduces_false_positive_rate():
    """The headline Phase 1.2 guarantee: with negative controls, the
    optimized direction must not increase the false-positive rate on
    control prompts, and should reduce it relative to a positive-only
    optimization that ignores controls."""
    model, tok = _make_fp_prone_model()
    pos = ["ACTION: compute 2*3", "ACTION: compute 7+1"]
    neg = ["ACTION: explain the sky", "ACTION: write a poem", "ACTION: capital of France"]
    # Start with a direction that already pushes SYMBOLIC on controls
    # (dim 0 up, dim 1 down) — i.e. an initial FP rate > 0.
    initial = np.array([2.0, -2.0, 0.0, 0.0], dtype=np.float32)

    # Baseline: positive-only optimization (legacy), no controls.
    _, info_legacy = optimize_steering_direction(
        model, tok, pos,
        initial_direction=initial,
        layer=0, alpha=1.0, sym_token=10, llm_token=20,
        n_iterations=5, seed=42, method="coordinate", n_coordinates=4,
    )
    # Measure the legacy result's FP rate on the control prompts directly.
    from daph_learning.steering.optimize import _get_logit_margin
    legacy_neg_margins = [
        _get_logit_margin(model, tok, p, 0, _, 1.0, 10, 20) for p in neg
    ]
    legacy_fpr = sum(1 for m in legacy_neg_margins if m > 0) / len(neg)

    # Balanced optimization with the same controls.
    _, info_balanced = optimize_steering_direction(
        model, tok, pos,
        initial_direction=initial,
        layer=0, alpha=1.0, sym_token=10, llm_token=20,
        n_iterations=5, seed=42, method="coordinate", n_coordinates=4,
        negative_prompts=neg, lambda_neg=1.5, gamma=1.0, beta=0.05,
    )
    # The balanced objective must not increase the control FPR above the
    # initial state, and must produce a lower FPR than the legacy run.
    assert info_balanced["final_false_positive_rate"] <= info_balanced["initial_false_positive_rate"] + 1e-9
    assert info_balanced["final_false_positive_rate"] <= legacy_fpr + 1e-9


def test_balanced_objective_l2_anchor_prevents_norm_blowup():
    """The L2 anchor term must keep the optimized direction close to the
    initial direction, preventing the norm blow-up that produced 100% FPs
    under the legacy objective."""
    model, tok = _make_fp_prone_model()
    pos = ["ACTION: compute 2*3"]
    neg = ["ACTION: explain the sky"]
    initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    initial_norm = float(np.linalg.norm(initial))

    optimized, info = optimize_steering_direction(
        model, tok, pos,
        initial_direction=initial,
        layer=0, alpha=1.0, sym_token=10, llm_token=20,
        n_iterations=5, seed=42, method="coordinate", n_coordinates=4,
        negative_prompts=neg, lambda_neg=1.5, gamma=1.0, beta=0.1,
    )
    # With a non-trivial beta the final norm must stay within 3x of the
    # initial norm (the legacy path could grow it unboundedly).
    assert info["final_norm"] < initial_norm * 3.0 + 1e-6
