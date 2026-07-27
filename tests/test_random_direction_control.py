"""Tests for the matched-norm random-direction control experiment.

See scripts/random_direction_control.py and CLAIMS.md §11.
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.steering.types import SteeringSpec, SteeringVector
from scripts.random_direction_control import (
    _make_random_direction,
    _make_random_vector,
    run_control_experiment,
)


def test_make_random_direction_same_norm():
    """The random direction must have the same L2 norm as the reference."""
    rng = np.random.RandomState(42)
    reference = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)  # norm = 5
    random_dir = _make_random_direction(reference, rng)
    ref_norm = float(np.linalg.norm(reference))
    rand_norm = float(np.linalg.norm(random_dir))
    assert abs(rand_norm - ref_norm) < 1e-5


def test_make_random_direction_different_direction():
    """The random direction should not be parallel to the reference
    (overwhelmingly unlikely for random Gaussian)."""
    rng = np.random.RandomState(42)
    reference = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    random_dir = _make_random_direction(reference, rng)
    # Cosine similarity should not be 1.0 (not parallel)
    cos_sim = float(np.dot(reference, random_dir) / (
        np.linalg.norm(reference) * np.linalg.norm(random_dir)
    ))
    assert abs(cos_sim) < 0.99


def test_make_random_direction_deterministic_with_seed():
    """Same seed should produce the same random direction."""
    reference = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    d1 = _make_random_direction(reference, rng1)
    d2 = _make_random_direction(reference, rng2)
    np.testing.assert_array_equal(d1, d2)


def test_make_random_vector_preserves_spec():
    """The random control vector should preserve the real vector's spec
    (layer, alpha, anchor) but change the vector_id."""
    rng = np.random.RandomState(42)
    spec = SteeringSpec(
        vector_id="real_v1",
        family="tool_policy",
        behavior="invoke_symbolic_tool",
        layer=24,
        alpha=1.5,
        anchor="ACTION",
        model_id="fake",
        hidden_size=4,
    )
    real = SteeringVector(spec=spec, values=np.ones(4, dtype=np.float32))
    random_vec = _make_random_vector(real, rng)
    assert random_vec.spec.layer == 24
    assert random_vec.spec.alpha == 1.5
    assert random_vec.spec.anchor == "ACTION"
    assert random_vec.spec.model_id == "fake"
    assert random_vec.spec.vector_id == "real_v1_random_control"
    # Values should have the same norm but different direction
    assert abs(np.linalg.norm(random_vec.values) - np.linalg.norm(real.values)) < 1e-5


def test_run_control_experiment_with_fake_model():
    """End-to-end: run the control experiment with a fake model and verify
    the output structure."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    # Fake model with single-token SYMBOLIC/LLM labels
    class FakeTok:
        pad_token_id = 0
        pad_token = "<pad>"
        padding_side = "left"
        def encode(self, text, add_special_tokens=False):
            mapping = {" SYMBOLIC": [10], " LLM": [20], "SYMBOLIC": [10], "LLM": [20]}
            if text in mapping:
                return mapping[text]
            if text == "ACTION:":
                return [1, 2, 3]
            if text == "ACTION: SYMBOLIC":
                return [1, 2, 3, 10]
            if text == "ACTION: LLM":
                return [1, 2, 3, 20]
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

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 4)
            self.config = type("C", (), {"hidden_size": 4, "_name_or_path": "fake-model"})()
            inner = type("I", (nn.Module,), {})()
            class FakeLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(4, 4)
                def forward(self, x):
                    return (self.linear(x),)
            inner.layers = nn.ModuleList([FakeLayer()])
            self.model = inner
            self.lm_head = nn.Linear(4, 100)
        def get_input_embeddings(self):
            return self.embed
        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)[0]
            return type("O", (), {"logits": self.lm_head(x)})()

    model = FakeModel()
    tok = FakeTok()

    # Create a real vector
    spec = SteeringSpec(
        vector_id="real_v1",
        family="tool_policy",
        behavior="invoke_symbolic_tool",
        layer=0,
        alpha=1.0,
        anchor="ACTION",
        model_id="fake-model",
        hidden_size=4,
    )
    real_vec = SteeringVector(spec=spec, values=np.array([1, 0, 0, 0], dtype=np.float32))

    # Create simple tasks
    tasks = []
    for i in range(6):
        tasks.append({
            "task_id": f"t{i}",
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": i, "b": i + 1, "op": "+"},
            "specification": f"Compute {i} + {i+1}. Return only the integer.",
            "expected": 2 * i + 1,
            "route_label": "symbolic",
            "capability_oracle": "symbolic",
            "accuracy_oracle": "llm",
            "utility_oracle": "symbolic",
        })

    result = run_control_experiment(
        real_vec,
        tasks,
        model,
        tok,
        n_random=3,
        seed=42,
        batch_size=6,
        prompt_format="raw",
        label_field="route_label",
        label_oracle_kind="policy_heuristic",
    )

    # Verify output structure
    assert "real_metrics" in result
    assert "random_metrics" in result
    assert "random_aggregate" in result
    assert "comparison" in result
    assert len(result["random_metrics"]) == 3
    assert result["n_random"] == 3

    # The comparison should have an interpretation
    comp = result["comparison"]
    assert comp["interpretation"] in {
        "real_vector_outperforms",
        "real_vector_within_random_range",
        "real_vector_underperforms",
    }
    assert "f1_lift" in comp
    assert "f1_z_score" in comp

    # The random aggregate should have mean and std for each metric
    for key in ["f1", "precision", "recall", "route_accuracy", "decision_coverage"]:
        assert key in result["random_aggregate"]
        assert "mean" in result["random_aggregate"][key]
        assert "std" in result["random_aggregate"][key]


def test_run_control_experiment_deterministic_with_seed():
    """Same seed should produce the same random directions and thus the
    same results."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class FakeTok:
        pad_token_id = 0
        pad_token = "<pad>"
        padding_side = "left"
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

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 4)
            self.config = type("C", (), {"hidden_size": 4, "_name_or_path": "fake-model"})()
            inner = type("I", (nn.Module,), {})()
            class FakeLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(4, 4)
                def forward(self, x):
                    return (self.linear(x),)
            inner.layers = nn.ModuleList([FakeLayer()])
            self.model = inner
            self.lm_head = nn.Linear(4, 100)
        def get_input_embeddings(self):
            return self.embed
        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)[0]
            return type("O", (), {"logits": self.lm_head(x)})()

    model = FakeModel()
    tok = FakeTok()

    spec = SteeringSpec(
        vector_id="v", family="tool_policy", behavior="b",
        layer=0, alpha=1.0, anchor="ACTION",
        model_id="fake-model", hidden_size=4,
    )
    real_vec = SteeringVector(spec=spec, values=np.array([1, 0, 0, 0], dtype=np.float32))
    tasks = [
        {"task_id": f"t{i}", "specification": f"Compute {i}.", "route_label": "symbolic",
         "capability_ids": ["integer_arithmetic"], "inputs": {}, "expected": i}
        for i in range(4)
    ]

    r1 = run_control_experiment(real_vec, tasks, model, tok, n_random=2, seed=42, prompt_format="raw", label_field="route_label")
    r2 = run_control_experiment(real_vec, tasks, model, tok, n_random=2, seed=42, prompt_format="raw", label_field="route_label")
    # The random metrics should be identical (same seed)
    for m1, m2 in zip(r1["random_metrics"], r2["random_metrics"]):
        assert m1["f1"] == m2["f1"]
