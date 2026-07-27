from __future__ import annotations

import pytest

from daph_learning.steering.hooks import (
    multi_layer_residual_addition_hook,
    residual_addition_hook,
)


def _make_fake_model(hidden_size=4, num_layers=2):
    """Build a fake model with a layer ModuleList that calls hooks."""
    import torch
    import torch.nn as nn

    class FakeLayer(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.linear = nn.Linear(hidden, hidden)

        def forward(self, x):
            return (self.linear(x),)

    class FakeModel(nn.Module):
        def __init__(self, hidden, num_layers):
            super().__init__()
            self.config = type("C", (), {"hidden_size": hidden, "_name_or_path": "fake"})()
            self.embed = nn.Embedding(100, hidden)
            layers = nn.ModuleList([FakeLayer(hidden) for _ in range(num_layers)])
            # Expose as .model.layers (the Llama-style layout)
            inner = type("Inner", (nn.Module,), {})()
            inner.layers = layers
            self.model = inner

        def get_input_embeddings(self):
            return self.embed

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)[0]
            return type("O", (), {"logits": x})()

    return FakeModel(hidden_size, num_layers)


def test_telemetry_sink_collects_stats_for_all_scope():
    """residual_addition_hook with token_scope='all' and a telemetry_sink
    must append per-forward-pass stats."""
    torch = pytest.importorskip("torch")
    model = _make_fake_model(hidden_size=4, num_layers=2)
    input_ids = torch.tensor([[1, 2, 3]])
    vec = torch.ones(4, dtype=torch.float32).numpy()
    sink: list = []
    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=2.0,
        token_scope="all",
        telemetry_sink=sink,
    ):
        model(input_ids)
    assert len(sink) == 1
    stats = sink[0]
    assert stats["layer_index"] == 0
    assert stats["n_steered_positions"] == 3  # batch=1, seq=3
    assert stats["h_norm_mean"] > 0
    assert stats["av_norm_mean"] > 0
    # alpha=2.0, vec=ones(4) => |av| = 2.0 * sqrt(4) = 4.0
    assert abs(stats["av_norm_mean"] - 4.0) < 1e-5
    assert stats["relative_perturbation"] > 0
    assert stats["cosine_shift_mean"] >= 0  # 1 - cos in [0, 2]


def test_telemetry_sink_collects_stats_for_last_scope():
    """token_scope='last' should record telemetry only for the steered
    position, not all positions."""
    torch = pytest.importorskip("torch")
    model = _make_fake_model(hidden_size=4, num_layers=2)
    input_ids = torch.tensor([[1, 2, 3]])
    vec = torch.ones(4, dtype=torch.float32).numpy()
    sink: list = []
    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=1.0,
        token_scope="last",
        target_token_index=-1,
        telemetry_sink=sink,
    ):
        model(input_ids)
    assert len(sink) == 1
    stats = sink[0]
    assert stats["n_steered_positions"] == 1  # only the last token


def test_telemetry_sink_none_does_not_collect():
    """When telemetry_sink is None, no stats are collected (backward compat)."""
    torch = pytest.importorskip("torch")
    model = _make_fake_model(hidden_size=4, num_layers=2)
    input_ids = torch.tensor([[1, 2, 3]])
    vec = torch.ones(4, dtype=torch.float32).numpy()
    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=1.0,
        token_scope="all",
        # no telemetry_sink
    ):
        model(input_ids)
    # No error, no telemetry — backward compatible.


def test_telemetry_multi_layer_collects_per_layer():
    """multi_layer_residual_addition_hook must collect telemetry from each
    layer separately."""
    torch = pytest.importorskip("torch")
    model = _make_fake_model(hidden_size=4, num_layers=2)
    input_ids = torch.tensor([[1, 2, 3]])
    vec = torch.ones(4, dtype=torch.float32).numpy()
    sink: list = []
    configs = [
        {"layer_index": 0, "vector": vec, "alpha": 1.0, "token_scope": "all"},
        {"layer_index": 1, "vector": vec, "alpha": 2.0, "token_scope": "all"},
    ]
    with torch.inference_mode(), multi_layer_residual_addition_hook(
        model, configs, telemetry_sink=sink
    ):
        model(input_ids)
    assert len(sink) == 2
    layer_ids = {s["layer_index"] for s in sink}
    assert layer_ids == {0, 1}
    # Layer 1 has alpha=2.0, so |av| should be 2x layer 0's |av|
    by_layer = {s["layer_index"]: s for s in sink}
    assert abs(by_layer[1]["av_norm_mean"] - 2 * by_layer[0]["av_norm_mean"]) < 1e-5


def test_telemetry_relative_perturbation_increases_with_alpha():
    """Larger alpha should produce larger relative_perturbation."""
    torch = pytest.importorskip("torch")
    model = _make_fake_model(hidden_size=4, num_layers=2)
    input_ids = torch.tensor([[1, 2, 3]])
    vec = torch.ones(4, dtype=torch.float32).numpy()

    sink_small: list = []
    with torch.inference_mode(), residual_addition_hook(
        model, layer_index=0, vector=vec, alpha=0.1, token_scope="all", telemetry_sink=sink_small
    ):
        model(input_ids)

    sink_large: list = []
    with torch.inference_mode(), residual_addition_hook(
        model, layer_index=0, vector=vec, alpha=10.0, token_scope="all", telemetry_sink=sink_large
    ):
        model(input_ids)

    assert sink_large[0]["relative_perturbation"] > sink_small[0]["relative_perturbation"]


def test_telemetry_cosine_shift_zero_for_zero_alpha():
    """alpha=0 means no steering, so cosine_shift should be ~0."""
    torch = pytest.importorskip("torch")
    model = _make_fake_model(hidden_size=4, num_layers=2)
    input_ids = torch.tensor([[1, 2, 3]])
    vec = torch.ones(4, dtype=torch.float32).numpy()
    sink: list = []
    with torch.inference_mode(), residual_addition_hook(
        model, layer_index=0, vector=vec, alpha=0.0, token_scope="all", telemetry_sink=sink
    ):
        model(input_ids)
    assert len(sink) == 1
    assert abs(sink[0]["cosine_shift_mean"]) < 1e-6
    assert abs(sink[0]["av_norm_mean"]) < 1e-6


def test_telemetry_records_batched_target_indices():
    """token_scope='last' with per-batch-item target_token_index must
    record telemetry for each steered position across the batch."""
    torch = pytest.importorskip("torch")
    model = _make_fake_model(hidden_size=4, num_layers=2)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    vec = torch.ones(4, dtype=torch.float32).numpy()
    sink: list = []
    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=0,
        vector=vec,
        alpha=1.0,
        token_scope="last",
        target_token_index=[2, 1],  # different position per batch item
        telemetry_sink=sink,
    ):
        model(input_ids)
    assert len(sink) == 1
    assert sink[0]["n_steered_positions"] == 2  # one per batch item


def test_telemetry_via_score_route_batch():
    """score_route_batch_from_logits must forward telemetry_sink to the hooks."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from daph_learning.routing.logit_router import score_route_batch_from_logits
    from daph_learning.steering.types import SteeringSpec, SteeringVector
    import numpy as np

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
            self.config = type("C", (), {"hidden_size": 4})()
            inner = type("I", (nn.Module,), {})()
            # The hook wraps each layer; output is (hidden,) tuple.
            class FakeLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(4, 4)
                def forward(self, x):
                    return (self.linear(x),)
            inner.layers = nn.ModuleList([FakeLayer()])
            self.model = inner
            self.lm_head = nn.Linear(4, 100)  # project to vocab
        def get_input_embeddings(self):
            return self.embed
        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)[0]
            logits = self.lm_head(x)  # [batch, seq, vocab=100]
            return type("O", (), {"logits": logits})()

    model = FakeModel()
    tok = FakeTok()
    spec = SteeringSpec(vector_id="v", family="f", behavior="b", layer=0, alpha=1.0, hidden_size=4)
    vec = SteeringVector(spec=spec, values=np.ones(4, dtype=np.float32))
    sink: list = []
    score_route_batch_from_logits(
        ["ACTION:"], model, tok, vector=vec, telemetry_sink=sink
    )
    assert len(sink) >= 1
    assert "h_norm_mean" in sink[0]
