from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from daph_learning.routing.steered_router import route_action_from_logits
from daph_learning.steering.hooks import multi_layer_residual_addition_hook


def test_route_action_from_logits():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.0, 3.5, 1.0, -2.0])
    action, margin = route_action_from_logits(
        logits,
        symbolic_token_id=1,
        llm_token_id=2,
    )
    assert action == "symbolic"
    assert margin == pytest.approx(2.5)

    action, margin = route_action_from_logits(
        logits,
        symbolic_token_id=1,
        llm_token_id=2,
        threshold=3.0,
    )
    assert action == "llm"
    assert margin == pytest.approx(2.5)


class _Layer:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)

        class Handle:
            def remove(_self):
                self.hooks.remove(hook)

        return Handle()

    def run(self, hidden):
        output = hidden
        for hook in list(self.hooks):
            output = hook(None, None, output)
        return output


class _Inner:
    def __init__(self):
        self.layers = [_Layer(), _Layer()]


class _Config:
    hidden_size = 2
    _name_or_path = "mock"


class _Model:
    def __init__(self):
        self.model = _Inner()
        self.config = _Config()


def test_multi_layer_residual_addition_hook():
    torch = pytest.importorskip("torch")
    model = _Model()
    hidden = torch.zeros((1, 3, 2), dtype=torch.float32)
    configs = [
        {
            "layer_index": 0,
            "vector": np.array([1.0, 0.0], dtype=np.float32),
            "alpha": 2.0,
            "token_scope": "last",
            "target_token_index": 1,
        },
        {
            "layer_index": 1,
            "vector": np.array([0.0, 1.0], dtype=np.float32),
            "alpha": 3.0,
            "token_scope": "last",
            "target_token_index": 1,
        },
    ]
    with multi_layer_residual_addition_hook(model, configs):
        out0 = model.model.layers[0].run(hidden)
        out1 = model.model.layers[1].run(out0)

    assert torch.equal(out1[0, 1], torch.tensor([2.0, 3.0]))
    assert torch.equal(out1[0, 0], torch.zeros(2))
    assert torch.equal(out1[0, 2], torch.zeros(2))


def test_batch_specific_target_indices():
    torch = pytest.importorskip("torch")
    model = _Model()
    hidden = torch.zeros((2, 4, 2), dtype=torch.float32)
    cfg = [{
        "layer_index": 0,
        "vector": np.array([1.0, 1.0], dtype=np.float32),
        "alpha": 1.0,
        "token_scope": "last",
        "target_token_index": [1, 2],
    }]
    with multi_layer_residual_addition_hook(model, cfg):
        out = model.model.layers[0].run(hidden)
    assert torch.equal(out[0, 1], torch.ones(2))
    assert torch.equal(out[1, 2], torch.ones(2))
    assert torch.equal(out[0, 2], torch.zeros(2))
    assert torch.equal(out[1, 1], torch.zeros(2))


def test_generate_ood_benchmark_determinism(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_ood_benchmark.py"
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    for output in (a, b):
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--output",
                str(output),
                "--num-tasks",
                "505",
                "--seed",
                "1234",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()

    rows = [json.loads(line) for line in a.read_text(encoding="utf-8").splitlines()]
    counts = {}
    for row in rows:
        counts[row["metadata"]["category"]] = counts.get(row["metadata"]["category"], 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1
