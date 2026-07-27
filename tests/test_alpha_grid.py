"""Tests for independent per-vector composite coefficients (--alpha-grid).

See CLAIMS.md §10 and audit §10: the v0.3.4 tune_steering.py only supported
a global multiplier over stored alphas in composite bundle mode. v0.3.5
adds --alpha-grid for independent per-vector coefficient sweeping.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts.tune_steering import main as tune_main


def _write_tasks(path: Path, n: int = 4) -> None:
    """Write n simple arithmetic tasks with route_label=symbolic."""
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "task_id": f"t{i}",
                "capability_ids": ["integer_arithmetic"],
                "inputs": {"a": i + 1, "b": i + 2, "op": "+"},
                "specification": f"Compute {i+1} + {i+2}. Return only the integer.",
                "expected": (i + 1) + (i + 2),
                "route_label": "symbolic",
                "capability_oracle": "symbolic",
                "accuracy_oracle": "llm",
                "utility_oracle": "symbolic",
            }) + "\n")


def _make_bundle(tmp_path: Path, n_vectors: int = 2, model_id: str = "sshleifer/tiny-gpt2") -> str:
    """Create a composite bundle of n_vectors .npz files and return the
    comma-separated path string."""
    import numpy as np
    from daph_learning.steering.types import SteeringSpec, SteeringVector
    from daph_learning.steering.io import save_vector

    paths = []
    for i in range(n_vectors):
        spec = SteeringSpec(
            vector_id=f"v{i}",
            family="tool_policy",
            behavior="invoke_symbolic_tool",
            layer=i,
            alpha=1.0,
            anchor="ACTION",
            model_id=model_id,
            hidden_size=4,
        )
        vec = SteeringVector(spec=spec, values=np.ones(4, dtype=np.float32))
        p = tmp_path / f"v{i}.npz"
        save_vector(vec, p)
        paths.append(str(p))
    return ",".join(paths)


def test_alpha_grid_requires_bundle(tmp_path: Path, monkeypatch):
    """--alpha-grid without --vector-bundle must error."""
    # We can't easily test this without a model, so just check arg parsing.
    # Actually, the validation happens after model load. Let's test the
    # validation logic directly.
    monkeypatch.setattr(sys, "argv", [
        "tune_steering.py",
        "--model", "fake",
        "--val-tasks", str(tmp_path / "tasks.jsonl"),
        "--vector-pattern", str(tmp_path / "v{layer}.npz"),
        "--layers", "0",
        "--alpha-grid", "0.5,1.0",
        "--output", str(tmp_path / "out.jsonl"),
    ])
    with pytest.raises((ValueError, SystemExit), match="alpha-grid requires --vector-bundle|error:"):
        tune_main()


def test_alpha_grid_wrong_arity_rejected(tmp_path: Path, monkeypatch):
    """A grid entry with the wrong number of alphas must be rejected."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    _write_tasks(tmp_path / "tasks.jsonl")
    bundle = _make_bundle(tmp_path, n_vectors=2, model_id="fake-model")

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
            inner.layers = nn.ModuleList([FakeLayer(), FakeLayer()])
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

    import scripts.tune_steering as ts
    monkeypatch.setattr(ts, "_load_llm", lambda _m: (model, tok))

    monkeypatch.setattr(sys, "argv", [
        "tune_steering.py",
        "--model", "fake-model",
        "--val-tasks", str(tmp_path / "tasks.jsonl"),
        "--vector-bundle", bundle,
        "--alpha-grid", "0.5,1.0,2.0",  # 3 values but bundle has 2 vectors
        "--prompt-format", "raw",
        "--output", str(tmp_path / "out.jsonl"),
        "--no-manifest",
    ])
    with pytest.raises(ValueError, match="alpha-grid entry.*has 3 values.*bundle has 2"):
        tune_main()


def test_alpha_grid_sweeps_independent_configurations(tmp_path: Path, monkeypatch):
    """--alpha-grid should produce one result per grid entry with the
    specified per-vector alphas."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import numpy as np

    _write_tasks(tmp_path / "tasks.jsonl")
    bundle = _make_bundle(tmp_path, n_vectors=2, model_id="fake-model")
    output = tmp_path / "out.jsonl"

    # Build a fake model + tokenizer with single-token SYMBOLIC/LLM labels.
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
            inner.layers = nn.ModuleList([FakeLayer(), FakeLayer()])
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

    import scripts.tune_steering as ts
    monkeypatch.setattr(ts, "_load_llm", lambda _m: (model, tok))

    monkeypatch.setattr(sys, "argv", [
        "tune_steering.py",
        "--model", "fake-model",
        "--val-tasks", str(tmp_path / "tasks.jsonl"),
        "--vector-bundle", bundle,
        "--alpha-grid", "0.5,1.0", "1.0,2.0", "2.0,0.5",
        "--prompt-format", "raw",
        "--output", str(output),
        "--no-manifest",
    ])
    tune_main()

    # The output file is a single JSON object with a "results" array.
    payload = json.loads(output.read_text(encoding="utf-8"))
    lines = payload["results"]
    assert len(lines) == 3  # three grid entries
    # Each result should have scale_semantics=independent_per_vector
    for line in lines:
        assert line["scale_semantics"] == "independent_per_vector"
    # The effective_alphas should match the grid entries
    assert lines[0]["effective_alphas"] == [0.5, 1.0]
    assert lines[1]["effective_alphas"] == [1.0, 2.0]
    assert lines[2]["effective_alphas"] == [2.0, 0.5]
    # sweep_alpha should be the grid entry (as a list)
    assert lines[0]["sweep_alpha"] == [0.5, 1.0]
    assert lines[1]["sweep_alpha"] == [1.0, 2.0]


def test_alpha_grid_non_finite_rejected(tmp_path: Path, monkeypatch):
    """Non-finite values in --alpha-grid must be rejected at parse time."""
    _write_tasks(tmp_path / "tasks.jsonl")
    bundle = _make_bundle(tmp_path, n_vectors=2)
    monkeypatch.setattr(sys, "argv", [
        "tune_steering.py",
        "--model", "fake",
        "--val-tasks", str(tmp_path / "tasks.jsonl"),
        "--vector-bundle", bundle,
        "--alpha-grid", "0.5,inf",
        "--output", str(tmp_path / "out.jsonl"),
    ])
    with pytest.raises((ValueError, SystemExit), match="non-finite|error:"):
        tune_main()


def test_alpha_grid_empty_entry_rejected(tmp_path: Path, monkeypatch):
    _write_tasks(tmp_path / "tasks.jsonl")
    bundle = _make_bundle(tmp_path, n_vectors=2)
    monkeypatch.setattr(sys, "argv", [
        "tune_steering.py",
        "--model", "fake",
        "--val-tasks", str(tmp_path / "tasks.jsonl"),
        "--vector-bundle", bundle,
        "--alpha-grid", "",
        "--output", str(tmp_path / "out.jsonl"),
    ])
    with pytest.raises((ValueError, SystemExit), match="parsed to no values|error:"):
        tune_main()
