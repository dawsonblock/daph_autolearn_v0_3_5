"""Tests for v0.3.8 Task 3.2 functional margin calibration (calibrate_margin.py).

The calibration probe loads a frozen model and computes
    m = 0.5 * (r_shuffled_mean - r_matched_mean)
on a small probe set with an untrained instance encoder. These tests
validate the calibration logic without requiring a real HuggingFace model
by monkeypatching ``load_frozen_model`` to return a tiny fake model + a
fake tokenizer that emulates the tokenize_batch contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Emulates the subset of the HF tokenizer API used by the calibration
    probe and tokenize_batch: __call__(return_tensors='pt', padding,
    truncation, max_length) and plain __call__ for state text."""
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<pad>"

    def __init__(self, vocab_size=30, hidden=8):
        self._vocab = vocab_size
        self._hidden = hidden

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False,
                 max_length=None, add_special_tokens=True):
        if isinstance(texts, str):
            ids = self._encode_one(texts)
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([[1] * len(ids)])}
        batch = [self._encode_one(t) for t in texts]
        max_len = max(len(b) for b in batch)
        padded = [b + [0] * (max_len - len(b)) for b in batch]
        masks = [[1] * len(b) + [0] * (max_len - len(b)) for b in batch]
        ids = torch.tensor(padded)
        attn = torch.tensor(masks)
        if return_tensors == "pt":
            return {"input_ids": ids, "attention_mask": attn}
        return {"input_ids": padded, "attention_mask": masks}

    def _encode_one(self, text: str) -> list[int]:
        # Deterministic pseudo-tokenization: hash each char to a vocab id.
        return [1 + (ord(c) % (self._vocab - 1)) for c in text[:16]] + [2]

    def __len__(self):
        return self._vocab


class _FakeEmbed(nn.Module):
    def __init__(self, vocab=30, hidden=8):
        super().__init__()
        self.embedding_dim = hidden
        self.weight = nn.Parameter(torch.randn(vocab, hidden) * 0.1)

    def forward(self, ids):
        return self.weight[ids]


class _FakeModel(nn.Module):
    """Tiny causal-LM-shaped model with a transformer-like forward that
    returns logits so compute_R can run cross-entropy."""

    def __init__(self, vocab=30, hidden=8, num_layers=2):
        super().__init__()
        self._embed = _FakeEmbed(vocab, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        class _Cfg:
            hidden_size = hidden
            num_hidden_layers = num_layers
            _name_or_path = "fake"

        self.config = _Cfg()

    def get_input_embeddings(self):
        return self._embed

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                inputs_embeds=None, use_cache=False, **kwargs):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self._embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return type("O", (), {"logits": logits, "loss": loss})()


# ---------------------------------------------------------------------------
# Test fixture: monkeypatch load_frozen_model + build a tiny dataset
# ---------------------------------------------------------------------------

CFG = {
    "seed": 1337,
    "model": {"name": "fake", "dtype": "float32", "trust_remote_code": True, "device_map": "cpu"},
    "memory": {
        "latent_tokens": 4, "encoder_width": 16, "encoder_layers": 1, "dropout": 0.0,
        "max_state_tokens": 32, "skill_dim": 8, "instance_dim": 8, "num_skills": 6,
        "composition": "gated",
    },
    "training": {
        "epochs": 1, "batch_size": 2, "learning_rate": 1e-4, "weight_decay": 0.0,
        "answer_weight": 1.0, "functional_weight": 0.5, "disentangle_weight": 0.1,
        "invariance_weight": 0.2, "functional_margin": 0.0, "functional_negatives": 3,
        "max_prompt_tokens": 64, "max_answer_tokens": 16, "gradient_clip": 1.0,
        "unfreeze_fraction": 0.3, "latent_relative_norm_limit": 0.0,
    },
    "evaluation": {"generation_max_new_tokens": 16, "seed_count": 5},
}


def _make_dataset_jsonl(path: Path, n: int = 20):
    from daph_latent_memory.state.schema import MathExample, MathWorkingState
    examples = []
    for i in range(n):
        split = "iid" if i < n // 2 else "train"
        ex = MathExample(
            example_id=f"ex-{i}",
            split=split,
            question=f"What is {i} plus {i+1}?",
            answer=str(2 * i + 1),
            state=MathWorkingState(
                domain="arithmetic",
                operation="add",
                variables={"a": i, "b": i + 1},
                constraints=[], retrieved_concepts=[],
                intermediate_steps=[], unresolved_subgoals=[],
            ),
            template="add_template",
            difficulty=1,
            skill_label="add",
        )
        examples.append(ex)
    with open(path, "w") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
    return path


@pytest.fixture
def patched_calibrate(monkeypatch, tmp_path):
    """Monkeypatch load_frozen_model to return a fake model + tokenizer,
    and provide a tiny dataset on disk."""
    dataset_path = _make_dataset_jsonl(tmp_path / "dataset.jsonl", n=20)

    def _fake_load(cfg):
        model = _FakeModel(vocab=30, hidden=8, num_layers=2)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, _FakeTokenizer(vocab_size=30, hidden=8)

    import daph_latent_memory.runtime as runtime
    monkeypatch.setattr(runtime, "load_frozen_model", _fake_load)

    # The calibrate module imports load_frozen_model at call time via the
    # runtime module, so monkeypatching runtime.load_frozen_model is enough.
    return dataset_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_calibrate_returns_required_fields(patched_calibrate):
    from scripts.calibrate_margin import calibrate
    report = calibrate(CFG, str(patched_calibrate), probe_size=8, seed=1337)
    for key in ("functional_margin", "r_matched_mean", "r_shuffled_mean",
                "gap", "n_probe", "used_floor", "seed"):
        assert key in report, f"missing field {key}"
    assert report["n_probe"] == 8
    assert report["seed"] == 1337


def test_calibrate_margin_is_half_gap(patched_calibrate):
    from scripts.calibrate_margin import calibrate
    report = calibrate(CFG, str(patched_calibrate), probe_size=8, seed=1337)
    if not report["used_floor"]:
        expected = 0.5 * report["gap"]
        assert abs(report["functional_margin"] - expected) < 1e-6, (
            f"margin should be 0.5*gap={expected}, got {report['functional_margin']}"
        )


def test_calibrate_uses_floor_when_gap_nonpositive(monkeypatch, patched_calibrate):
    """If the matched NLL is >= shuffled NLL (no causal signal at init),
    the margin must fall back to the positive floor."""
    from scripts import calibrate_margin as cm

    # Force a non-positive gap by patching compute_R to return equal values.
    def _flat_R(*args, **kwargs):
        return torch.zeros(1)
    monkeypatch.setattr(cm, "compute_R", _flat_R)
    report = cm.calibrate(CFG, str(patched_calibrate), probe_size=8, seed=1337)
    assert report["gap"] == 0.0
    assert report["used_floor"] is True
    assert report["functional_margin"] == cm.DEFAULT_FLOOR_MARGIN


def test_calibrate_main_writes_json(monkeypatch, patched_calibrate, tmp_path):
    """The CLI main() should write a valid JSON report to --output."""
    import sys
    from scripts import calibrate_margin as cm

    out = tmp_path / "calib.json"
    sys.argv = [
        "calibrate_margin.py",
        "--config", str(tmp_path / "nonexistent.yaml"),  # not read; we patch load
        "--dataset", str(patched_calibrate),
        "--probe-size", "6",
        "--output", str(out),
    ]
    # Patch load_config to return our CFG.
    monkeypatch.setattr(cm, "load_config", lambda _p: CFG)
    cm.main()
    assert out.exists()
    with open(out) as f:
        report = json.load(f)
    assert "functional_margin" in report
    assert report["n_probe"] == 6
