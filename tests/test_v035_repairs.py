from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from daph_learning.evaluation.paired import (
    clopper_pearson_interval,
    paired_discordant_analysis,
)
from daph_learning.routing.steered_router import resolve_route_token_ids
from daph_learning.steering.io import (
    load_vector_bundle,
    save_vector,
)
from daph_learning.steering.types import SteeringSpec, SteeringVector
from scripts.generate_v0_outputs import _generate_batch


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {
            " SYMBOLIC": [101],
            " LLM": [202],
            "SYMBOLIC": [101, 102],
            "LLM": [202, 203],
        }
        return mapping.get(text, [999])


def test_resolve_route_token_ids_autodetect():
    tok = DummyTokenizer()
    sym_id, llm_id = resolve_route_token_ids(tok, leading_space=None)
    assert sym_id == 101
    assert llm_id == 202


class RawOnlyTokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {
            " SYMBOLIC": [1, 2],
            " LLM": [3, 4],
            "SYMBOLIC": [11],
            "LLM": [22],
        }
        return mapping[text]


def test_resolve_route_token_ids_autodetect_falls_back_to_raw():
    assert resolve_route_token_ids(RawOnlyTokenizer(), leading_space=None) == (11, 22)


def test_paired_discordant_analysis():
    result = paired_discordant_analysis(7, 2)
    assert result["p_value"] == pytest.approx(0.1796875)
    assert result["odds_ratio"] == pytest.approx(3.5)
    assert result["mcnemar_statistic"] == pytest.approx(16.0 / 9.0)
    assert 0.0 <= result["mcnemar_p_value"] <= 1.0
    lo, hi = result["improved_probability_ci_95"]
    assert lo < 7 / 9 < hi
    assert lo < 0.5 < hi


def test_mcnemar_equal_discordance_has_zero_corrected_statistic():
    result = paired_discordant_analysis(4, 4)
    assert result["mcnemar_statistic"] == 0.0
    assert result["mcnemar_p_value"] == 1.0


def test_clopper_pearson_zero_successes():
    lo, hi = clopper_pearson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(1.0 - 0.025 ** (1.0 / 10.0), rel=1e-8)


def _vector(layer: int, alpha: float, vector_id: str) -> SteeringVector:
    return SteeringVector(
        SteeringSpec(
            vector_id=vector_id,
            family="tool_policy",
            behavior="invoke_symbolic_tool",
            layer=layer,
            alpha=alpha,
            hidden_size=3,
        ),
        np.array([1.0, 0.0, -1.0], dtype=np.float32),
    )


def test_load_vector_bundle_comma_and_json(tmp_path: Path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    save_vector(_vector(2, 1.0, "a"), a)
    save_vector(_vector(5, 1.5, "b"), b)

    comma = load_vector_bundle(f"{a},{b}")
    assert [v.spec.layer for v in comma] == [2, 5]

    manifest = tmp_path / "bundle.json"
    manifest.write_text(
        json.dumps(
            {
                "vectors": [
                    {"path": "a.npz", "alpha": 0.25},
                    {"path": "b.npz", "alpha": 0.75},
                ]
            }
        ),
        encoding="utf-8",
    )
    bundle = load_vector_bundle(str(manifest))
    assert [v.spec.layer for v in bundle] == [2, 5]
    assert [v.spec.alpha for v in bundle] == pytest.approx([0.25, 0.75])


def test_load_vector_bundle_rejects_duplicate_layers(tmp_path: Path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    save_vector(_vector(2, 1.0, "a"), a)
    save_vector(_vector(2, 1.5, "b"), b)
    with pytest.raises(ValueError, match="duplicate layers"):
        load_vector_bundle(f"{a},{b}")


class _FakeEmbedding:
    def __init__(self, torch):
        self.weight = torch.zeros((4, 2))


class _FakeModel:
    def __init__(self, torch):
        self._torch = torch
        self._embedding = _FakeEmbedding(torch)

    def get_input_embeddings(self):
        return self._embedding

    def generate(self, input_ids, attention_mask=None, **kwargs):
        batch = input_ids.shape[0]
        generated = self._torch.full(
            (batch, 1),
            99,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return self._torch.cat([input_ids, generated], dim=1)


class _FakeBatchTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    padding_side = "right"

    def __init__(self, torch):
        self.torch = torch

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            ids = [ord(ch) % 50 + 1 for ch in texts]
            return {"input_ids": ids}
        rows = [[ord(ch) % 50 + 1 for ch in text] for text in texts]
        width = max(len(row) for row in rows)
        padded = []
        masks = []
        for row in rows:
            pad = [0] * (width - len(row))
            if self.padding_side == "left":
                padded.append(pad + row)
                masks.append([0] * len(pad) + [1] * len(row))
            else:
                padded.append(row + pad)
                masks.append([1] * len(row) + [0] * len(pad))
        return {
            "input_ids": self.torch.tensor(padded, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(masks, dtype=self.torch.long),
        }

    def decode(self, tokens, skip_special_tokens=True):
        values = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
        return "FINAL: 42" if values == [99] else "?"


def test_generate_batch_returns_one_completion_per_prompt():
    torch = pytest.importorskip("torch")
    model = _FakeModel(torch)
    tokenizer = _FakeBatchTokenizer(torch)
    outputs = _generate_batch(
        ["a", "longer", "mid"],
        model,
        tokenizer,
        max_new_tokens=4,
        seed=123,
        prompt_format="raw",
    )
    assert outputs == ["FINAL: 42", "FINAL: 42", "FINAL: 42"]


def test_gate_emits_expanded_paired_telemetry(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"

    task_rows = []
    base_rows = []
    treat_rows = []
    # Construct 7 improvements and 2 regressions exactly.
    for i in range(9):
        tid = f"t{i}"
        task_rows.append({"task_id": tid, "expected": 1})
        if i < 7:
            base_rows.append({"task_id": tid, "output": "0"})
            treat_rows.append({"task_id": tid, "output": "1"})
        else:
            base_rows.append({"task_id": tid, "output": "1"})
            treat_rows.append({"task_id": tid, "output": "0"})

    for path, rows in (
        (tasks, task_rows),
        (baseline, base_rows),
        (treatment, treat_rows),
    ):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    script = Path(__file__).resolve().parents[1] / "scripts" / "v0_hint_ablation_gate.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--tasks",
            str(tasks),
            "--baseline-outputs",
            str(baseline),
            "--treatment-outputs",
            str(treatment),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["p_value_two_sided"] == pytest.approx(0.1796875)
    assert payload["discordant_odds_ratio"] == pytest.approx(3.5)
    assert payload["mcnemar_statistic_continuity_corrected"] == pytest.approx(16 / 9)
    assert len(payload["discordant_improved_probability_ci_95"]) == 2
