"""Tests for the v0.3.6 Phase 2.1 empirical oracle builder.

See scripts/build_empirical_oracles.py. These tests cover the
derivation logic (``_derive_oracles``) and the symbolic measurement
(``_measure_symbolic``) which are pure and do not require a real model.
The LLM measurement path is exercised via a tiny fake model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_empirical_oracles import (
    _derive_oracles,
    _measure_symbolic,
    build_empirical_oracles,
)


def _sym(ran, value=None, latency=1.0, error=None):
    return {
        "symbolic_ran": ran,
        "symbolic_value": value,
        "symbolic_latency_ms": latency,
        "symbolic_error": error,
    }


def _llm(ran, parsed=None, correct=None, latency=100.0, error=None):
    return {
        "llm_ran": ran,
        "llm_text": None,
        "llm_parsed_int": parsed,
        "llm_correct": correct,
        "llm_latency_ms": latency,
        "llm_error": error,
    }


def test_capability_oracle_symbolic_when_executor_succeeds():
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    out = _derive_oracles(task, _sym(True, 42), _llm(False), symbolic_latency_budget_ms=50.0)
    assert out["capability_oracle"] == "symbolic"


def test_capability_oracle_llm_when_executor_raises():
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    out = _derive_oracles(task, _sym(False, error="ValueError: x"), _llm(False), symbolic_latency_budget_ms=50.0)
    assert out["capability_oracle"] == "llm"


def test_accuracy_oracle_symbolic_when_llm_wrong():
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    out = _derive_oracles(task, _sym(True, 42), _llm(True, parsed=99, correct=False), symbolic_latency_budget_ms=50.0)
    assert out["accuracy_oracle"] == "symbolic"


def test_accuracy_oracle_llm_when_symbolic_wrong():
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    out = _derive_oracles(task, _sym(True, 41), _llm(True, parsed=42, correct=True), symbolic_latency_budget_ms=50.0)
    assert out["accuracy_oracle"] == "llm"


def test_accuracy_oracle_symbolic_on_tie_both_correct():
    """When both backends get the right answer, prefer symbolic (exact + cheaper)."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    out = _derive_oracles(task, _sym(True, 42), _llm(True, parsed=42, correct=True), symbolic_latency_budget_ms=50.0)
    assert out["accuracy_oracle"] == "symbolic"


def test_accuracy_oracle_llm_for_non_numeric_task():
    """Tasks with no expected integer answer route to LLM for accuracy."""
    task = {"task_id": "t1", "expected": None, "capability_ids": []}
    out = _derive_oracles(task, _sym(False, error="unsupported_capability"), _llm(True), symbolic_latency_budget_ms=50.0)
    assert out["accuracy_oracle"] == "llm"


def test_utility_oracle_symbolic_when_fast_and_correct():
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    out = _derive_oracles(task, _sym(True, 42, latency=2.0), _llm(True, parsed=42, correct=True, latency=500.0), symbolic_latency_budget_ms=50.0)
    assert out["utility_oracle"] == "symbolic"


def test_utility_oracle_llm_when_symbolic_too_slow():
    """Even if symbolic is correct, if it exceeds the latency budget and the
    LLM is also correct, utility prefers the LLM."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    out = _derive_oracles(task, _sym(True, 42, latency=200.0), _llm(True, parsed=42, correct=True, latency=100.0), symbolic_latency_budget_ms=50.0)
    assert out["utility_oracle"] == "llm"


def test_utility_oracle_symbolic_when_llm_wrong_even_if_slow():
    """If the LLM is wrong, symbolic wins utility regardless of latency."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    out = _derive_oracles(task, _sym(True, 42, latency=999.0), _llm(True, parsed=99, correct=False), symbolic_latency_budget_ms=50.0)
    assert out["utility_oracle"] == "symbolic"


def test_empirical_route_label_equals_utility_oracle():
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    out = _derive_oracles(task, _sym(True, 42), _llm(False), symbolic_latency_budget_ms=50.0)
    assert out["empirical_route_label"] == out["utility_oracle"]


def test_empirical_provenance_recorded():
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    out = _derive_oracles(task, _sym(True, 42), _llm(True, parsed=42, correct=True), symbolic_latency_budget_ms=50.0)
    assert out["empirical"]["script_version"] == "v0.3.6"
    assert out["empirical"]["symbolic_correct"] is True
    assert out["empirical"]["llm_correct"] is True


def test_measure_symbolic_runs_supported_capability():
    task = {
        "task_id": "t1",
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": 12, "b": 34, "op": "*"},
        "specification": "Compute 12 * 34.",
        "expected": 408,
    }
    m = _measure_symbolic(task)
    assert m["symbolic_ran"] is True
    assert m["symbolic_value"] == 408
    assert m["symbolic_error"] is None
    assert m["symbolic_latency_ms"] >= 0.0


def test_measure_symbolic_unsupported_capability_skips():
    task = {
        "task_id": "t1",
        "capability_ids": ["code_generation"],
        "specification": "Write a function.",
        "expected": None,
    }
    m = _measure_symbolic(task)
    assert m["symbolic_ran"] is False
    assert m["symbolic_error"] == "unsupported_capability"


def test_measure_symbolic_records_error_on_malformed():
    task = {
        "task_id": "t1",
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": 5, "b": 0, "op": "/"},
        "specification": "Compute 5 / 0.",
        "expected": None,
    }
    m = _measure_symbolic(task)
    assert m["symbolic_ran"] is False
    assert m["symbolic_error"] is not None


def test_build_empirical_oracles_symbolic_only_mode():
    """Without a model, the builder runs in symbolic-only mode and still
    produces annotated tasks with empirical oracle fields."""
    tasks = [
        {
            "task_id": "t1",
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": 2, "b": 3, "op": "+"},
            "specification": "Compute 2 + 3.",
            "expected": 5,
            "route_label": "symbolic",
        },
        {
            "task_id": "t2",
            "capability_ids": [],
            "specification": "Explain the sky.",
            "expected": None,
            "route_label": "llm",
        },
    ]
    annotated, measurements = build_empirical_oracles(
        tasks, model=None, tokenizer=None, run_llm=False,
    )
    assert len(annotated) == 2
    assert len(measurements) == 2
    # t1: symbolic ran and got 5 -> capability=symbolic, accuracy=symbolic
    assert annotated[0]["capability_oracle"] == "symbolic"
    assert annotated[0]["accuracy_oracle"] == "symbolic"
    assert annotated[0]["utility_oracle"] == "symbolic"
    assert annotated[0]["empirical_route_label"] == "symbolic"
    assert annotated[0]["oracle_provenance"] == "empirical_v0_3_6"
    # t2: no supported capability -> capability=llm, accuracy=llm
    assert annotated[1]["capability_oracle"] == "llm"
    assert annotated[1]["accuracy_oracle"] == "llm"
    # Original route_label preserved
    assert annotated[0]["route_label"] == "symbolic"
    assert annotated[1]["route_label"] == "llm"
    # Measurements sidecar records the raw signals
    assert measurements[0]["task_id"] == "t1"
    assert measurements[0]["symbolic_ran"] is True
    assert measurements[0]["symbolic_value"] == 5
    assert measurements[1]["llm_ran"] is False


def test_build_empirical_oracles_overwrites_heuristic_labels():
    """The empirical builder must overwrite the v0.3.5 heuristic oracle
    fields, not append to them. This is the whole point of Phase 2.1."""
    tasks = [
        {
            "task_id": "t1",
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": 2, "b": 3, "op": "+"},
            "specification": "Compute 2 + 3.",
            "expected": 5,
            # v0.3.5 heuristic labels (wrong for this task — symbolic is
            # actually capable and correct here, but the heuristic
            # generator marked easy_add as accuracy=llm).
            "capability_oracle": "symbolic",
            "accuracy_oracle": "llm",
            "utility_oracle": "llm",
            "route_label": "llm",
        },
    ]
    annotated, _ = build_empirical_oracles(
        tasks, model=None, tokenizer=None, run_llm=False,
    )
    # Empirical measurement: symbolic ran and produced 5 == expected.
    # accuracy_oracle must now be "symbolic", overwriting the heuristic "llm".
    assert annotated[0]["accuracy_oracle"] == "symbolic"
    assert annotated[0]["utility_oracle"] == "symbolic"


class _FakeLLM:
    """A fake model/tokenizer pair that always emits FINAL: <expected>."""

    def __init__(self, correct: bool = True):
        self._correct = correct
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.pad_token = "<pad>"
        self.chat_template = None

        class _Cfg:
            hidden_size = 4
            _name_or_path = "fake"
        self.config = _Cfg()

        import torch
        self._torch = torch
        self._embed = type("E", (), {"weight": torch.zeros((4, 2))})()

    def get_input_embeddings(self):
        return self._embed

    def generate(self, input_ids, **kwargs):
        t = self._torch
        # Append a single fake token; the tokenizer.decode below ignores it.
        extra = t.full((input_ids.shape[0], 1), 99, dtype=input_ids.dtype, device=input_ids.device)
        return t.cat([input_ids, extra], dim=1)

    def __call__(self, text, **kwargs):
        t = self._torch
        ids = [ord(ch) % 50 + 1 for ch in text]
        return {"input_ids": t.tensor([ids]), "attention_mask": t.ones(1, len(ids), dtype=t.long)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def decode(self, tokens, skip_special_tokens=True):
        return "FINAL: 5" if self._correct else "FINAL: 99"


def test_build_empirical_oracles_with_fake_llm_correct():
    tasks = [
        {
            "task_id": "t1",
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": 2, "b": 3, "op": "+"},
            "specification": "Compute 2 + 3.",
            "expected": 5,
            "route_label": "symbolic",
        },
    ]
    fake = _FakeLLM(correct=True)
    annotated, measurements = build_empirical_oracles(
        tasks, model=fake, tokenizer=fake, run_llm=True,
        prompt_format="raw", max_new_tokens=8,
    )
    # Both backends correct -> accuracy=symbolic (preferred on tie)
    assert annotated[0]["accuracy_oracle"] == "symbolic"
    assert measurements[0]["llm_ran"] is True
    assert measurements[0]["llm_correct"] is True


def test_build_empirical_oracles_with_fake_llm_wrong():
    tasks = [
        {
            "task_id": "t1",
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": 2, "b": 3, "op": "+"},
            "specification": "Compute 2 + 3.",
            "expected": 5,
            "route_label": "symbolic",
        },
    ]
    fake = _FakeLLM(correct=False)  # LLM emits FINAL: 99
    annotated, measurements = build_empirical_oracles(
        tasks, model=fake, tokenizer=fake, run_llm=True,
        prompt_format="raw", max_new_tokens=8,
    )
    # Symbolic correct, LLM wrong -> accuracy=symbolic
    assert annotated[0]["accuracy_oracle"] == "symbolic"
    assert measurements[0]["llm_correct"] is False


def test_build_empirical_oracles_cli_writes_files(tmp_path: Path):
    """End-to-end: the CLI main() writes the annotated JSONL and the
    measurements sidecar in symbolic-only mode."""
    import sys
    from scripts.build_empirical_oracles import main

    inp = tmp_path / "in.jsonl"
    inp.write_text(
        json.dumps({
            "task_id": "t1",
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": 2, "b": 3, "op": "+"},
            "specification": "Compute 2 + 3.",
            "expected": 5,
        }) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    old_argv = sys.argv
    sys.argv = [
        "build_empirical_oracles.py",
        "--input", str(inp),
        "--output", str(out),
        "--symbolic-only",
    ]
    try:
        main()
    finally:
        sys.argv = old_argv
    assert out.exists()
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[0]["accuracy_oracle"] == "symbolic"
    meas = tmp_path / (out.name + ".measurements.jsonl")
    assert meas.exists()
